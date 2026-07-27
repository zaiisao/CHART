"""WHICH TERM prefers the wrong tempo? Reuses wire_test's gate-validated free-q machinery
(its free ELBO matched innovq's to 0.0017 nats).

Start at ORACLE tempo, optimize the SAME ELBO, and split the change into:
  recon_beat, recon_db, recon_obs | kl_phase_innov, kl_level, kl_meter
The term that DROPS most is what pays for the move.

  KL drops most   -> the PRIOR breaks the tie the wrong way; reshaping it is the fix.
  recon drops most-> the LIKELIHOOD prefers wrong tempo; no prior can fix it.
"""
import sys, math, torch, torch.nn.functional as F
for p in ("/home/sogang/jaehoon/VBPM_reintegration","/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix",
          "/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq","/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem"):
    sys.path.insert(0,p)
import pm_common as P, innovq as IQ
from wire_test import free_rollout
from rollout_vec_s import draw_noise
dev="cuda:0"; torch.manual_seed(0); T=1500; TWO_PI=2*math.pi; K=4
GP=0.06; P.PHYS["gamma_phase"]=GP; IQ.RHO_P=math.exp(-GP); IQ.GP1=1-IQ.RHO_P; IQ.R0=IQ.softplus_inv(IQ.GP1)
tr=P.build_crops(P.load_songs("train"),n_per_song=1,seed=0,crop=T,dev=dev); N=tr["b"].shape[0]
sel=torch.arange(48,device=dev); B=len(sel)
h,b,db,obs=tr["h"][sel],tr["b"][sel],tr["db"][sel],tr["obs"][sel]
phi_true,lt_frame=tr["phi"][sel],tr["lt"][sel]; lt_true=lt_frame.mean(1)
mo=F.one_hot(tr["m"].long().clamp(0,K-1),K).float()
if mo.dim()==2: mo=mo.unsqueeze(1).expand(-1,T,-1)
Zor=torch.cat([torch.cos(tr["phi"]).unsqueeze(-1),torch.sin(tr["phi"]).unsqueeze(-1),
               tr["lt"].unsqueeze(-1),mo],-1)
d0,h0=P.new_decoders(dev); dec,hdec=IQ.Cut(d0),IQ.Cut(h0)
od=torch.optim.Adam(list(d0.parameters())+list(h0.parameters()),lr=3e-3)
for s in range(700):
    od.zero_grad(); ss=torch.randperm(N,device=dev)[:16]
    a,b_,c_=P.recon_terms(dec,hdec,Zor[ss],tr["b"][ss],tr["db"][ss],tr["obs"][ss],recon="bce")
    (a+b_+c_).mean().backward(); od.step()
with torch.no_grad():
    g=float(sum(P.recon_terms(dec,hdec,Zor[:32],tr["b"][:32],tr["db"][:32],tr["obs"][:32],recon="bce")[i].mean() for i in (0,1)))
    scr=(tr["phi"][:32]+0.05*torch.arange(T,device=dev).float())%TWO_PI
    Zb=torch.cat([torch.cos(scr).unsqueeze(-1),torch.sin(scr).unsqueeze(-1),tr["lt"][:32].unsqueeze(-1),mo[:32]],-1)
    bd=float(sum(P.recon_terms(dec,hdec,Zb,tr["b"][:32],tr["db"][:32],tr["obs"][:32],recon="bce")[i].mean() for i in (0,1)))
sens=100*(bd-g)/abs(g); print(f"[gate] decoder phase sensitivity {sens:+.1f}%")
assert sens>20, "decoder not phase-sensitive; aborting"
for q in list(d0.parameters())+list(h0.parameters()): q.requires_grad_(False)
mref=IQ.InnovQ().to(dev); Pi=mref.Pi_phys
pars=dict(mp1=torch.zeros(B,device=dev,requires_grad=True),
          r1r=torch.zeros(B,device=dev,requires_grad=True),
          ml1=lt_true.clone().detach().requires_grad_(True),
          sl1r=torch.full((B,),IQ.softplus_inv(0.25),device=dev,requires_grad=True),
          mlog=torch.zeros(B,K,device=dev,requires_grad=True),
          inn=torch.zeros(B,T-1,4,device=dev,requires_grad=True))
opt=torch.optim.Adam(list(pars.values()),lr=3e-3)
def breakdown(tag):
    accs=None
    with torch.no_grad():
        for r in range(4):
            nz=draw_noise(B,T,K,dev,IQ.DOF)
            ro=free_rollout(pars,nz,Pi,s_phi=0.05,s_lt=0.0025,rho1_max=0.9,sample=True)
            rb,rd,roo=P.recon_terms(dec,hdec,ro["Z"],b,db,obs,recon="bce")
            v=[rb.mean(),rd.mean(),roo.mean(),ro["kl_p"].mean(),ro["kl_l"].mean(),ro["kl_m"].mean()]
            v=[float(x) for x in v]
            accs=v if accs is None else [a+c for a,c in zip(accs,v)]
        accs=[a/4 for a in accs]
        ro=free_rollout(pars,draw_noise(B,T,K,dev,IQ.DOF),Pi,s_phi=0.05,s_lt=0.0025,rho1_max=0.9,sample=False)
        corr=float(torch.abs(torch.exp(1j*(ro["phi"]-phi_true)).mean(1)).mean())
        mae=100*float((ro["lt"].mean(1)-lt_true).abs().mean())
    return accs,corr,mae
a0,c0,m0=breakdown("start")
for s in range(400):
    opt.zero_grad()
    nz=draw_noise(B,T,K,dev,IQ.DOF)
    ro=free_rollout(pars,nz,Pi,s_phi=0.05,s_lt=0.0025,rho1_max=0.9,sample=True)
    rb,rd,roo=P.recon_terms(dec,hdec,ro["Z"],b,db,obs,recon="bce")
    ((rb+rd+roo+ro["kl_p"]+ro["kl_l"]+ro["kl_m"]).mean()).backward(); opt.step()
a1,c1,m1=breakdown("end")
names=["recon_beat","recon_db","recon_obs","kl_phase","kl_level","kl_meter"]
print(f"\n{'term':>12} {'ORACLE start':>13} {'found end':>12} {'change':>10}")
for n,x,y in zip(names,a0,a1):
    print(f"{n:>12} {x:13.2f} {y:12.2f} {y-x:+10.2f}")
print(f"{'TOTAL':>12} {sum(a0):13.2f} {sum(a1):12.2f} {sum(a1)-sum(a0):+10.2f}")
print(f"\ncorr {c0:.4f} -> {c1:.4f} | lvlMAE {m0:.2f}% -> {m1:.2f}%")
dr=(a1[0]+a1[1]+a1[2])-(a0[0]+a0[1]+a0[2]); dk=(a1[3]+a1[4]+a1[5])-(a0[3]+a0[4]+a0[5])
print(f"\nrecon change {dr:+.2f} | KL change {dk:+.2f}")
print("DRIVER:", "KL -> prior breaks the tie the wrong way; RESHAPING THE PRIOR IS THE FIX"
      if dk<dr else "RECON -> the likelihood prefers wrong tempo; no prior can fix it")
