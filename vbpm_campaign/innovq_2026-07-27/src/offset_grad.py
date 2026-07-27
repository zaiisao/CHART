"""The offset IS q(phi_1). Why doesn't the ELBO learn it?
 (1) does mu_phi1 MOVE during training, and toward truth?
 (2) how big is the recon gradient on mu_phi1 vs the other heads?
 (3) is the recon LANDSCAPE over offset even informative? sweep phi_1 and plot recon.
"""
import sys, math, torch, torch.nn.functional as F, numpy as np
for p in ("/home/sogang/jaehoon/VBPM_reintegration","/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix",
          "/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq","/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem"):
    sys.path.insert(0,p)
import pm_common as P, innovq as IQ
from wire_test import free_traj, free_heads
from rollout_vec_s import draw_noise
dev="cuda:0"; torch.manual_seed(0); T=1500; TWO_PI=2*math.pi; K=4; FPS=50.0
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
for q in list(d0.parameters())+list(h0.parameters()): q.requires_grad_(False)
Pi=IQ.InnovQ().to(dev).Pi_phys
def mkpars():
    return dict(mp1=torch.zeros(B,device=dev,requires_grad=True),
                r1r=torch.zeros(B,device=dev,requires_grad=True),
                ml1=lt_true.clone().detach().requires_grad_(True),
                sl1r=torch.full((B,),IQ.softplus_inv(0.25),device=dev,requires_grad=True),
                mlog=torch.zeros(B,K,device=dev,requires_grad=True),
                inn=torch.zeros(B,T-1,4,device=dev,requires_grad=True))
def offerr(mp1):
    d=torch.remainder(mp1-phi_true[:,0]+math.pi,TWO_PI)-math.pi
    return float(d.abs().mean())
pars=mkpars()
print(f"(1) offset error at init: {offerr(pars['mp1'].detach()):.4f} rad "
      f"(random guess would be ~{math.pi/2:.4f})")
opt=torch.optim.Adam(list(pars.values()),lr=3e-3)
gnorms={"mp1":[], "ml1":[], "inn":[]}
for s in range(400):
    opt.zero_grad()
    ro=free_traj(**free_heads(pars,0.05,0.0025,0.9,0.05),noise=draw_noise(B,T,K,dev,IQ.DOF),Pi=Pi,sample=True)
    rb,rd,roo=P.recon_terms(dec,hdec,ro["Z"],b,db,obs,recon="bce")
    ((rb+rd+roo).mean()).backward(retain_graph=True)          # RECON-ONLY gradient
    for k in gnorms: gnorms[k].append(float(pars[k].grad.abs().mean()))
    opt.zero_grad()
    ro=free_traj(**free_heads(pars,0.05,0.0025,0.9,0.05),noise=draw_noise(B,T,K,dev,IQ.DOF),Pi=Pi,sample=True)
    rb,rd,roo=P.recon_terms(dec,hdec,ro["Z"],b,db,obs,recon="bce")
    ((rb+rd+roo+ro["kl_p"]+ro["kl_l"]+ro["kl_m"]).mean()).backward(); opt.step()
    if s in (99,399):
        print(f"    step {s+1}: offset error {offerr(pars['mp1'].detach()):.4f} rad")
print(f"(2) mean |recon grad| : mu_phi1 {np.mean(gnorms['mp1']):.3e} | "
      f"level {np.mean(gnorms['ml1']):.3e} | innovations {np.mean(gnorms['inn']):.3e}")
# (4) can the offset be found if the INNOVATIONS cannot absorb it?
print("\n(4) train ONLY mu_phi1 (true tempo; innovations held at zero):")
for tag,frozen in (("innovations FROZEN at 0",True),("innovations free",False)):
    torch.manual_seed(0); pp=mkpars()
    with torch.no_grad(): pp["ml1"].copy_(lt_true)
    tr_par=[pp["mp1"]] if frozen else [pp["mp1"],pp["inn"]]
    o=torch.optim.Adam(tr_par,lr=3e-2)
    for s_ in range(300):
        o.zero_grad()
        ro=free_traj(**free_heads(pp,0.05,0.0025,0.9,0.05),noise=draw_noise(B,T,K,dev,IQ.DOF),Pi=Pi,sample=True)
        rb,rd,roo=P.recon_terms(dec,hdec,ro["Z"],b,db,obs,recon="bce")
        ((rb+rd+roo+ro["kl_p"]+ro["kl_l"]+ro["kl_m"]).mean()).backward(); o.step()
    with torch.no_grad():
        ro=free_traj(**free_heads(pp,0.05,0.0025,0.9,0.05),noise=draw_noise(B,T,K,dev,IQ.DOF),Pi=Pi,sample=False)
        Fs=[]
        for i in range(B):
            ref=np.where(b[i].cpu().numpy()>0.5)[0]/FPS
            if len(ref)>=4:
                from vbpm.evaluate import beats_from_barphase, f_measure
                Fs.append(f_measure(ref,beats_from_barphase(ro["phi"][i].cpu().numpy(),int(tr["m"][sel][i])+1,FPS)))
    print(f"    {tag:26s} offset err {offerr(pp['mp1'].detach()):.4f} rad  beat F {np.mean(Fs):.4f}")

# (3) is the recon landscape over offset informative at all?
base=mkpars()
with torch.no_grad(): base["ml1"].copy_(lt_true)
print("(3) recon vs phase offset (true tempo, zero innovations):")
vals=[]
for k in range(12):
    with torch.no_grad():
        base["mp1"].copy_(phi_true[:,0]+k*TWO_PI/12.0)
        ro=free_traj(**free_heads(base,0.05,0.0025,0.9,0.05),noise=draw_noise(B,T,K,dev,IQ.DOF),Pi=Pi,sample=False)
        rb,rd,roo=P.recon_terms(dec,hdec,ro["Z"],b,db,obs,recon="bce")
        vals.append(float((rb+rd+roo).mean()))
best=int(np.argmin(vals))
print("    offset(2pi/12 units):  "+" ".join(f"{i:6d}" for i in range(12)))
print("    recon              :  "+" ".join(f"{v-min(vals):6.1f}" for v in vals))
print(f"    minimum at offset {best} (0 = TRUE)  -> {'CORRECT' if best==0 else 'WRONG'}"
      f" | depth {max(vals)-min(vals):.1f} nats")
