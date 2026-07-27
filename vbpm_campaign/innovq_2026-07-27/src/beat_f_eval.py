"""Convert today's corr numbers into BEAT F. Density-matched blind control is MANDATORY.

Arms:
  init        tempogram level init, before any ELBO training  (the corr 0.57-0.64 model)
  per-frame   after ELBO with the current per-frame tempo walk
  beat-gated  after ELBO with the doc's between-beats-constant tempo
  BLIND       uniform grid at the SAME beat count, best of 12 offsets  <- the floor any
              trajectory must clear to mean anything
"""
import sys, math, torch, torch.nn.functional as F, numpy as np
for p in ("/home/sogang/jaehoon/VBPM_reintegration","/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix",
          "/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq","/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem"):
    sys.path.insert(0,p)
import pm_common as P, innovq as IQ
from vbpm.evaluate import beats_from_barphase, f_measure
from wire_test import free_traj, free_heads
from beatgate_ab import free_traj_gated
from rollout_vec_s import draw_noise
dev="cuda:0"; torch.manual_seed(0); T=1500; TWO_PI=2*math.pi; K=4; FPS=50.0
GP=0.06; P.PHYS["gamma_phase"]=GP; IQ.RHO_P=math.exp(-GP); IQ.GP1=1-IQ.RHO_P; IQ.R0=IQ.softplus_inv(IQ.GP1)
tr=P.build_crops(P.load_songs("train"),n_per_song=1,seed=0,crop=T,dev=dev); N=tr["b"].shape[0]
sel=torch.arange(48,device=dev); B=len(sel)
h,b,db,obs=tr["h"][sel],tr["b"][sel],tr["db"][sel],tr["obs"][sel]
phi_true,lt_frame=tr["phi"][sel],tr["lt"][sel]; lt_true=lt_frame.mean(1); mi=tr["m"][sel]
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
def train(fn,steps=400):
    torch.manual_seed(0); pars=mkpars()
    opt=torch.optim.Adam(list(pars.values()),lr=3e-3)
    for s in range(steps):
        opt.zero_grad()
        ro=fn(**free_heads(pars,0.05,0.0025,0.9,0.05),noise=draw_noise(B,T,K,dev,IQ.DOF),Pi=Pi,sample=True)
        rb,rd,roo=P.recon_terms(dec,hdec,ro["Z"],b,db,obs,recon="bce")
        ((rb+rd+roo+ro["kl_p"]+ro["kl_l"]+ro["kl_m"]).mean()).backward(); opt.step()
    return pars
def evaluate(pars,fn,tag):
    with torch.no_grad():
        ro=fn(**free_heads(pars,0.05,0.0025,0.9,0.05),noise=draw_noise(B,T,K,dev,IQ.DOF),Pi=Pi,sample=False)
    Fs=[];Fb=[];cs=[]
    for i in range(B):
        ref=np.where(b[i].cpu().numpy()>0.5)[0]/FPS
        if len(ref)<4: continue
        ph=ro["phi"][i].cpu().numpy()
        est=beats_from_barphase(ph,int(mi[i])+1,FPS)
        Fs.append(f_measure(ref,est))
        # density-matched blind control: uniform grid, same count, best of 12 offsets
        n=len(est)
        if n>=2:
            span=T/FPS; per=span/max(n,1); best=0.0
            for k in range(12):
                g=np.arange(n)*per+k*per/12.0
                best=max(best,f_measure(ref,g))
            Fb.append(best)
        cs.append(float(torch.abs(torch.exp(1j*(ro["phi"][i]-phi_true[i])).mean())))
    print(f"  {tag:24s} beat F {np.mean(Fs):.4f}  | blind {np.mean(Fb):.4f}  | "
          f"lift {np.mean(Fs)-np.mean(Fb):+.4f} | per-crop corr {np.mean(cs):.4f}")
    return np.mean(Fs),np.mean(Fb)
print(f"\n{'':26s} {'beat F':>8s}")
evaluate(mkpars(),free_traj,"init (tempogram, no ELBO)")
evaluate(train(free_traj),free_traj,"after ELBO, per-frame")
evaluate(train(free_traj_gated),free_traj_gated,"after ELBO, BEAT-GATED")
# reference: what does the ORACLE latent score through the same decode path?
with torch.no_grad():
    Fo=[]
    for i in range(B):
        ref=np.where(b[i].cpu().numpy()>0.5)[0]/FPS
        if len(ref)<4: continue
        Fo.append(f_measure(ref,beats_from_barphase(phi_true[i].cpu().numpy(),int(mi[i])+1,FPS)))
print(f"  {'ORACLE latent (ceiling)':24s} beat F {np.mean(Fo):.4f}")

# --- is the gap OFFSET? give the init model the TRUE phi_1, change nothing else ---
pars=mkpars()
with torch.no_grad(): pars["mp1"].copy_(phi_true[:,0])
evaluate(pars,free_traj,"init + TRUE phase offset")
# --- and: best-of-12 offset search on the init model (what the blind control is allowed) ---
base=mkpars()
with torch.no_grad():
    ro=free_traj(**free_heads(base,0.05,0.0025,0.9,0.05),noise=draw_noise(B,T,K,dev,IQ.DOF),Pi=Pi,sample=False)
Fbest=[]
for i in range(B):
    ref=np.where(b[i].cpu().numpy()>0.5)[0]/FPS
    if len(ref)<4: continue
    ph=ro["phi"][i].cpu().numpy(); bb=0.0
    for k in range(12):
        est=beats_from_barphase((ph+k*TWO_PI/12.0)%TWO_PI,int(mi[i])+1,FPS)
        bb=max(bb,f_measure(ref,est))
    Fbest.append(bb)
print(f"  {'init + best-of-12 offset':24s} beat F {np.mean(Fbest):.4f}  <- same freedom the blind control gets")
