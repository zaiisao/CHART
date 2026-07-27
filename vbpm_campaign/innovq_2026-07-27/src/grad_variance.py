"""Is the mu_phi1 gradient unusable because the wrapped-Cauchy reparameterization has
infinite variance? Compare the gradient estimator's mean vs spread across noise draws,
sampled vs deterministic."""
import sys, math, torch, torch.nn.functional as F, numpy as np
for p in ("/home/sogang/jaehoon/VBPM_reintegration","/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix",
          "/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq","/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem"):
    sys.path.insert(0,p)
import pm_common as P, innovq as IQ
from wire_test import free_traj, free_heads
from rollout_vec_s import draw_noise
dev="cuda:0"; torch.manual_seed(0); T=1500; TWO_PI=2*math.pi; K=4
GP=0.06; P.PHYS["gamma_phase"]=GP; IQ.RHO_P=math.exp(-GP); IQ.GP1=1-IQ.RHO_P; IQ.R0=IQ.softplus_inv(IQ.GP1)
tr=P.build_crops(P.load_songs("train"),n_per_song=1,seed=0,crop=T,dev=dev); N=tr["b"].shape[0]
sel=torch.arange(48,device=dev); B=len(sel)
b,db,obs=tr["b"][sel],tr["db"][sel],tr["obs"][sel]
phi_true,lt_true=tr["phi"][sel],tr["lt"][sel].mean(1)
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
def grads(sample, r1r_val, n=60):
    G=[]
    for r in range(n):
        pars=dict(mp1=(phi_true[:,0]+1.0).clone().detach().requires_grad_(True),   # 1 rad off
                  r1r=torch.full((B,),r1r_val,device=dev),
                  ml1=lt_true.clone(),
                  sl1r=torch.full((B,),IQ.softplus_inv(0.25),device=dev),
                  mlog=torch.zeros(B,K,device=dev),
                  inn=torch.zeros(B,T-1,4,device=dev))
        ro=free_traj(**free_heads(pars,0.05,0.0025,0.9,0.05),
                     noise=draw_noise(B,T,K,dev,IQ.DOF),Pi=Pi,sample=sample)
        rb,rd,roo=P.recon_terms(dec,hdec,ro["Z"],b,db,obs,recon="bce")
        (rb+rd+roo).mean().backward()
        G.append(pars["mp1"].grad.clone())
    G=torch.stack(G)                                   # [n,B]
    m=G.mean(0); sd=G.std(0)
    snr=float((m.abs()/(sd+1e-12)).mean())
    # is the SIGN of the mean gradient correct? (offset is +1 rad, so grad should be >0)
    signok=float((m>0).float().mean())
    return float(m.abs().mean()), float(sd.mean()), snr, signok, float(G.abs().max())
print(f"{'setting':34s} {'|mean|':>10s} {'sd':>10s} {'SNR':>7s} {'sign ok':>8s} {'max|g|':>10s}")
for tag,samp,r1 in (("deterministic (sample=False)",False,0.0),
                    ("sampled, rho1=0.45 (default)",True,0.0),
                    ("sampled, rho1=0.81 (tighter)",True,2.0),
                    ("sampled, rho1=0.89 (tightest)",True,4.0)):
    m,sd,snr,so,mx=grads(samp,r1)
    print(f"{tag:34s} {m:10.3e} {sd:10.3e} {snr:7.4f} {100*so:7.1f}% {mx:10.3e}")
print("\n(SNR = |mean|/sd per crop. SNR << 1 means the estimator is noise; sign ok = fraction")
print(" of crops whose mean gradient points the right way. Offset is +1.0 rad from truth.)")
