"""Is phase-blindness STRUCTURAL or a training-dynamics artifact?
  A) random decoder            -> sensitivity at training init
  B) decoder trained on ORACLE phase (teacher-forced) -> is it sensitive once fit?
  C) decoder trained on MODEL rollout (what actually happens in training)
Sensitivity = how much recon rises when phase is drifted off."""
import sys, math, numpy as np, torch, torch.nn.functional as F
for p in ("/home/sogang/jaehoon/VBPM_reintegration","/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix",
          "/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq","/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem"):
    sys.path.insert(0,p)
import pm_common as P, innovq as IQ
dev="cuda:0"; torch.manual_seed(0); T=1500; TWO_PI=2*math.pi
P.PHYS["gamma_phase"]=0.06; P.PHYS["dev_sigma"]=0.02
D=P.build_crops(P.load_songs("train"),n_per_song=2,seed=0,crop=T,dev=dev)
N=D["b"].shape[0]; idx=torch.arange(min(16,N),device=dev)
def zfrom(phi,lt,meter):
    return torch.cat([torch.cos(phi).unsqueeze(-1),torch.sin(phi).unsqueeze(-1),
                      lt.unsqueeze(-1),meter],-1)
K=4
meter=torch.zeros(N,T,K,device=dev); meter[...,0]=1.0
Zor=zfrom(D["phi"],D["lt"],meter)                       # ORACLE latent trajectory
def sens(dec,hdec,Z,recon,tag):
    out=[]
    for bias in (0.0,0.002,0.01,0.05):
        phi2=(torch.atan2(Z[...,1],Z[...,0])+bias*torch.arange(T,device=dev).float())%TWO_PI
        Z2=zfrom(phi2,Z[...,2],Z[...,3:])
        with torch.no_grad():
            rb,rd,_=P.recon_terms(dec,hdec,Z2[idx],D["b"][idx],D["db"][idx],D["obs"][idx],recon=recon)
        out.append(float((rb+rd).mean()))
    rel=100*(out[-1]-out[0])/abs(out[0])
    print(f"   {tag:34s} {recon:7s} "+" ".join(f"{v:9.1f}" for v in out)+f"   rise {rel:+7.2f}%")
    return rel
for recon in ("bce","cramer"):
    d0,h0=P.new_decoders(dev); dec,hdec=IQ.Cut(d0),IQ.Cut(h0)
    sens(dec,hdec,Zor,recon,"A) random decoder")
    opt=torch.optim.Adam(list(d0.parameters())+list(h0.parameters()),lr=3e-3)
    for s in range(600):
        opt.zero_grad(); sel=torch.randperm(N,device=dev)[:16]
        rb,rd,ro_=P.recon_terms(dec,hdec,Zor[sel],D["b"][sel],D["db"][sel],D["obs"][sel],recon=recon)
        (rb+rd+ro_).mean().backward(); opt.step()
    sens(dec,hdec,Zor,recon,"B) decoder fit on ORACLE phase")
