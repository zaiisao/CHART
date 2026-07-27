"""Does ORACLE z still maximize the ELBO under innovation-q + current settings
(gamma=0.06, dev=0.02, import os; T=int(os.environ.get('CROP','1500')))? Compare, with decoders refit per condition:
  TRUTH  : q = true trajectory (small nonzero innovations)
  COAST  : perfect metronome from true init (ZERO innovations -> minimal rate)
  RANDOM : wrong initial phase, zero innovations
Report recon + innovation-KL + total for each."""
import sys, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem")
import pm_common as P, innovq as IQ
from innovq import TWO_PI, kl_phase_innov
dev="cuda:0"; torch.manual_seed(0)
import os; T=int(os.environ.get('CROP','1500'))
P.PHYS["gamma_phase"]=0.06; P.PHYS["dev_sigma"]=0.02
D=P.build_crops(P.load_songs("train"),n_per_song=1,seed=0,crop=T,dev=dev)
B=D["phi"].shape[0]; mh=F.one_hot(D["m"],4).float().unsqueeze(1).expand(-1,T,-1)
def wrap(x): return (x+math.pi)%TWO_PI-math.pi
def innov_of(phi,lt):
    """the per-step innovation a q would have to emit to produce this trajectory"""
    return wrap(phi[:,1:]-phi[:,:-1]-torch.exp(lt[:,:-1].clamp(-12,6)))
def coast_traj(phi,lt):
    rate=torch.exp(lt).median(dim=1,keepdim=True).values
    t=torch.arange(T,device=phi.device).float().unsqueeze(0)
    return (phi[:,:1]+rate*t)%TWO_PI, lt.median(dim=1,keepdim=True).values.expand(-1,T)
def rand_traj(phi,lt):
    p,l=coast_traj(phi,lt)
    off=torch.rand(B,1,device=phi.device)*TWO_PI
    return (p+off)%TWO_PI, l
def evaluate(name,phi,lt,steps=3000,width=128):
    torch.manual_seed(0)
    dec=nn.Sequential(nn.Linear(7,width),nn.Tanh(),nn.Linear(width,2)).to(dev)
    opt=torch.optim.AdamW(dec.parameters(),lr=1e-3)
    Z=torch.cat([torch.cos(phi).unsqueeze(-1),torch.sin(phi).unsqueeze(-1),lt.unsqueeze(-1),mh],-1)
    for i in range(steps):
        o=dec(Z)
        l=(F.binary_cross_entropy_with_logits(o[...,0],D["b"],reduction='none').sum(-1)
          +F.binary_cross_entropy_with_logits(o[...,1],D["db"],reduction='none').sum(-1)).mean()
        opt.zero_grad(); l.backward(); opt.step()
    recon=float(l)
    eps=innov_of(phi,lt)
    sq=torch.full_like(eps,1e-3)                 # q as sharp as the parameterization allows
    klp=float(kl_phase_innov(eps,sq).sum(-1).mean()) if kl_phase_innov(eps,sq).dim()>1 else float(kl_phase_innov(eps,sq).mean())
    corr=float(torch.abs(torch.exp(1j*(phi-D["phi"])).mean()))
    print(f"{name:8s} recon={recon:8.2f}  kl_phase={klp:9.2f}  TOTAL={recon+klp:9.2f}  |innov|={float(eps.abs().mean()):.2e}  corr={corr:.3f}")
    return recon+klp
print(f"T={T} frames ({T/50:.0f}s), {B} crops, gamma=0.06\n")
a=evaluate("TRUTH", D["phi"], D["lt"])
pc,lc=coast_traj(D["phi"],D["lt"]); b_=evaluate("COAST", pc, lc)
pr,lr=rand_traj(D["phi"],D["lt"]);  c=evaluate("RANDOM", pr, lr)
print(f"\nTRUTH - COAST = {a-b_:+.2f} nats/crop  ({'TRUTH WINS' if a<b_ else 'COAST WINS -- landscape inverted!'})")
print(f"TRUTH - RANDOM = {a-c:+.2f} nats/crop")
