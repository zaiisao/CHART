"""Is tempo gradient-REACHABLE or only identifiable? Sweep a tempo multiplier around truth
and plot the reconstruction loss (decoders refit at truth, then frozen). If the landscape is
multimodal / flat away from the optimum, gradient descent cannot find tempo -- explaining why
it must be searched (PF) or supervised, not learned by SGD."""
import sys, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem")
import pm_common as P
dev="cuda:0"; torch.manual_seed(0); T=1500; TWO_PI=2*math.pi
D=P.build_crops(P.load_songs("train"),n_per_song=1,seed=0,crop=T,dev=dev)
B=D["phi"].shape[0]; mh=F.one_hot(D["m"],4).float().unsqueeze(1).expand(-1,T,-1)
def traj(mult):
    lt=D["lt"]+math.log(mult)
    steps=torch.exp(lt.clamp(-12,6))
    phi=(D["phi"][:,:1]+torch.cumsum(F.pad(steps[:,:-1],(1,0)),1))%TWO_PI
    return phi,lt
# decoders fit at TRUE tempo, then frozen -> the landscape the model would descend
phi,lt=traj(1.0)
Z=torch.cat([torch.cos(phi).unsqueeze(-1),torch.sin(phi).unsqueeze(-1),lt.unsqueeze(-1),mh],-1)
dec=nn.Sequential(nn.Linear(7,128),nn.Tanh(),nn.Linear(128,2)).to(dev)
opt=torch.optim.AdamW(dec.parameters(),lr=1e-3)
for i in range(4000):
    o=dec(Z); l=(F.binary_cross_entropy_with_logits(o[...,0],D["b"],reduction='none').sum(-1)
                +F.binary_cross_entropy_with_logits(o[...,1],D["db"],reduction='none').sum(-1)).mean()
    opt.zero_grad(); l.backward(); opt.step()
print(f"decoders fit at true tempo: recon={float(l):.1f}\n")
print(f"{'mult':>6} {'recon':>9}  {'':<32}")
res=[]
with torch.no_grad():
    for mult in [0.5,0.6,0.667,0.75,0.85,0.9,0.95,0.98,1.0,1.02,1.05,1.1,1.2,1.33,1.5,1.75,2.0]:
        p,l2=traj(mult)
        Z2=torch.cat([torch.cos(p).unsqueeze(-1),torch.sin(p).unsqueeze(-1),l2.unsqueeze(-1),mh],-1)
        o=dec(Z2)
        r=float((F.binary_cross_entropy_with_logits(o[...,0],D["b"],reduction='none').sum(-1)
               +F.binary_cross_entropy_with_logits(o[...,1],D["db"],reduction='none').sum(-1)).mean())
        res.append((mult,r))
        bar="#"*int(max(0,(r-180)/8))
        print(f"{mult:6.3f} {r:9.1f}  {bar}")
r0=[r for m_,r in res if abs(m_-1.0)<1e-9][0]
print(f"\ntrue-tempo recon={r0:.1f}; local minima (recon lower than both neighbours):")
for i in range(1,len(res)-1):
    if res[i][1]<res[i-1][1] and res[i][1]<res[i+1][1]:
        print(f"   mult={res[i][0]:.3f} recon={res[i][1]:.1f}")
d=[(res[i+1][1]-res[i][1])/(res[i+1][0]-res[i][0]) for i in range(len(res)-1)]
print(f"gradient magnitude near truth (mult .95->1.05): {abs(d[6]):.0f} | far (mult .5->.6): {abs(d[0]):.0f}")
