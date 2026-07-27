"""Can we WIDEN the tempo pit? Two levers:
  (1) WINDOW LENGTH T -- drift ~ T, so pit width ~ 1/T. Short windows = wide pit.
  (2) TARGET BLUR -- smear the beat targets in time; a wrong tempo still overlaps.
Measure pit width = range of tempo multiplier where recon stays within 25% of its depth."""
import sys, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem")
import pm_common as P
dev="cuda:0"; torch.manual_seed(0); TWO_PI=2*math.pi
MULTS=np.concatenate([np.linspace(0.5,0.95,10),np.linspace(0.96,1.04,17),np.linspace(1.05,1.6,12)])
def blur(y,sig):
    if sig<=0: return y
    k=int(6*sig)|1; x=torch.arange(k,device=y.device).float()-k//2
    g=torch.exp(-x**2/(2*sig*sig)); g=g/g.max()
    return torch.clamp(F.conv1d(y.unsqueeze(1),g.view(1,1,-1),padding=k//2).squeeze(1),0,1)
def landscape(T,sig):
    D=P.build_crops(P.load_songs("train"),n_per_song=1,seed=0,crop=T,dev=dev)
    mh=F.one_hot(D["m"],4).float().unsqueeze(1).expand(-1,T,-1)
    bt,dbt=blur(D["b"],sig),blur(D["db"],sig)
    def Z(mult):
        lt=D["lt"]+math.log(mult); st=torch.exp(lt.clamp(-12,6))
        phi=(D["phi"][:,:1]+torch.cumsum(F.pad(st[:,:-1],(1,0)),1))%TWO_PI
        return torch.cat([torch.cos(phi).unsqueeze(-1),torch.sin(phi).unsqueeze(-1),lt.unsqueeze(-1),mh],-1)
    dec=nn.Sequential(nn.Linear(7,128),nn.Tanh(),nn.Linear(128,2)).to(dev)
    opt=torch.optim.AdamW(dec.parameters(),lr=1e-3); z0=Z(1.0)
    for i in range(2500):
        o=dec(z0); l=(F.binary_cross_entropy_with_logits(o[...,0],bt,reduction='none').sum(-1)
                     +F.binary_cross_entropy_with_logits(o[...,1],dbt,reduction='none').sum(-1)).mean()
        opt.zero_grad(); l.backward(); opt.step()
    rs=[]
    with torch.no_grad():
        for mlt in MULTS:
            o=dec(Z(float(mlt)))
            rs.append(float((F.binary_cross_entropy_with_logits(o[...,0],bt,reduction='none').sum(-1)
                            +F.binary_cross_entropy_with_logits(o[...,1],dbt,reduction='none').sum(-1)).mean()))
    rs=np.array(rs); lo,hi=rs.min(),np.median(rs)
    thr=lo+0.25*(hi-lo)
    inside=MULTS[rs<=thr]
    return (inside.min(),inside.max(),100*(inside.max()-inside.min())/2), lo, hi
print(f"{'T':>6} {'blur':>5} | {'pit range (mult)':>20} {'half-width %':>12} | depth")
for T in (256,512,1500):
    (a,b,w),lo,hi=landscape(T,0.0)
    print(f"{T:6d} {0:5.1f} | [{a:.3f},{b:.3f}]{'':>6} {w:11.1f}% | {hi-lo:.0f}")
for sig in (3.0,10.0,25.0):
    (a,b,w),lo,hi=landscape(1500,sig)
    print(f"{1500:6d} {sig:5.1f} | [{a:.3f},{b:.3f}]{'':>6} {w:11.1f}% | {hi-lo:.0f}")
