"""Test the hypothesis: BCE is POINTWISE (no distance info) -> Dirac pit.
A DISPLACEMENT loss (Chamfer between predicted and true beat times) should vary SMOOTHLY
with tempo error, giving gradient at any distance. Compare the two landscapes."""
import sys, math
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem")
import pm_common as P
dev="cuda:0"; torch.manual_seed(0); T=1500; TWO_PI=2*math.pi
D=P.build_crops(P.load_songs("train"),n_per_song=1,seed=0,crop=T,dev=dev)
B=D["phi"].shape[0]
def beats_at(mult):
    """predicted beat FRAMES: where the (m-subdivided) phase wraps, at tempo x mult"""
    lt=D["lt"]+math.log(mult); st=torch.exp(lt.clamp(-12,6))
    phi=(D["phi"][:,:1]+torch.cumsum(F.pad(st[:,:-1],(1,0)),1))
    m=D["m"].float().unsqueeze(1)
    psi=(phi*m)%TWO_PI
    return [(torch.where(torch.diff(psi[i])< -math.pi)[0]+1).float() for i in range(B)]
true=[torch.where(D["b"][i]>0.5)[0].float() for i in range(B)]
def chamfer(pred,tru):
    """mean |displacement| from each TRUE beat to nearest PREDICTED beat (frames)"""
    out=[]
    for p,t in zip(pred,tru):
        if len(p)==0 or len(t)==0: continue
        d=(t.unsqueeze(1)-p.unsqueeze(0)).abs().min(1).values
        out.append(d.mean())
    return float(torch.stack(out).mean())
def bce_proxy(pred,tru,w=2.0):
    """pointwise overlap: fraction of true beats with a prediction within w frames"""
    out=[]
    for p,t in zip(pred,tru):
        if len(p)==0 or len(t)==0: continue
        d=(t.unsqueeze(1)-p.unsqueeze(0)).abs().min(1).values
        out.append((d<=w).float().mean())
    return float(torch.stack(out).mean())
print(f"{'mult':>6} | {'pointwise hit-rate':>18} | {'CHAMFER dist (frames)':>22}")
print(f"{'':6} | {'(what BCE sees)':>18} | {'(distance loss sees)':>22}")
prev=None
for mult in (0.5,0.6,0.7,0.8,0.9,0.95,0.98,1.0,1.02,1.05,1.1,1.2,1.4,1.7,2.0):
    pr=beats_at(mult); hit=bce_proxy(pr,true); ch=chamfer(pr,true)
    bar="#"*int(ch/3)
    print(f"{mult:6.2f} | {hit:18.3f} | {ch:9.1f} {bar}")
