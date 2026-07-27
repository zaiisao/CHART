"""SYMMETRIC Chamfer: true->pred AND pred->true. Extra beats now pay their own distance.
Does it give a smooth basin centred at mult=1.0 (long-range gradient, no annealing needed)?
Also test a soft/differentiable version usable as an actual training loss."""
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
    lt=D["lt"]+math.log(mult); st=torch.exp(lt.clamp(-12,6))
    phi=(D["phi"][:,:1]+torch.cumsum(F.pad(st[:,:-1],(1,0)),1))
    psi=(phi*D["m"].float().unsqueeze(1))%TWO_PI
    return [(torch.where(torch.diff(psi[i])< -math.pi)[0]+1).float() for i in range(B)]
true=[torch.where(D["b"][i]>0.5)[0].float() for i in range(B)]
def sym_chamfer(pred,tru):
    fwd=[];bwd=[]
    for p,t in zip(pred,tru):
        if len(p)==0 or len(t)==0: continue
        d=(t.unsqueeze(1)-p.unsqueeze(0)).abs()
        fwd.append(d.min(1).values.mean())     # true -> nearest pred
        bwd.append(d.min(0).values.mean())     # pred -> nearest true  (penalises extra beats)
    return float(torch.stack(fwd).mean()), float(torch.stack(bwd).mean())
print(f"{'mult':>6} | {'true->pred':>10} {'pred->true':>10} | {'SYMMETRIC':>10}")
rows=[]
for mult in (0.5,0.6,0.7,0.8,0.9,0.95,0.98,1.0,1.02,1.05,1.1,1.2,1.4,1.7,2.0):
    f,b=sym_chamfer(beats_at(mult),true); s=f+b; rows.append((mult,s))
    print(f"{mult:6.2f} | {f:10.1f} {b:10.1f} | {s:8.1f} {'#'*int(s/2)}")
best=min(rows,key=lambda r:r[1])
print(f"\nminimum at mult={best[0]:.2f} (value {best[1]:.1f})   <- want 1.00")
mono=all(rows[i][1]>=rows[i+1][1] for i in range(rows.index(best)) ) and all(rows[i][1]<=rows[i+1][1] for i in range(rows.index(best),len(rows)-1))
print(f"monotone descent toward the minimum from both sides: {mono}")
g=[(rows[i+1][1]-rows[i][1])/(rows[i+1][0]-rows[i][0]) for i in range(len(rows)-1)]
print(f"slope far from optimum (0.5->0.6): {g[0]:+.1f} | near (0.98->1.0): {g[6]:+.1f}   <- BCE had ~0 far slope")
