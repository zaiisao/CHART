"""VERIFY the harness: does beats_at(1.0) reproduce the ground-truth beats?
If not, every loss-landscape result tonight is measuring the wrong thing."""
import sys, math
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem")
import pm_common as P
dev="cuda:0"; T=1500; TWO_PI=2*math.pi
D=P.build_crops(P.load_songs("train"),n_per_song=1,seed=0,crop=T,dev=dev)
B=D["phi"].shape[0]
def beats_at(mult,use_m=True):
    lt=D["lt"]+math.log(mult); st=torch.exp(lt.clamp(-12,6))
    phi=(D["phi"][:,:1]+torch.cumsum(F.pad(st[:,:-1],(1,0)),1))
    m=D["m"].float().unsqueeze(1) if use_m else 1.0
    psi=(phi*m)%TWO_PI
    return [(torch.where(torch.diff(psi[i])< -math.pi)[0]+1).float() for i in range(B)]
true=[torch.where(D["b"][i]>0.5)[0].float() for i in range(B)]
pred=beats_at(1.0)
nt=np.array([len(t) for t in true]); npd=np.array([len(p) for p in pred])
print(f"beat COUNTS at mult=1.0:  true mean {nt.mean():.1f}  pred mean {npd.mean():.1f}  ratio {npd.mean()/nt.mean():.3f}")
print(f"  per-crop ratio: median {np.median(npd/np.maximum(nt,1)):.3f}  deciles {np.round(np.percentile(npd/np.maximum(nt,1),[10,50,90]),3)}")
print(f"  meter D['m'] distribution: {np.bincount(D['m'].cpu().numpy(),minlength=5)}")
# what multiplier makes the COUNTS match?
for mult in (1.0,1.2,1.33,1.4,1.5,1.7,2.0):
    p=beats_at(mult); r=np.mean([len(a)/max(len(b_),1) for a,b_ in zip(p,true)])
    print(f"   mult {mult:.2f} -> pred/true count ratio {r:.3f}")
# alignment quality at 1.0
d=[]
for p,t in zip(pred,true):
    if len(p) and len(t): d.append(float((t.unsqueeze(1)-p.unsqueeze(0)).abs().min(1).values.mean()))
print(f"\nmean |true beat -> nearest pred| at mult=1.0: {np.mean(d):.2f} frames  (should be ~1 if correct)")
# is the true bar phase itself consistent with the beats? (sanity on D['phi'])
psi_true=(D["phi"]*D["m"].float().unsqueeze(1))%TWO_PI
pt=[(torch.where(torch.diff(psi_true[i])< -math.pi)[0]+1).float() for i in range(B)]
d2=[float((t.unsqueeze(1)-p.unsqueeze(0)).abs().min(1).values.mean()) for p,t in zip(pt,true) if len(p) and len(t)]
print(f"mean |true beat -> nearest wrap of TRUE phase*m|: {np.mean(d2):.2f} frames  (the reference itself)")
print(f"  counts from true phase: mean {np.mean([len(p) for p in pt]):.1f} vs true beats {nt.mean():.1f}")
