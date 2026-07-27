"""OPTIMAL TRANSPORT landscape: mass-preserving displacement cost between predicted and true
beat sets. Requirements: (a) long-range gradient, (b) density can't be gamed, (c) min at 1.0.
Variants: W1 on sorted event times (equal counts via mass normalisation), and a
count-penalised Chamfer control."""
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
def w1_quantile(pred,tru,K=64):
    """1-D Wasserstein via quantile matching: both event sets -> K quantiles of their
    empirical CDF, then mean |difference|. Mass-preserving by construction; density
    cannot be gamed because both sides are normalised to the same total mass."""
    q=torch.linspace(0,1,K,device=dev); out=[]
    for p,t in zip(pred,tru):
        if len(p)<2 or len(t)<2: continue
        out.append((torch.quantile(p,q)-torch.quantile(t,q)).abs().mean())
    return float(torch.stack(out).mean())
def chamfer_counted(pred,tru,lam=8.0):
    out=[]
    for p,t in zip(pred,tru):
        if len(p)==0 or len(t)==0: continue
        d=(t.unsqueeze(1)-p.unsqueeze(0)).abs()
        c=d.min(1).values.mean()+d.min(0).values.mean()
        c=c+lam*abs(len(p)-len(t))/len(t)          # explicit cardinality penalty
        out.append(c)
    return float(torch.stack(out).mean())
def ibi_w1(pred,tru,K=48):
    """W1 between the INTER-BEAT-INTERVAL distributions -- tempo-native, shift-invariant."""
    q=torch.linspace(0,1,K,device=dev); out=[]
    for p,t in zip(pred,tru):
        if len(p)<3 or len(t)<3: continue
        out.append((torch.quantile(torch.diff(p),q)-torch.quantile(torch.diff(t),q)).abs().mean())
    return float(torch.stack(out).mean())
print(f"{'mult':>6} | {'W1(times)':>10} | {'Chamfer+count':>14} | {'W1(IBI)':>9}")
rows=[]
for mult in (0.5,0.6,0.7,0.8,0.9,0.95,0.98,1.0,1.02,1.05,1.1,1.2,1.4,1.7,2.0):
    pr=beats_at(mult)
    a,b,c=w1_quantile(pr,true),chamfer_counted(pr,true),ibi_w1(pr,true)
    rows.append((mult,a,b,c))
    print(f"{mult:6.2f} | {a:10.2f} | {b:14.2f} | {c:9.3f} {'#'*int(c*3)}")
for j,name in ((1,"W1(times)"),(2,"Chamfer+count"),(3,"W1(IBI)")):
    best=min(rows,key=lambda r:r[j])
    far=abs(rows[1][j]-rows[0][j])/0.1
    print(f"\n{name:14s} min at mult={best[0]:.2f}  | far-field slope {far:.2f} (BCE~0)  {'<-- CORRECT' if abs(best[0]-1.0)<1e-9 else ''}")
