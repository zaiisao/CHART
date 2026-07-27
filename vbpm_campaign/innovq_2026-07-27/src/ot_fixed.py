"""REBUILT with a verification GATE first. Beat readout must reproduce ground truth at
mult=1.0 (count ratio ~1.00, displacement ~1 frame) BEFORE any landscape is trusted."""
import sys, math
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem")
import pm_common as P
dev="cuda:0"; torch.manual_seed(0); T=1500; TWO_PI=2*math.pi
D=P.build_crops(P.load_songs("train"),n_per_song=1,seed=0,crop=T,dev=dev)
B=D["phi"].shape[0]
true=[torch.where(D["b"][i]>0.5)[0].float() for i in range(B)]
# ---- BEAT-LEVEL phase, derived per crop so that mult=1 reproduces the true beats ----
# beat rate = true beats per frame, measured from the annotations themselves
rate=torch.tensor([ (len(t)-1)/max(float(t[-1]-t[0]),1e-6) if len(t)>1 else 0.0 for t in true],device=dev)
ph0 =torch.tensor([ float(t[0])*float(r) for t,r in zip(true,rate.tolist())],device=dev)
def beats_at(mult):
    """beat phase advances at rate*mult cycles/frame; beats = integer crossings"""
    t=torch.arange(T,device=dev).float().unsqueeze(0)
    ph=(rate.unsqueeze(1)*mult)*t - ph0.unsqueeze(1)
    out=[]
    for i in range(B):
        k=torch.floor(ph[i]); idx=torch.where(torch.diff(k)>0)[0]+1
        out.append(idx.float())
    return out
# ---- GATE ----
pr=beats_at(1.0)
cr=np.mean([len(p)/max(len(t),1) for p,t in zip(pr,true)])
dsp=np.mean([float((t.unsqueeze(1)-p.unsqueeze(0)).abs().min(1).values.mean()) for p,t in zip(pr,true) if len(p) and len(t)])
print(f"GATE  count ratio {cr:.3f} (want ~1.00) | mean displacement {dsp:.2f} frames (want <2)")
if not (0.9<cr<1.1 and dsp<3.0):
    print("GATE FAILED -- not running landscapes."); sys.exit(0)
print("GATE PASSED\n")
def w1(pred,tru,K=64):
    q=torch.linspace(0,1,K,device=dev); o=[]
    for p,t in zip(pred,tru):
        if len(p)<2 or len(t)<2: continue
        o.append((torch.quantile(p,q)-torch.quantile(t,q)).abs().mean())
    return float(torch.stack(o).mean())
def cham_cnt(pred,tru,lam=8.0):
    o=[]
    for p,t in zip(pred,tru):
        if len(p)==0 or len(t)==0: continue
        d=(t.unsqueeze(1)-p.unsqueeze(0)).abs()
        o.append(d.min(1).values.mean()+d.min(0).values.mean()+lam*abs(len(p)-len(t))/len(t))
    return float(torch.stack(o).mean())
def ibi_w1(pred,tru,K=48):
    q=torch.linspace(0,1,K,device=dev); o=[]
    for p,t in zip(pred,tru):
        if len(p)<3 or len(t)<3: continue
        o.append((torch.quantile(torch.diff(p),q)-torch.quantile(torch.diff(t),q)).abs().mean())
    return float(torch.stack(o).mean())
print(f"{'mult':>6} | {'W1(times)':>10} {'Cham+cnt':>10} {'W1(IBI)':>9}")
rows=[]
for m_ in (0.5,0.6,0.7,0.8,0.9,0.95,0.98,1.0,1.02,1.05,1.1,1.2,1.4,1.7,2.0):
    p=beats_at(m_); a,b,c=w1(p,true),cham_cnt(p,true),ibi_w1(p,true)
    rows.append((m_,a,b,c)); print(f"{m_:6.2f} | {a:10.2f} {b:10.2f} {c:9.3f} {'#'*int(c*4)}")
for j,nm in ((1,"W1(times)"),(2,"Cham+cnt"),(3,"W1(IBI)")):
    bst=min(rows,key=lambda r:r[j]); far=abs(rows[1][j]-rows[0][j])/0.1
    ok="<-- CORRECT" if abs(bst[0]-1.0)<1e-9 else ""
    print(f"\n{nm:10s} min at mult={bst[0]:.2f} | far-field slope {far:.2f} (BCE ~0) {ok}")
