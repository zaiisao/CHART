"""Give the tempo head a PERIODICITY-NATIVE feature: autocorrelation of the beat activation.
Baseline (mean-pooled GRU features): MAE 0.153 (15.3%). TARGET <=0.02 (2%) for corr>=0.82.
Test A: can a tiny head on autocorr features alone predict log-tempo? (representation test)
Test B: what MAE does the classic argmax-of-autocorr estimator get? (no learning at all)"""
import sys, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem")
import pm_common as P
dev="cuda:0"; torch.manual_seed(0); rng=np.random.default_rng(0); T=1500; TWO_PI=2*math.pi
tr=P.build_crops(P.load_songs("train"),n_per_song=1,seed=0,crop=T,dev=dev)
ev=P.build_crops(P.load_songs("eval"), n_per_song=1,seed=1,crop=T,dev=dev)
LAGS=torch.arange(10,400,device=dev)          # 0.2s..8s at 50fps: covers 30-300 BPM beat & bar
def autocorr(h):
    """h [B,T,2] -> [B,len(LAGS)] normalized autocorrelation of the beat channel"""
    x=h[...,0]-h[...,0].mean(1,keepdim=True)
    n=x.shape[1]; out=[]
    for L in LAGS.tolist():
        out.append((x[:,:n-L]*x[:,L:]).mean(1))
    a=torch.stack(out,1)
    return a/ (a.abs().max(1,keepdim=True).values+1e-8)
with torch.no_grad():
    Atr,Aev=autocorr(tr["h"]),autocorr(ev["h"])
    ytr=torch.log(torch.exp(tr["lt"]).median(1).values); yev=torch.log(torch.exp(ev["lt"]).median(1).values)
print(f"autocorr features: {Atr.shape[1]} lags | train {Atr.shape[0]} eval {Aev.shape[0]}")
# --- Test B: parameter-free argmax-of-autocorr -> bar period -> log bar-advance
with torch.no_grad():
    peak=LAGS[Aev.argmax(1)].float()                       # frames per (dominant) period
    for mult,name in ((1,"period=bar"),(4,"period=beat,x4 for bar")):
        est=torch.log(TWO_PI/(peak*mult))
        print(f"  [B] argmax-autocorr ({name}): MAE {float((est-yev).abs().mean()):.3f}  corr {float(torch.corrcoef(torch.stack([est,yev]))[0,1]):+.3f}")
# --- Test A: tiny MLP on autocorr features
net=nn.Sequential(nn.Linear(Atr.shape[1],128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1)).to(dev)
opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
for s in range(1,3001):
    i=torch.tensor(rng.integers(0,Atr.shape[0],32),device=dev,dtype=torch.long)
    loss=F.mse_loss(net(Atr[i]).squeeze(-1),ytr[i])
    opt.zero_grad(); loss.backward(); opt.step()
    if s%1000==0:
        with torch.no_grad():
            p=net(Aev).squeeze(-1)
            mae=float((p-yev).abs().mean()); r=float(torch.corrcoef(torch.stack([p,yev]))[0,1])
        print(f"  [A] MLP-on-autocorr s{s}: eval MAE {mae:.3f} ({100*mae:.1f}%)  corr {r:+.3f}",flush=True)
print(f"\n  baseline (GRU mean-pool head): MAE 0.153 (15.3%), corr +0.78")
print(f"  TARGET for phase corr>=0.82 @T=1500: MAE <= 0.020 (2%)")
