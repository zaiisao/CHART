"""Two questions:
 (1) With ORACLE octave, how precise is the ACF? -> precision ceiling of the 2-ch signal.
 (2) Is the octave LEARNABLE from the tempogram? -> practical MAE, no rich features needed."""
import sys, math
import numpy as np, torch, torch.nn as nn
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem")
import pm_common as P
dev="cuda:0"; T=1500
LO,HI=10,150
def tempogram(x):
    """log-lag ACF vector: the standard periodicity representation."""
    x=x-x.mean(); n=1<<int(np.ceil(np.log2(2*len(x))))
    f=np.fft.rfft(x,n); a=np.fft.irfft(f*np.conj(f),n)[:HI+2]
    a=a/(a[0]+1e-9); a=np.convolve(a,np.ones(3)/3,mode="same")
    return a[LO:HI]
def load(split,seed):
    d=P.build_crops(P.load_songs(split),n_per_song=1,seed=seed,crop=T,dev=dev)
    B=d["h"].shape[0]
    A=np.stack([tempogram(d["h"][i,:,0].cpu().numpy()) for i in range(B)])
    A2=np.stack([tempogram(d["h"][i,:,1].cpu().numpy()) for i in range(B)])
    ibi=np.array([np.median(np.diff(np.where(d["b"][i].cpu().numpy()>0.5)[0])) for i in range(B)])
    return np.concatenate([A,A2],1).astype(np.float32), ibi.astype(np.float32)
Xtr,ytr=load("train",0); Xev,yev=load("eval",1)
print(f"tempogram {Xtr.shape[1]}-d | train {len(ytr)} eval {len(yev)}")
# ---- (1) oracle-octave precision ceiling
def peak(a):
    i=int(np.argmax(a))
    if 0<i<len(a)-1:
        d=(a[i-1]-a[i+1])/(2*(a[i-1]-2*a[i]+a[i+1])+1e-12); return i+float(np.clip(d,-1,1))+LO
    return i+LO
for nm,X,y in (("train",Xtr,ytr),("eval",Xev,yev)):
    pk=np.array([peak(X[i,:HI-LO]) for i in range(len(y))])
    best=np.array([min([pk[i]*f for f in (0.25,1/3,0.5,2/3,1,1.5,2,3,4)],
                       key=lambda c: abs(math.log(c/y[i]))) for i in range(len(y))])
    m=np.abs(np.log(best/y)).mean()
    print(f"  [{nm}] ORACLE-OCTAVE ACF MAE {m:.4f} ({100*m:.2f}%)  <- precision ceiling of 2-ch signal")
# ---- (2) learn log-tempo directly from the tempogram (small MLP, no GRU, no pooling)
torch.manual_seed(0)
Xt=torch.tensor(Xtr,device=dev); yt=torch.log(torch.tensor(ytr,device=dev))
Xe=torch.tensor(Xev,device=dev); ye=torch.log(torch.tensor(yev,device=dev))
mu,sd=Xt.mean(0,keepdim=True),Xt.std(0,keepdim=True)+1e-6
Xt=(Xt-mu)/sd; Xe=(Xe-mu)/sd
net=nn.Sequential(nn.Linear(Xt.shape[1],512),nn.GELU(),nn.Dropout(0.2),
                  nn.Linear(512,256),nn.GELU(),nn.Linear(256,1)).to(dev)
opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-3)
best=(9,9)
for s in range(3000):
    net.train(); opt.zero_grad()
    l=(net(Xt).squeeze(-1)-yt).abs().mean(); l.backward(); opt.step()
    if s%250==0 or s==2999:
        net.eval()
        with torch.no_grad():
            te=(net(Xt).squeeze(-1)-yt).abs().mean().item()
            ee=(net(Xe).squeeze(-1)-ye).abs().mean().item()
        if ee<best[1]: best=(te,ee)
        print(f"    step {s:4d}  train MAE {te:.4f} ({100*te:.1f}%)  eval MAE {ee:.4f} ({100*ee:.1f}%)")
print(f"\n  BEST tempogram-MLP: train {100*best[0]:.1f}%  eval {100*best[1]:.1f}%")
print(f"  vs pooled-GRU encoder: train 8.7%  eval 14.4%     TARGET 2%")
