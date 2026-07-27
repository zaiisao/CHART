"""Our tempogram head vs madmom's tempo estimators, on IDENTICAL beat activations.
Metrics: Acc1 (within 4% of true tempo) and Acc2 (octave-tolerant, the standard pair).
NOTE asymmetry: ours is TRAINED on the train split; madmom's are hand-designed with a
built-in tempo prior and see no training data. Both see the same Beat This activations.
"""
import sys, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
for p in ("/home/sogang/jaehoon/VBPM_reintegration","/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix",
          "/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq","/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem"):
    sys.path.insert(0,p)
import pm_common as P
from madmom.features.tempo import TempoEstimationProcessor
dev="cuda:0"; T=1500; FPS=50.0; LO,HI=10,150
def tgram(x):
    x=x-x.mean(); n=1<<int(np.ceil(np.log2(2*len(x))))
    f=np.fft.rfft(x,n); a=np.fft.irfft(f*np.conj(f),n)[:HI+2]
    a=a/(a[0]+1e-9); return np.convolve(a,np.ones(3)/3,mode="same")[LO:HI]
def load(split,n_per,seed):
    d=P.build_crops(P.load_songs(split),n_per_song=n_per,seed=seed,crop=T,dev=dev)
    X,y,A=[],[],[]
    for i in range(d["h"].shape[0]):
        idx=np.where(d["b"][i].cpu().numpy()>0.5)[0]
        if len(idx)<8: continue
        act=d["h"][i,:,0].cpu().numpy().astype(np.float64)
        X.append(np.concatenate([tgram(act),tgram(d["h"][i,:,1].cpu().numpy())]))
        y.append(60.0*FPS/np.median(np.diff(idx)))          # true BPM
        A.append(act)
    return np.stack(X).astype(np.float32),np.array(y),A
Xtr,ytr,_=load("train",12,0); Xev,yev,Aev=load("eval",4,1)
print(f"train {len(ytr)} crops, eval {len(yev)} crops; true BPM median {np.median(yev):.1f}",flush=True)
# ---- ours: tempogram -> log-BPM classifier
Xt=torch.tensor(Xtr,device=dev); Xe=torch.tensor(Xev,device=dev)
mu,sd=Xt.mean(0,keepdim=True),Xt.std(0,keepdim=True)+1e-6; Xt=(Xt-mu)/sd; Xe=(Xe-mu)/sd
lt=torch.log(torch.tensor(ytr,dtype=torch.float32,device=dev))
NB=96; lo,hi=math.log(40),math.log(260); edges=torch.linspace(lo,hi,NB,device=dev)
tgt=F.softmax(-0.5*((lt[:,None]-edges[None,:])/((hi-lo)/NB*1.5))**2,1)
torch.manual_seed(0)
net=nn.Sequential(nn.Linear(Xt.shape[1],512),nn.GELU(),nn.Dropout(0.3),
                  nn.Linear(512,256),nn.GELU(),nn.Dropout(0.1),nn.Linear(256,NB)).to(dev)
opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-2)
for s in range(400):
    net.train(); opt.zero_grad(); pm=torch.randperm(len(lt),device=dev)[:256]
    (-(tgt[pm]*F.log_softmax(net(Xt[pm]),1)).sum(1).mean()).backward(); opt.step()
net.eval()
with torch.no_grad():
    lg=net(Xe); p=F.softmax(lg,1); hard=edges[lg.argmax(1)]
    w=(edges[None,:]-hard[:,None]).abs()<((hi-lo)/NB*3); pm2=p*w; pm2=pm2/(pm2.sum(1,keepdim=True)+1e-9)
    ours=torch.exp((pm2*edges[None,:]).sum(1)).cpu().numpy()
# ---- madmom
res={"ours (tempogram, trained)":ours}
for m in ("acf","comb","dbn"):
    proc=TempoEstimationProcessor(method=m,min_bpm=40,max_bpm=250,fps=FPS)
    est=[]
    for a in Aev:
        try:
            t=proc(a)
            est.append(float(t[0][0]) if len(t) else np.nan)
        except Exception as e:
            est.append(np.nan)
    res[f"madmom {m}"]=np.array(est)
def acc(e,t):
    ok=~np.isnan(e)
    lr=np.log(e[ok]/t[ok])
    a1=np.mean(np.abs(lr)<math.log(1.04))
    oct_=np.minimum(np.abs(lr),np.minimum(np.abs(lr-math.log(2)),
         np.minimum(np.abs(lr+math.log(2)),np.minimum(np.abs(lr-math.log(3)),np.abs(lr+math.log(3))))))
    a2=np.mean(oct_<math.log(1.04))
    return a1,a2,np.abs(lr).mean(),ok.mean()
print(f"\n{'estimator':30s} {'Acc1':>7s} {'Acc2':>7s} {'MAE(log)':>9s} {'valid':>7s}")
for k,v in res.items():
    a1,a2,mae,val=acc(v,yev)
    print(f"{k:30s} {100*a1:6.1f}% {100*a2:6.1f}% {100*mae:8.1f}% {100*val:6.0f}%")
