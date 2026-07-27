"""CLAIM UNDER TEST: the observation likelihood p(o|z) resolves the tempo octave.
At 2x the model puts a beat between every true beat (lands on low activation);
at 0.5x it explains only half. Score each octave candidate by the activation
evidence at the predicted beat frames (best offset), pick argmax, measure.

Deploy-legal: uses only h (the activation), never b/annotations."""
import sys, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
for p in ("/home/sogang/jaehoon/VBPM_reintegration","/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix",
          "/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq","/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem"):
    sys.path.insert(0,p)
import pm_common as P
dev="cuda:0"; T=1500; TWO_PI=2*math.pi; LO,HI=10,150
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
        X.append(np.concatenate([tgram(d["h"][i,:,0].cpu().numpy()),tgram(d["h"][i,:,1].cpu().numpy())]))
        y.append(np.median(np.diff(idx))); A.append(d["h"][i,:,0].cpu().numpy())
    return np.stack(X).astype(np.float32),np.array(y,np.float32),np.stack(A).astype(np.float32)
Xtr,ytr,_=load("train",12,0); Xev,yev,Aev=load("eval",4,1)
Xt=torch.tensor(Xtr,device=dev); Xe=torch.tensor(Xev,device=dev)
mu,sd=Xt.mean(0,keepdim=True),Xt.std(0,keepdim=True)+1e-6; Xt=(Xt-mu)/sd; Xe=(Xe-mu)/sd
lt=torch.log(torch.tensor(ytr,device=dev)); NB=96; lo,hi=math.log(LO),math.log(HI)
edges=torch.linspace(lo,hi,NB,device=dev)
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
    base=torch.exp((pm2*edges[None,:]).sum(1)).cpu().numpy()          # frames/beat
FACT=[1/3,0.5,2/3,1.0,1.5,2.0,3.0]
def evidence(a,ibi):
    """max over offset of mean activation at predicted beats, MINUS mean activation
       (so denser grids are not rewarded for free) -> p(o|z) surrogate."""
    best=-9
    for off in np.arange(0,ibi,max(ibi/16,0.5)):
        idx=np.round(np.arange(off,len(a)-1,ibi)).astype(int)
        if len(idx)<4: continue
        best=max(best,float(a[idx].mean()))
    return best
n=len(yev); pick=np.zeros(n); prior_only=base.copy()
for i in range(n):
    sc=[evidence(Aev[i],base[i]*f) for f in FACT]
    pick[i]=base[i]*FACT[int(np.argmax(sc))]
def rep(nm,e):
    lr=np.log(e/yev); mae=np.abs(lr).mean()
    ok=np.mean(np.abs(lr)<0.04)
    print(f"  {nm:34s} MAE {100*mae:5.2f}%   within-4% {100*ok:5.1f}%")
print(f"eval crops {n}")
rep("tempogram alone (no likelihood)",prior_only)
rep("tempogram + p(o|z) octave pick",pick)
oracle=np.array([min([base[i]*f for f in FACT],key=lambda c:abs(math.log(c/yev[i]))) for i in range(n)])
rep("oracle octave (upper bound)",oracle)
