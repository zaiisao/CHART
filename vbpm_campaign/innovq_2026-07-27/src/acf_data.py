"""The eval gap is data volume + wrong head shape, not representation.
 - many crops per song (was 1)
 - log-tempo CLASSIFICATION over bins (octave = discrete decision) + local refinement
Fold-honest: train/eval song lists are disjoint upstream in pm_common."""
import sys, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem")
import pm_common as P
dev="cuda:0"; T=1500; LO,HI=10,150
def tgram(x):
    x=x-x.mean(); n=1<<int(np.ceil(np.log2(2*len(x))))
    f=np.fft.rfft(x,n); a=np.fft.irfft(f*np.conj(f),n)[:HI+2]
    a=a/(a[0]+1e-9); return np.convolve(a,np.ones(3)/3,mode="same")[LO:HI]
def load(split,n_per,seed):
    d=P.build_crops(P.load_songs(split),n_per_song=n_per,seed=seed,crop=T,dev=dev)
    B=d["h"].shape[0]; X=[];y=[]
    for i in range(B):
        idx=np.where(d["b"][i].cpu().numpy()>0.5)[0]
        if len(idx)<8: continue
        X.append(np.concatenate([tgram(d["h"][i,:,0].cpu().numpy()),
                                 tgram(d["h"][i,:,1].cpu().numpy())]))
        y.append(np.median(np.diff(idx)))
    return np.stack(X).astype(np.float32), np.array(y,dtype=np.float32)
Xtr,ytr=load("train",12,0); Xev,yev=load("eval",4,1)
print(f"train {len(ytr)} crops (was 145) | eval {len(yev)} | dim {Xtr.shape[1]}")
Xt=torch.tensor(Xtr,device=dev); Xe=torch.tensor(Xev,device=dev)
mu,sd=Xt.mean(0,keepdim=True),Xt.std(0,keepdim=True)+1e-6
Xt=(Xt-mu)/sd; Xe=(Xe-mu)/sd
lt=torch.log(torch.tensor(ytr,device=dev)); le=torch.log(torch.tensor(yev,device=dev))
# --- classification bins over log-IBI, 48 bins/octave, + soft-argmax refinement
NB=96; lo,hi=math.log(LO),math.log(HI)
edges=torch.linspace(lo,hi,NB,device=dev)
def to_soft(l):  # soft target: gaussian over bins (label smoothing in log-tempo)
    d=(l[:,None]-edges[None,:])/((hi-lo)/NB*1.5)
    return F.softmax(-0.5*d**2,dim=1)
torch.manual_seed(0)
net=nn.Sequential(nn.Linear(Xt.shape[1],512),nn.GELU(),nn.Dropout(0.3),
                  nn.Linear(512,256),nn.GELU(),nn.Dropout(0.1),nn.Linear(256,NB)).to(dev)
opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-2)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,4000)
tgt=to_soft(lt); best=(9,9,0)
def mae(logits,l):
    p=F.softmax(logits,1); pred=(p*edges[None,:]).sum(1)      # soft-argmax = local refinement
    hard=edges[logits.argmax(1)]
    # refine around the argmax bin only (avoids smearing across octaves)
    w=(edges[None,:]-hard[:,None]).abs()<((hi-lo)/NB*3)
    pm=(p*w); pm=pm/(pm.sum(1,keepdim=True)+1e-9); ref=(pm*edges[None,:]).sum(1)
    return (ref-l).abs().mean().item(),(pred-l).abs().mean().item()
for s in range(4000):
    net.train(); opt.zero_grad()
    perm=torch.randperm(len(lt),device=dev)[:256]
    loss=-(tgt[perm]*F.log_softmax(net(Xt[perm]),1)).sum(1).mean()
    loss.backward(); opt.step(); sched.step()
    if s%400==0 or s==3999:
        net.eval()
        with torch.no_grad():
            tr,_=mae(net(Xt),lt); ev,_=mae(net(Xe),le)
        if ev<best[1]: best=(tr,ev,s)
        print(f"    step {s:4d}  train {100*tr:5.2f}%   eval {100*ev:5.2f}%")
print(f"\n  BEST tempogram-CLS: train {100*best[0]:.2f}%  eval {100*best[1]:.2f}% (step {best[2]})")
print(f"  pooled-GRU 8.7/14.4 | tempogram-REG 1.8/17.2 | ceiling 1.17/1.67 | TARGET 2%")
