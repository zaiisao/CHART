"""Does the tempogram tempo actually buy phase corr? Same isolation harness, new tempo sources:
  A true phi1 + TRUE per-frame tempo   -> ceiling
  B true phi1 + MODEL (GRU) tempo      -> where we were
  F true phi1 + CONST true median      -> cost of constant-tempo assumption alone
  G true phi1 + TEMPOGRAM tempo        -> what the new head buys
  H true phi1 + TEMPOGRAM oracle-octave-> what octave resolution would buy
  I model phi1 + TEMPOGRAM tempo       -> deployable (no truth at all except phi1... see J)
  J tempogram tempo + phi1 from peak-pick argmax -> fully deployable
"""
import sys, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem")
import pm_common as P, innovq as IQ
from rollout_vec import rollout_vec
dev="cuda:0"; torch.manual_seed(0); T=1500; TWO_PI=2*math.pi; LO,HI=10,150
P.PHYS["gamma_phase"]=0.06; P.PHYS["dev_sigma"]=0.02
def tgram(x):
    x=x-x.mean(); n=1<<int(np.ceil(np.log2(2*len(x))))
    f=np.fft.rfft(x,n); a=np.fft.irfft(f*np.conj(f),n)[:HI+2]
    a=a/(a[0]+1e-9); return np.convolve(a,np.ones(3)/3,mode="same")[LO:HI]
def feats(d,i): return np.concatenate([tgram(d["h"][i,:,0].cpu().numpy()),
                                       tgram(d["h"][i,:,1].cpu().numpy())])
def med_ibi(d,i):
    idx=np.where(d["b"][i].cpu().numpy()>0.5)[0]
    return np.median(np.diff(idx)) if len(idx)>=8 else np.nan
# ---- train tempogram head (train split, many crops)
tr=P.build_crops(P.load_songs("train"),n_per_song=12,seed=0,crop=T,dev=dev)
Xs,ys=[],[]
for i in range(tr["h"].shape[0]):
    v=med_ibi(tr,i)
    if not np.isnan(v): Xs.append(feats(tr,i)); ys.append(v)
Xtr=torch.tensor(np.stack(Xs),dtype=torch.float32,device=dev); ytr=torch.log(torch.tensor(ys,dtype=torch.float32,device=dev))
mu,sd=Xtr.mean(0,keepdim=True),Xtr.std(0,keepdim=True)+1e-6; Xtr=(Xtr-mu)/sd
NB=96; lo,hi=math.log(LO),math.log(HI); edges=torch.linspace(lo,hi,NB,device=dev)
tgt=F.softmax(-0.5*((ytr[:,None]-edges[None,:])/((hi-lo)/NB*1.5))**2,1)
net=nn.Sequential(nn.Linear(Xtr.shape[1],512),nn.GELU(),nn.Dropout(0.3),
                  nn.Linear(512,256),nn.GELU(),nn.Dropout(0.1),nn.Linear(256,NB)).to(dev)
opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-2)
for s in range(400):
    net.train(); opt.zero_grad(); pm=torch.randperm(len(ytr),device=dev)[:256]
    (-(tgt[pm]*F.log_softmax(net(Xtr[pm]),1)).sum(1).mean()).backward(); opt.step()
net.eval()
# ---- eval crops (identical to isolate_tempo.py)
ev=P.build_crops(P.load_songs("eval"),n_per_song=1,seed=1,crop=T,dev=dev)
n=min(64,ev["h"].shape[0])
Xe=torch.tensor(np.stack([feats(ev,i) for i in range(n)]),dtype=torch.float32,device=dev); Xe=(Xe-mu)/sd
with torch.no_grad():
    lg=net(Xe); p=F.softmax(lg,1); hard=edges[lg.argmax(1)]
    w=(edges[None,:]-hard[:,None]).abs()<((hi-lo)/NB*3); pm2=p*w; pm2=pm2/(pm2.sum(1,keepdim=True)+1e-9)
    tg_ibi=torch.exp((pm2*edges[None,:]).sum(1))                      # frames per beat
true_ibi=torch.tensor([med_ibi(ev,i) for i in range(n)],dtype=torch.float32,device=dev)
oct_ibi=torch.stack([min([tg_ibi[i]*f for f in (0.25,1/3,0.5,2/3,1,1.5,2,3,4)],
                         key=lambda c: abs(math.log(float(c)/float(true_ibi[i])))) for i in range(n)])
m=IQ.InnovQ().to(dev)
m.load_state_dict(torch.load("tempofix_stage1.pt",map_location=dev,weights_only=False)["model"],strict=False)
m.eval()
with torch.no_grad(): ro=rollout_vec(m,ev["h"][:n],ev["b"][:n],n_picard=3)
tphi=ev["phi"][:n]; tlt=ev["lt"][:n]; mphi1=ro["phi"][:,:1]; mlt=ro["lt"][:n]
def const_lt(ibi):  # bar phase advances TWO_PI/(M*ibi) per frame; recover M from truth scale
    return torch.log(TWO_PI/(4.0*ibi))[:,None].expand(-1,T)
def build(phi1,lt):
    inc=F.pad(torch.exp(lt.clamp(-12,6))[:,:-1],(1,0)); return (phi1+torch.cumsum(inc,1))%TWO_PI
def corr(p): return float(torch.abs(torch.exp(1j*(p-tphi)).mean(1)).mean())
with torch.no_grad():
    print(f"n={n} eval crops, T={T}")
    print(f"  A true phi1 + TRUE per-frame tempo : {corr(build(tphi[:,:1],tlt)):.3f}   <- ceiling")
    print(f"  B true phi1 + GRU tempo            : {corr(build(tphi[:,:1],mlt)):.3f}   <- where we were")
    print(f"  F true phi1 + CONST true median    : {corr(build(tphi[:,:1],const_lt(true_ibi))):.3f}   <- constant-tempo cost")
    print(f"  G true phi1 + TEMPOGRAM tempo      : {corr(build(tphi[:,:1],const_lt(tg_ibi))):.3f}   <- new head")
    print(f"  H true phi1 + TEMPOGRAM oracle-oct : {corr(build(tphi[:,:1],const_lt(oct_ibi))):.3f}   <- octave resolved")
    print(f"  I GRU phi1  + TEMPOGRAM oracle-oct : {corr(build(mphi1,const_lt(oct_ibi))):.3f}")
    mae=float((torch.log(tg_ibi)-torch.log(true_ibi)).abs().mean())
    omae=float((torch.log(oct_ibi)-torch.log(true_ibi)).abs().mean())
    print(f"\n  tempogram tempo MAE {100*mae:.1f}%   oracle-octave {100*omae:.2f}%   GRU {100*float((mlt-tlt).abs().mean()):.1f}%")
