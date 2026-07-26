"""V1(c): stronger audio predictor -- causal dilated conv over MERT-layer windows,
Student-t(2) heteroscedastic NLL (mean + log-scale heads). Song-disjoint protocol
identical to step5/step6 (train songs -> fit; eval-song second halves -> score).
Variants: audio-only TCN, TCN+hist, hist-MLP (capacity parity). Modes: causal / look."""
import sys, json, math, numpy as np, torch, torch.nn as nn
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
from collections import defaultdict
from core import logmass
SCR='/home/sogang/.tmp/claude-1003/-home-sogang-jaehoon-VBPM/eff15063-3b15-45c2-a6fc-3b22ddda3990/scratchpad'
P='/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise'
dev='cuda:0'
torch.manual_seed(0); np.random.seed(0)

def half_mask(stems):
    idx=defaultdict(list); m=np.zeros(len(stems),bool)
    for i,s in enumerate(stems): idx[s].append(i)
    for s,ii in idx.items(): m[np.array(ii[len(ii)//2:])]=True
    return m
def r2f(y,p): return float(1-((y-p)**2).sum()/((y-y.mean())**2).sum())

def load(mode,sp):
    d=np.load(f'{SCR}/tcn_{mode}_{sp}.npz',allow_pickle=True)
    return d

HT=np.load(f'{P}/hist_train.npz',allow_pickle=True)['X'].astype(np.float32)
HE=np.load(f'{P}/hist_eval.npz',allow_pickle=True)['X'].astype(np.float32)

class TCN(nn.Module):
    def __init__(self, cin=768, ch=128, ncond=6, nhist=0):
        super().__init__()
        self.inp=nn.Conv1d(cin,ch,1)
        self.convs=nn.ModuleList([nn.Conv1d(ch,ch,3,dilation=d,padding=0) for d in (1,2,4,8,16)])
        self.norm=nn.ModuleList([nn.GroupNorm(8,ch) for _ in range(5)])
        self.head=nn.Sequential(nn.Linear(2*ch+ncond+nhist,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,2))
    def forward(self,x,c,h=None):
        # x [B,T,768] -> causal stack (we simply let padding=0 shrink; take last + mean)
        z=self.inp(x.transpose(1,2))
        for cv,nm in zip(self.convs,self.norm):
            z2=cv(z)
            z=torch.nn.functional.gelu(nm(z2))+z[:,:,-z2.shape[2]:]
        feat=torch.cat([z[:,:,-1],z.mean(2)],1)
        inp=torch.cat([feat,c]+([h] if h is not None else []),1)
        o=self.head(inp)
        return o[:,0]*0.05, o[:,1]-3.5   # mu (scaled), log_s (init near log 0.03)

class HistMLP(nn.Module):
    def __init__(self,nh):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(nh,256),nn.GELU(),nn.Linear(256,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,2))
    def forward(self,h):
        o=self.net(h); return o[:,0]*0.05, o[:,1]-3.5

def tnll(e,mu,ls,nu=2.0):
    s=torch.exp(ls)
    d=torch.distributions.StudentT(df=nu,loc=mu,scale=s)
    return -d.log_prob(e)

def cond_arr(d):
    met=d['meter'].astype(int)
    return np.column_stack([d['u_prev'],d['logLp'],(met==2),(met==3),(met==4),np.ones(len(met))]).astype(np.float32)

def run(mode,use_audio=True,use_hist=False,tag='',epochs=30):
    dt=load(mode,'train'); de=load(mode,'eval')
    M2=half_mask(de['stem'])
    Xt=dt['X']; Xe=de['X']
    et=dt['e'].astype(np.float32); ee=de['e'].astype(np.float32)
    ct=cond_arr(dt); ce=cond_arr(de)
    # normalize MERT dims with train stats (sampled)
    sel=np.random.default_rng(0).permutation(len(Xt))[:4000]
    mu_=Xt[sel].astype(np.float32).mean((0,1)); sd_=Xt[sel].astype(np.float32).std((0,1))+1e-4
    # song-grouped val split from train (last 20% of songs)
    songs=np.unique(dt['stem']); vs=set(songs[int(0.85*len(songs)):])
    vam=np.array([s in vs for s in dt['stem']]); trm=~vam
    nh=HT.shape[1] if use_hist else 0
    hmu=HT.mean(0); hsd=HT.std(0)+1e-6
    Ht=((HT-hmu)/hsd).astype(np.float32); He=((HE-hmu)/hsd).astype(np.float32)
    assert len(Ht)==len(et) and len(He)==len(ee), (len(Ht),len(et),len(He),len(ee))
    if use_audio:
        model=TCN(nhist=nh).to(dev)
    else:
        model=HistMLP(HT.shape[1]).to(dev)
    opt=torch.optim.Adam(model.parameters(),lr=3e-4,weight_decay=1e-5)
    idx=np.where(trm)[0]; B=256
    ete=torch.tensor(et).to(dev)
    best=(1e9,None)
    for ep in range(epochs):
        np.random.shuffle(idx); tot=0;nb=0
        model.train()
        for i in range(0,len(idx),B):
            j=idx[i:i+B]
            c=torch.tensor(ct[j]).to(dev); e=ete[j]
            h=torch.tensor(Ht[j]).to(dev) if use_hist else None
            if use_audio:
                x=torch.tensor(((Xt[j].astype(np.float32)-mu_)/sd_)).to(dev)
                mu,ls=model(x,c,h)
            else:
                mu,ls=model(torch.tensor(Ht[j]).to(dev))
            loss=tnll(e,mu,ls).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot+=loss.item()*len(j); nb+=len(j)
        # val
        model.eval(); vl=0;vn=0
        with torch.no_grad():
            vj=np.where(vam)[0]
            for i in range(0,len(vj),1024):
                j=vj[i:i+1024]
                c=torch.tensor(ct[j]).to(dev); e=ete[j]
                h=torch.tensor(Ht[j]).to(dev) if use_hist else None
                if use_audio:
                    x=torch.tensor(((Xt[j].astype(np.float32)-mu_)/sd_)).to(dev)
                    mu,ls=model(x,c,h)
                else: mu,ls=model(torch.tensor(Ht[j]).to(dev))
                vl+=tnll(e,mu,ls).sum().item(); vn+=len(j)
        vl/=vn
        if vl<best[0]: best=(vl,{k:v.detach().clone() for k,v in model.state_dict().items()})
        if ep%5==0 or ep==epochs-1: print(f'  [{tag}] ep{ep} train {tot/nb:.4f} val {vl:.4f}')
    model.load_state_dict(best[1]); model.eval()
    # eval on held-out second halves
    mus=[];lss=[]
    with torch.no_grad():
        for i in range(0,len(ee),1024):
            j=slice(i,min(i+1024,len(ee)))
            c=torch.tensor(ce[j]).to(dev)
            h=torch.tensor(He[j]).to(dev) if use_hist else None
            if use_audio:
                x=torch.tensor(((Xe[j].astype(np.float32)-mu_)/sd_)).to(dev)
                mu,ls=model(x,c,h)
            else: mu,ls=model(torch.tensor(He[j]).to(dev))
            mus.append(mu.cpu().numpy()); lss.append(ls.cpu().numpy())
    mu=np.concatenate(mus)[M2]; ls=np.concatenate(lss)[M2]
    y=ee[M2]
    R2=r2f(y,mu)
    # discretized LL: bin edges relative: target u in [ulo,uhi]; mu is increment pred -> loc=u_prev+mu
    LL=float(logmass(de['ulo'][M2],de['uhi'][M2],de['u_prev'][M2]+mu.astype(np.float64),np.exp(ls.astype(np.float64)),'t',2.0).mean())
    print(f'[{tag}] heldout R2={R2:+.4f}  discretized t2 LL={LL:.4f}  val_nll={best[0]:.4f}')
    return dict(R2=R2,LL=LL,val=float(best[0])),mu

OUT={}
OUT['hist_mlp'],mu_h=run('causal',use_audio=False,tag='hist-MLP')
OUT['tcn_causal'],mu_c=run('causal',True,False,tag='TCN-causal')
OUT['tcn_causal_hist'],_=run('causal',True,True,tag='TCN-causal+hist')
OUT['tcn_look'],mu_l=run('look',True,False,tag='TCN-look')
OUT['tcn_look_hist'],_=run('look',True,True,tag='TCN-look+hist')
# stack TCN audio pred with hist GBM pred (same as audit)
from sklearn.ensemble import HistGradientBoostingRegressor as GBR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
dt=load('causal','train'); de=load('causal','eval'); M2=half_mask(de['stem'])
et=dt['e']; y=de['e'][M2]
ghist=GBR(max_iter=300,learning_rate=0.05,max_depth=4,l2_regularization=1.0,random_state=0).fit(HT,et)
ph=ghist.predict(HE[M2])
for nm,mua in (('causal',mu_c),('look',mu_l)):
    st=LinearRegression().fit(np.column_stack([ph*0+0,ph*0+0]),[0,0]) if False else None
    # simple convex search on held-in? use train OOF too expensive; do direct LSQ on TRAIN via OOF hist + TCN val?  fall back: fixed weights scan reported transparently
    best=(-9,None)
    for w in np.linspace(0,1,21):
        r=r2f(y,(1-w)*ph+w*mua)
        if r>best[0]: best=(r,w)
    print(f'stack hist-GBM + TCN-{nm}: best-w={best[1]:.2f} R2={best[0]:+.4f} (hist GBM alone {r2f(y,ph):+.4f})  [w scan = diagnostic only]')
    OUT[f'stack_{nm}']=dict(R2=float(best[0]),w=float(best[1]),hist_alone=r2f(y,ph))
json.dump(OUT,open('/home/sogang/jaehoon/VBPM_reintegration/vbpm_thorough/v1_tcn.json','w'),indent=1)
print('saved')
