"""Extract per-anchor MERT windows for the TCN attack. Layer probe first (grouped ridge),
then dump windows for the best layer: causal [f-99..f] and lookahead [f-50..f+49]."""
import sys, numpy as np, math, json
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
from core import prep
from data import build, FPS
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
SCR='/home/sogang/.tmp/claude-1003/-home-sogang-jaehoon-VBPM/eff15063-3b15-45c2-a6fc-3b22ddda3990/scratchpad'
CACHE='/disk1/jaehoon/vbpm_mert_cache'

def anchors(d):
    n=len(d['u'])
    ks=[k for k in range(1,n) if d['valid'][k] and d['valid'][k-1]]
    return ks

# ---------- pass 1: layer probe on train subset
Dt=prep(build('train'))
rng=np.random.default_rng(0)
sub=list(rng.permutation(len(Dt))[:60])
Xl=[[] for _ in range(13)]; y=[]; g=[]
for si in sub:
    d=Dt[si]
    z=np.load(f"{CACHE}/{d['stem']}.npz")['feats']  # [13,T,768] fp16
    T=z.shape[1]
    for k in anchors(d):
        f=int(round(d['beats'][k]*FPS)); f=max(2,min(T-3,f))
        w=z[:,f-2:f+1,:].astype(np.float32).mean(1)   # [13,768]
        for l in range(13): Xl[l].append(w[l])
        y.append(d['u'][k]-d['u'][k-1]); g.append(d['stem'])
y=np.asarray(y); g=np.asarray(g)
print('probe n=',len(y))
gkf=GroupKFold(n_splits=5)
best=(None,-9)
probe={}
for l in range(13):
    X=np.asarray(Xl[l]); X=(X-X.mean(0))/(X.std(0)+1e-6)
    ss=[]
    for tr,va in gkf.split(X,y,groups=g):
        m=Ridge(alpha=100.0).fit(X[tr],y[tr]); p=m.predict(X[va])
        ss.append(1-((y[va]-p)**2).sum()/((y[va]-y[va].mean())**2).sum())
    r=float(np.mean(ss)); probe[l]=r
    print(f'  layer {l:2d}: grouped-CV R2 {r:+.4f}')
    if r>best[1]: best=(l,r)
L=best[0]
print('best layer',L)
json.dump(dict(probe=probe,best=L),open(f'{SCR}/layer_probe.json','w'))

# ---------- pass 2: extract windows
def extract(split,D,mode,W=100):
    Xw=[]; meta=[]
    for d in D:
        z=np.load(f"{CACHE}/{d['stem']}.npz")['feats'][L]  # [T,768] fp16
        T=len(z)
        for k in anchors(d):
            f=int(round(d['beats'][k]*FPS)); f=max(2,min(T-3,f))
            if mode=='causal': s0,s1=f-W+1,f+1
            else: s0,s1=f-W//2,f+W//2
            s0c,s1c=max(0,s0),min(T,s1)
            w=z[s0c:s1c]
            if s0c>s0: w=np.concatenate([np.repeat(w[:1],s0c-s0,0),w],0)
            if s1c<s1: w=np.concatenate([w,np.repeat(w[-1:],s1-s1c,0)],0)
            Xw.append(w)
            Lp=d['I'][k-1]*FPS
            meta.append((d['u'][k]-d['u'][k-1], d['u'][k-1], math.log(Lp), d['meter'], d['stem'],
                         d['ulo'][k], d['uhi'][k], d['u'][k], d['dataset']))
    Xw=np.stack(Xw).astype(np.float16)
    e,up,lLp,met,stem,ulo,uhi,u,ds=zip(*meta)
    np.savez(f'{SCR}/tcn_{mode}_{split}.npz', X=Xw, e=np.array(e), u_prev=np.array(up),
             logLp=np.array(lLp), meter=np.array(met), stem=np.array(stem),
             ulo=np.array(ulo), uhi=np.array(uhi), u=np.array(u), dataset=np.array(ds))
    print(mode,split,Xw.shape,Xw.nbytes/1e9,'GB')

De=prep(build('eval'))
for mode in ('causal','look'):
    extract('train',Dt,mode); extract('eval',De,mode)
print('done')
