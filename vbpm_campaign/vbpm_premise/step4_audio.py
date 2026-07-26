"""(d) AUDIO-CONDITIONED transition: p(u_k | u_{k-1}, activation) with learned mean AND scale.
Trained on TRAIN songs (20% of train songs held out for early stopping), scored on the
SECOND HALF of EVAL songs -- the identical held-out set used for (a),(b),(c)."""
import sys, math, json, numpy as np, torch, torch.nn as nn
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
from core import *
from feats import song_feats
from scipy.optimize import minimize_scalar
torch.manual_seed(0); np.random.seed(0)

def collect(D, mode='full'):
    Xs=[]; P=[]
    for d in D:
        X,ks = song_feats(d, 300, mode)
        if not len(ks): continue
        Xs.append(X); P += [(d,k) for k in ks]
    return np.concatenate(Xs,0), P

Dtr = prep(build('train')); Dev = prep(build('eval'))
CACHE={}
for mode in ('full','causal'):
    Xtr,Ptr = collect(Dtr,mode); Xev,Pev = collect(Dev,mode)
    CACHE[mode]=(Xtr,Ptr,Xev,Pev)
    print(mode,'Xtr',Xtr.shape,'Xev',Xev.shape)
Gtr = gather(CACHE['full'][1]); Gev = gather(CACHE['full'][3])
# masks selecting the held-out (2nd-half) eval pairs, in the SAME order as Xev
def half_mask(P):
    from collections import defaultdict
    idx=defaultdict(list)
    for i,(d,k) in enumerate(P): idx[d['stem']].append(i)
    m2=np.zeros(len(P),bool)
    for st,ii in idx.items():
        h=len(ii)//2
        m2[np.array(ii[h:])]=True
    return m2
M2 = half_mask(CACHE['full'][3])
print('held-out (2nd half) eval pairs:', int(M2.sum()))

# ---- global fixed law (a), refit here on identical pairs -------------------------
from scipy.optimize import minimize
def fit_t(G):
    f=lambda th: -logmass(G['ulo'],G['uhi'],G['u_prev']+th[0],math.exp(th[1]),'t',math.exp(th[2])+0.2).mean()
    r=minimize(f,[0.,math.log(.03),math.log(2.)],method='Nelder-Mead',options=dict(maxiter=6000,xatol=1e-7,fatol=1e-9))
    return dict(c=r.x[0], s=math.exp(r.x[1]), nu=math.exp(r.x[2])+0.2)
A = fit_t(Gtr)
sub=lambda G,m: {k:(v[m] if isinstance(v,np.ndarray) else v) for k,v in G.items() if k!='n'}
Gho = sub(Gev,M2)
ll_a = logmass(Gho['ulo'],Gho['uhi'],Gho['u_prev']+A['c'],A['s'],'t',A['nu'])
print(f"(a) GLOBAL fixed  c={A['c']:+.5f} s={A['s']:.5f} nu={A['nu']:.2f}   HO {ll_a.mean():+.4f}")

# ---- neural conditional ---------------------------------------------------------
def lap_logmass_t(lo,hi,mu,s):
    z1=(hi-mu)/s; z0=(lo-mu)/s
    F=lambda z: torch.where(z<0, 0.5*torch.exp(z), 1-0.5*torch.exp(-z))
    return torch.log(torch.clamp(F(z1)-F(z0),min=1e-300))

def run(mode, head='both', shuffle=False, seed=0, epochs=200):
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    Xtr,Ptr,Xev,Pev = CACHE[mode]
    Gt, Ge = gather(Ptr), gather(Pev)
    Xtr=Xtr.copy(); Xev=Xev.copy()
    if shuffle:   # CONTROL: audio features from a random other song-position
        Xtr[:, :-4] = Xtr[rng.permutation(len(Xtr))][:, :-4]
        Xev[:, :-4] = Xev[rng.permutation(len(Xev))][:, :-4]
    mu_,sd_ = Xtr.mean(0), Xtr.std(0)+1e-6
    Zt=torch.tensor((Xtr-mu_)/sd_); Ze=torch.tensor((Xev-mu_)/sd_)
    stems=np.array([d['stem'] for d,k in Ptr]); us=np.unique(stems)
    vs=set(rng.choice(us, size=max(1,len(us)//5), replace=False))
    vm=np.array([s in vs for s in stems]); tm=~vm
    T=lambda G,k: torch.tensor(G[k], dtype=torch.float64)
    net=nn.Sequential(nn.Linear(Zt.shape[1],64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,2)).double()
    nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)
    opt=torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    base_ls=math.log(A['s'])
    def pred(Z,G):
        o=net(Z); mu=T(G,'u_prev')+A['c']; ls=torch.full_like(mu,base_ls)
        if head in ('both','mean'):  mu = mu + 0.05*o[:,0]
        if head in ('both','scale'): ls = ls + 0.5*torch.tanh(o[:,1])*3.0
        return mu, torch.exp(ls)
    def nll(Z,G,m=None):
        mu,s=pred(Z,G); lo,hi=T(G,'ulo'),T(G,'uhi')
        v=-lap_logmass_t(lo,hi,mu,s)
        return v[torch.tensor(m)].mean() if m is not None else v.mean()
    best=(1e9,None)
    for ep in range(epochs):
        opt.zero_grad(); l=nll(Zt,Gt,tm); l.backward(); opt.step()
        with torch.no_grad(): v=float(nll(Zt,Gt,vm))
        if v<best[0]-1e-6: best=(v,[p.detach().clone() for p in net.parameters()])
    for p,q in zip(net.parameters(),best[1]): p.data.copy_(q)
    with torch.no_grad():
        mu_t,s_t = pred(Zt,Gt); mu_e,s_e = pred(Ze,Ge)
    mu_t,s_t,mu_e,s_e = [x.numpy() for x in (mu_t,s_t,mu_e,s_e)]
    # recalibrate ONE global scale multiplier on TRAIN, then score in the SAME t-family as (a)
    f=lambda lk: -logmass(Gt['ulo'],Gt['uhi'],mu_t,s_t*math.exp(lk),'t',A['nu']).mean()
    r=minimize_scalar(f,bounds=(-2,2),method='bounded'); kk=math.exp(r.x)
    ll = logmass(Ge['ulo'],Ge['uhi'],mu_e,s_e*kk,'t',A['nu'])[M2]
    return ll, float(np.std(mu_e[M2]-(Ge['u_prev'][M2]+A['c']))), float(np.std(np.log(s_e[M2])))

DS=Gho['dataset']; DSU=sorted(set(DS))
def show(name, ll):
    print(f"{name:44s} HO {ll.mean():+.4f}  gain {(ll-ll_a).mean():+.4f} | "
          + " ".join(f"{d}:{(ll-ll_a)[DS==d].mean():+.4f}(n={int((DS==d).sum())})" for d in DSU))
    return dict(ll=float(ll.mean()), gain=float((ll-ll_a).mean()),
                per_ds={d:dict(n=int((DS==d).sum()), gain=float((ll-ll_a)[DS==d].mean())) for d in DSU})

OUT={'a':dict(ll=float(ll_a.mean()), n=int(len(ll_a)), **A)}
for mode in ('full','causal'):
    for head in ('both','mean','scale'):
        lls=[]
        for sd in (0,1,2):
            ll,dm,dls = run(mode,head,False,sd); lls.append(ll)
        ll=np.mean(lls,0)
        OUT[f'd_{mode}_{head}']=show(f'(d) AUDIO {mode:6s} head={head:5s} (3 seeds)', ll)
        OUT[f'd_{mode}_{head}']['sd_mean_shift']=dm; OUT[f'd_{mode}_{head}']['sd_log_scale']=dls
        print(f'      -> sd of learned mean-shift {dm:.5f} (vs typical |e| ~0.023) ; sd of learned log-scale {dls:.4f}')
ll,_,_ = run('full','both',True,0)
OUT['d_shuffled_control']=show('(d-CTRL) shuffled audio, full, both', ll)
json.dump(OUT, open('/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise/step4.json','w'), indent=1)
