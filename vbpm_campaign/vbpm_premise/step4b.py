import sys, math, json, numpy as np, torch, torch.nn as nn
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
from core import logmass
from scipy.optimize import minimize, minimize_scalar
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(8)

def load(mode,sp):
    d=np.load(f'feat_{mode}_{sp}.npz', allow_pickle=True)
    G={k:d[k] for k in d.files if k!='X'}; G['n']=len(G['u'])
    return d['X'].astype(np.float64), G
def half_mask(G):
    m2=np.zeros(G['n'],bool); 
    from collections import defaultdict
    idx=defaultdict(list)
    for i,st in enumerate(G['stem']): idx[st].append(i)
    for st,ii in idx.items(): m2[np.array(ii[len(ii)//2:])]=True
    return m2
Xf_tr,Gtr = load('full','train'); Xf_ev,Gev = load('full','eval')
Xc_tr,_   = load('causal','train'); Xc_ev,_ = load('causal','eval')
M2 = half_mask(Gev); print('held-out 2nd-half eval pairs', int(M2.sum()), 'train pairs', Gtr['n'])
def fit_t(G):
    f=lambda th: -logmass(G['ulo'],G['uhi'],G['u_prev']+th[0],math.exp(th[1]),'t',math.exp(th[2])+0.2).mean()
    r=minimize(f,[0.,math.log(.03),math.log(2.)],method='Nelder-Mead',options=dict(maxiter=6000,xatol=1e-7,fatol=1e-9))
    return dict(c=float(r.x[0]), s=float(math.exp(r.x[1])), nu=float(math.exp(r.x[2])+0.2))
A=fit_t(Gtr)
sub=lambda G,m:{k:(v[m] if isinstance(v,np.ndarray) else v) for k,v in G.items() if k!='n'}
Gho=sub(Gev,M2)
ll_a=logmass(Gho['ulo'],Gho['uhi'],Gho['u_prev']+A['c'],A['s'],'t',A['nu'])
print(f"(a) GLOBAL fixed c={A['c']:+.5f} s={A['s']:.5f} nu={A['nu']:.2f}  HO {ll_a.mean():+.4f}")
DS=Gho['dataset']; DSU=sorted(set(DS))

def lapmass(lo,hi,mu,s):
    F=lambda z: torch.where(z<0,0.5*torch.exp(z),1-0.5*torch.exp(-z))
    return torch.log(torch.clamp(F((hi-mu)/s)-F((lo-mu)/s),min=1e-300))

def run(mode,head,shuffle,seed,epochs=400):
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    Xt = (Xf_tr if mode=='full' else Xc_tr).copy(); Xe=(Xf_ev if mode=='full' else Xc_ev).copy()
    if shuffle:
        Xt[:,:-4]=Xt[rng.permutation(len(Xt))][:,:-4]; Xe[:,:-4]=Xe[rng.permutation(len(Xe))][:,:-4]
    mu_,sd_=Xt.mean(0),Xt.std(0)+1e-6
    Zt=torch.tensor((Xt-mu_)/sd_); Ze=torch.tensor((Xe-mu_)/sd_)
    us=np.unique(Gtr['stem']); vs=set(rng.choice(us,max(1,len(us)//5),replace=False))
    vm=torch.tensor(np.array([s in vs for s in Gtr['stem']])); tm=~vm
    up_t=torch.tensor(Gtr['u_prev']); lo_t=torch.tensor(Gtr['ulo']); hi_t=torch.tensor(Gtr['uhi'])
    up_e=torch.tensor(Gev['u_prev']); 
    net=nn.Sequential(nn.Linear(Zt.shape[1],64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,2)).double()
    nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)
    opt=torch.optim.Adam(net.parameters(),lr=5e-3,weight_decay=1e-5)
    bls=math.log(A['s'])
    def pred(Z,up):
        o=net(Z); mu=up+A['c']; ls=torch.full_like(mu,bls)
        if head in('both','mean'): mu=mu+0.1*o[:,0]
        if head in('both','scale'): ls=ls+2.0*torch.tanh(o[:,1])
        return mu,torch.exp(ls)
    best=(1e9,None)
    for ep in range(epochs):
        opt.zero_grad(); mu,s=pred(Zt,up_t)
        v=-lapmass(lo_t,hi_t,mu,s); v[tm].mean().backward(); opt.step()
        with torch.no_grad():
            mu,s=pred(Zt,up_t); q=float((-lapmass(lo_t,hi_t,mu,s))[vm].mean())
        if q<best[0]-1e-7: best=(q,[p.detach().clone() for p in net.parameters()],ep)
    for p,q in zip(net.parameters(),best[1]): p.data.copy_(q)
    with torch.no_grad():
        mt,st=pred(Zt,up_t); me,se=pred(Ze,up_e)
    mt,st,me,se=[x.numpy() for x in (mt,st,me,se)]
    f=lambda lk:-logmass(Gtr['ulo'],Gtr['uhi'],mt,st*math.exp(lk),'t',A['nu']).mean()
    kk=math.exp(minimize_scalar(f,bounds=(-2,2),method='bounded').x)
    ll=logmass(Gev['ulo'],Gev['uhi'],me,se*kk,'t',A['nu'])[M2]
    return ll,float(np.std(me[M2]-(Gev['u_prev'][M2]+A['c']))),float(np.std(np.log(se[M2]))),best[2]

def show(n,ll):
    print(f"{n:46s} HO {ll.mean():+.4f} gain {(ll-ll_a).mean():+.4f} | "
          +" ".join(f"{d}:{(ll-ll_a)[DS==d].mean():+.4f}(n={int((DS==d).sum())})" for d in DSU),flush=True)
    return dict(ll=float(ll.mean()),gain=float((ll-ll_a).mean()),
                per_ds={d:dict(n=int((DS==d).sum()),gain=float((ll-ll_a)[DS==d].mean())) for d in DSU})
OUT={'a':dict(ll=float(ll_a.mean()),n=int(len(ll_a)),**A)}
for mode in ('full','causal'):
    for head in ('both','mean','scale'):
        L=[];ms=[];ss=[]
        for sd in (0,1,2):
            ll,dm,dls,be=run(mode,head,False,sd); L.append(ll); ms.append(dm); ss.append(dls)
        ll=np.mean(L,0); r=show(f'(d) AUDIO {mode:6s} head={head:5s} 3-seed avg',ll)
        r['sd_mean_shift']=float(np.mean(ms)); r['sd_log_scale']=float(np.mean(ss))
        r['per_seed_gain']=[float((x-ll_a).mean()) for x in L]
        OUT[f'd_{mode}_{head}']=r
        print(f"      learned mean-shift sd={np.mean(ms):.5f} (mean |e|=0.0229) ; log-scale sd={np.mean(ss):.4f} ; per-seed gains={[round(g,4) for g in r['per_seed_gain']]}",flush=True)
L=[run('full','both',True,s)[0] for s in (0,1,2)]
OUT['d_shuffled_ctrl']=show('(d-CTRL) SHUFFLED audio, full, both',np.mean(L,0))
json.dump(OUT,open('step4.json','w'),indent=1)
print('DONE')
