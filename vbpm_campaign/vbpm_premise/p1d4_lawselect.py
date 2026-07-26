"""PREMISE 1(d) DECISIVE: which tempo law should p_physical use?
Strictly CAUSAL one-step-ahead predictive log-likelihood of log-IBI.
ALL parameters fit on the 147 TRAIN songs; scored on the 79 held-out EVAL songs."""
import sys, numpy as np
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
from common import load_labels, per_ds, fmt_ds
from scipy import stats

def logibi(s):
    ibi=np.diff(s["beats"]); ibi=ibi[(ibi>0.1)&(ibi<2.0)]
    return np.log(ibi) if len(ibi)>=24 else None
TR=[(s["dataset"],logibi(s)) for s in load_labels("train")]
EV=[(s["dataset"],logibi(s)) for s in load_labels("eval")]
TR=[(d,x) for d,x in TR if x is not None]; EV=[(d,x) for d,x in EV if x is not None]
BURN=8

def norm_ll(x,mu,sd): return stats.norm.logpdf(x,mu,sd)
def lap_ll(x,mu,b):   return stats.laplace.logpdf(x,mu,b)
def t_ll(x,mu,sc,nu): return stats.t.logpdf(x,nu,mu,sc)

def kalman_pred(x, sw, sn):
    """local-level: s_k = s_{k-1}+w, x_k = s_k+n. Returns causal predictive mean & sd."""
    s=x[0]; P=sn**2
    mu=np.zeros(len(x)); sd=np.zeros(len(x))
    for k in range(1,len(x)):
        Pp=P+sw**2
        mu[k]=s; sd[k]=np.sqrt(Pp+sn**2)
        K=Pp/(Pp+sn**2); s=s+K*(x[k]-s); P=(1-K)*Pp
    return mu,sd

def predictors(x, a, sw, sn):
    n=len(x); run=np.cumsum(x)/np.arange(1,n+1)
    out={}
    out["RW"]        = (x[BURN-1:-1], np.ones(n-BURN))
    out["SHRINK"]    = (a*x[BURN-1:-1]+(1-a)*run[BURN-1:-1], np.ones(n-BURN))
    mu,sd = kalman_pred(x, sw, sn)
    out["LOCALLEVEL"]= (mu[BURN:], sd[BURN:]/sd[BURN:].mean())
    return out, x[BURN:]

def score(data, a, sw, sn, scales):
    acc={}
    for d,x in data:
        if len(x)<=BURN+4: continue
        P,y = predictors(x,a,sw,sn)
        for name,(mu,rel) in P.items():
            for fam in ("gauss","laplace","t3"):
                s = scales[(name,fam)]*rel
                if fam=="gauss": ll=norm_ll(y,mu,s)
                elif fam=="laplace": ll=lap_ll(y,mu,s/np.sqrt(2))
                else: ll=t_ll(y,mu,s*np.sqrt(1/3.),3.0)
                acc.setdefault((name,fam),[]).append(dict(dataset=d, ll=float(ll.mean())))
    return acc

# --- fit scales (and a, sw, sn) on TRAIN ---
def train_resid(data,a,sw,sn):
    R={}
    for d,x in data:
        if len(x)<=BURN+4: continue
        P,y=predictors(x,a,sw,sn)
        for name,(mu,rel) in P.items():
            R.setdefault(name,[]).append((y-mu)/rel)
    return {k:np.concatenate(v) for k,v in R.items()}

best=None
for sw in (0.001,0.002,0.004,0.008,0.016):
    for sn in (0.010,0.015,0.020,0.025,0.035):
        R=train_resid(TR,0.85,sw,sn)["LOCALLEVEL"]
        ll=norm_ll(R,0,np.std(R)).mean()
        if best is None or ll>best[0]: best=(ll,sw,sn)
_,SW,SN=best
alls=[]
for a in np.round(np.arange(0.5,1.01,0.05),2):
    R=train_resid(TR,a,SW,SN)["SHRINK"]; alls.append((norm_ll(R,0,np.std(R)).mean(),a))
A=max(alls)[1]
R=train_resid(TR,A,SW,SN)
scales={}
for name,r in R.items():
    for fam in ("gauss","laplace","t3"):
        if fam=="gauss": scales[(name,fam)]=float(np.std(r))
        elif fam=="laplace": scales[(name,fam)]=float(np.sqrt(2)*np.mean(np.abs(r-np.median(r))))
        else:
            # MLE scale for t_3
            from scipy.optimize import minimize_scalar
            f=lambda ls:-t_ll(r,np.median(r),np.exp(ls),3.0).mean()
            res=minimize_scalar(f,bounds=(np.log(1e-4),np.log(1.0)),method='bounded')
            scales[(name,fam)]=float(np.exp(res.x)*np.sqrt(3.))
print(f"TRAIN-fitted hyper-params: local-level sigma_w={SW}, sigma_n={SN}; shrinkage a={A}")
print("TRAIN-fitted predictive scales (log units):")
for k,v in sorted(scales.items()): print(f"   {k}: {v:.4f}")

acc=score(EV,A,SW,SN,scales)
print(f"\n== HELD-OUT one-step predictive log-lik of log-IBI, nats/beat ({len(EV)} eval songs)")
base=None
tab={}
for (name,fam),rows in sorted(acc.items()):
    d=per_ds(rows,'ll'); tab[(name,fam)]=d
    if (name,fam)==("RW","gauss"): base=d
for (name,fam),d in tab.items():
    delta={k:(v-base[k][0],n) for k,(v,n) in d.items()}
    print(f"  {name:11s} {fam:8s}: " + "  ".join(f"{k}={v:+.3f}" for k,(v,n) in delta.items())
          + f"   [abs POOLED={d['POOLED'][0]:.3f}]")
print("  (numbers are nats/beat RELATIVE to the Gaussian random walk = p_physical's usual choice)")
# win counts vs Gaussian RW
rw={ (r['dataset'],i):r['ll'] for i,r in enumerate(acc[("RW","gauss")]) }
for key in tab:
    if key==("RW","gauss"): continue
    w=sum(1 for i,(r,b) in enumerate(zip(acc[key],acc[("RW","gauss")])) if r['ll']>b['ll'])
    print(f"    songs where {key[0]}/{key[1]} beats Gaussian-RW: {w}/{len(acc[key])}")
