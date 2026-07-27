"""Is the gate failure my uncalibrated emission, or the principle?
FIT (q_peak, w, q_floor) by max-likelihood at the TRUE ibi on TRAIN crops,
then run the SAME gate on EVAL. No tuning against the gate itself."""
import sys, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
for p in ("/home/sogang/jaehoon/VBPM_reintegration","/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix",
          "/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq","/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem"):
    sys.path.insert(0,p)
import pm_common as P
dev="cuda:0"; T=1500; LO,HI=10,150
exec(open("octave_elbo.py").read().split("FACT=")[0].replace('print(f"eval','#print(f"eval'))
_,ytr2,Atr=load("train",4,7)
def ll(a,ibi,qp,w,qf,noff=24):
    o=np.clip(a,0.0,1.0); t=np.arange(len(o)); best=-1e18
    for off in np.arange(0,ibi,max(ibi/noff,0.5)):
        ph=(t-off)%ibi; d=np.minimum(ph,ibi-ph)
        q=np.clip(qf+(qp-qf)*np.exp(-0.5*(d/w)**2),1e-4,1-1e-4)
        best=max(best,float((o*np.log(q)+(1-o)*np.log(1-q)).sum()))
    return best
print(f"activation stats: mean {Atr.mean():.3f}  p99 {np.percentile(Atr,99):.3f}  max {Atr.max():.3f}")
best=None
for qp in (0.2,0.35,0.5,0.7,0.9):
    for w in (1.0,2.0,3.0,5.0):
        for qf in (0.005,0.02,0.05):
            s=np.mean([ll(Atr[i],ytr2[i],qp,w,qf) for i in range(min(40,len(ytr2)))])
            if best is None or s>best[0]: best=(s,qp,w,qf)
_,QP,W,QF=best
print(f"FITTED on train at TRUE ibi: q_peak {QP} width {W} floor {QF}  (ll {best[0]:.0f})")
g=[]
for i in range(min(80,len(yev))):
    s={f:ll(Aev[i],yev[i]*f,QP,W,QF) for f in (0.5,1.0,2.0)}
    g.append(s[1.0]>=max(s[0.5],s[2.0]))
rate=np.mean(g)
print(f"GATE (eval, calibrated) prefers true ibi over 0.5x/2x: {100*rate:.1f}%   "
      f"{'PASS' if rate>0.8 else 'FAIL'}")
if rate>0.8:
    FACT=[1/3,0.5,2/3,1.0,1.5,2.0,3.0]; n=len(yev); pick=np.zeros(n)
    for i in range(n):
        pick[i]=base[i]*FACT[int(np.argmax([ll(Aev[i],base[i]*f,QP,W,QF) for f in FACT]))]
    for nm,e in (("tempogram alone",base),("tempogram + calibrated pick",pick)):
        lr=np.log(e/yev); print(f"  {nm:32s} MAE {100*np.abs(lr).mean():5.2f}%  within-4% {100*np.mean(np.abs(lr)<0.04):5.1f}%")
else:
    # which way does it lean? diagnose before concluding
    lean=[max(((f,ll(Aev[i],yev[i]*f,QP,W,QF)) for f in (0.5,1.0,2.0)),key=lambda x:x[1])[0]
          for i in range(min(80,len(yev)))]
    from collections import Counter
    print("  preferred factor when truth is known:",dict(Counter(lean)))
