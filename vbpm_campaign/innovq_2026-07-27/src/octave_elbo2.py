"""Retest with a PROPER likelihood: full-frame Bernoulli over the activation.
  q_t = eps + (1-eps)*exp(-0.5*(dist_t/w)^2),  dist_t = frames to nearest predicted beat
  score = sum_t [ o_t log q_t + (1-o_t) log(1-q_t) ]
Pays for unexplained activation (grid too sparse) AND beats on silence (too dense).
GATE first: the score must prefer the TRUE ibi over 0.5x/2x when handed the truth."""
import sys, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
for p in ("/home/sogang/jaehoon/VBPM_reintegration","/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix",
          "/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq","/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem"):
    sys.path.insert(0,p)
import pm_common as P
dev="cuda:0"; T=1500; LO,HI=10,150
exec(open("octave_elbo.py").read().split("FACT=")[0].replace('print(f"eval','#print(f"eval'))
def loglik(a,ibi,w=2.0,eps=1e-3):
    o=np.clip(a,0.0,1.0); L=len(o); best=-1e18
    t=np.arange(L)
    for off in np.arange(0,ibi,max(ibi/24,0.5)):
        ph=((t-off)%ibi); d=np.minimum(ph,ibi-ph)
        q=eps+(1-2*eps)*np.exp(-0.5*(d/w)**2)
        best=max(best,float((o*np.log(q)+(1-o)*np.log(1-q)).sum()))
    return best
# ---- GATE: with TRUE ibi in hand, does the score rank 1.0x above 0.5x and 2.0x?
g=[]
for i in range(min(60,len(yev))):
    s={f:loglik(Aev[i],yev[i]*f) for f in (0.5,1.0,2.0)}
    g.append(s[1.0]>=max(s[0.5],s[2.0]))
print(f"GATE  score prefers true ibi over 0.5x/2x on {100*np.mean(g):.1f}% of crops"
      f"   {'PASS' if np.mean(g)>0.8 else 'FAIL -> likelihood cannot resolve octave'}")
FACT=[1/3,0.5,2/3,1.0,1.5,2.0,3.0]
n=len(yev); pick=np.zeros(n)
for i in range(n):
    sc=[loglik(Aev[i],base[i]*f) for f in FACT]
    pick[i]=base[i]*FACT[int(np.argmax(sc))]
def rep(nm,e):
    lr=np.log(e/yev); print(f"  {nm:34s} MAE {100*np.abs(lr).mean():5.2f}%   within-4% {100*np.mean(np.abs(lr)<0.04):5.1f}%")
print(f"eval crops {n}")
rep("tempogram alone",base)
rep("tempogram + Bernoulli octave pick",pick)
rep("oracle octave",np.array([min([base[i]*f for f in FACT],key=lambda c:abs(math.log(c/yev[i]))) for i in range(n)]))
