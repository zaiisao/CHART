"""Is the gate failure caused by DRIFT over the 30s window (my harness), or by the
likelihood itself? Run the identical gate at increasing window lengths.
 drift hypothesis -> passes short, degrades long, and the leaning is toward 0.5x."""
import sys, math, numpy as np
for p in ("/home/sogang/jaehoon/VBPM_reintegration","/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix",
          "/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq","/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem"):
    sys.path.insert(0,p)
import pm_common as P
dev="cuda:0"; T=1500; LO,HI=10,150
exec(open("octave_elbo.py").read().split("FACT=")[0].replace('print(f"eval','#print(f"eval'))
QP,W,QF=0.2,3.0,0.05
def ll(o,ibi,noff=32):
    t=np.arange(len(o)); best=-1e18
    for off in np.arange(0,ibi,max(ibi/noff,0.5)):
        ph=(t-off)%ibi; d=np.minimum(ph,ibi-ph)
        q=np.clip(QF+(QP-QF)*np.exp(-0.5*(d/W)**2),1e-4,1-1e-4)
        best=max(best,float((o*np.log(q)+(1-o)*np.log(1-q)).sum()))
    return best
from collections import Counter
print(f"{'window':>8} {'seconds':>8} {'gate':>7}   leaning")
for Wn in (128,256,384,512,768,1500):
    g=[]; lean=[]
    for i in range(min(80,len(yev))):
        seg=np.clip(Aev[i][:Wn],0,1)
        s={f:ll(seg,yev[i]*f) for f in (0.5,1.0,2.0)}
        g.append(s[1.0]>=max(s[0.5],s[2.0]))
        lean.append(max(s,key=s.get))
    c=Counter(lean)
    print(f"{Wn:>8} {Wn/50:>7.1f}s {100*np.mean(g):>6.1f}%   "
          f"0.5x:{c.get(0.5,0):3d}  1.0x:{c.get(1.0,0):3d}  2.0x:{c.get(2.0,0):3d}")
