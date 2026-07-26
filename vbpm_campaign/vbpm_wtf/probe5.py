"""CONTROL: does my contrast instrument produce >>1 for an emission that IS phase-tuned?"""
import sys, math
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms')
from audit_common import load_split, FPS
from vbpm.evaluate import _estimate_meter

def my_barphase(downs,T):
    t=(np.arange(T)+0.5)/FPS; d=np.asarray(downs,float)
    bp=np.interp(t,d,np.arange(len(d)),left=np.nan,right=np.nan)
    pre=t<d[0]; post=t>d[-1]
    bp[pre]=(t[pre]-d[0])/max(d[1]-d[0],1e-6)
    bp[post]=len(d)-1+(t[post]-d[-1])/max(d[-1]-d[-2],1e-6)
    return (bp%1.0)*2*np.pi

TWO_PI=2*math.pi
ARMS='/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms'
A=np.load(f'{ARMS}/act_eval.npz',allow_pickle=True); ev=load_split('eval',with_feats=False)
def oracle_p(phi,m,kap=8.0,lo=0.01,hi=0.95):
    """beat-tuned: peaks at every 2pi/m of bar phase."""
    s=np.exp(kap*(np.cos(m*phi)-1.0)); return lo+(hi-lo)*s
rows=[]
for s in ev:
    T=s['T']
    if len(s['downs'])<3: continue
    a=np.clip(np.asarray(A[s['stem']+'|act'],np.float32),1e-4,1-1e-4)[:T]
    phi=my_barphase(s['downs'],T); m=_estimate_meter(s['beats'],s['downs'])
    def ll(p):
        q=oracle_p(p,m)
        return float(np.mean(a[:,0]*np.log(q)+(1-a[:,0])*np.log(1-q)))
    t=ll(phi); off=[ll((phi+TWO_PI*k/12)%TWO_PI) for k in range(1,12)]
    rows.append(t-np.mean(off))
r=np.array(rows)
print(f'ORACLE phase-tuned emission, SAME instrument, n={len(r)}: '
      f'd_ll/frame={r.mean():+.4f}  contrast=exp={math.exp(r.mean()):.4f}  frac>0={np.mean(r>0):.2f}')
