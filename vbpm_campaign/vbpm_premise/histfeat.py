"""AUDIO-BLIND latent-history features: everything a higher-order FIXED physical law
(state = recent tempo history) can use.  No activation is touched."""
import sys, numpy as np, math
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
from core import prep
from data import build
def hist_feats(D):
    Xs=[]; stems=[]
    for d in D:
        n=len(d['u']); u=d['u']
        ks=[k for k in range(1,n) if d['valid'][k] and d['valid'][k-1]]
        for k in ks:
            up=u[k-1]
            lag=lambda j: u[k-j] if k-j>=0 else up
            past=u[max(0,k-8):k]
            past2=u[max(0,k-4):k]
            allp=u[:k]
            de=np.diff(allp) if len(allp)>1 else np.array([0.0])
            f=[up, up-lag(2), lag(2)-lag(3), lag(3)-lag(4),
               float(np.mean(past))-up, float(np.mean(past2))-up,
               float(np.mean(np.abs(np.diff(past)))) if len(past)>2 else 0.0,
               float(np.mean(np.abs(de))), float(np.std(de)), float(np.median(np.abs(de))),
               float(np.mean(allp))-up, float(np.std(allp)), float(k),
               float(d['meter']==2),float(d['meter']==3),float(d['meter']==4)]
            Xs.append(f); stems.append(d['stem'])
    return np.asarray(Xs,np.float64), np.array(stems)
if __name__=='__main__':
    for sp in ('train','eval'):
        D=prep(build(sp)); X,st=hist_feats(D); np.savez(f'hist_{sp}.npz',X=X,stem=st); print(sp,X.shape)
