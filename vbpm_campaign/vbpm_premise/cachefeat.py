import sys, numpy as np, pickle
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
from core import *
from feats import song_feats
for mode in ('full','causal'):
    for sp in ('train','eval'):
        D = prep(build(sp)); Xs=[]; P=[]
        for d in D:
            X,ks = song_feats(d,300,mode)
            if not len(ks): continue
            Xs.append(X); P += [(d['stem'],k,d) for k in ks]
        X=np.concatenate(Xs,0)
        G=gather([(d,k) for st,k,d in P])
        np.savez(f'feat_{mode}_{sp}.npz', X=X, **{k:v for k,v in G.items() if isinstance(v,np.ndarray)})
        print(mode,sp,X.shape)
