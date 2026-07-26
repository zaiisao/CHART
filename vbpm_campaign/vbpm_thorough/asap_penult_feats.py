"""X2(b) extra arm: give audio its BEST shot -- Beat This 512-d penultimate features,
pooled causally at each prediction anchor (mean over last 3 s + anchor frame),
same row order as the other feature sets (identical ks per song)."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_thorough')
from core import prep
from asap_data import build_asap, FPS
PC=Path('/home/sogang/jaehoon/VBPM/percussion_bias/asap_penult_cache')
W=150
for sp in ('train','eval'):
    D=prep(build_asap(sp))
    Xs=[]; stems=[]
    for d in D:
        pen=np.load(PC/f"{d['stem']}.npy").astype(np.float32)  # [T,512]
        T=min(len(pen), d['T'])
        n=len(d['u'])
        ks=[k for k in range(1,n) if d['valid'][k] and d['valid'][k-1]]
        b=d['beats']
        for k in ks:
            f=int(round(b[k]*FPS)); f=max(2,min(T-3,f))
            m=pen[max(0,f-W):f].mean(0)
            Xs.append(np.concatenate([m, pen[f]]).astype(np.float16))
            stems.append(d['stem'])
    X=np.stack(Xs)
    np.savez(f'pen_asap_{sp}.npz', X=X, stem=np.array(stems))
    print(sp, X.shape, flush=True)
print('PEN FEATS DONE')
