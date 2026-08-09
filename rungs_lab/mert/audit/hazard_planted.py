"""Can the hazard probe recover a PLANTED volatility-tracking signal?
Replicates the probe's exact pipeline but substitutes synthetic hazard series."""
import numpy as np, torch
from scipy.stats import spearmanr
cache=torch.load("/disk4/jaehoon/VBPM_cache/mert/cache_mert_fold0_c1400.pt",weights_only=False)
FPS=44100/1024
rng=np.random.default_rng(0)
res={k:[] for k in ["perfect","noisy","null"]}
stats=[]
for e in cache["val_entries"]:
    bf=e["beat_frames"]; T=cache["val_acts"][e["stem"]].shape[0]
    iv=np.diff(bf).astype(float)
    if len(iv)<3: continue
    vol_beat=np.abs(np.diff(iv))/iv[:-1]
    vol=np.zeros(T)
    for i,v in enumerate(vol_beat):
        lo,hi=bf[i+1],min(bf[i+2],T)
        if lo<T: vol[lo:hi]=v
    lo,hi=bf[0],min(bf[-1],T)
    if not(hi-lo>100 and vol[lo:hi].std()>0): continue
    v=vol[lo:hi]
    stats.append(dict(T=hi-lo,nuniq=len(np.unique(v)),frac_zero=float((v==0).mean()),
                      mode_frac=float(np.bincount(np.unique(v,return_inverse=True)[1]).max()/len(v))))
    res["perfect"].append(spearmanr(v,v).statistic)
    res["noisy"].append(spearmanr(v+rng.normal(0,v.std(),len(v)),v).statistic)
    res["null"].append(spearmanr(rng.normal(size=len(v)),v).statistic)
for k,a in res.items():
    a=np.array(a); print(k,"mean rho",round(float(np.nanmean(a)),4),"median",round(float(np.nanmedian(a)),4),"n",len(a))
import pandas as pd
d=pd.DataFrame(stats); print(d.describe().round(3))
