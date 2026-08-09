import numpy as np, torch, sys
from scipy.stats import spearmanr
from scipy.ndimage import gaussian_filter1d
cache=torch.load("/disk4/jaehoon/VBPM_cache/mert/cache_mert_fold0_c1400.pt",weights_only=False)
FPS=44100/1024; rng=np.random.default_rng(0)
keys=["perfect","smooth0.5s","smooth1s","smooth2s","lag0.5s","binary_top","null"]
res={k:[] for k in keys}; S=[]
for e in cache["val_entries"]:
    bf=e["beat_frames"]; T=cache["val_acts"][e["stem"]].shape[0]
    iv=np.diff(bf).astype(float)
    if len(iv)<3: continue
    vb=np.abs(np.diff(iv))/iv[:-1]; vol=np.zeros(T)
    for i,v in enumerate(vb):
        lo,hi=bf[i+1],min(bf[i+2],T)
        if lo<T: vol[lo:hi]=v
    lo,hi=bf[0],min(bf[-1],T)
    if not(hi-lo>100 and vol[lo:hi].std()>0): continue
    v=vol[lo:hi]
    S.append((len(v),len(np.unique(v)),(v==0).mean(),np.bincount(np.unique(v,return_inverse=True)[1]).max()/len(v)))
    for k in keys:
        if k=="perfect": h=v
        elif k.startswith("smooth"): h=gaussian_filter1d(v,float(k[6:-1])*FPS)
        elif k=="lag0.5s": h=np.roll(v,int(0.5*FPS))
        elif k=="binary_top": h=(v>np.percentile(v,90)).astype(float)+1e-6*rng.normal(size=len(v))
        else: h=rng.normal(size=len(v))
        res[k].append(spearmanr(h,v).statistic)
for k in keys:
    a=np.array(res[k],float); print(f"{k:12s} mean {np.nanmean(a):+.4f} median {np.nanmedian(a):+.4f} frac>0.1 {(a>0.1).mean():.2f}")
S=np.array(S); print("\nvol series: T med",np.median(S[:,0]),"n_unique med",np.median(S[:,1]),
      "frac_zero med",round(np.median(S[:,2]),3),"largest-tie-frac med",round(np.median(S[:,3]),3))
