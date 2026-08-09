import numpy as np, torch
from scipy.stats import spearmanr, pearsonr
FPS=44100/1024
cache=torch.load("/disk4/jaehoon/VBPM_cache/mert/cache_mert_fold0_c1400.pt",weights_only=False)
E=cache["val_entries"]
def volseries(e,T):
    bf=np.asarray(e["beat_frames"]); iv=np.diff(bf).astype(float)
    vb=np.abs(np.diff(iv))/iv[:-1]
    vol=np.zeros(T)
    for i,v in enumerate(vb):
        lo,hi=bf[i+1],min(bf[i+2],T)
        if lo<T: vol[lo:hi]=v
    return vol,bf,vb
def volseries_time(e,T):
    bt=np.asarray(e["beat_times"]); bf=np.asarray(e["beat_frames"]); iv=np.diff(bt)
    vb=np.abs(np.diff(iv))/iv[:-1]
    vol=np.zeros(T)
    for i,v in enumerate(vb):
        lo,hi=bf[i+1],min(bf[i+2],T)
        if lo<T: vol[lo:hi]=v
    return vol
rng=np.random.default_rng(0)
res={k:[] for k in["frame_vs_time","planted_perfect","planted_blur","planted_lag","planted_noisy","planted_causal"]}
quant=[]
for e in E:
    T=int(e["beat_frames"][-1])+50
    vol,bf,vb=volseries(e,T); volt=volseries_time(e,T)
    lo,hi=bf[0],min(bf[-1],T)
    s=slice(lo,hi)
    if hi-lo<=100 or vol[s].std()==0: continue
    res["frame_vs_time"].append(spearmanr(vol[s],volt[s]).statistic)
    # planted signals, correlated with TIME-based (true) volatility, scored against the probe's frame-based target
    p=volt.copy()
    res["planted_perfect"].append(spearmanr(p[s],vol[s]).statistic)
    k=np.ones(int(FPS));  # 1s box blur
    pb=np.convolve(p,k/k.sum(),'same')
    res["planted_blur"].append(spearmanr(pb[s],vol[s]).statistic)
    pl=np.roll(p,int(FPS//2))
    res["planted_lag"].append(spearmanr(pl[s],vol[s]).statistic)
    pn=p+rng.normal(0,p.std()+1e-9,T)
    res["planted_noisy"].append(spearmanr(pn[s],vol[s]).statistic)
    # causal: hazard rises only AFTER the change is observable (shift by one beat)
    pc=np.zeros(T)
    for i,v in enumerate(vb):
        a,b=bf[min(i+2,len(bf)-1)],min(bf[min(i+3,len(bf)-1)],T)
        if a<T: pc[a:b]=v
    res["planted_causal"].append(spearmanr(pc[s],vol[s]).statistic)
    quant.append((np.median(vb), np.median(np.abs(np.diff(np.diff(np.asarray(e["beat_times"])*FPS - np.asarray(e["beat_frames"])[:0] ))) ) if False else 0))
for k,v in res.items():
    v=np.array(v); print(f"{k:16s} n={len(v)} mean={np.nanmean(v):.3f} median={np.nanmedian(v):.3f}")
# tie structure
e=E[0];T=int(e["beat_frames"][-1])+50;vol,bf,vb=volseries(e,T)
print("example song: n distinct vol values",len(np.unique(vol)),"frac zero",(vol==0).mean())
allvb=np.concatenate([np.abs(np.diff(np.diff(np.asarray(x["beat_frames"]).astype(float))))/np.diff(np.asarray(x["beat_frames"]).astype(float))[:-1] for x in E])
print("frame-vol median",np.median(allvb),"frac exactly 0",(allvb==0).mean())
allvt=np.concatenate([np.abs(np.diff(np.diff(np.asarray(x["beat_times"]))))/np.diff(np.asarray(x["beat_times"]))[:-1] for x in E])
print("time-vol median",np.median(allvt),"frac<0.005",(allvt<0.005).mean(),"quantization step ~",1/ (0.5*FPS))
