import json,sys
from pathlib import Path
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
LAB=Path("/home/sogang/jaehoon/VBPM/rungs_lab"); sys.path.insert(0,str(LAB))
from data.songs import iter_songs
from smc_data import load_smc
RACE=Path("/disk4/jaehoon/VBPM_cache/mert/race"); ERA=Path("/disk1/jaehoon/vbpm_mert_layers")
tempo_of,dataset_of={},{}
for s in iter_songs():
    bt,_=s.beats()
    if len(bt)>=8: tempo_of[s.stem]=60.0/float(np.median(np.diff(bt))); dataset_of[s.stem]=s.dataset
for e in load_smc():
    if len(e["beat_times"])>=8: tempo_of[e["stem"]]=60.0/float(np.median(np.diff(e["beat_times"]))); dataset_of[e["stem"]]="smc"
acts={}
for fn in ("bt_acts_beat.npz","bt_acts_beat_v2.npz"):
    d=np.load(RACE/fn); FPS=float(d["fps"])
    for k in d.files:
        if k!="fps": acts[k]=d[k]
emb={}; prov={}
for p in RACE.glob("*.npy"): emb[p.stem]=np.load(p).astype(np.float32).mean(axis=1); prov[p.stem]="race"
for sub in ("corpus","smc"):
    for p in (ERA/sub).glob("*.pt"):
        if p.stem in emb: continue
        d=torch.load(p,map_location="cpu",weights_only=False); emb[p.stem]=d["layers"].float().mean(dim=1).numpy(); prov[p.stem]="era_"+sub
def acf_candidates(a,fps):
    a=a-a.mean(); n=len(a); ac=np.correlate(a,a,"full")[n-1:]; ac=ac/(ac[0]+1e-12)
    lo,hi=int(fps*60/300),min(int(fps*60/40),n-1); lag=lo+int(np.argmax(ac[lo:hi+1])); t1=60.0*fps/lag
    part={}
    for r in (0.5,2.0):
        pl=int(round(lag*r))
        if lo<=pl<=hi: part[r]=ac[pl]
    r=max(part,key=part.get); return t1,60.0*fps/int(round(lag*r))
rows,X=[],[]
for stem,a in acts.items():
    if stem not in tempo_of or stem not in emb: continue
    t1,t2=acf_candidates(a,FPS); lo,hi=min(t1,t2),max(t1,t2); ta=tempo_of[stem]
    dl,dh=abs(np.log2(ta/lo)),abs(np.log2(ta/hi))
    if min(dl,dh)>np.log2(1.19): continue
    rows.append(dict(stem=stem,dataset=dataset_of[stem],lo=lo,hi=hi,y=int(dh<dl),pick=int(t1==hi),prov=prov[stem]))
    X.append(emb[stem])
X=np.stack(X); y=np.array([r["y"] for r in rows]); ds=np.array([r["dataset"] for r in rows])
pv=np.array([r["prov"] for r in rows]); gm=np.log2([np.sqrt(r["lo"]*r["hi"]) for r in rows])[:,None]
print("n",len(rows));
import collections
print("dataset x prov:",collections.Counter(zip(ds,pv)))
for d in np.unique(ds): print(" ",d,"n",(ds==d).sum(),"P(y=1)",y[ds==d].mean())
def cv(Xf,yv,C=0.05,seed=0,groups=None):
    pred=np.zeros(len(yv))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(Xf,yv):
        sc=StandardScaler().fit(Xf[tr]); clf=LogisticRegression(max_iter=3000,C=C,class_weight="balanced").fit(sc.transform(Xf[tr]),yv[tr]); pred[te]=clf.predict(sc.transform(Xf[te]))
    return balanced_accuracy_score(yv,pred),pred
# dataset-only baseline
D=np.stack([(ds==d).astype(float) for d in np.unique(ds)],1)
print("dataset-onehot only balacc",cv(np.hstack([D,gm]),y,C=1.0)[0])
print("prov-onehot only balacc",cv(np.hstack([np.stack([(pv==p).astype(float) for p in np.unique(pv)],1),gm]),y,C=1.0)[0])
print("tempo only",cv(gm,y,C=1.0)[0])
for l in (0,5,7,8,11):
    b,pred=cv(np.hstack([X[:,l],gm]),y)
    # leave-one-dataset-out
    lodo={}
    for d in np.unique(ds):
        te=ds==d; tr=~te
        if len(np.unique(y[te]))<2: continue
        sc=StandardScaler().fit(np.hstack([X[tr,l],gm[tr]])); clf=LogisticRegression(max_iter=3000,C=0.05,class_weight="balanced").fit(sc.transform(np.hstack([X[tr,l],gm[tr]])),y[tr])
        lodo[d]=round(float(balanced_accuracy_score(y[te],clf.predict(sc.transform(np.hstack([X[te,l],gm[te]]))))),3)
    # within-dataset CV
    wd={}
    for d in np.unique(ds):
        m=ds==d
        if m.sum()<40 or len(np.unique(y[m]))<2 or min(np.bincount(y[m]))<5: continue
        wd[d]=round(float(cv(np.hstack([X[m][:,l],gm[m]]),y[m])[0]),3)
    print(f"L{l} pooled {b:.3f} | LODO {lodo} | within {wd}")
