"""Is 'MERT features contain octave information' a genre/dataset-prior confound?"""
import sys,numpy as np,torch
from pathlib import Path
sys.path.insert(0,'/home/sogang/jaehoon/VBPM/rungs_lab')
from data.songs import iter_songs
from smc_data import load_smc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
RACE=Path("/disk4/jaehoon/VBPM_cache/mert/race"); ERA=Path("/disk1/jaehoon/vbpm_mert_layers")
tempo_of,dataset_of={},{}
for s in iter_songs():
    bt,_=s.beats()
    if len(bt)>=8: tempo_of[s.stem]=60/float(np.median(np.diff(bt))); dataset_of[s.stem]=s.dataset
for e in load_smc():
    if len(e["beat_times"])>=8: tempo_of[e["stem"]]=60/float(np.median(np.diff(e["beat_times"]))); dataset_of[e["stem"]]="smc"
acts={}
for fn in ("bt_acts_beat.npz","bt_acts_beat_v2.npz"):
    d=np.load(RACE/fn); FPS=float(d["fps"])
    for k in d.files:
        if k!="fps": acts[k]=d[k]
emb={}
for p in RACE.glob("*.npy"): emb[p.stem]=np.load(p).astype(np.float32).mean(axis=1)
for sub in ("corpus","smc"):
    for p in (ERA/sub).glob("*.pt"):
        if p.stem not in emb: emb[p.stem]=torch.load(p,map_location="cpu",weights_only=False)["layers"].float().mean(dim=1).numpy()
def acf_cand(a,fps):
    a=a-a.mean(); n=len(a); ac=np.correlate(a,a,"full")[n-1:]; ac=ac/(ac[0]+1e-12)
    lo,hi=int(fps*60/300),min(int(fps*60/40),n-1); lag=lo+int(np.argmax(ac[lo:hi+1])); t1=60*fps/lag
    part={}
    for r in (0.5,2.0):
        pl=int(round(lag*r))
        if lo<=pl<=hi: part[r]=ac[pl]
    r=max(part,key=part.get); return t1,60*fps/int(round(lag*r))
rows=[];X=[]
for stem,a in acts.items():
    if stem not in tempo_of or stem not in emb: continue
    t1,t2=acf_cand(a,FPS); lo,hi=min(t1,t2),max(t1,t2); ta=tempo_of[stem]
    dl,dh=abs(np.log2(ta/lo)),abs(np.log2(ta/hi))
    if min(dl,dh)>np.log2(1.19): continue
    rows.append((dataset_of[stem],int(dh<dl),lo,hi)); X.append(emb[stem])
X=np.stack(X); ds=np.array([r[0] for r in rows]); y=np.array([r[1] for r in rows])
gm=np.log2([np.sqrt(r[2]*r[3]) for r in rows])[:,None]
def cv(Xf,yv,groups=None,C=0.05):
    pred=np.zeros(len(yv))
    sp=(StratifiedGroupKFold(5,shuffle=True,random_state=0).split(Xf,yv,groups) if groups is not None
        else StratifiedKFold(5,shuffle=True,random_state=0).split(Xf,yv))
    for tr,te in sp:
        sc=StandardScaler().fit(Xf[tr])
        pred[te]=LogisticRegression(max_iter=3000,C=C,class_weight="balanced").fit(sc.transform(Xf[tr]),yv[tr]).predict(sc.transform(Xf[te]))
    return balanced_accuracy_score(yv,pred),pred
oh=np.stack([(ds==d).astype(float) for d in np.unique(ds)],1)
print("n",len(y))
print("tempo-only (gm)            %.3f"%cv(gm,y,C=1.0)[0])
print("DATASET ONE-HOT only       %.3f"%cv(oh,y,C=1.0)[0])
print("dataset one-hot + gm       %.3f"%cv(np.hstack([oh,gm]),y,C=1.0)[0])
for l in (5,7,8,11):
    b,pred=cv(np.hstack([X[:,l],gm]),y)
    bg,_=cv(np.hstack([X[:,l],gm]),y,groups=ds)   # leave-dataset-out-ish
    m=ds=="smc"
    bs,_=cv(np.hstack([X[m][:,l],gm[m]]),y[m])
    print(f"L{l:2d} pooled {b:.3f} | grouped-by-dataset CV {bg:.3f} | SMC-only {bs:.3f}")
m=ds=="smc"; print("SMC-only tempo-only %.3f  (smc y=higher %.2f)"%(cv(gm[m],y[m],C=1.0)[0],y[m].mean()))
