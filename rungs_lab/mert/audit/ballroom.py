"""Reconstruct the race2_octave trial table and explain the ballroom column."""
import sys,numpy as np,torch
from pathlib import Path
sys.path.insert(0,'/home/sogang/jaehoon/VBPM/rungs_lab')
from data.songs import iter_songs
from smc_data import load_smc
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
emb=set(p.stem for p in RACE.glob("*.npy"))
for sub in ("corpus","smc"): emb|=set(p.stem for p in (ERA/sub).glob("*.pt"))
def acf_cand(a,fps):
    a=a-a.mean(); n=len(a); ac=np.correlate(a,a,"full")[n-1:]; ac=ac/(ac[0]+1e-12)
    lo,hi=int(fps*60/300),min(int(fps*60/40),n-1); lag=lo+int(np.argmax(ac[lo:hi+1])); t1=60*fps/lag
    part={}
    for r in (0.5,2.0):
        pl=int(round(lag*r))
        if lo<=pl<=hi: part[r]=ac[pl]
    r=max(part,key=part.get); return t1,60*fps/int(round(lag*r))
rows=[];drop={}
for stem,a in acts.items():
    if stem not in tempo_of or stem not in emb: continue
    t1,t2=acf_cand(a,FPS); lo,hi=min(t1,t2),max(t1,t2); ta=tempo_of[stem]
    dl,dh=abs(np.log2(ta/lo)),abs(np.log2(ta/hi))
    ds=dataset_of[stem]
    if min(dl,dh)>np.log2(1.19): drop[ds]=drop.get(ds,0)+1; continue
    rows.append((ds,int(dh<dl),int(t1==hi)))
import collections
ds=np.array([r[0] for r in rows]); y=np.array([r[1] for r in rows]); pk=np.array([r[2] for r in rows])
print("kept",len(rows),"dropped-per-ds",drop)
for d in sorted(set(ds)):
    m=ds==d; print(f"{d:12s} n={m.sum():4d}  y=higher {y[m].mean():.3f}  n_y1={int(y[m].sum()):4d} n_y0={int((~y[m].astype(bool)).sum()):4d}  ACFbal {0.5*((pk[m][y[m]==1]==1).mean()+(pk[m][y[m]==0]==0).mean()):.3f}")
