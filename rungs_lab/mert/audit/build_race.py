import json,sys
from pathlib import Path
import numpy as np, torch
LAB=Path('/home/sogang/jaehoon/VBPM/rungs_lab'); sys.path.insert(0,str(LAB))
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
def acf_candidates(a,fps):
    a=a-a.mean(); n=len(a); ac=np.correlate(a,a,"full")[n-1:]; ac=ac/(ac[0]+1e-12)
    lo,hi=int(fps*60/300),min(int(fps*60/40),n-1); lag=lo+int(np.argmax(ac[lo:hi+1])); t1=60.0*fps/lag
    part={}
    for r in (0.5,2.0):
        pl=int(round(lag*r))
        if lo<=pl<=hi: part[r]=ac[pl]
    r=max(part,key=part.get); return t1,60.0*fps/int(round(lag*r))
emb={}
for p in RACE.glob("*.npy"): emb[p.stem]=np.load(p).astype(np.float32).mean(axis=1)
for sub in ("corpus","smc"):
    for p in (ERA/sub).glob("*.pt"):
        if p.stem in emb: continue
        d=torch.load(p,map_location="cpu",weights_only=False); emb[p.stem]=d["layers"].float().mean(dim=1).numpy()
rows,X=[],[]
for stem,a in acts.items():
    if stem not in tempo_of or stem not in emb: continue
    t1,t2=acf_candidates(a,FPS); lo,hi=min(t1,t2),max(t1,t2); ta=tempo_of[stem]
    d_lo,d_hi=abs(np.log2(ta/lo)),abs(np.log2(ta/hi))
    if min(d_lo,d_hi)>np.log2(1.19): continue
    rows.append(dict(stem=stem,dataset=dataset_of[stem],lo=lo,hi=hi,y=int(d_hi<d_lo),acf_pick_high=int(t1==hi)))
    X.append(emb[stem])
np.savez('/home/sogang/jaehoon/VBPM/rungs_lab/mert/audit/race_xy.npz',X=np.stack(X),
  y=np.array([r['y'] for r in rows]),pick=np.array([r['acf_pick_high'] for r in rows]),
  ds=np.array([r['dataset'] for r in rows]),lo=np.array([r['lo'] for r in rows]),hi=np.array([r['hi'] for r in rows]))
print(len(rows),'saved')
