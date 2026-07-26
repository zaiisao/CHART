"""Density-matched blind control for the MERT PF result.
The PF emits n_ratio=2.39x too many beats, which inflates F-measure recall. Fair question:
does a BLIND uniform grid emitting the SAME number of beats score just as well?"""
import sys, glob, math
import numpy as np
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
from vbpm.evaluate import f_measure, metronome
CACHE="/disk1/jaehoon/vbpm_mert_cache"; fps=50.0
ev=[]
for f in sorted(glob.glob(f"{CACHE}/eval__*.npz"))[:25]:      # same n_eval=25 as the varB MERT run
    d=np.load(f,allow_pickle=True)
    ev.append((int(d["feats"].shape[1]), np.asarray(d["beats"],float)))
MAXF=1200                                                     # same max_frames as the varB run
RATIO=2.39                                                    # the PF's measured n_est/n_true
gridF=[]; metF=[]; best_gridF=[]
for T0,beats in ev:
    T=min(T0,MAXF); dur=T/fps
    ref=beats[beats<dur]
    if len(ref)<2: continue
    n=max(int(round(len(ref)*RATIO)),2)
    # blind uniform grid with the SAME beat count as the PF emitted (phase 0, no audio)
    est=np.linspace(0, dur, n, endpoint=False)
    gridF.append(f_measure(ref,est))
    metF.append(f_measure(ref,metronome(T,fps)))
    # also: best-case blind grid (sweep the phase offset) = generous upper bound for "blind"
    bb=0.0
    for off in np.linspace(0, dur/max(n,1), 12, endpoint=False):
        bb=max(bb, f_measure(ref, est+off))
    best_gridF.append(bb)
print(f"n songs = {len(gridF)}")
print(f"  metronome (120bpm)                    : {np.mean(metF):.3f}")
print(f"  BLIND uniform grid @ PF density(2.39x): {np.mean(gridF):.3f}")
print(f"  BLIND grid @ PF density, best offset  : {np.mean(best_gridF):.3f}   (generous blind upper bound)")
print()
print(f"  >>> Variant-B PF (smooth) on MERT      : 0.449")
print(f"  >>> margin over density-matched blind  : {0.449-np.mean(gridF):+.3f}  (vs best-offset blind: {0.449-np.mean(best_gridF):+.3f})")
print(f"  >>> conv probe on same features        : 0.805")
