"""AUDIT 7 -- how big does a difference have to be to mean anything? (N=79 songs)"""
import sys
import numpy as np
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
from audit_common import (load_split, ideal_barphase, truncate, score_phase, banner,
                          metronome, f_measure, FPS)

ev = load_split("eval")
rows = {k: [] for k in ["ideal ceiling", "120BPM metronome", "oracle-tempo+oracle-phase (open loop)",
                        "spam grid 0.15 s"]}
for s in ev:
    T, ref, dref = truncate(s, None)
    if len(ref) < 3 or len(dref) < 2: continue
    dur = T / FPS
    ph = ideal_barphase(dref, T, mode="extrap")
    rows["ideal ceiling"].append(score_phase(ph, ref, dref, T)["beat_F"])
    rows["120BPM metronome"].append(f_measure(ref, metronome(T, FPS)))
    ibi = float(np.median(np.diff(ref)))
    rows["oracle-tempo+oracle-phase (open loop)"].append(
        max(f_measure(ref, np.arange(o, dur, ibi)) for o in np.linspace(0, ibi, 25, endpoint=False)))
    rows["spam grid 0.15 s"].append(f_measure(ref, np.arange(0.0, dur, 0.15)))

banner("PER-SONG SPREAD (N=79) -- the noise floor on any reported mean")
for k, v in rows.items():
    v = np.asarray(v)
    print(f"  {k:42s} mean={v.mean():.3f}  sd={v.std(ddof=1):.3f}  SEM={v.std(ddof=1)/np.sqrt(len(v)):.3f}")
print("\n  RULE: with N=79 and SEM ~0.02-0.03, paired differences below ~0.05 are not")
print("  distinguishable from noise unless reported per-song / paired.")
