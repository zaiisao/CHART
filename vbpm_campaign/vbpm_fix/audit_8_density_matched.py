"""AUDIT 8 -- DENSITY-MATCHED SPAM FLOOR.

The trained MERT VBPM free-run emits n_est/n_true = 3.92. A fair floor for a
model at that density is not the 120 BPM metronome but a blind constant grid of
the SAME density. This computes that floor per song for a range of densities.
"""
import sys
import numpy as np
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
from audit_common import load_split, truncate, banner, f_measure, FPS

ev = load_split("eval")
banner("BLIND CONSTANT GRID AT A MATCHED DENSITY (per song: period = median_IBI / k)")
print(f"{'density k':>10s} {'beat_F':>8s}   (a blind grid emitting k x as many beats as truth)")
for k in [1.0, 1.5, 2.0, 3.0, 3.92, 5.0]:
    Fs = []
    for s in ev:
        T, ref, _ = truncate(s, None)
        if len(ref) < 3: continue
        ibi = float(np.median(np.diff(ref)))
        per = max(ibi / k, 0.10)                    # read-out cannot emit closer than 0.10 s
        Fs.append(f_measure(ref, np.arange(0.0, T / FPS, per)))
    print(f"{k:10.2f} {np.mean(Fs):8.3f}")
print("\n  The trained MERT VBPM free-run scores 0.336 at k=3.92 -- i.e. BELOW the blind")
print("  density-matched grid. Its 0.31-0.34 is not a beat-tracking score.")
