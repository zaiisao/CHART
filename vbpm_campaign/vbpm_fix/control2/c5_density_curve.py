"""DENSITY-MATCHED FLOOR CURVE: what beat_F does an AUDIO-BLIND constant grid get as a
function of how many beats it emits (n_est/n_true)?

Any estimator that over-emits gets F for free, so a claimed score must be compared against
the blind grid AT ITS OWN DENSITY, not against the 120-BPM metronome. Read off the row that
matches the variant's reported n_est/n_true.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/control2")
from cc import FPS, load_split, truncate, f_measure, banner

ev = load_split("eval")
PROTOS = [("eval[:30] cap1600", ev[:30], 1600), ("ALL79 cap1600", ev, 1600), ("ALL79 FULL", ev, None)]
DENS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.4, 3.0, 3.5, 4.0, 5.0]
NSEED = 5
out = {}

banner("blind constant grid, random start phase -- mean over 5 random phases")
print(f"  {'density x true':>16s} | " + " | ".join(f"{n:>17s}" for n, _, _ in PROTOS), flush=True)
for d in DENS:
    cells = []
    for name, songs, cap in PROTOS:
        vals = []
        for seed in range(NSEED):
            rng = np.random.default_rng(seed)
            per = []
            for s in songs:
                T, ref, dref = truncate(s, cap)
                if len(ref) < 2:
                    continue
                n = max(int(round(len(ref) * d)), 2)
                step = (T / FPS) / n
                per.append(f_measure(ref, np.arange(n) * step + rng.random() * step))
            vals.append(float(np.mean(per)))
        cells.append(float(np.mean(vals)))
        out[f"d{d}_{name}"] = float(np.mean(vals))
    print(f"  {d:16.2f} | " + " | ".join(f"{c:17.3f}" for c in cells), flush=True)

json.dump(out, open("/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/control2/c5_out.json", "w"), indent=1)
print("\nWROTE c5_out.json", flush=True)
