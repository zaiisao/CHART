"""P2: what is the BEST possible free-run beat_F given that the deploy read-out chain
(free_run's phase_mu) is provably a CONSTANT-tempo metronome? Pure numpy oracle."""
import sys, glob, math
import numpy as np
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
from vbpm.distributions import TWO_PI
from vbpm.evaluate import beats_from_barphase, downbeats_from_barphase, f_measure, metronome, _estimate_meter

CACHE = "/disk1/jaehoon/vbpm_mert_cache"; fps = 50.0
ev = []
for f in sorted(glob.glob(f"{CACHE}/eval__*.npz"))[:30]:
    d = np.load(f, allow_pickle=True)
    ev.append(dict(T=int(d["feats"].shape[1]), beats=np.asarray(d["beats"], float), downs=np.asarray(d["downs"], float)))

def score(T, phidot, off, m, ref):
    pm = (np.arange(T) * phidot + off) % TWO_PI
    return f_measure(ref, beats_from_barphase(pm, m, fps))

rows = {k: [] for k in ["metronome", "true_tempo_off0", "true_tempo_bestoff", "grid_best",
                        "ideal_barphase", "true_tempo_bestoff_short"]}
for s in ev:
    T = min(s["T"], 1600)
    ref = s["beats"][s["beats"] < T / fps]; dref = s["downs"][s["downs"] < T / fps]
    if len(ref) < 2: continue
    m = _estimate_meter(ref, dref)
    phidot = TWO_PI / (np.median(np.diff(s["beats"])) * m * fps)
    rows["metronome"].append(f_measure(ref, metronome(T, fps)))
    rows["true_tempo_off0"].append(score(T, phidot, 0.0, m, ref))
    offs = np.linspace(0, TWO_PI, 64, endpoint=False)
    rows["true_tempo_bestoff"].append(max(score(T, phidot, o, m, ref) for o in offs))
    # joint grid over tempo AND offset: the true ceiling of a constant-tempo mean chain
    best = 0.0
    for pd in phidot * np.linspace(0.97, 1.03, 41):
        for o in offs:
            best = max(best, score(T, pd, o, m, ref))
    rows["grid_best"].append(best)
    # short window (first 300 frames = 6 s): does drift explain the gap?
    T2 = 300; ref2 = s["beats"][s["beats"] < T2 / fps]
    if len(ref2) >= 2:
        rows["true_tempo_bestoff_short"].append(max(score(T2, phidot, o, m, ref2) for o in offs))
    # IDEAL bar phase from the GT downbeats (piecewise-linear 0->2pi per bar) = read-out sanity
    dd = s["downs"][s["downs"] < T / fps]
    if len(dd) >= 2:
        tt = np.arange(T) / fps
        ph = np.interp(tt, dd, np.arange(len(dd)) * TWO_PI,
                       left=np.nan, right=np.nan)
        ph = np.nan_to_num(ph, nan=0.0) % TWO_PI
        rows["ideal_barphase"].append(f_measure(ref, beats_from_barphase(ph, m, fps)))

for k, v in rows.items():
    print(f"{k:>26}: {np.mean(v):.3f}  (n={len(v)})")
