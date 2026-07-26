"""AUDIT 1 -- REFERENCE CEILING + PROTOCOL AUDIT.

Q1: what beat_F does the OFFICIAL deploy read-out achieve from an IDEAL bar phase
    (0 at each downbeat, linear to 2pi at the next)?  Previously reported 0.955.
Q2: is the protocol previous numbers were produced under (`eval[:30]`, max 1600
    frames) representative?  eval__* sorts ballroom < beatles < hainsworth, so
    [:30] is 30/30 BALLROOM -- this is checked explicitly.
Q3: how much of the read-out depends on the ORACLE meter m = _estimate_meter(GT)?
"""
import sys
import numpy as np

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
from audit_common import (load_split, ideal_barphase, ideal_beatlinear_barphase, truncate,
                          score_phase, agg, by_dataset, ratio, banner, metronome,
                          f_measure, _estimate_meter, FPS, PRIOR_MAX_FRAMES, PRIOR_N_EVAL)

ev_all = load_split("eval")
tr = load_split("train")
banner("SPLIT SANITY")
print(f"train {len(tr)} songs, eval {len(ev_all)} songs")
print(f"train folds: {sorted(set(s['fold'] for s in tr))}   eval folds: {sorted(set(s['fold'] for s in ev_all))}")
ov = set(s['stem'].replace('train__', '') for s in tr) & set(s['stem'].replace('eval__', '') for s in ev_all)
print(f"stem overlap train/eval: {len(ov)}  -> {'CLEAN' if not ov else '*** LEAK ***'}")
import collections
print("eval by dataset:", dict(collections.Counter(s['dataset'] for s in ev_all)))
print(f"eval[:{PRIOR_N_EVAL}] by dataset:", dict(collections.Counter(s['dataset'] for s in ev_all[:PRIOR_N_EVAL])),
      "   <-- PROTOCOL BIAS CHECK")
Ts = np.array([s['T'] for s in ev_all])
print(f"eval T frames: min {Ts.min()} med {int(np.median(Ts))} max {Ts.max()} "
      f"| songs longer than {PRIOR_MAX_FRAMES}: {(Ts > PRIOR_MAX_FRAMES).sum()}/{len(Ts)}")


def run(songs, max_frames, phase_fn, meter_mode="oracle", label=""):
    rows = []
    for s in songs:
        T, ref, dref = truncate(s, max_frames)
        if len(ref) < 2 or len(dref) < 2:
            continue
        ph = phase_fn(s, T, ref, dref)
        if ph is None:
            continue
        m = None if meter_mode == "oracle" else int(meter_mode)
        r = score_phase(ph, ref, dref, T, meter=m)
        r["dataset"] = s["dataset"]; r["stem"] = s["stem"]
        r["metronome_F"] = f_measure(ref, metronome(T, FPS))
        rows.append(r)
    a = agg(rows, ["beat_F", "downbeat_F", "metronome_F"])
    print(f"{label:52s} beat_F={a['beat_F']:.3f}  db_F={a['downbeat_F']:.3f}  "
          f"metro={a['metronome_F']:.3f}  n_est/n_true={ratio(rows):.3f}  N={len(rows)}")
    return rows, a


PH_EXTRAP = lambda s, T, ref, dref: ideal_barphase(dref, T, mode="extrap")
PH_STRICT = lambda s, T, ref, dref: ideal_barphase(dref, T, mode="strict")
PH_BEATLIN = lambda s, T, ref, dref: ideal_beatlinear_barphase(ref, dref, T)

banner("Q1  IDEAL-BAR-PHASE CEILING through the official read-out")
print("(phase source | eval subset | frame cap)")
r_prior, a_prior = run(ev_all[:PRIOR_N_EVAL], PRIOR_MAX_FRAMES, PH_EXTRAP,
                       label="ideal(extrap) | eval[:30] (all ballroom) | <=1600")
run(ev_all[:PRIOR_N_EVAL], PRIOR_MAX_FRAMES, PH_STRICT,
    label="ideal(strict, probe_stages) | eval[:30] | <=1600")
r_1600, a_1600 = run(ev_all, PRIOR_MAX_FRAMES, PH_EXTRAP,
                     label="ideal(extrap) | ALL 79 eval | <=1600")
r_full, a_full = run(ev_all, None, PH_EXTRAP,
                     label="ideal(extrap) | ALL 79 eval | FULL length")
run(ev_all, None, PH_STRICT, label="ideal(strict, probe_stages) | ALL 79 | FULL")
run(ev_all, None, PH_BEATLIN, label="ideal BEAT-LINEAR phase | ALL 79 | FULL   [read-out code limit]")

banner("Q1b PER-DATASET (ideal extrap, ALL eval, FULL length)")
for k, (v, n) in by_dataset(r_full, "beat_F").items():
    print(f"  {k:12s} ideal beat_F = {v:.3f}  (n={n})")
for k, (v, n) in by_dataset(r_full, "metronome_F").items():
    print(f"  {k:12s} metronome_F  = {v:.3f}  (n={n})")

banner("Q3  HOW MUCH OF THE READ-OUT IS THE ORACLE METER?")
_, a_or = run(ev_all, None, PH_EXTRAP, meter_mode="oracle", label="ideal | m = _estimate_meter(GT downbeats)")
_, a_m4 = run(ev_all, None, PH_EXTRAP, meter_mode="4", label="ideal | m forced to 4 (no GT meter)")
_, a_m3 = run(ev_all, None, PH_EXTRAP, meter_mode="3", label="ideal | m forced to 3")
print(f"  -> oracle-meter advantage over always-4: {a_or['beat_F'] - a_m4['beat_F']:+.3f} beat_F")
mc = {}
for s in ev_all:
    T, ref, dref = truncate(s, None)
    if len(ref) < 2 or len(dref) < 2:
        continue
    mc[_estimate_meter(ref, dref)] = mc.get(_estimate_meter(ref, dref), 0) + 1
print(f"  oracle meter histogram on eval: {dict(sorted(mc.items()))}")

banner("METRONOME FLOOR (the number every variant must beat)")
for label, rows in [("eval[:30] <=1600", r_prior), ("ALL 79 <=1600", r_1600), ("ALL 79 FULL", r_full)]:
    a = agg(rows, ["metronome_F"])
    print(f"  120BPM metronome, {label:20s} = {a['metronome_F']:.3f}   (N={len(rows)})")
