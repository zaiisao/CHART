"""(1) REFERENCE CEILING + the floors any claimed fix has to clear.

  * ideal bar phase (0 at each downbeat -> 2pi at the next) through the OFFICIAL read-out
  * 120 BPM metronome floor
  * blind constant-grid floors at matched DENSITY (a spam grid can score high)
  * perfect OPEN LOOP (oracle tempo + oracle start phase, audio-blind afterwards)
Protocols: eval[:30] cap 1600 (what every prior probe used) and ALL 79 songs, full length.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/control2")
from cc import (FPS, TWO_PI, load_split, truncate, ideal_barphase, score_phase, agg, ratio,
                by_dataset, line, banner, metronome, f_measure, _estimate_meter, sem)

ev = load_split("eval")
print(f"eval songs {len(ev)} | lengths median {np.median([s['T'] for s in ev]):.0f} "
      f"max {max(s['T'] for s in ev)} frames", flush=True)
print("dataset counts:", {k: sum(1 for s in ev if s['dataset'] == k) for k in sorted({s['dataset'] for s in ev})})
print("eval[:30] datasets:", {k: sum(1 for s in ev[:30] if s['dataset'] == k)
                              for k in sorted({s['dataset'] for s in ev[:30]})}, flush=True)

PROTOS = [("eval[:30] cap1600", ev[:30], 1600), ("ALL79 cap1600", ev, 1600), ("ALL79 FULL", ev, None)]
out = {}

banner("(1) REFERENCE CEILING -- ideal bar phase through the official read-out")
for name, songs, cap in PROTOS:
    rows = []
    for s in songs:
        T, ref, dref = truncate(s, cap)
        if len(ref) < 2 or len(dref) < 2:
            continue
        ph = ideal_barphase(dref, T)
        if ph is None:
            continue
        r = score_phase(ph, ref, dref)
        r["dataset"] = s["dataset"]
        rows.append(r)
    line(f"IDEAL bar phase [{name}]", rows, extra=f"sem={sem(rows,'beat_F'):.3f}")
    out[f"ideal_{name}"] = dict(beat_F=agg(rows, "beat_F"), db_F=agg(rows, "downbeat_F"),
                                ratio=ratio(rows), N=len(rows), sem=sem(rows, "beat_F"))
    if cap is None:
        print("   per dataset:", {k: round(v[0], 3) for k, v in by_dataset(rows, "beat_F").items()}, flush=True)

banner("(2) FLOORS -- what you get with NO tracking at all")


def const_grid_rows(songs, cap, period_sec):
    rows = []
    for s in songs:
        T, ref, dref = truncate(s, cap)
        if len(ref) < 2:
            continue
        est = np.arange(0.0, T / FPS, period_sec)
        rows.append(dict(beat_F=f_measure(ref, est), downbeat_F=float("nan"),
                         n_est=len(est), n_true=len(ref), dataset=s["dataset"]))
    return rows


def open_loop_rows(songs, cap, oracle_phase=True):
    """perfect open loop: constant true-tempo grid, started AT a true beat (or at 0)."""
    rows = []
    for s in songs:
        T, ref, dref = truncate(s, cap)
        if len(ref) < 2:
            continue
        ibi = float(np.median(np.diff(ref)))
        t0 = ref[0] if oracle_phase else 0.0
        est = np.arange(t0, T / FPS, ibi)
        rows.append(dict(beat_F=f_measure(ref, est), downbeat_F=float("nan"),
                         n_est=len(est), n_true=len(ref), dataset=s["dataset"]))
    return rows


for name, songs, cap in PROTOS:
    mrows = []
    for s in songs:
        T, ref, dref = truncate(s, cap)
        if len(ref) < 2:
            continue
        mrows.append(dict(beat_F=f_measure(ref, metronome(T, FPS)), downbeat_F=float("nan"),
                          n_est=len(metronome(T, FPS)), n_true=len(ref), dataset=s["dataset"]))
    line(f"120 BPM metronome [{name}]", mrows)
    out[f"metronome_{name}"] = dict(beat_F=agg(mrows, "beat_F"), ratio=ratio(mrows), N=len(mrows))

for name, songs, cap in [("eval[:30] cap1600", ev[:30], 1600), ("ALL79 FULL", ev, None)]:
    for per, lab in [(0.10, "blind grid 0.10 s (min-dist of beats_from_barphase)"),
                     (0.15, "blind grid 0.15 s (min-dist of beats_from_activation)"),
                     (0.21, "blind grid 0.21 s (~2.4x true density: variant-B PF density)")]:
        rows = const_grid_rows(songs, cap, per)
        line(f"{lab} [{name}]", rows)
        out[f"grid{per}_{name}"] = dict(beat_F=agg(rows, "beat_F"), ratio=ratio(rows))
    r = open_loop_rows(songs, cap, oracle_phase=True)
    line(f"PERFECT OPEN LOOP (oracle tempo + oracle start phase) [{name}]", r)
    out[f"openloop_{name}"] = dict(beat_F=agg(r, "beat_F"), ratio=ratio(r))
    r0 = open_loop_rows(songs, cap, oracle_phase=False)
    line(f"open loop, oracle tempo, phase 0 [{name}]", r0)
    out[f"openloop_nophase_{name}"] = dict(beat_F=agg(r0, "beat_F"), ratio=ratio(r0))

banner("(3) how much does the GT-given meter help the read-out? (always-4 vs _estimate_meter)")
rows_m, rows_4 = [], []
for s in ev:
    T, ref, dref = truncate(s, None)
    if len(ref) < 2 or len(dref) < 2:
        continue
    ph = ideal_barphase(dref, T)
    if ph is None:
        continue
    rows_m.append(score_phase(ph, ref, dref))
    rows_4.append(score_phase(ph, ref, dref, meter=4))
print(f"  GT meter {agg(rows_m,'beat_F'):.3f}   always-4 {agg(rows_4,'beat_F'):.3f}   "
      f"gift = {agg(rows_m,'beat_F') - agg(rows_4,'beat_F'):+.3f}", flush=True)
out["meter_gift"] = agg(rows_m, "beat_F") - agg(rows_4, "beat_F")

json.dump(out, open("/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/control2/c1_out.json", "w"), indent=1)
print("\nWROTE c1_out.json", flush=True)
