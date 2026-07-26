"""REUSABLE TIME-ROLL LEAK CONTROL.

Any claimed fix can be pushed through this. It evaluates the deploy read-out twice
with the SAME rng seed: once with aligned features, once with the features rolled
along time by ``roll`` frames while the LABELS stay put. If the score does not fall
to ~the metronome floor, the evaluation is reading the labels through some other
channel and the result is not a beat-tracking result.

    from roll_control import roll_control
    roll_control(lambda h_np: my_free_run_phase(h_np), songs, roll=1000)

``phase_fn`` takes a float32 [T, h_dim] numpy feature block and returns a bar-phase
trajectory [T] (numpy). ``feat_fn`` maps a cached song record + T -> [T, h_dim].
"""
from __future__ import annotations

import numpy as np
import torch

from audit_common import (truncate, score_phase, agg, ratio, metronome, f_measure, FPS)


def roll_control(phase_fn, feat_fn, songs, roll=1000, cap=None, seed=0, label=""):
    res = {}
    for tag, r in [("aligned", 0), (f"rolled(+{roll})", roll)]:
        rows = []
        for s in songs:
            T, ref, dref = truncate(s, cap)
            if len(ref) < 2 or len(dref) < 2:
                continue
            f = feat_fn(s, T)
            if r:
                f = np.roll(f, r, axis=0)
            torch.manual_seed(seed)          # identical noise draw in both arms
            ph = phase_fn(f)
            row = score_phase(np.asarray(ph)[:T], ref, dref, T)
            row["metronome_F"] = f_measure(ref, metronome(T, FPS))
            rows.append(row)
        a = agg(rows, ["beat_F", "downbeat_F", "metronome_F"])
        a["ratio"] = ratio(rows); a["N"] = len(rows); a["rows"] = rows
        res[tag] = a
        print(f"  {label:26s} {tag:14s} beat_F={a['beat_F']:.3f} db_F={a['downbeat_F']:.3f} "
              f"metro={a['metronome_F']:.3f} n_est/n_true={a['ratio']:.3f} N={a['N']}", flush=True)
    al = res["aligned"]; ro = res[f"rolled(+{roll})"]
    floor = al["metronome_F"]
    drop = al["beat_F"] - ro["beat_F"]
    if abs(drop) < 0.02:
        verdict = ("AUDIO-BLIND: the score is INVARIANT to a 20 s feature shift. "
                   "Nothing leaked, but nothing was used either -- the deploy path is open loop.")
    elif al["beat_F"] <= floor + 0.02:
        verdict = "N/A (aligned score is already at/below the metronome floor -- nothing to leak)"
    elif ro["beat_F"] <= floor + 0.05:
        verdict = "CLEAN (aligned gain is real and disappears when the features slide)"
    else:
        verdict = f"*** LEAK SUSPECTED: rolled stays {ro['beat_F'] - floor:+.3f} above the floor ***"
    print(f"  {label:26s} -> drop {drop:+.3f}; {verdict}", flush=True)
    return res
