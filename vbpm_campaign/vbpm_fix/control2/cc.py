"""CONTROL AUDIT (independent re-run) -- shared helpers.

Only IMPORTS from vbpm/ (never mutates it). Everything is scored through the official
deploy read-out: vbpm.evaluate.beats_from_barphase / beats_from_activation + mir_eval F.
"""
from __future__ import annotations

import glob
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")

from vbpm.evaluate import (  # noqa: E402
    beats_from_barphase, downbeats_from_barphase, beats_from_activation,
    metronome, f_measure, _estimate_meter,
)

CACHE = "/disk1/jaehoon/vbpm_mert_cache"
FPS = 50.0
TWO_PI = 2.0 * math.pi


def load_split(split, with_feats=False, cap=None):
    out = []
    for f in sorted(glob.glob(f"{CACHE}/{split}__*.npz")):
        d = np.load(f, allow_pickle=True)
        rec = dict(stem=Path(f).stem, path=f, T=int(d["feats"].shape[1]),
                   beats=np.asarray(d["beats"], float), downs=np.asarray(d["downs"], float),
                   fps=float(d["fps"]), dataset=str(d["dataset"]))
        if with_feats:
            rec["feats"] = d["feats"]
        out.append(rec)
        if cap and len(out) >= cap:
            break
    return out


def truncate(song, max_frames):
    T = song["T"] if max_frames is None else min(song["T"], max_frames)
    ref = song["beats"][song["beats"] < T / FPS]
    dref = song["downs"][song["downs"] < T / FPS]
    return T, ref, dref


def ideal_barphase(downs, T, fps=FPS):
    """phi = 0 at each downbeat, linear to 2pi at the next; first/last bar extrapolated."""
    if len(downs) < 2:
        return None
    t = (np.arange(T) + 0.5) / fps
    ph = np.zeros(T, float)
    for i in range(len(downs) - 1):
        a, b = downs[i], downs[i + 1]
        msk = (t >= a) & (t < b)
        ph[msk] = TWO_PI * (t[msk] - a) / max(b - a, 1e-6)
    d0 = max(downs[1] - downs[0], 1e-6)
    pre = t < downs[0]
    ph[pre] = (TWO_PI * (t[pre] - downs[0]) / d0) % TWO_PI
    dl = max(downs[-1] - downs[-2], 1e-6)
    post = t >= downs[-1]
    ph[post] = (TWO_PI * (t[post] - downs[-1]) / dl) % TWO_PI
    return ph


def score_phase(phase, ref, dref, meter=None):
    m = _estimate_meter(ref, dref) if meter is None else meter
    est_b = beats_from_barphase(np.asarray(phase), m, FPS)
    est_d = downbeats_from_barphase(np.asarray(phase), FPS)
    return dict(meter=int(m), beat_F=f_measure(ref, est_b),
                downbeat_F=(f_measure(dref, est_d) if len(dref) >= 2 else float("nan")),
                n_est=int(len(est_b)), n_true=int(len(ref)),
                n_est_db=int(len(est_d)), n_true_db=int(len(dref)))


def score_activation(prob, ref, dref, thr=0.5):
    est_b = beats_from_activation(np.asarray(prob), FPS, thr=thr)
    return dict(beat_F=f_measure(ref, est_b), n_est=int(len(est_b)), n_true=int(len(ref)),
                downbeat_F=float("nan"), n_est_db=0, n_true_db=int(len(dref)))


def agg(rows, key):
    v = [r[key] for r in rows if isinstance(r.get(key), float) and not math.isnan(r[key])]
    return float(np.mean(v)) if v else float("nan")


def sem(rows, key):
    v = [r[key] for r in rows if isinstance(r.get(key), float) and not math.isnan(r[key])]
    return float(np.std(v) / math.sqrt(max(len(v), 1))) if v else float("nan")


def ratio(rows, ke="n_est", kt="n_true"):
    return sum(r[ke] for r in rows) / max(sum(r[kt] for r in rows), 1)


def by_dataset(rows, key):
    ds = {}
    for r in rows:
        ds.setdefault(r["dataset"], []).append(r[key])
    return {k: (float(np.nanmean(v)), len(v)) for k, v in sorted(ds.items())}


def line(tag, rows, extra=""):
    print(f"  {tag:46s} beat_F={agg(rows,'beat_F'):.3f} db_F={agg(rows,'downbeat_F'):.3f} "
          f"n_est/n_true={ratio(rows):.3f} N={len(rows)} {extra}", flush=True)


def banner(s):
    print("\n" + "=" * 84)
    print(s)
    print("=" * 84, flush=True)
