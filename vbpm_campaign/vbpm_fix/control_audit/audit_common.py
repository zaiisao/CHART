"""CONTROL / ANTI-FOOLING AUDIT -- shared harness.

Everything the other agents' "fixes" get measured against lives here. Nothing in
vbpm/ is touched; this module only IMPORTS the official read-out + scorer so the
yardstick is literally the same code path the variants use.
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

# The protocol previous numbers were produced under (probe_linear / probe_dirac / train_mert):
#   ev = load("eval")[:30]   and   max_frames = 1600
# BOTH of those choices are audited -- see audit_1_ceiling.py.
PRIOR_MAX_FRAMES = 1600
PRIOR_N_EVAL = 30


def load_split(split, with_feats=False, cap=None):
    """Load the fold-honest cache. with_feats=False keeps only labels+T (fast)."""
    out = []
    for f in sorted(glob.glob(f"{CACHE}/{split}__*.npz")):
        d = np.load(f, allow_pickle=True)
        rec = dict(stem=Path(f).stem, path=f,
                   T=int(d["feats"].shape[1]),
                   beats=np.asarray(d["beats"], float),
                   downs=np.asarray(d["downs"], float),
                   fps=float(d["fps"]), dataset=str(d["dataset"]), fold=int(d["fold"]))
        if with_feats:
            rec["feats"] = d["feats"]
        out.append(rec)
        if cap and len(out) >= cap:
            break
    return out


# ----------------------------------------------------------------------------------
# IDEAL (oracle) bar phase -- the reference-ceiling generator
# ----------------------------------------------------------------------------------
def ideal_barphase(downs, T, fps=FPS, mode="extrap"):
    """phi = 0 at each downbeat, linear to 2*pi at the next downbeat.

    mode="strict" : phi held at 0 outside [first_down, last_down]  (what probe_stages.py did)
    mode="extrap" : first/last bar length extended outside the annotated span so beats
                    before the first / after the last downbeat are still emitted.
    """
    if len(downs) < 2:
        return None
    t = (np.arange(T) + 0.5) / fps
    ph = np.zeros(T, float)
    for i in range(len(downs) - 1):
        a, b = downs[i], downs[i + 1]
        msk = (t >= a) & (t < b)
        ph[msk] = TWO_PI * (t[msk] - a) / max(b - a, 1e-6)
    if mode == "extrap":
        d0 = max(downs[1] - downs[0], 1e-6)
        pre = t < downs[0]
        ph[pre] = (TWO_PI * (t[pre] - downs[0]) / d0) % TWO_PI
        dl = max(downs[-1] - downs[-2], 1e-6)
        post = t >= downs[-1]
        ph[post] = (TWO_PI * (t[post] - downs[-1]) / dl) % TWO_PI
    return ph


def ideal_beatlinear_barphase(beats, downs, T, fps=FPS):
    """Bar phase piecewise-linear BETWEEN BEATS (madmom's actual pointer semantics):
    knots at every beat, beat k inside a bar sits exactly at 2*pi*k/m. Isolates
    'read-out code' from 'beats are not equally spaced inside the bar'."""
    if len(downs) < 2 or len(beats) < 4:
        return None
    t = (np.arange(T) + 0.5) / fps
    ph = np.zeros(T, float)
    for i in range(len(downs) - 1):
        a, b = downs[i], downs[i + 1]
        inb = beats[(beats >= a - 1e-6) & (beats < b - 1e-6)]
        if len(inb) < 1:
            continue
        knots = np.concatenate([inb, [b]])
        m = len(inb)
        vals = np.concatenate([np.arange(m) * TWO_PI / m, [TWO_PI]])
        msk = (t >= a) & (t < b)
        ph[msk] = np.interp(t[msk], knots, vals)
    return ph


def truncate(song, max_frames):
    T = song["T"] if max_frames is None else min(song["T"], max_frames)
    ref = song["beats"][song["beats"] < T / FPS]
    dref = song["downs"][song["downs"] < T / FPS]
    return T, ref, dref


def score_phase(phase, ref, dref, T, meter=None):
    """Official deploy read-out applied to a bar-phase trajectory."""
    m = _estimate_meter(ref, dref) if meter is None else meter
    est_b = beats_from_barphase(phase, m, FPS)
    est_d = downbeats_from_barphase(phase, FPS)
    return dict(
        meter=int(m),
        beat_F=f_measure(ref, est_b),
        downbeat_F=f_measure(dref, est_d) if len(dref) >= 2 else float("nan"),
        n_est=int(len(est_b)), n_true=int(len(ref)),
        n_est_db=int(len(est_d)), n_true_db=int(len(dref)),
    )


def agg(rows, keys):
    out = {}
    for k in keys:
        v = [r[k] for r in rows if isinstance(r.get(k), float) and not math.isnan(r[k])]
        out[k] = float(np.mean(v)) if v else float("nan")
    return out


def by_dataset(rows, key):
    ds = {}
    for r in rows:
        ds.setdefault(r["dataset"], []).append(r[key])
    return {k: (float(np.nanmean(v)), len(v)) for k, v in sorted(ds.items())}


def ratio(rows, ke="n_est", kt="n_true"):
    ne = sum(r[ke] for r in rows)
    nt = sum(r[kt] for r in rows)
    return ne / max(nt, 1)


def banner(s):
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78, flush=True)
