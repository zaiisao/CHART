"""Shared data / eval helpers for the Variant-B particle-filter experiments."""
from __future__ import annotations

import glob
import math
from pathlib import Path

import numpy as np
import torch

from vbpm.evaluate import (beats_from_barphase, downbeats_from_barphase,
                           metronome, f_measure, _estimate_meter)

CACHE = "/disk1/jaehoon/vbpm_mert_cache"
FPS = 50.0
H_DIM_DIRAC = 8


def load(split, cap=None, with_feats=False):
    out = []
    for f in sorted(glob.glob(f"{CACHE}/{split}__*.npz")):
        d = np.load(f, allow_pickle=True)
        r = dict(stem=Path(f).stem, T=int(d["feats"].shape[1]),
                 beats=np.asarray(d["beats"], float), downs=np.asarray(d["downs"], float))
        if with_feats:
            r["feats"] = d["feats"]
        out.append(r)
        if cap and len(out) >= cap:
            break
    return out


def dirac_h(beats, downs, start, n, rng, shift=0):
    """h[:,0]=beat impulses, h[:,1]=downbeat impulses (+ tiny noise).  ORACLE input.

    `rng` is required: for the SHIFT TEST the background noise must be bit-identical
    between the shifted and unshifted builds, so the ONLY difference is impulse position.
    """
    h = rng.standard_normal((n, H_DIM_DIRAC)).astype(np.float32) * 0.01
    for t in beats:
        i = int(round(t * FPS)) - start + shift
        if 0 <= i < n:
            h[i, 0] += 1.0
    for t in downs:
        i = int(round(t * FPS)) - start + shift
        if 0 <= i < n:
            h[i, 1] += 1.0
    return h


def targets(beats, downs, start, n):
    b = np.zeros(n, np.float32); db = np.zeros(n, np.float32)
    for t in beats:
        i = int(round(t * FPS)) - start
        if 0 <= i < n:
            b[i] = 1.0
    for t in downs:
        i = int(round(t * FPS)) - start
        if 0 <= i < n:
            db[i] = 1.0
    return b, db


def dirac_obs(h: torch.Tensor) -> torch.Tensor:
    """Observation target for the Dirac regime: the two binary impulse channels."""
    return (h[..., :2] > 0.5).to(h.dtype)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def blind_grid_F(ref, T, n_est, n_off=16, seed=0):
    """DENSITY-MATCHED PHASE-BLIND FLOOR: an evenly spaced grid with the SAME number of
    beats as the estimate, at a random phase.  If a deploy read-out only matches this, it
    is not tracking -- it is emitting at a lucky density."""
    if n_est < 2:
        return 0.0
    dur = T / FPS
    per = dur / n_est
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_off):
        off = rng.random() * per
        out.append(f_measure(ref, np.arange(off, dur, per)))
    return float(np.mean(out))


def monotonise(phase):
    """Project a sampled phase path onto the bar-pointer's own MONOTONE support.

    The generative advance is phi_{t-1} + phidot with phidot > 0, so the bar pointer can
    never run backwards; backward steps in a particle path come only from the heavy
    wrapped-Cauchy tail, and each one manufactures a spurious 2*pi wrap in the read-out.
    Unwrap the increments, clamp negatives to 0, re-accumulate, re-wrap.  The result is
    still fed to the MANDATED vbpm.evaluate.beats_from_barphase.
    """
    p = np.asarray(phase, dtype=float)
    d = np.diff(p)
    d = (d + math.pi) % (2 * math.pi) - math.pi        # wrap increments to (-pi, pi]
    d = np.clip(d, 0.0, None)
    return np.concatenate([[p[0]], p[0] + np.cumsum(d)]) % (2 * math.pi)


def score_phase(phase, ref, dref, m, T, tag_seed=0):
    est = beats_from_barphase(phase, m, FPS)
    dest = downbeats_from_barphase(phase, FPS)
    return dict(
        beat_F=f_measure(ref, est),
        db_F=(f_measure(dref, dest) if len(dref) >= 2 else float("nan")),
        n_est=len(est), n_true=len(ref),
        blind=blind_grid_F(ref, T, len(est), seed=tag_seed),
    )


def agg(rows, key):
    v = [r[key] for r in rows if r[key] == r[key]]
    return float(np.mean(v)) if v else float("nan")


def summarize(rows):
    n_est = sum(r["n_est"] for r in rows); n_true = sum(r["n_true"] for r in rows)
    return dict(beat_F=agg(rows, "beat_F"), db_F=agg(rows, "db_F"),
                n_ratio=(n_est / max(n_true, 1)), blind_floor=agg(rows, "blind"),
                N=len(rows))


def circ_maxdiff(a, b):
    d = np.abs(np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b)))))
    return float(d.max()), float(d.mean())
