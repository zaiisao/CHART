"""Shared data / eval helpers for the Variant-B experiments (no vbpm/ file is modified)."""
import glob, math
from pathlib import Path
import numpy as np, torch

from vbpm.evaluate import (beats_from_barphase, downbeats_from_barphase,
                           metronome, f_measure, _estimate_meter)

CACHE = "/disk1/jaehoon/vbpm_mert_cache"
FPS = 50.0
H_DIM_DIRAC = 8
TWO_PI = 2 * math.pi


def load_split(split, cap=None, with_feats=False):
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


def dirac_h(beats, downs, start, n, shift_frames=0, rng=None, h_dim=H_DIM_DIRAC):
    """h[:,0]=beat impulses, h[:,1]=downbeat impulses, rest small noise.
    `shift_frames` moves BOTH impulse trains later in time (the mechanism/shift test)."""
    r = rng if rng is not None else np.random
    h = r.standard_normal((n, h_dim)).astype(np.float32) * 0.01
    for t in beats:
        i = int(round(t * FPS)) - start + shift_frames
        if 0 <= i < n:
            h[i, 0] += 1.0
    for t in downs:
        i = int(round(t * FPS)) - start + shift_frames
        if 0 <= i < n:
            h[i, 1] += 1.0
    return h


def targets(beats, downs, start, n):
    b = np.zeros(n, np.float32); db = np.zeros(n, np.float32)
    for t in beats:
        i = int(round(t * FPS)) - start
        if 0 <= i < n: b[i] = 1.0
    for t in downs:
        i = int(round(t * FPS)) - start
        if 0 <= i < n: db[i] = 1.0
    return b, db


def smooth_phase(traj, win=5):
    """Read-out helper: unwrap -> moving average -> rewrap. Removes particle jitter that
    would otherwise create spurious 2*pi wraps. A READ-OUT choice, not a model change."""
    if win <= 1:
        return np.asarray(traj, float)
    u = np.unwrap(np.asarray(traj, float))
    k = np.ones(win) / win
    s = np.convolve(u, k, mode="same")
    s[:win] = u[:win]; s[-win:] = u[-win:]
    return s % TWO_PI


def score_phase(phase, song, T, fps=FPS, smooth=0):
    """Deploy read-out exactly as mandated: beats_from_barphase + f_measure vs GT."""
    ref = song["beats"][song["beats"] < T / fps]
    dref = song["downs"][song["downs"] < T / fps]
    if len(ref) < 2:
        return None
    ph = smooth_phase(phase, smooth) if smooth else np.asarray(phase, float)
    m = _estimate_meter(ref, dref)
    est = beats_from_barphase(ph, m, fps)
    dest = downbeats_from_barphase(ph, fps)
    return {"beat_F": f_measure(ref, est),
            "db_F": f_measure(dref, dest) if len(dref) >= 2 else float("nan"),
            "n_est": len(est), "n_true": len(ref),
            "metronome": f_measure(ref, metronome(T, fps)), "m": m}


def circ_absdiff(a, b):
    d = (np.asarray(a) - np.asarray(b) + math.pi) % TWO_PI - math.pi
    return np.abs(d)


def best_lag(traj0, traj1, max_lag=45):
    """Lag L (frames) that best aligns traj1(t) with traj0(t-L).

    A deploy path that TRACKS the audio must return L ~= +shift when the input is delayed by
    `shift` frames.  An open-loop metronome returns an arbitrary L with a flat objective.
    Also returns the circular-distance curve min/max so flatness is visible.
    """
    t0 = np.asarray(traj0, float); t1 = np.asarray(traj1, float)
    lags = np.arange(-max_lag, max_lag + 1)
    costs = []
    for L in lags:
        if L >= 0:
            a, b = t0[:len(t0) - L], t1[L:]
        else:
            a, b = t0[-L:], t1[:len(t1) + L]
        costs.append(circ_absdiff(a, b).mean())
    costs = np.asarray(costs)
    return int(lags[costs.argmin()]), float(costs.min()), float(costs.max())


def shift_stats(traj0, traj1, max_lag=45):
    d = circ_absdiff(traj0, traj1)
    L, cmin, cmax = best_lag(traj0, traj1, max_lag)
    return {"max": float(d.max()), "mean": float(d.mean()),
            "lag": L, "lag_cost_min": cmin, "lag_cost_max": cmax}


def agg(rows, key):
    v = [r[key] for r in rows if r is not None and not math.isnan(r[key])]
    return float(np.mean(v)) if v else float("nan")


def ratio(rows):
    return float(np.mean([r["n_est"] / max(r["n_true"], 1) for r in rows if r]))
