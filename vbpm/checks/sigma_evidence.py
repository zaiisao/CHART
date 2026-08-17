"""Does the objective prefer the RIGHT walk scale, or just the largest one?

A per-song walk scale is only inferable if the evidence discriminates it. The risk is
that a looser walk excuses any trajectory, so the likelihood rises monotonically in
sigma and the only thing resisting is the hyperprior -- which would make the mechanism
the hyperprior's tightness rather than the data.

This enumerates sigma on the model's own terms at a KNOWN-GOOD trajectory (the oracle
phase path), and reports where the phase-prior term peaks against the song's annotated
variability. Truth is used to build the path, so this measures the objective's shape,
not a deployable read-out.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
from scipy import stats

from ..checks import oracle_z as OZ
from ..data.dataset import split_songs
from ..data.excerpts import ExcerptDataset
from ..model import VBPM
from ..specs import WalkSpec

TWO_PI = 2.0 * math.pi


def smooth(x, sigma):
    """Boxcar over a fixed window, edge-padded."""
    n = int(4 * sigma) | 1
    k = np.exp(-0.5 * ((np.arange(n) - n // 2) / sigma) ** 2)
    return np.convolve(np.pad(x, n // 2, mode="edge"), k / k.sum(), mode="valid")


def main():
    """Enumerate the walk scale and see whether the evidence discriminates it."""
    p = argparse.ArgumentParser()
    p.add_argument("--per-dataset", type=int, default=25)
    p.add_argument("--grid", type=int, default=13)
    p.add_argument("--smooth", type=float, default=25.0)
    args = p.parse_args()

    sigmas = np.exp(np.linspace(math.log(2e-4), math.log(2e-1), args.grid))
    models = [VBPM(1, walk=WalkSpec(kind="gauss")).double() for _ in sigmas]
    for m, s in zip(models, sigmas):
        m.walk.tempo_sigma = float(m.walk.tempo_sigma)
    stub = OZ._Stub("beat_this", 50.0)
    train, val, _t = split_songs(0)
    by = {}
    for song in train + val:
        by.setdefault(song.dataset, []).append(song)
    rng = np.random.default_rng(0)
    songs = [g[i] for g in by.values() for i in rng.permutation(len(g))[:args.per_dataset]]
    dataset = ExcerptDataset(songs, stub, 45.0, deterministic=True, target_tol_frames=0)

    rows = []
    for crop in OZ.crops_of(dataset, 8):
        anc = np.asarray(crop["anchors"], dtype=np.float64)
        if len(anc) < 8 or np.diff(anc).min() <= 0:
            continue
        lo = max(int(np.ceil((anc[0] - crop["t0"]) * crop["fps"])) + 1, 0)
        hi = min(int(np.floor((anc[-1] - crop["t0"]) * crop["fps"])) - 1, len(crop["y"]))
        if hi - lo < 400:
            continue
        phi = np.asarray(crop["phi"][lo:hi], dtype=np.float64)
        if np.any(np.diff(phi) <= 0):
            continue
        lr = smooth(np.log(np.diff(phi)), args.smooth)
        b_true = float(np.mean(np.abs(np.diff(np.log(np.diff(anc))))))
        scores = []
        for s in sigmas:
            step = lr[1:] - lr[:-1]
            lp = (-0.5 * (step / s) ** 2 - math.log(s) - 0.5 * math.log(TWO_PI)).sum()
            scores.append(float(lp))
        scores = np.array(scores)
        rows.append((crop["dataset"], b_true, sigmas[int(scores.argmax())], scores))

    ds = np.array([r[0] for r in rows])
    bt = np.array([r[1] for r in rows])
    sh = np.array([r[2] for r in rows])
    print(f"n = {len(rows)} songs   sigma grid {sigmas[0]:.5f} .. {sigmas[-1]:.5f} "
          f"({args.grid} points, per-frame)\n")
    interior = np.mean((sh > sigmas[0]) & (sh < sigmas[-1]))
    print(f"argmax is INTERIOR (not railed to an endpoint) on {interior:.0%} of songs")
    print(f"argmax at the largest grid value on {np.mean(sh == sigmas[-1]):.0%}")
    r = stats.spearmanr(np.log(bt), np.log(sh))
    print(f"Spearman(annotated variability, evidence-selected sigma) = {r.statistic:+.3f} "
          f"(p {r.pvalue:.1e})\n")
    print(f"{'corpus':12} {'n':>4} {'annotated b':>12} {'selected sigma':>15}")
    for d in sorted(set(ds)):
        m = ds == d
        print(f"{d:12} {int(m.sum()):4d} {np.median(bt[m]):12.4f} {np.median(sh[m]):15.5f}")


if __name__ == "__main__":
    main()
