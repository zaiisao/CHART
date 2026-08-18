"""Does the ELBO want the correction it needs?

The matched-factorisation q gives the per-step prior cost of an offset in closed form,
kappa_phys * A(kappa_q) * (1 - cos delta), and the emission gives the reward. So the
offset the objective actually wants is the argmax of

    J(delta) = loglik(path displaced by d - delta) - loglik(path at d) - cost(delta)

with d the placement error to be repaired. No training, no encoder: the annotations
build the oracle path, and the only question is whether the objective's own optimum
reaches d. Swept over the observation noise sigma_obs, whose curvature 1/sigma^2 sets
the reward's slope.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from ..data.dataset import load_catalog
from ..observation import gauss_time_loglik

TWO_PI = 2.0 * math.pi


def oracle_path(downbeats, fps, seconds):
    """Per-frame phase from the annotated downbeats, plus the annotation frames."""
    T = int(round(seconds * fps))
    t = np.arange(T) / fps
    turns = TWO_PI * np.arange(len(downbeats))
    phi = np.interp(t, downbeats, turns)
    rate = np.gradient(phi) * 1.0
    keep = (downbeats >= t[0]) & (downbeats <= t[-1])
    return phi, rate, downbeats[keep] * fps


def main():
    """Sweep sigma_obs and the placement error; print the offset the ELBO wants."""
    p = argparse.ArgumentParser()
    p.add_argument("--songs", type=int, default=24)
    p.add_argument("--seconds", type=float, default=45.0)
    p.add_argument("--fps", type=float, default=50.0)
    p.add_argument("--kappa-phys", type=float, default=100000.0)
    p.add_argument("--sigmas", default="0.100,0.050,0.025,0.0125")
    p.add_argument("--errors", default="0.20,0.79,1.57")
    args = p.parse_args()

    folds = load_catalog()
    songs = [s for f, ss in folds.items() if f is not None for s in ss]
    rng = np.random.default_rng(0)
    rng.shuffle(songs)

    paths = []
    for s in songs:
        if len(paths) >= args.songs:
            break
        try:
            raw = np.loadtxt(s.beats_path, usecols=(0, 1), ndmin=2)
        except Exception:
            continue
        db = raw[raw[:, 1] == 1][:, 0]
        db = db[db < args.seconds]
        if len(db) < 8:
            continue
        phi, rate, ann = oracle_path(db, args.fps, args.seconds)
        paths.append((phi, ann))

    phi = torch.tensor(np.stack([p_[0] for p_ in paths]), dtype=torch.float64)
    n = max(len(p_[1]) for p_ in paths)
    ann_f = torch.zeros(len(paths), n, dtype=torch.float64)
    ann_v = torch.zeros(len(paths), n, dtype=torch.float64)
    for i, (_, a) in enumerate(paths):
        ann_f[i, :len(a)] = torch.tensor(a)
        ann_v[i, :len(a)] = 1.0

    grid = torch.linspace(-math.pi, math.pi, 721, dtype=torch.float64)
    print(f"{len(paths)} songs, {args.seconds:.0f}s, kappa_phys={args.kappa_phys:g}\n")
    print(f"{'sigma_obs':>10} {'d (rad)':>9} {'delta*':>9} {'frac repaired':>14} {'J(delta*)':>11}")
    for sig in [float(x) for x in args.sigmas.split(",")]:
        for d in [float(x) for x in args.errors.split(",")]:
            base = gauss_time_loglik(phi + d, ann_f, ann_v, sig, fps=args.fps)
            best_d, best_J = [], []
            for b in range(phi.shape[0]):
                J = []
                for g in grid:
                    ll = gauss_time_loglik((phi[b] + d - g)[None], ann_f[b:b+1],
                                           ann_v[b:b+1], sig, fps=args.fps)[0]
                    J.append(float(ll - base[b] - args.kappa_phys * (1 - math.cos(float(g)))))
                J = np.array(J)
                k = int(J.argmax())
                best_d.append(float(grid[k]))
                best_J.append(J[k])
            bd = np.median(best_d)
            print(f"{sig:10.4f} {d:9.3f} {bd:9.4f} {bd/d:14.3f} {np.median(best_J):11.2f}")


if __name__ == "__main__":
    main()
