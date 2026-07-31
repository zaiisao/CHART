"""Rich-features experiment: [T,512] penultimate features vs the compressed [T,2] activation.

Does meter evidence survive where the compression loses it (asap 0.350 vs synth ceiling 0.988)?

Frontend stays FROZEN (§6.1); we train only our own head (linear psi on a fixed reduction).
[T,512] for 18.9k crops is ~60 GB, so reductions are computed inside the load pass and only
the summary vectors are kept. Crops/labels/folds identical to the compressed-h run.

Run: CUDA_VISIBLE_DEVICES=1 /disk4/anaconda3/envs/chart/bin/python \
         experiments/stage0_rich_features.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from vbpm.data import FPS, VALUES, load_crops  # noqa: E402
from vbpm.stage0 import Stage0  # noqa: E402
from vbpm.train_real import (cv_out_of_fold, fit_vectorized, predict_m,  # noqa: E402
                             score, score_per_dataset)


def reduce_meanmax(X):
    return np.concatenate([X.mean(0), X.max(0)])                       # 1024


def reduce_meanmaxstd(X):
    return np.concatenate([X.mean(0), X.max(0), X.std(0)])             # 1536


def reduce_novelty(X):
    """8-dim: novelty-curve autocorrelation at the estimated beat lag and its multiples.

    Label-free and timing-tolerant: meter should show as relative autocorrelation strength
    at 2/3/4x the beat lag, whatever the absolute tempo.
    """
    d = np.diff(X, axis=0)
    nov = np.sqrt((d * d).sum(1))
    nov = (nov - nov.mean()) / (nov.std() + 1e-9)
    T = len(nov)
    L = min(T - 2, 320)
    if L < 20:
        return np.zeros(8)
    r = np.array([float((nov[:-lag] * nov[lag:]).mean()) for lag in range(1, L + 1)])
    lo, hi = 10, min(100, L)                       # beat period 0.2-2.0 s at 50 fps
    bl = lo + int(np.argmax(r[lo - 1:hi]))

    def at(mult):
        lag = int(round(bl * mult))
        return r[lag - 1] if lag <= L else 0.0

    return np.array([bl / FPS, at(1), at(2), at(3), at(4), at(6),
                     float(nov.mean()), float(np.log1p(T))])


REDUCTIONS = {"rich-meanmax": (reduce_meanmax, 1024),
              "rich-mms": (reduce_meanmaxstd, 1536),
              "rich-novelty": (reduce_novelty, 8)}


def make_entry(song, crop, h_crop, t0):
    X = h_crop.astype(np.float64)
    return {"s": {k: fn(X) for k, (fn, _) in REDUCTIONS.items()},
            "y": crop["y"], "m_true": crop["m_true"], "dataset": song.dataset,
            "fold": song.fold, "stem": song.stem, "crop": crop["crop"]}


def identity_reducer(v):
    """Precomputed summaries pass through psi untouched."""
    return torch.as_tensor(np.asarray(v, dtype=np.float64))


def main():
    smoke = "--smoke" in sys.argv
    crops, report = load_crops(limit_per_fold=6 if smoke else None,
                               output="features", make_entry=make_entry)
    print(f"crops: {len(crops)}  rejects: {report['rejects']}")

    cv = [c for c in crops if c["fold"] is not None]
    test = [c for c in crops if c["fold"] is None]

    for name, (_, s_dim) in REDUCTIONS.items():
        def fit_fn(train):
            entries = [{"h": c["s"][name], "y": c["y"]} for c in train]
            return fit_vectorized(Stage0(VALUES, reducer=identity_reducer, s_dim=s_dim),
                                  entries)

        pooled, preds, test_preds = cv_out_of_fold(
            cv, test, fit_fn,
            lambda model, cs: [predict_m(model, c["s"][name]) for c in cs],
            verbose=False)
        print(f"---- {name} (s_dim={s_dim}) ----")
        score_per_dataset(pooled, preds, VALUES)
        if test:
            score("gtzan", test, test_preds, VALUES)


if __name__ == "__main__":
    main()
