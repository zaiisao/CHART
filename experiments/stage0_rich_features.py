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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "v2"))

from vbpm.data import FPS, VALUES, extract_crops, iter_frontend_features, slice_h  # noqa: E402
from vbpm.stage0 import Stage0  # noqa: E402
from vbpm.train_real import fit_vectorized, score  # noqa: E402


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


def load(limit_per_fold=None, device="cuda"):
    crops = []
    for s, H in iter_frontend_features(device=device, limit_per_fold=limit_per_fold,
                                       output="features"):
        song_crops, _ = extract_crops(*s.beats())
        for c in song_crops:
            X, _ = slice_h(H, c["beats"])
            X = X.astype(np.float64)
            crops.append({"s": {k: fn(X) for k, (fn, _) in REDUCTIONS.items()},
                          "y": c["y"], "m_true": c["m_true"], "dataset": s.dataset,
                          "fold": s.fold, "stem": s.stem, "crop": c["crop"]})
    return crops


def main():
    smoke = "--smoke" in sys.argv
    crops = load(limit_per_fold=6 if smoke else None)
    print(f"crops: {len(crops)}")

    ident = lambda v: torch.as_tensor(np.asarray(v, dtype=np.float64))  # noqa: E731
    cv = [c for c in crops if c["fold"] is not None]
    test = [c for c in crops if c["fold"] is None]

    for name, (_, s_dim) in REDUCTIONS.items():
        entries = lambda cs: [{"h": c["s"][name], "y": c["y"], "m_true": c["m_true"]}  # noqa: E731
                              for c in cs]
        pooled, preds = [], []
        for fold in sorted({c["fold"] for c in cv}):
            train = entries([c for c in cv if c["fold"] != fold])
            held = [c for c in cv if c["fold"] == fold]
            model = fit_vectorized(Stage0(VALUES, reducer=ident, s_dim=s_dim), train)
            preds += [model.to_value(int(model.predict(c["s"][name]).argmax())) for c in held]
            pooled += held
        print(f"---- {name} (s_dim={s_dim}) ----")
        for ds in sorted({c["dataset"] for c in pooled}):
            sel = [i for i, c in enumerate(pooled) if c["dataset"] == ds]
            score(ds, [pooled[i] for i in sel], [preds[i] for i in sel], VALUES)
        score("ALL-CV", pooled, preds, VALUES)
        if test:
            model = fit_vectorized(Stage0(VALUES, reducer=ident, s_dim=s_dim), entries(cv))
            t_preds = [model.to_value(int(model.predict(c["s"][name]).argmax()))
                       for c in test]
            score("gtzan", test, t_preds, VALUES)


if __name__ == "__main__":
    main()
