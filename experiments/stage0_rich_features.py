"""Rich-features experiment: does meter evidence survive in Beat This [T,512] penultimate
features where the compressed [T,2] activation loses it (asap 0.350 vs synth ceiling 0.988)?

Frontend stays FROZEN (§6.1); we train only our own head (linear psi on a fixed reduction).
[T,512] for 18.9k crops is ~60 GB, so reductions are computed inside the load pass and only
the summary vectors are kept. Crops/labels/folds identical to the compressed-h run.

Run: CUDA_VISIBLE_DEVICES=1 /disk4/anaconda3/envs/chart/bin/python experiments/stage0_rich_features.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "v2"))

from data.songs import iter_songs  # noqa: E402
from vbpm.data import MIN_BEATS, VALUES, derive_m_true, derive_y, make_crops  # noqa: E402
from vbpm.stage0 import Stage0  # noqa: E402
from vbpm.train_real import fit_vectorized, score  # noqa: E402

FPS = 50.0


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
    r = np.array([float((nov[:-l] * nov[l:]).mean()) for l in range(1, L + 1)])
    lo, hi = 10, min(100, L)                       # beat period 0.2-2.0 s at 50 fps
    bl = lo + int(np.argmax(r[lo - 1:hi]))

    def at(mult):
        l = int(round(bl * mult))
        return r[l - 1] if l <= L else 0.0

    return np.array([bl / FPS, at(1), at(2), at(3), at(4), at(6),
                     float(nov.mean()), float(np.log1p(T))])


REDUCTIONS = {"rich-meanmax": (reduce_meanmax, 1024),
              "rich-mms": (reduce_meanmaxstd, 1536),
              "rich-novelty": (reduce_novelty, 8)}


def load(limit_per_fold=None, device="cuda"):
    import soundfile
    from frontends.beat_this import BeatThisFrontend

    by_fold = {}
    for s in iter_songs():
        by_fold.setdefault(s.fold, []).append(s)

    crops = []
    for fold, members in sorted(by_fold.items(), key=lambda kv: (kv[0] is None, kv[0])):
        checkpoint = "final0" if fold is None else f"fold{fold}"
        frontend = BeatThisFrontend(checkpoint=checkpoint, device=device, output="features")
        if limit_per_fold is not None:
            members = members[:limit_per_fold]
        for s in members:
            beat_times, downbeat_times = s.beats()
            if len(downbeat_times) < 2:
                continue
            song_crops = []
            for ci, (cb, bounds) in enumerate(make_crops(beat_times, downbeat_times)):
                m_true = derive_m_true(cb, bounds)
                if m_true is None or m_true not in VALUES or len(cb) < MIN_BEATS:
                    continue
                y, _ = derive_y(cb, bounds[:-1])
                song_crops.append((ci, cb, y, m_true))
            if not song_crops:
                continue
            signal, sample_rate = soundfile.read(str(s.audio_path), dtype="float32")
            if signal.ndim > 1:
                signal = signal.mean(axis=1)
            H = frontend.get_features(signal, sample_rate).numpy()      # [T, 512]
            for ci, cb, y, m_true in song_crops:
                lo = max(0, int(math.floor(cb[0] * FPS)))
                hi = min(len(H), int(math.ceil(cb[-1] * FPS)) + 1)
                X = H[lo:hi].astype(np.float64)
                crops.append({"s": {k: fn(X) for k, (fn, _) in REDUCTIONS.items()},
                              "y": y, "m_true": m_true, "dataset": s.dataset,
                              "fold": s.fold, "stem": s.stem, "crop": ci})
        del frontend
        print(f"  {checkpoint}: done ({len(crops)} crops so far)", flush=True)
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
