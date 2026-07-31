"""Stage 0 as a downbeat tracker given beats: deployable decode, F-measure at +-70 ms.

Decode (h + beat grid only, no y):
    m_hat = argmax p_psi(m|h)                       (peaks-reducer prior, trained fold-honest)
    r_hat = argmax_r sum_{i = r mod m_hat} sigmoid(act_down at beat i)
    predicted downbeats = beats[r_hat :: m_hat]

Arms: raw peak-pick of the downbeat channel (unconstrained baseline), grid decode with
predicted m, grid decode with oracle m (separates meter cost from offset cost).

Run: CUDA_VISIBLE_DEVICES=3 /disk4/anaconda3/envs/chart/bin/python \
         experiments/stage0_downbeat_decode.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "v2"))
import reference as R  # noqa: E402

from vbpm.data import FPS, VALUES, extract_crops, iter_frontend_features, slice_h  # noqa: E402
from vbpm.reducers import REDUCERS  # noqa: E402
from vbpm.stage0 import Stage0  # noqa: E402
from vbpm.train_real import fit_vectorized  # noqa: E402
TOL_S = 0.07


def f_measure(pred_times, true_times, tol=TOL_S):
    """Greedy one-to-one matching within +-tol, the standard beat-eval convention."""
    pred, true = list(pred_times), list(true_times)
    used = np.zeros(len(true), dtype=bool)
    tp = 0
    for p in pred:
        best, best_d = -1, tol
        for j, t in enumerate(true):
            d = abs(p - t)
            if not used[j] and d <= best_d:
                best, best_d = j, d
        if best >= 0:
            used[best] = True
            tp += 1
    if not pred or not true:
        return 0.0 if (pred or true) else 1.0
    prec, rec = tp / len(pred), tp / len(true)
    return 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0


def act_at_beats(h_prob_down, crop_beats, t0):
    idx = np.clip(np.round((crop_beats - t0) * FPS - 0.5).astype(int),
                  0, len(h_prob_down) - 1)
    return h_prob_down[idx]


def grid_decode(m, down_at_beats, crop_beats):
    scores = [down_at_beats[r::m].sum() / max(len(down_at_beats[r::m]), 1)
              for r in range(m)]
    r_hat = int(np.argmax(scores))
    return crop_beats[r_hat::m]


def load(device="cuda"):
    """One crop entry per §5-valid crop, with activation h, beat grid and downbeats."""
    crops = []
    for s, H in iter_frontend_features(device=device):
        song_crops, _ = extract_crops(*s.beats())
        for c in song_crops:
            h_crop, t0 = slice_h(H, c["beats"])
            crops.append({"h": h_crop, "y": c["y"], "m_true": c["m_true"],
                          "beats": c["beats"], "downs": c["bounds"][:-1], "t0": t0,
                          "dataset": s.dataset, "fold": s.fold})
    return crops


def main():
    crops = load()
    cv = [c for c in crops if c["fold"] is not None]
    test = [c for c in crops if c["fold"] is None]
    reducer, s_dim = REDUCERS["peaks"]

    # fold-honest predicted m, stored on each crop
    for fold in sorted({c["fold"] for c in cv}):
        model = fit_vectorized(Stage0(VALUES, reducer=reducer, s_dim=s_dim),
                               [c for c in cv if c["fold"] != fold])
        for c in (c for c in cv if c["fold"] == fold):
            c["m_hat"] = model.to_value(int(model.predict(c["h"]).argmax()))
        print(f"fold {fold}: m predicted", flush=True)
    model_all = fit_vectorized(Stage0(VALUES, reducer=reducer, s_dim=s_dim), cv)
    for c in test:
        c["m_hat"] = model_all.to_value(int(model_all.predict(c["h"]).argmax()))

    print(f"\n== downbeat F at +-{TOL_S * 1000:.0f} ms, per dataset ==")
    print(f"{'dataset':12s} {'n':>6s} {'peakpick':>9s} {'grid-mhat':>10s} {'grid-oracle':>12s}")
    for ds in sorted({c["dataset"] for c in crops}):
        sel = [c for c in crops if c["dataset"] == ds]
        f_pp, f_grid, f_oracle = [], [], []
        for c in sel:
            prob_down = 1 / (1 + np.exp(-c["h"][:, 1]))
            peaks = R.pick_peaks(prob_down, threshold=0.5)
            f_pp.append(f_measure(c["t0"] + (peaks + 0.5) / FPS, c["downs"]))
            down_at_beats = act_at_beats(prob_down, c["beats"], c["t0"])
            f_grid.append(f_measure(
                grid_decode(c["m_hat"], down_at_beats, c["beats"]), c["downs"]))
            f_oracle.append(f_measure(
                grid_decode(c["m_true"], down_at_beats, c["beats"]), c["downs"]))
        print(f"{ds:12s} {len(sel):6d} {np.mean(f_pp):9.3f} {np.mean(f_grid):10.3f} "
              f"{np.mean(f_oracle):12.3f}", flush=True)


if __name__ == "__main__":
    main()
