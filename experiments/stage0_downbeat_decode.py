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

import mir_eval.beat
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "v2"))
import reference as R  # noqa: E402

from vbpm.data import FPS, VALUES, load_crops, to_prob  # noqa: E402
from vbpm.reducers import REDUCERS  # noqa: E402
from vbpm.stage0 import Stage0  # noqa: E402
from vbpm.train_real import cv_out_of_fold, fit_vectorized, predict_m  # noqa: E402

TOL_S = 0.07


def f_measure(pred_times, true_times, tol=TOL_S):
    """mir_eval's standard F-measure, without its 5-s trim (crops are crop-local)."""
    return mir_eval.beat.f_measure(np.asarray(true_times), np.asarray(pred_times),
                                   f_measure_threshold=tol)


def act_at_beats(h_prob_down, crop_beats, t0):
    idx = np.clip(np.round((crop_beats - t0) * FPS - 0.5).astype(int),
                  0, len(h_prob_down) - 1)
    return h_prob_down[idx]


def grid_decode(m, down_at_beats, crop_beats):
    scores = [down_at_beats[r::m].sum() / max(len(down_at_beats[r::m]), 1)
              for r in range(m)]
    r_hat = int(np.argmax(scores))
    return crop_beats[r_hat::m]


def make_entry(song, crop, h_crop, t0):
    """Standard fields plus the beat grid, downbeats and frame origin the decode needs."""
    return {"h": h_crop, "y": crop["y"], "m_true": crop["m_true"],
            "beats": crop["beats"], "downs": crop["bounds"][:-1], "t0": t0,
            "dataset": song.dataset, "fold": song.fold}


def main():
    crops, report = load_crops(make_entry=make_entry)
    print(f"crops: {report['usable']}  rejects: {report['rejects']}")
    cv = [c for c in crops if c["fold"] is not None]
    test = [c for c in crops if c["fold"] is None]
    reducer, s_dim = REDUCERS["peaks"]

    # fold-honest predicted m, stored on each crop
    pooled, preds, test_preds = cv_out_of_fold(
        cv, test,
        lambda train: fit_vectorized(Stage0(VALUES, reducer=reducer, s_dim=s_dim), train),
        lambda model, cs: [predict_m(model, c["h"]) for c in cs])
    for c, m_hat in zip(pooled + test, preds + test_preds):
        c["m_hat"] = m_hat

    print(f"\n== downbeat F at +-{TOL_S * 1000:.0f} ms, per dataset ==")
    print(f"{'dataset':12s} {'n':>6s} {'peakpick':>9s} {'grid-mhat':>10s} {'grid-oracle':>12s}")
    for ds in sorted({c["dataset"] for c in crops}):
        sel = [c for c in crops if c["dataset"] == ds]
        f_pp, f_grid, f_oracle = [], [], []
        for c in sel:
            prob_down = to_prob(c["h"][:, 1])
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
