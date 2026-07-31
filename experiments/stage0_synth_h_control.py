"""Causal control (all datasets): swap ONLY h for clean synthetic activations.

Originally written for the asap-at-chance result, then extended to every corpus.

Real asap annotations, real crop boundaries, real per-crop labels, same folds, same
reducers, same linear psi, same fit — but h is Gaussian bumps at the ANNOTATED beat
(ch 0) and downbeat (ch 1) times, converted to logits (the pipeline's activation form).
Rubato timing is preserved because the bump times are the real annotated times.

  * synthetic-h accuracy HIGH  -> everything downstream of h is sound; the real
    frontend activation content is the bottleneck on asap.
  * synthetic-h accuracy ~chance -> the fault is ours (labels/crops/reducer/psi), not
    the frontend.

Run: /disk4/anaconda3/envs/chart/bin/python experiments/stage0_synth_h_control.py
"""
import math

import numpy as np

from vbpm import metrics as R

from data.songs import iter_songs
from vbpm.data import FPS, VALUES, extract_crops, to_prob
from vbpm.reducers import REDUCERS
from vbpm.stage0 import Stage0
from vbpm.fitting import (cv_out_of_fold, fit_vectorized, predict_m,
                             score, score_per_dataset)

SIGMA_S = 0.06          # the synthetic bench's bump width (§6.4)


def synth_h(crop_beats, crop_downs):
    """[T, 2] LOGIT activations: bumps at annotated beats (ch 0) / downbeats (ch 1)."""
    t0 = crop_beats[0] - 0.5
    duration = (crop_beats[-1] - t0) + 0.5
    T = int(math.ceil(duration * FPS))
    t = t0 + (np.arange(T) + 0.5) / FPS

    def bumps(times):
        acc = np.zeros(T)
        for tc in times:
            acc += np.exp(-0.5 * ((t - tc) / SIGMA_S) ** 2)
        return np.clip(acc, 0.0, 1.0)

    prob = np.stack([bumps(crop_beats), bumps(crop_downs)], axis=1)
    prob = prob * (1 - 2e-3) + 1e-3                       # bound away from {0,1}
    return np.log(prob / (1 - prob)).astype(np.float64)   # logits, like the frontend


def main(datasets=None):
    crops = []
    for s in iter_songs(datasets=datasets):
        beat_times, downbeat_times = s.beats()
        song_crops, _ = extract_crops(beat_times, downbeat_times)
        for c in song_crops:
            crops.append({"h": synth_h(c["beats"], c["bounds"][:-1]), "y": c["y"],
                          "m_true": c["m_true"], "dataset": s.dataset, "fold": s.fold,
                          "stem": s.stem, "crop": c["crop"]})
    n_by_m = {m: sum(1 for c in crops if c["m_true"] == m) for m in VALUES}
    print(f"synthetic-h crops: {len(crops)}  by class: {n_by_m}")

    cv = [c for c in crops if c["fold"] is not None]
    test = [c for c in crops if c["fold"] is None]
    for name in ("meanmax", "peaks"):
        reducer, s_dim = REDUCERS[name]
        pooled, preds, test_preds = cv_out_of_fold(
            cv, test,
            lambda train: fit_vectorized(Stage0(VALUES, reducer=reducer, s_dim=s_dim), train),
            lambda model, cs: [predict_m(model, c["h"]) for c in cs],
            verbose=False)
        print(f"---- reducer {name} ----")
        score_per_dataset(pooled, preds, VALUES)
        if name == "peaks":
            meter_change_breakdown(pooled, preds)
        if test:
            score("gtzan", test, test_preds, VALUES)

    pk = [R.peak_count_estimate(to_prob(c["h"]), VALUES) for c in crops]
    print("---- baseline ----")
    for ds in sorted({c["dataset"] for c in crops}):
        sel = [i for i, c in enumerate(crops) if c["dataset"] == ds]
        score(f"pk-{ds}", [crops[i] for i in sel], [pk[i] for i in sel], VALUES)


def meter_change_breakdown(pooled, preds):
    """Split the pooled score by song-level meter change.

    Constant-meter songs vs changing songs, and within changing songs by whether the
    crop sits at a label switch.
    """
    by_song = {}
    for i, c in enumerate(pooled):
        by_song.setdefault((c["dataset"], c["stem"]), []).append(i)

    const_idx, away_idx, switch_idx = [], [], []
    n_changing_songs = 0
    for idx in by_song.values():
        idx = sorted(idx, key=lambda i: pooled[i]["crop"])
        labels = [pooled[i]["m_true"] for i in idx]
        if len(set(labels)) == 1:
            const_idx += idx
            continue
        n_changing_songs += 1
        for j, i in enumerate(idx):
            at_switch = ((j > 0 and labels[j] != labels[j - 1])
                         or (j + 1 < len(idx) and labels[j] != labels[j + 1]))
            (switch_idx if at_switch else away_idx).append(i)

    print(f"\n---- meter-change breakdown ({n_changing_songs} changing songs of "
          f"{len(by_song)}) ----")
    for tag, idx in (("constant-m", const_idx), ("chg/away", away_idx),
                     ("chg/switch", switch_idx)):
        if idx:
            score(tag, [pooled[i] for i in idx], [preds[i] for i in idx], VALUES)


if __name__ == "__main__":
    main()
