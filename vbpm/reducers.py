"""Reducers: s(h) -> R^d, the swappable compression between features and prior.

The prior needs a fixed-size input but h is [T, D] with variable T; a reducer is the
function that compresses one to the other, and the spec requires it be swappable so a
pooling change is not a rewrite. `mean_max` is the default (re-exported from stage0).
`peak_summary` is the hand-built alternative: rates and ratio of beat vs downbeat
activation peaks, plus downbeat-channel autocorrelation at multiples of the beat
period. Both are deployable (they read audio features only, never annotations), and
non-differentiable is fine — the prior's gradient flows through W and b, s(h) is a
per-crop constant.
"""
from __future__ import annotations

import numpy as np
import torch

from .data import FPS, to_prob
from .metrics import pick_peaks
from .stage0 import reduce_h as mean_max  # noqa: F401  (the spec's default reducer)

MEAN_MAX_DIM = 4        # 2D with D=2
PEAK_SUMMARY_DIM = 10


def peak_summary(h) -> torch.Tensor:
    """[T, 2] activation logits -> a 10-dim summary built around peak counting.

    Meter lives in the RATE of downbeat peaks relative to beat peaks; mean/max pooling
    discards peakiness entirely, which is why this exists (measured 2026-07-31: the
    mean⊕max default collapsed on real crops while a peak-count rule scored 0.702).
    """
    prob = to_prob(h)
    T = max(len(prob), 1)
    beat, down = prob[:, 0], prob[:, 1]
    beat_peaks, down_peaks = pick_peaks(beat), pick_peaks(down)
    n_beat, n_down = len(beat_peaks), len(down_peaks)
    peak_ratio = n_beat / max(n_down, 1)          # ~beats per bar, the meter's signature

    # downbeat-channel autocorrelation at lags k * (median inter-beat-peak interval)
    autocorr = np.zeros(3)
    if n_beat >= 3:
        beat_lag = int(round(float(np.median(np.diff(beat_peaks)))))
        centred = down - down.mean()
        denom = float((centred ** 2).sum())
        for j, k in enumerate((2, 3, 4)):
            lag = k * beat_lag
            if 0 < lag < T and denom > 0:
                autocorr[j] = float((centred[:-lag] * centred[lag:]).sum()) / denom

    summary = np.array([beat.mean(), down.mean(),
                        beat.max(initial=0.0), down.max(initial=0.0),
                        n_beat / T * FPS, n_down / T * FPS, peak_ratio,
                        autocorr[0], autocorr[1], autocorr[2]], dtype=np.float64)
    return torch.tensor(summary, dtype=torch.float64)


REDUCERS = {
    "meanmax": (mean_max, MEAN_MAX_DIM),
    "peaks": (peak_summary, PEAK_SUMMARY_DIM),
}
