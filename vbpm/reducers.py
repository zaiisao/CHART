"""§4.4 reducers: s(h) -> R^{d}, swappable behind one function.

`mean_max` is the Stage-0 default (re-exported from stage0). `peak_summary` is the
hand-built alternative §4.4 names as an experimental axis: rates and ratio of beat vs
downbeat activation peaks, plus downbeat-channel autocorrelation at multiples of the
beat period. Deployable: reads h only (C2); non-differentiable is fine — ψ's gradient
flows through W and b, s(h) is a per-song constant.
"""
from __future__ import annotations

import numpy as np
import torch

from .stage0 import reduce_h as mean_max  # noqa: F401  (the §4.4 default)

MEAN_MAX_DIM = 4        # 2D with D=2
PEAK_SUMMARY_DIM = 10


def _pick_peaks(x: np.ndarray, threshold: float = 0.5, min_distance: int = 2) -> np.ndarray:
    """Strict local maxima above threshold, greedily thinned (same rule as the §8 baseline)."""
    cand = [i for i in range(1, len(x) - 1)
            if x[i] > threshold and x[i] >= x[i - 1] and x[i] > x[i + 1]]
    out: list = []
    for i in cand:
        if not out or i - out[-1] >= min_distance:
            out.append(i)
        elif x[i] > x[out[-1]]:
            out[-1] = i
    return np.asarray(out, dtype=int)


def peak_summary(h) -> torch.Tensor:
    """[T, 2] activation logits -> R^10 summary. Expresses the beat/downbeat rate ratio
    that mean⊕max provably cannot (measured 2026-07-31: 0.405 vs peak-count 0.702)."""
    logits = np.asarray(h, dtype=np.float64)
    prob = 1.0 / (1.0 + np.exp(-logits))
    T = max(len(prob), 1)
    beat, down = prob[:, 0], prob[:, 1]
    beat_peaks, down_peaks = _pick_peaks(beat), _pick_peaks(down)
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
                        n_beat / T * 50.0, n_down / T * 50.0, peak_ratio,
                        autocorr[0], autocorr[1], autocorr[2]], dtype=np.float64)
    return torch.tensor(summary, dtype=torch.float64)


REDUCERS = {
    "meanmax": (mean_max, MEAN_MAX_DIM),
    "peaks": (peak_summary, PEAK_SUMMARY_DIM),
}
