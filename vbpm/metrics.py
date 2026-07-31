"""§8 metrics and deployable baselines — the RUNTIME owner.

tests/reference.py carries its own copies by design: the spec-side oracle must stay
self-contained and must never import the implementation it judges, and the runtime
package must never import from tests/. That duplication is sanctioned (same rule as the
oracle's model math); this module is the single owner for every non-test consumer.
"""
from __future__ import annotations

import numpy as np

DEFAULT_VALUES = (2, 3, 4)


def balanced_accuracy(true_m, pred_m, values=DEFAULT_VALUES) -> float:
    """Mean over PRESENT classes of per-class recall (§8 primary metric)."""
    true_m = np.asarray(true_m)
    pred_m = np.asarray(pred_m)
    recalls = []
    for m in values:
        sel = true_m == m
        if sel.sum() == 0:
            continue
        recalls.append(float((pred_m[sel] == m).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def confusion(true_m, pred_m, values=DEFAULT_VALUES) -> np.ndarray:
    """[K, K] counts, rows = true class, columns = predicted class."""
    idx = {m: i for i, m in enumerate(values)}
    counts = np.zeros((len(values), len(values)), dtype=int)
    for t, p in zip(np.asarray(true_m), np.asarray(pred_m)):
        counts[idx[int(t)], idx[int(p)]] += 1
    return counts


def distinct_predicted(pred_m) -> int:
    """§8 collapse detector: number of distinct predicted classes."""
    return int(len(set(int(p) for p in np.asarray(pred_m).ravel())))


def majority_predict(train_m, values=DEFAULT_VALUES) -> int:
    """§8 baseline: the training majority class (ties broken toward the smaller count)."""
    counts = {m: 0 for m in values}
    for m in train_m:
        counts[int(m)] += 1
    return max(values, key=lambda m: (counts[m], -m))


def pick_peaks(x, threshold: float = 0.5, min_distance: int = 2) -> np.ndarray:
    """Strict local maxima above ``threshold``, greedily thinned to ``min_distance``."""
    x = np.asarray(x, dtype=np.float64)
    candidates = [i for i in range(1, len(x) - 1)
                  if x[i] > threshold and x[i] >= x[i - 1] and x[i] > x[i + 1]]
    out: list = []
    for i in candidates:
        if not out or i - out[-1] >= min_distance:
            out.append(i)
        elif x[i] > x[out[-1]]:
            out[-1] = i
    return np.asarray(out, dtype=int)


def peak_count_estimate(h, values=DEFAULT_VALUES, threshold: float = 0.5) -> int:
    """§8 peak-count baseline: DEPLOYABLE, reads only h (as probabilities).

    beats-per-bar ~= (#peaks in the beat channel) / (#peaks in the downbeat channel).
    """
    n_beat = len(pick_peaks(h[:, 0], threshold))
    n_down = len(pick_peaks(h[:, 1], threshold))
    if n_down == 0:
        return max(values)
    estimate = n_beat / n_down
    return min(values, key=lambda m: (abs(m - estimate), m))
