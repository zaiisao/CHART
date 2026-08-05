"""Downbeat F-measure in TIME, and the nulls it has to beat.

No beat grid and no offset vocabulary: the model emits a per-frame downbeat probability,
peaks of that curve become downbeat TIMES, and those are matched against the annotated
times with the standard +-70 ms window. Nothing here needs to know how many beats are in
a bar.
"""
from __future__ import annotations

import numpy as np

TOLERANCE_S = 0.070


def peak_times(probs, fps: float, period_s: float, threshold: float = 0.5):
    """Frames whose probability is a local maximum above ``threshold`` -> times (s).

    Peaks are separated by at least half a bar, so one bar contributes one downbeat: the
    emission a + b cos(phi) is broad by construction and would otherwise fire on several
    adjacent frames of the same wrap.
    """
    probs = np.asarray(probs, dtype=np.float64)
    # RELATIVE to the curve's own maximum. An absolute 0.5 was unreachable: the emission
    # a + b cos(phi) tops out near 0.16 at every (a, b) this model has ever learned, so
    # the picker returned [] on every crop and every "emission-D F 0.000" ever reported
    # was a threshold artifact rather than a measurement.
    ceiling = float(probs.max()) if probs.size else 0.0
    if ceiling <= 0.0:
        return np.zeros(0, dtype=np.float64)
    probs = probs / ceiling
    min_gap = max(1, int(round(0.5 * period_s * fps)))
    order = np.argsort(-probs)
    taken: list[int] = []
    for i in order:
        if probs[i] < threshold:
            break
        if all(abs(i - j) >= min_gap for j in taken):
            taken.append(int(i))
    return np.sort(np.asarray(taken, dtype=np.float64)) / fps


def f_measure(predicted, annotated, tolerance: float = TOLERANCE_S):
    """(f, precision, recall) with greedy one-to-one matching inside ``tolerance``."""
    predicted = np.asarray(predicted, dtype=np.float64)
    annotated = np.asarray(annotated, dtype=np.float64)
    if len(annotated) == 0:
        return (1.0, 1.0, 1.0) if len(predicted) == 0 else (0.0, 0.0, 1.0)
    if len(predicted) == 0:
        return 0.0, 1.0, 0.0

    used = np.zeros(len(annotated), dtype=bool)
    hits = 0
    for t in predicted:
        gap = np.abs(annotated - t)
        gap[used] = np.inf
        j = int(np.argmin(gap))
        if gap[j] <= tolerance:
            used[j] = True
            hits += 1
    precision = hits / len(predicted)
    recall = hits / len(annotated)
    f = 0.0 if hits == 0 else 2 * precision * recall / (precision + recall)
    return f, precision, recall


def null_times(crop, kind: str, rng):
    """A baseline downbeat sequence with the right RATE but no learned phase.

    ``kind="random"`` starts the grid at a uniformly random phase; ``kind="zero"`` starts
    it at the crop boundary. Both know the bar period -- which the model is also given --
    so beating them is a statement about PHASE alone, not about tempo.
    """
    period = crop["bar_period"]
    span = len(crop["y"]) / crop["fps"] if "fps" in crop else None
    duration = span if span is not None else (crop["downbeat_times"][-1] - crop["t0"])
    offset = rng.uniform(0.0, period) if kind == "random" else 0.0
    return crop["t0"] + offset + np.arange(0.0, max(duration, 0.0), period)
