"""Scoring: the downbeat metrics themselves, and the model-facing evaluation loops.

METRICS (was metrics_db.py):Downbeat F-measure in TIME, and the nulls it has to beat.

No beat grid and no offset vocabulary: the model emits a per-frame downbeat probability,
peaks of that curve become downbeat TIMES, and those are matched against the annotated
times with the standard +-70 ms window. Nothing here needs to know how many beats are in
a bar.

EVALUATION: score a trained model per dataset -- both read-outs, continuity metrics,
and the nulls.
"""
from __future__ import annotations

from collections import defaultdict

import mir_eval
import numpy as np
import torch

from ..data.features import FPS

from ..data.dataset import Batches
from ..model import downbeat_frames

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


def rule_g_times(mu, mask, raw, fps: float = FPS):
    """Rule-g downbeat TIMES per crop: wrap frames of ``mu`` -> seconds from t0.

    THE wraps-to-times conversion. It existed in four hand-rolled copies before
    (evaluation, both check scripts, controls), one of which dropped ``delta`` on the
    way to ``infer_phase`` -- the same bug class the emission-D audit caught. Every
    scorer now goes through this one.
    """
    wraps = downbeat_frames(mu, mask).cpu().numpy()
    return [(np.flatnonzero(wraps[i, :len(c["y"]) - 1]) + 1) / fps + c["t0"]
            for i, c in enumerate(raw)]


def evaluate_pooled(model, crops, device, batch_size: int):
    """Pooled (F, CMLt, AMLt, est/ref) over ``crops`` on the deployed path.

    The cheap per-epoch diagnostic read-out (check_warm_start): same conversion and
    the same F authority as ``evaluate``, without the per-dataset/null machinery.
    AMLt is a DIAGNOSTIC, never a target: F low with AMLt high is a metrical mistake,
    both low is noise, and est/ref ~1.0 says the rate is right whatever the phase does.
    """
    model.eval()
    rows = []

    with torch.no_grad():
        for raw, batch in Batches(crops, batch_size, device)():
            mu = model.infer_phase(batch["h"], batch["delta"])
            for est, crop in zip(rule_g_times(mu, batch["mask"], raw), raw):
                ref = np.asarray(crop["downbeat_times"])
                if len(ref) < 2 or len(est) < 2:
                    rows.append((0.0, 0.0, 0.0, len(est) / max(len(ref), 1)))
                    continue
                _cc, _ac, cmlt, amlt = mir_eval.beat.continuity(ref, est)
                rows.append((f_measure(est, ref)[0], cmlt, amlt, len(est) / len(ref)))

    return tuple(np.array(rows).mean(0))


def evaluate(model, crops, device, batch_size: int, seed: int = 0):
    """Per-dataset downbeat metrics for both read-outs, beside the nulls.

    rule-g reads the encoder mean alone; emission-D is the tutorial's Alternative D.
    CMLt/AMLt sit beside F so a metrical mistake (F low, AMLt high) is distinguishable
    from noise (both low); AMLt is never a training target.
    """
    model.eval()
    rows: dict = defaultdict(lambda: defaultdict(list))
    rng = np.random.default_rng(seed)

    with torch.no_grad():
        for raw, batch in Batches(crops, batch_size, device)():
            mu = model.infer_phase(batch["h"], batch["delta"])
            times = rule_g_times(mu, batch["mask"], raw)
            probs = model.emission_probs(batch["h"], batch["delta"]).cpu().numpy()

            for i, crop in enumerate(raw):
                t = len(crop["y"])
                truth = np.asarray(crop["downbeat_times"])
                rule_g = times[i]
                per = rows[crop["dataset"]]
                per["rule-g"].append(f_measure(rule_g, truth)[0])
                if len(rule_g) > 1 and len(truth) > 1:
                    _cc, _ac, cmlt, amlt = mir_eval.beat.continuity(truth, rule_g)
                else:
                    cmlt = amlt = 0.0
                per["rule-g CMLt"].append(cmlt)
                per["rule-g AMLt"].append(amlt)
                per["est/ref"].append(len(rule_g) / max(len(truth), 1))

                alt_d = peak_times(probs[i, :t], FPS, crop["bar_period"]) + crop["t0"]
                per["emission-D"].append(f_measure(alt_d, truth)[0])

                crop["fps"] = FPS
                for kind in ("random", "zero"):
                    per[f"null-{kind}"].append(
                        f_measure(null_times(crop, kind, rng), truth)[0])

    return {ds: {k: (float(np.mean(v)), len(v)) for k, v in per_mode.items()}
            for ds, per_mode in rows.items()}


def print_table(per_seed):
    """The final mean +- sd table over seeds, one row per (split, dataset, mode)."""
    print("\n==== downbeat F (+-70 ms), mean +- sd over seeds ====")
    for key in sorted(per_seed):
        split, dataset, mode = key
        values = [v for v, _ in per_seed[key].values()]
        count = next(iter(per_seed[key].values()))[1]
        print(f"  {split:6s} {mode:15s} {dataset:11s} F {np.mean(values):.3f} "
              f"+-{np.std(values):.3f}  (n={count})")
