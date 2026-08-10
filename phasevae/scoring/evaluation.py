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

from ..data.excerpts import collate_excerpts
from ..model import TWO_PI, downbeat_frames

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


def trajectory_period(mu, mask, fps):
    """[B, T] phase -> [B] bar period in seconds, read off the model's OWN trajectory.

    The mean per-frame advance IS the model's inferred rate, so 2*pi/(rate*fps) is its
    inferred bar length. Nothing annotation-derived and no external estimator: since the
    rate became the model's own inference (there is no ``delta`` any more), this is the
    only bar period in the pipeline, and it is what the nulls and the peak picker need.

    Increments are wrapped to (-pi, pi] because ``mu`` is emitted per frame through
    atan2: at a bar boundary the raw difference jumps by ~2*pi and would otherwise
    cancel the advance it should be counting.
    """
    inc = mu[:, 1:] - mu[:, :-1]
    inc = torch.atan2(torch.sin(inc), torch.cos(inc))
    weight = mask[:, 1:] * mask[:, :-1]
    rate = (inc * weight).sum(1) / weight.sum(1).clamp(min=1.0)
    # a non-advancing (or backward) trajectory has no period; fall back to the window
    # length so the grid degenerates to a single time rather than dividing by zero
    span = mask.sum(1).clamp(min=1.0) / fps
    period = torch.where(rate > 1e-6, TWO_PI / (rate.clamp(min=1e-6) * fps), span)
    return period.cpu().numpy()


def trajectory_health(mu, kappa, mask, crops):
    """(advance, kappa, phase_err, coverage): what F cannot see about the trajectory.

    F(+-70 ms) only asks whether wrap TIMES land near downbeats, so it cannot distinguish
    a learned bar pointer from an arbitrary path that happens to cross zero in the right
    places. These four numbers can, and each has a recorded failure to compare against:

      advance   mean wrapped per-frame increment, rad/frame. Real bars of 2-5 s at 50 fps
                give 0.025-0.063. The spike train measured 0.715 (10x too fast, phase 0
                on downbeats and arbitrary between); a KL-dominated collapse gives ~0.
      kappa     mean posterior concentration. Starts at kappa_physical (2000); the
                faithful/ collapse drove it to 0.022 (A(kappa) 0.011, i.e. uniform),
                which buys cheap KL by being uncertain everywhere.
      phase_err mean circular distance |mu - true_phase| over annotated frames, rad.
                Chance is pi/2 = 1.571; the chimera run sat at 1.44 -- i.e. no better
                than chance -- while its ADVANCE looked correct at 0.069 vs 0.065.
      coverage  fraction of the circle mu actually visits (16 bins). A trajectory pinned
                near one phase reports ~0.06 even when F looks alive.

    ``crops`` are scoring_records; items whose annotated span does not cover the window
    contribute only their valid frames.
    """
    from ..data.dataset import true_phase

    inc = mu[:, 1:] - mu[:, :-1]
    inc = torch.atan2(torch.sin(inc), torch.cos(inc))
    w = mask[:, 1:] * mask[:, :-1]
    advance = float((inc * w).sum() / w.sum().clamp(min=1.0))
    kappa_mean = float((kappa * mask).sum() / mask.sum().clamp(min=1.0))

    errs, visited = [], np.zeros(16, dtype=bool)
    for i, crop in enumerate(crops):
        t = len(crop["y"])
        phi_true, ok = true_phase(crop)
        if not ok.any():
            continue
        m = mu[i, :t].cpu().numpy()[ok]
        diff = np.angle(np.exp(1j * (m - phi_true[ok])))
        errs.append(np.abs(diff))
        visited |= np.bincount(((m % TWO_PI) / TWO_PI * 16).astype(int) % 16,
                               minlength=16).astype(bool)

    phase_err = float(np.mean(np.concatenate(errs))) if errs else float("nan")
    return advance, kappa_mean, phase_err, float(visited.mean())


def null_times(crop, kind: str, rng):
    """A baseline downbeat sequence with the right RATE but no learned phase.

    ``kind="random"`` starts the grid at a uniformly random phase; ``kind="zero"`` starts
    it at the crop boundary. Both get the model's OWN inferred bar period (see
    trajectory_period), so beating them is a statement about PHASE alone, not about
    tempo -- the null is handed the same rate the model found.
    """
    period = crop["bar_period"]
    span = len(crop["y"]) / crop["fps"] if "fps" in crop else None
    duration = span if span is not None else (crop["downbeat_times"][-1] - crop["t0"])
    offset = rng.uniform(0.0, period) if kind == "random" else 0.0
    return crop["t0"] + offset + np.arange(0.0, max(duration, 0.0), period)


def rule_g_times(mu, mask, raw):
    """Rule-g downbeat TIMES per crop: wrap frames of ``mu`` -> seconds from t0.

    THE wraps-to-times conversion. It existed in four hand-rolled copies before
    (evaluation, both check scripts, controls), one of which dropped ``delta`` on the
    way to ``infer_phase`` -- the same bug class the emission-D audit caught. Every
    scorer now goes through this one. The frame grid comes from each crop's ``fps``
    key -- whatever the frontend that made it ticks at.
    """
    wraps = downbeat_frames(mu, mask).cpu().numpy()
    return [(np.flatnonzero(wraps[i, :len(c["y"]) - 1]) + 1) / c["fps"] + c["t0"]
            for i, c in enumerate(raw)]


def scoring_records(raw) -> list:
    """Collated excerpt batch -> per-item scoring records, trimmed to valid frames.

    THE excerpt-to-scorer bridge: the record vocabulary (y/fps/t0/downbeat_times/...) is
    what every metric helper and control consumes. A fully-masked backstop item
    (nothing scorable) yields None so row alignment with the batch tensors survives.

    ``bar_period`` is NOT here: it no longer exists in the dataset, and the only rate in
    the pipeline is the model's own. ``evaluate`` fills the key in from
    trajectory_period once mu is available, for the nulls and the peak picker.
    """
    records = []
    for i in range(len(raw["y"])):
        valid = int(raw["mask"][i].sum())
        if valid == 0:
            records.append(None)
            continue
        records.append({"y": raw["y"][i, :valid].numpy(),
                        "fps": float(raw["fps"][i]), "t0": float(raw["t0"][i]),
                        "downbeat_times": np.asarray(raw["downbeat_times"][i]),
                        "anchors": np.asarray(raw["anchors"][i]),
                        "dataset": raw["dataset"][i], "song_id": raw["song_id"][i]})
    return records


def evaluate(model, dataset, frontend, device, batch_size: int, seed: int = 0):
    """Per-dataset downbeat metrics for both read-outs, beside the nulls.

    Consumes the deterministic excerpt dataset directly: the frozen frontend turns
    each batch's windows into features inline (the same call training uses), and each
    item is scored on its valid frames. rule-g reads the encoder mean alone;
    emission-D is the tutorial's Alternative D. CMLt/AMLt sit beside F so a metrical
    mistake (F low, AMLt high) is distinguishable from noise (both low); AMLt is
    never a training target.
    """
    assert dataset.deterministic, "evaluation scores FIXED windows"
    model.eval()
    rows: dict = defaultdict(lambda: defaultdict(list))
    rng = np.random.default_rng(seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         collate_fn=collate_excerpts)

    with torch.no_grad():
        for raw in loader:
            records = scoring_records(raw)
            keep = [i for i, c in enumerate(records) if c is not None]
            if not keep:
                continue
            crops = [records[i] for i in keep]
            # the same frontend call training makes; features never touch disk
            h = frontend.forward_features(raw["input"]).clone()
            mask = raw["mask"].to(device, non_blocking=True)

            mu = model.infer_phase(h, mask)[keep]
            times = rule_g_times(mu, mask[keep], crops)
            probs = model.emission_probs(h, mask)[keep].cpu().numpy()

            # the peak picker and the nulls need a bar period; take the model's OWN
            # inferred rate, the only period left in the pipeline now that delta is gone
            period = trajectory_period(mu, mask[keep], float(raw["fps"][0]))
            for i, crop in enumerate(crops):
                crop["bar_period"] = float(period[i])

            for i, crop in enumerate(crops):
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

                alt_d = peak_times(probs[i, :t], crop["fps"], crop["bar_period"]) + crop["t0"]
                per["emission-D"].append(f_measure(alt_d, truth)[0])

                for kind in ("random", "zero"):
                    per[f"null-{kind}"].append(
                        f_measure(null_times(crop, kind, rng), truth)[0])

    return {ds: {k: (float(np.mean(v)), len(v)) for k, v in per_mode.items()}
            for ds, per_mode in rows.items()}


def print_table(results):
    """One row per (split, dataset, mode). One run = one seed; sweeps aggregate outside."""
    print("\n==== downbeat F (+-70 ms) ====")
    rows = sorted((split, dataset, mode, value, count)
                  for split, per_dataset in results.items()
                  for dataset, modes in per_dataset.items()
                  for mode, (value, count) in modes.items())
    for split, dataset, mode, value, count in rows:
        print(f"  {split:6s} {mode:15s} {dataset:11s} F {value:.3f}  (n={count})")
