"""Score a trained model per dataset: both read-outs, continuity metrics, and the nulls."""
from __future__ import annotations

from collections import defaultdict

import mir_eval
import numpy as np
import torch

from vbpm.data import FPS

from .loading import Batches
from .metrics_db import f_measure, null_times, peak_times
from .model import downbeat_frames


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
            wraps = downbeat_frames(mu, batch["mask"]).cpu().numpy()
            probs = model.emission_probs(batch["h"]).cpu().numpy()
            for i, crop in enumerate(raw):
                t = len(crop["y"])
                truth = np.asarray(crop["downbeat_times"])
                rule_g = (np.flatnonzero(wraps[i, :t - 1]) + 1) / FPS + crop["t0"]
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
