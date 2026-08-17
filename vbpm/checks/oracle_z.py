"""The pipeline with z SUPPLIED rather than inferred.

Everything downstream of the encoder runs unchanged -- the shipped emission, the
shipped peak picker, the shipped metrics -- on the true bar phase read off the
annotations. What it measures is the ceiling the encoder is failing to reach, and
whether any of the machinery below the encoder is itself broken. It is an ORACLE:
z comes from y, so no number here is label-free.
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict

import numpy as np
import torch

from ..config import load_config
from ..data.dataset import split_songs
from ..data.excerpts import ExcerptDataset, collate_excerpts
from ..model import VBPM
from ..vonmises import sample_vonmises
from ..specs import EmissionSpec
from ..scoring.evaluation import (continuity_scores, f_measure, null_times, peak_times,
                                  rule_g_times, scoring_records, trajectory_period)


def oracle_phase(crop):
    """Unwrapped true bar phase: 2 pi k at the k-th annotated downbeat.

    Built from the annotations, so any number derived from it is an oracle bound.
    """
    n = len(crop["y"])
    times = crop["t0"] + np.arange(n) / crop["fps"]
    db = np.asarray(crop["anchors"], dtype=np.float64)
    if len(db) < 2:
        return None
    turns = 2.0 * np.pi * np.arange(len(db))
    left = turns[0] + (times[0] - db[0]) * 2 * np.pi / (db[1] - db[0])
    right = turns[-1] + (times[-1] - db[-1]) * 2 * np.pi / (db[-1] - db[-2])
    return np.interp(times, db, turns, left=left, right=right).astype(np.float32)


def crops_of(dataset, batch_size: int):
    """Deterministic scoring records with the oracle phase path attached."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         collate_fn=collate_excerpts)
    for raw in loader:
        records = scoring_records(raw)
        for i, crop in enumerate(records):
            if crop is None:
                continue
            phase = oracle_phase(crop)
            if phase is None:
                continue
            crop["phi"] = phase
            yield crop


def fit_emission(model, crops, steps: int, lr: float):
    """The emission's own parameters, trained on oracle z under the shipped recon."""
    # crops are not all the same length: songs shorter than the window keep only their
    # valid frames, so they batch by length rather than by stacking.
    buckets: dict = {}
    for c in crops:
        buckets.setdefault(len(c["y"]), []).append(c)
    batches = [(torch.tensor(np.stack([c["phi"] for c in group])),
                torch.tensor(np.stack([c["y"] for c in group])))
               for group in buckets.values()]
    opt = torch.optim.Adam([model.emission_a, model.emission_b_raw], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        total = sum(-model.recon_term(model.emission_logits(phi), y,
                                      torch.ones_like(y), 1.0).sum()
                    for phi, y in batches) / len(crops)
        total.backward()
        opt.step()
    return float(-total.item())


def score(model, crops, seed: int = 0, kappa: float | None = None):
    """kappa: draw z from vM(true phase, kappa) instead of handing over the exact path.

    This is the posterior q would actually deliver at that concentration, so it asks how
    much jitter each emission tolerates before the read-out breaks.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    rows: dict = defaultdict(lambda: defaultdict(list))
    with torch.no_grad():
        for crop in crops:
            mu = torch.tensor(crop["phi"])[None]
            if kappa is not None:
                mu = mu + sample_vonmises(torch.full_like(mu, float(kappa)))
            mask = torch.ones_like(mu)
            probs = torch.sigmoid(model.emission_logits(mu))[0].numpy()
            crop["bar_period"] = float(trajectory_period(mu, mask, crop["fps"])[0])
            truth = np.asarray(crop["downbeat_times"])
            per = rows[crop["dataset"]]

            rule_g = rule_g_times(mu, mask, [crop])[0]
            per["rule-g"].append(f_measure(rule_g, truth)[0])
            cmlt, amlt = continuity_scores(truth, rule_g)
            per["rule-g CMLt"].append(cmlt)
            per["rule-g AMLt"].append(amlt)

            alt_d = peak_times(probs, crop["fps"], crop["bar_period"]) + crop["t0"]
            per["emission-D"].append(f_measure(alt_d, truth)[0])
            cmlt, amlt = continuity_scores(truth, alt_d)
            per["emission-D CMLt"].append(cmlt)
            per["emission-D AMLt"].append(amlt)

            for kind in ("random", "zero"):
                per[f"null-{kind}"].append(
                    f_measure(null_times(crop, kind, rng), truth)[0])
    return {ds: {k: (float(np.mean(v)), float(np.std(v) / math.sqrt(len(v))), len(v))
                 for k, v in per_mode.items()} for ds, per_mode in rows.items()}


class _Stub:
    """The dataset needs a frontend's NAME and FPS to find the cache. Oracle z never

    reads x, so the network itself is never built and no audio is decoded.
    """
    def __init__(self, name: str, fps: float):
        self.name, self.FPS = name, fps


def main():
    """Score the pipeline with z supplied instead of inferred."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="vbpm/configs/baseline.yaml")
    p.add_argument("--emission", default=None)
    p.add_argument("--recon", default=None)
    p.add_argument("--tol", type=int, default=None)
    p.add_argument("--val-fold", type=int, default=0)
    p.add_argument("--limit-per-fold", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--jitter-kappa", type=float, nargs="*", default=None,
                   help="sweep: score at vM(true, kappa) instead of the exact path")
    args = p.parse_args()

    loaded = load_config(args.config)
    cfg = loaded[0] if isinstance(loaded, tuple) else loaded
    kind = args.emission or cfg.emission
    recon = args.recon or getattr(cfg, "emission_recon", "event")
    tol = args.tol if args.tol is not None else getattr(cfg, "target_tol_frames", 0)

    frontend = _Stub(cfg.frontend_name if hasattr(cfg, "frontend_name") else "beat_this",
                     50.0)
    train_songs, val_songs, test_songs = split_songs(args.val_fold, args.limit_per_fold)

    def make(songs, det):
        return ExcerptDataset(songs, frontend, cfg.excerpt_seconds,
                              deterministic=det, target_tol_frames=tol)
    train_crops = list(crops_of(make(train_songs, True), args.batch_size))
    test_crops = list(crops_of(make(test_songs, True), args.batch_size))

    print(f"emission={kind} recon={recon} target_tol_frames={tol}")
    print(f"{len(train_crops)} train crops, {len(test_crops)} test crops")

    model = VBPM(1, emission=EmissionSpec(kind=kind, recon=recon,
                                          bump_kappa=cfg.emission_bump_kappa))
    model.eval()
    fitted = fit_emission(model, train_crops, args.steps, args.lr)
    print(f"trained emission: a={model.emission_a.item():+.3f} "
          f"b={model.emission_b.item():.3f} "
          f"recon/crop={fitted:.2f}")

    if not args.jitter_kappa:
        for dataset, modes in sorted(score(model, test_crops).items()):
            print(f"\n-- {dataset} --")
            for mode, (mean, se, n) in sorted(modes.items()):
                print(f"   {mode:20s} {mean:.3f} +- {se:.3f}   (n={n})")
        return

    period = float(np.mean([c["bar_period"] for c in test_crops
                            if "bar_period" in c])) or 2.28
    print(f"\n{'kappa':>10} {'sd_ms':>7} {'emission-D':>12} {'CMLt':>8} {'AMLt':>8} "
          f"{'rule-g':>9}")
    for kappa in [None] + list(args.jitter_kappa):
        modes = score(model, test_crops, kappa=kappa)["gtzan"]
        sd = "exact" if kappa is None else f"{1000 * (1 / math.sqrt(kappa)) / (2 * math.pi) * 2.28:7.1f}"
        label = "exact" if kappa is None else f"{kappa:10.0f}"
        print(f"{label:>10} {sd:>7} {modes['emission-D'][0]:12.3f} "
              f"{modes['emission-D CMLt'][0]:8.3f} {modes['emission-D AMLt'][0]:8.3f} "
              f"{modes['rule-g'][0]:9.3f}")


if __name__ == "__main__":
    main()
