"""Fold-honest Stage-0 training + §8 evaluation on the real corpora (§6.3).

Protocol (§8): 8-fold CV over the CV-eligible datasets, pooled out-of-fold predictions,
balanced accuracy computed ONCE over the pool, reported per dataset (never pooled across
datasets for meter claims). gtzan is test-only: scored with a model trained on all CV crops.

The per-crop ELBO is Appendix-A `Stage0.elbo`; training uses a vectorized equivalent
(precomputed per-crop emission sufficient statistics + reduced s(h)) that is VERIFIED
against the per-crop method before any fit — an unverified second objective is how this
project got burned before.

Usage:
    python -m vbpm.train_real [--datasets asap rwc ...] [--device cuda:1] [--smoke]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "v2"))
import reference as R  # noqa: E402  (§8 metrics + baselines, spec-side code)

from .data import load_crops  # noqa: E402
from .reducers import REDUCERS  # noqa: E402
from .stage0 import Stage0  # noqa: E402

MAX_OFFSETS = 4     # widest bar-offset axis: r ranges over 0..m-1, and max(values) = 4


def crop_stats(crop, values, reducer):
    """Per-crop constants: emission counts [K, MAX_OFFSETS, 4], offset mask, s(h).

    log p(y|m,r) is linear in v = [lsig(a), lsig(-a), lsig(b), lsig(-b)] with integer
    coefficients: (#downbeat-slot ones, #downbeat-slot zeros, #other ones, #other zeros).
    """
    y = np.asarray(crop["y"], dtype=np.float64)
    n, total_ones = len(y), float(y.sum())
    counts = np.zeros((len(values), MAX_OFFSETS, 4))
    offset_mask = np.full((len(values), MAX_OFFSETS), -np.inf)   # additive: -inf = no such r
    for k, m in enumerate(values):
        for r in range(m):
            slots = np.arange(r, n, m)
            on_ones = float(y[slots].sum())
            counts[k, r] = (on_ones, len(slots) - on_ones,
                            total_ones - on_ones, (n - len(slots)) - (total_ones - on_ones))
            offset_mask[k, r] = 0.0
    return counts, offset_mask, reducer(crop["h"]).numpy()


class Batch:
    """All crops' constants stacked, so one training step is a few dense ops."""

    def __init__(self, crops, values, reducer):
        stats = [crop_stats(c, values, reducer) for c in crops]
        self.counts = torch.tensor(np.stack([s[0] for s in stats]))       # [N, K, R, 4]
        self.offset_mask = torch.tensor(np.stack([s[1] for s in stats]))  # [N, K, R]
        self.reduced = torch.tensor(np.stack([s[2] for s in stats]))      # [N, s_dim]
        self.log_m = torch.log(torch.tensor([float(m) for m in values]))

    def emission_logp(self, model):
        lsig = torch.nn.functional.logsigmoid
        v = torch.stack([lsig(model.alpha), lsig(-model.alpha),
                         lsig(model.beta), lsig(-model.beta)])
        return (torch.logsumexp(self.counts @ v + self.offset_mask, dim=-1)
                - self.log_m)                                             # [N, K]

    def elbo_mean(self, model):
        emission = self.emission_logp(model)
        prior = torch.log_softmax(self.reduced @ model.W.T + model.b, dim=-1)
        q_logp = torch.log_softmax(prior + model.c * emission, dim=-1)
        q = q_logp.exp()
        return (q * (emission - (q_logp - prior))).sum(-1).mean()


def fit_vectorized(model, crops, steps=500, lr=0.5):
    batch = Batch(crops, model.values, model.reducer)

    seen, params = set(), []
    for p in model.named_params().values():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            params.append(p)
    opt = torch.optim.Adam(params, lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        (-batch.elbo_mean(model)).backward()
        opt.step()

    return model


def verify_vectorized(crops, values, reducer, s_dim):
    """The vectorized objective must equal the Appendix-A per-crop ELBO exactly."""
    probe = crops[: min(8, len(crops))]
    model = Stage0(values, reducer=reducer, s_dim=s_dim)
    with torch.no_grad():   # non-degenerate params so agreement is not vacuous
        model.alpha.fill_(1.3), model.beta.fill_(-0.7), model.c.fill_(0.8)
        model.W.copy_(torch.randn_like(model.W) * 0.5)
        model.b.copy_(torch.randn_like(model.b) * 0.5)
    vectorized = float(Batch(probe, values, reducer).elbo_mean(model))
    per_crop = float(np.mean([float(model.elbo(c["h"], c["y"])) for c in probe]))
    # relative tolerance: the two accumulate the same sum in different orders
    assert abs(vectorized - per_crop) < 1e-9 * max(1.0, abs(per_crop)), \
        f"vectorized ELBO {vectorized!r} != per-crop ELBO {per_crop!r}"
    print(f"vectorized ELBO verified against Stage0.elbo on {len(probe)} crops "
          f"(|diff| = {abs(vectorized - per_crop):.2e})")


def score(tag, subset, preds, values):
    true = [c["m_true"] for c in subset]
    balanced = R.balanced_accuracy(true, preds, values)
    raw = float(np.mean(np.asarray(true) == np.asarray(preds)))
    print(f"{tag:12s} n={len(subset):4d}  balanced={balanced:.3f}  raw={raw:.3f}  "
          f"classes={R.distinct_predicted(preds)}  "
          f"confusion={R.confusion(true, preds, values).tolist()}")
    return balanced


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--smoke", action="store_true", help="6 songs per fold, quick wiring check")
    ap.add_argument("--reducers", nargs="*", default=["meanmax", "peaks"],
                    choices=sorted(REDUCERS), help="§4.4 reducer variants, one CV run each")
    args = ap.parse_args(argv)

    values = (2, 3, 4)
    print("loading crops (live fold-honest frontend pass)...", flush=True)
    crops, report = load_crops(datasets=args.datasets, device=args.device,
                               limit_per_fold=6 if args.smoke else None)
    print(f"usable: {report['usable']}  rejects: {report['rejects']}  "
          f"unmatched downbeats: {report['unmatched_downbeats']}")
    per_dataset: dict = {}
    for (dataset, m), n in sorted(report["per_dataset"].items()):
        per_dataset.setdefault(dataset, {})[m] = n
    for dataset, class_counts in per_dataset.items():
        print(f"  {dataset:12s} " + "  ".join(f"m={m}:{class_counts.get(m, 0)}"
                                              for m in values))

    cv = [c for c in crops if c["fold"] is not None]
    test = [c for c in crops if c["fold"] is None]

    for name in args.reducers:
        reducer, s_dim = REDUCERS[name]
        print(f"\n######## reducer: {name} (s_dim={s_dim}) ########")
        verify_vectorized(crops, values, reducer, s_dim)

        # -- 8-fold CV, pooled out-of-fold predictions (§8) ------------------------------
        pooled_crops, pooled_preds = [], []
        for fold in sorted({c["fold"] for c in cv}):
            train = [c for c in cv if c["fold"] != fold]
            held = [c for c in cv if c["fold"] == fold]
            model = fit_vectorized(Stage0(values, reducer=reducer, s_dim=s_dim),
                                   train, steps=args.steps, lr=args.lr)
            pooled_preds += [model.to_value(int(model.predict(c["h"]).argmax()))
                             for c in held]
            pooled_crops += held
            print(f"fold {fold}: trained on {len(train)}, predicted {len(held)}", flush=True)

        print("\n== pooled out-of-fold (CV datasets), per dataset ==")
        for dataset in sorted({c["dataset"] for c in pooled_crops}):
            sel = [i for i, c in enumerate(pooled_crops) if c["dataset"] == dataset]
            score(dataset, [pooled_crops[i] for i in sel],
                  [pooled_preds[i] for i in sel], values)
        score("ALL-CV", pooled_crops, pooled_preds, values)

        # -- gtzan: test-only, model trained on all CV crops ----------------------------
        if test:
            model = fit_vectorized(Stage0(values, reducer=reducer, s_dim=s_dim),
                                   cv, steps=args.steps, lr=args.lr)
            preds = [model.to_value(int(model.predict(c["h"]).argmax())) for c in test]
            print("\n== test-only (gtzan), model trained on all CV crops ==")
            score("gtzan", test, preds, values)

    # -- baselines (§8), reducer-independent ---------------------------------------------
    print("\n######## baselines (held-out, deployable) ########")
    majority = R.majority_predict([c["m_true"] for c in cv], values)
    score("majority", cv, [majority] * len(cv), values)
    # peak-count reads sigmoid(h): the frontend emits LOGITS, the baseline expects activations
    peak_preds = [R.peak_count_estimate(1 / (1 + np.exp(-c["h"])), values) for c in cv]
    score("peak-count", cv, peak_preds, values)
    if test:
        peak_test = [R.peak_count_estimate(1 / (1 + np.exp(-c["h"])), values) for c in test]
        score("peak-gtzan", test, peak_test, values)


if __name__ == "__main__":
    main()
