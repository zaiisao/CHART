"""Shared fitting machinery: vectorized §5 training, the §8 CV protocol, scoring.

The per-crop ELBO is Appendix-A `Stage0.elbo`; `Batch` is its vectorized equivalent,
VERIFIED against the per-crop method (`verify_vectorized`) — an unverified second
objective is how this project got burned before. Entry points live at the repo root
(train.py, train_real.py); this module is library only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import reference as R  # noqa: E402  (§8 metrics + baselines, spec-side code)

from .stage0 import Stage0  # noqa: E402

MAX_OFFSETS = 4     # widest bar-offset axis: r ranges over 0..m-1, and max(values) = 4


def emission_counts(y, values):
    """(counts [K, MAX_OFFSETS, 4], offset_mask [K, MAX_OFFSETS]): the emission's statistics.

    log p(y|m,r) is linear in v = [lsig(a), lsig(-a), lsig(b), lsig(-b)] with integer
    coefficients: (#downbeat-slot ones, #downbeat-slot zeros, #other ones, #other zeros).
    The single authority for this linearisation — it is objective math, and a second
    hand-copy is an unverified second objective (how this project got burned before).
    """
    y = np.asarray(y, dtype=np.float64)
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
    return counts, offset_mask


def crop_stats(crop, values, reducer):
    """Per-crop constants: emission counts, offset mask, s(h)."""
    counts, offset_mask = emission_counts(crop["y"], values)
    return counts, offset_mask, reducer(crop["h"]).numpy()


def emission_logp_from_counts(counts, offset_mask, log_m, alpha, beta):
    """[..., K] log p_theta(y|m) from precomputed counts. Dtype/device follow the inputs.

    The ONE composition of the linearised emission — every vectorized objective
    (Batch.elbo_mean, the e2e head) must call this, not re-derive it.
    """
    lsig = torch.nn.functional.logsigmoid
    v = torch.stack([lsig(alpha), lsig(-alpha), lsig(beta), lsig(-beta)])
    return torch.logsumexp(counts @ v + offset_mask, dim=-1) - log_m


def elbo_mean_from(emission, prior_logits, c):
    """Scalar mean ELBO from per-crop emission [..., K] and prior logits [..., K] (§4.6).

    The ONE composition of prior/encoder/objective for vectorized training paths.
    """
    prior = torch.log_softmax(prior_logits, dim=-1)
    q_logp = torch.log_softmax(prior + c * emission, dim=-1)
    q = q_logp.exp()
    return (q * (emission - (q_logp - prior))).sum(-1).mean()


class Batch:
    """All crops' constants stacked, so one training step is a few dense ops."""

    def __init__(self, crops, values, reducer):
        stats = [crop_stats(c, values, reducer) for c in crops]
        self.counts = torch.tensor(np.stack([s[0] for s in stats]))       # [N, K, R, 4]
        self.offset_mask = torch.tensor(np.stack([s[1] for s in stats]))  # [N, K, R]
        self.reduced = torch.tensor(np.stack([s[2] for s in stats]))      # [N, s_dim]
        self.log_m = torch.log(torch.tensor([float(m) for m in values]))

    def emission_logp(self, model):
        """[N, K] log p_theta(y|m) for every crop from the precomputed counts."""
        return emission_logp_from_counts(self.counts, self.offset_mask, self.log_m,
                                         model.alpha, model.beta)

    def elbo_mean(self, model):
        """Scalar mean ELBO over the batch — equals mean of Stage0.elbo per crop."""
        prior_logits = self.reduced @ model.W.T + model.b
        return elbo_mean_from(self.emission_logp(model), prior_logits, model.c)


def fit_vectorized(model, crops, steps=500, lr=0.5):
    """§5 training loop (Adam, full batch) on the vectorized objective."""
    batch = Batch(crops, model.values, model.reducer)
    opt = torch.optim.Adam(model.trainable_params(), lr=lr)

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
        model.alpha.fill_(1.3)
        model.beta.fill_(-0.7)
        model.c.fill_(0.8)
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
    """Print one §8 scoring line: balanced/raw accuracy, class count, confusion."""
    true = [c["m_true"] for c in subset]
    balanced = R.balanced_accuracy(true, preds, values)
    raw = float(np.mean(np.asarray(true) == np.asarray(preds)))
    print(f"{tag:12s} n={len(subset):4d}  balanced={balanced:.3f}  raw={raw:.3f}  "
          f"classes={R.distinct_predicted(preds)}  "
          f"confusion={R.confusion(true, preds, values).tolist()}")
    return balanced


def predict_m(model, h):
    """Deployable prediction as a COUNT (C1): argmax of predict(h), converted once."""
    return model.to_value(int(model.predict(h).argmax()))


def cv_out_of_fold(cv, test, fit_fn, predict_fn, verbose=True):
    """The §8 protocol, once.

    Per-fold train-on-complement, pooled out-of-fold predictions, plus a model trained
    on all CV crops for the test-only split.

    Returns (pooled_crops, pooled_preds, test_preds). fit_fn(crops) -> fitted model;
    predict_fn(model, crops) -> list of counts (list-at-a-time so callers can batch).
    """
    pooled_crops, pooled_preds = [], []
    for fold in sorted({c["fold"] for c in cv}):
        train = [c for c in cv if c["fold"] != fold]
        held = [c for c in cv if c["fold"] == fold]
        model = fit_fn(train)
        pooled_preds += predict_fn(model, held)
        pooled_crops += held
        if verbose:
            print(f"fold {fold}: trained on {len(train)}, predicted {len(held)}", flush=True)

    test_preds = []
    if test:
        model = fit_fn(cv)
        test_preds = predict_fn(model, test)
    return pooled_crops, pooled_preds, test_preds


def score_per_dataset(pooled_crops, pooled_preds, values):
    """§8: per-dataset lines (never pooled for meter claims) then the ALL-CV pool."""
    for dataset in sorted({c["dataset"] for c in pooled_crops}):
        sel = [i for i, c in enumerate(pooled_crops) if c["dataset"] == dataset]
        score(dataset, [pooled_crops[i] for i in sel],
              [pooled_preds[i] for i in sel], values)
    score("ALL-CV", pooled_crops, pooled_preds, values)
