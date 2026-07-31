"""Combined evidence: trained rich-feature head ⊕ hand-built peak summary, one ψ.

The 2026-07-31 verdict: the e2e head wins in-distribution (asap 0.501, ALL-CV 0.595) but
fails the gtzan transfer (0.301 < chance); the peaks reducer transfers (gtzan 0.624) but
is blind on asap (0.350). The two evidence streams are complementary — this experiment
asks whether one deployable path can keep both, with the gtzan transfer as the acid test.

Four fold-honest arms on identical crops, one frontend pass ("features+activations"):
    linear    trained linear psi on the 10-dim peak summary   (reproduces compressed-peaks)
    head      pure AutocorrHead on [T, 512]                   (reproduces the e2e run)
    concat    AutocorrHead with the standardized peak summary appended to its MLP input
    ensemble  mean of linear's and head's log-softmax priors  (product of experts)

Run: CUDA_VISIBLE_DEVICES=1 /disk4/anaconda3/envs/vbpm/bin/python \
         experiments/stage0_combined_head.py
"""
import sys

import numpy as np
import torch

from vbpm.data import VALUES, load_crops
from vbpm.fitting import (elbo_mean_from, emission_counts, emission_logp_from_counts,
                          score, score_per_dataset)
from vbpm.heads import AutocorrHead
from vbpm.metrics import balanced_accuracy  # noqa: F401  (handy in interactive use)
from vbpm.reducers import peak_summary

K = len(VALUES)
EPOCHS = 30
BATCH = 256
SEED = 0


def make_entry(song, crop, h_crop, t0):
    """Features fp16 + peak summary of the activation channels + emission stats."""
    features, acts = h_crop[:, :-2], h_crop[:, -2:]
    counts, mask = emission_counts(crop["y"], VALUES)
    return {"h16": features.astype(np.float16),
            "s10": peak_summary(acts).numpy().astype(np.float32),
            "C": counts.astype(np.float32), "mask": mask.astype(np.float32),
            "m_true": crop["m_true"], "dataset": song.dataset, "fold": song.fold}


# ---- batching (length-bucketed, fp16 on the wire) -------------------------------------
def bucket_chunks(crops, batch_size):
    order = sorted(range(len(crops)), key=lambda i: len(crops[i]["h16"]))
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


def pad_features(crops, idx, device):
    Ts = [len(crops[i]["h16"]) for i in idx]
    x16 = torch.zeros(len(idx), max(Ts), 512, dtype=torch.float16)
    for j, i in enumerate(idx):
        x16[j, :Ts[j]] = torch.from_numpy(crops[i]["h16"])
    return x16.to(device).float(), torch.tensor(Ts).to(device)


def stack_field(crops, idx, field, device):
    return torch.from_numpy(np.stack([crops[i][field] for i in idx])).to(device)


# ---- training -------------------------------------------------------------------------
def train_arm(train_crops, device, extra_stats=None, use_features=True):
    """One fold-honest arm: joint ELBO ascent on theta + psi(head or linear) + phi.

    extra_stats = (mean, std) standardizer for s10 when the arm consumes it.
    use_features=False trains the LINEAR arm: psi logits = W @ z(s10) + b.
    """
    torch.manual_seed(SEED)
    extra_dim = 10 if extra_stats is not None else 0
    if use_features:
        head = AutocorrHead(extra_dim=extra_dim).to(device)
        psi_params = list(head.parameters())
    else:
        head = torch.nn.Linear(10, K).to(device)   # linear psi on z(s10)
        psi_params = list(head.parameters())
    scalars = {"alpha": torch.tensor(0.5, device=device, requires_grad=True),
               "beta": torch.tensor(-0.5, device=device, requires_grad=True),
               "c": torch.tensor(1.0, device=device, requires_grad=True)}
    opt = torch.optim.Adam([{"params": psi_params, "lr": 3e-3},
                            {"params": list(scalars.values()), "lr": 0.1}])
    log_m = torch.log(torch.tensor([float(m) for m in VALUES], device=device))
    chunks = bucket_chunks(train_crops, BATCH)
    rng = np.random.default_rng(SEED)

    for epoch in range(EPOCHS):
        total, n_batches = 0.0, 0
        rng.shuffle(chunks)
        for idx in chunks:
            logits = arm_logits(head, train_crops, idx, device, extra_stats, use_features)
            Cs = stack_field(train_crops, idx, "C", device)
            masks = stack_field(train_crops, idx, "mask", device)
            em = emission_logp_from_counts(Cs, masks, log_m,
                                           scalars["alpha"], scalars["beta"])
            loss = -elbo_mean_from(em, logits, scalars["c"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, n_batches = total - float(loss), n_batches + 1
        if epoch % 10 == 9 or epoch == 0:
            print(f"    epoch {epoch + 1}: mean ELBO {total / n_batches:+.4f}", flush=True)

    dead = [n for n, p in (head.named_parameters() if use_features else
                           [("w", head.weight), ("b", head.bias)]) if p.grad is None]
    assert not dead, f"psi parameters with no gradient: {dead}"
    return head


def arm_logits(head, crops, idx, device, extra_stats, use_features):
    """[B, K] prior logits for one arm on one index batch."""
    z10 = None
    if extra_stats is not None:
        mean, std = extra_stats
        z10 = (stack_field(crops, idx, "s10", device) - mean) / std
    if not use_features:
        return head(z10)
    x, lengths = pad_features(crops, idx, device)
    return head(x, lengths, extra=z10) if head.extra_dim else head(x, lengths)


@torch.no_grad()
def predict_logits(head, crops, device, extra_stats=None, use_features=True):
    out = []
    for i in range(0, len(crops), BATCH):
        idx = list(range(i, min(i + BATCH, len(crops))))
        out.append(arm_logits(head, crops, idx, device, extra_stats, use_features).cpu())
    return torch.cat(out)                                             # [N, K]


def standardizer(train_crops, device):
    s = np.stack([c["s10"] for c in train_crops])
    return (torch.tensor(s.mean(0), dtype=torch.float32, device=device),
            torch.tensor(s.std(0) + 1e-8, dtype=torch.float32, device=device))


def to_preds(logits):
    return [VALUES[int(k)] for k in logits.argmax(-1)]


def main():
    smoke = "--smoke" in sys.argv
    device = "cuda"
    crops, report = load_crops(limit_per_fold=6 if smoke else None, device=device,
                               output="features+activations", make_entry=make_entry)
    print(f"crops: {len(crops)}  rejects: {report['rejects']}")

    cv = [c for c in crops if c["fold"] is not None]
    test = [c for c in crops if c["fold"] is None]
    arms = ("linear", "head", "concat", "ensemble")
    pooled = []
    preds = {arm: [] for arm in arms}
    test_preds = {}

    def run_split(train, held):
        """Train the three trainable arms on ``train``, return per-arm preds on ``held``."""
        stats = standardizer(train, device)
        linear = train_arm(train, device, extra_stats=stats, use_features=False)
        head = train_arm(train, device)
        concat = train_arm(train, device, extra_stats=stats)
        lin_lp = torch.log_softmax(predict_logits(linear, held, device, stats, False), -1)
        head_lp = torch.log_softmax(predict_logits(head, held, device), -1)
        return {"linear": to_preds(lin_lp), "head": to_preds(head_lp),
                "concat": to_preds(predict_logits(concat, held, device, stats)),
                "ensemble": to_preds(0.5 * (lin_lp + head_lp))}

    for fold in sorted({c["fold"] for c in cv}):
        train = [c for c in cv if c["fold"] != fold]
        held = [c for c in cv if c["fold"] == fold]
        print(f"fold {fold}: {len(train)} train / {len(held)} held", flush=True)
        fold_preds = run_split(train, held)
        for arm in arms:
            preds[arm] += fold_preds[arm]
        pooled += held

    if test:
        print("gtzan arm: training on all CV crops", flush=True)
        test_preds = run_split(cv, test)

    for arm in arms:
        print(f"\n######## arm: {arm} ########")
        score_per_dataset(pooled, preds[arm], VALUES)
        if test:
            score("gtzan", test, test_preds[arm], VALUES)


if __name__ == "__main__":
    main()
