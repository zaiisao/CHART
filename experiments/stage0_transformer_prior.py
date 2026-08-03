"""Transformer prior over h: attention across t = 1..T replaces the pooled reducer.

The tutorial's conditional-prior transformer (section 9.3), specialized to Stage 0's one
categorical latent, against the incumbents on identical crops. Four fold-honest arms:

    linear     trained linear psi on the 10-dim peak summary  (deployed reference)
    autocorr   AutocorrHead on [T, 512]                       (structured trained head)
    tf512      TransformerPrior on the rich [T, 512] features
    tf2        TransformerPrior on the compressed [T, 2] activations — isolates the
               cross-checkpoint feature-misalignment confound: the 2 channels are
               semantically pinned across checkpoints, the 512 are not
    tf512n     tf512 with per-crop feature normalisation (each channel centred and scaled
               over the crop's own frames) — the second half of the same confound test:
               it grants the transformer the protection the autocorr head gets for free,
               so a constant cross-checkpoint feature shift can no longer survive as a
               constant logit offset

Pre-registered criteria: PRIMARY = gtzan transfer (beat linear's 0.624 to claim the
reducer's crown; ALL-CV alone does not count). Secondary = per-dataset, especially asap
vs the autocorr head's 0.501. Everything else (emission, encoder, exact ELBO, folds,
crops) is unchanged.

Run: CUDA_VISIBLE_DEVICES=1 /disk4/anaconda3/envs/vbpm/bin/python \
         experiments/stage0_transformer_prior.py
"""
import sys

import numpy as np
import torch

from vbpm.data import VALUES, load_crops
from vbpm.fitting import (cv_out_of_fold, elbo_mean_from, emission_counts,
                          emission_logp_from_counts, score, score_per_dataset)
from vbpm.heads import AutocorrHead, TransformerPrior
from vbpm.reducers import peak_summary

K = len(VALUES)
EPOCHS = 30
BATCH = 256
SEED = 0

# arm name -> (model builder, crop field holding its input, channel width, normalise)
ARMS = {
    "linear": (lambda: torch.nn.Linear(10, K), "s10", None, False),
    "autocorr": (lambda: AutocorrHead(), "h512", 512, False),
    "tf512": (lambda: TransformerPrior(in_dim=512), "h512", 512, False),
    "tf2": (lambda: TransformerPrior(in_dim=2), "h2", 2, False),
    "tf512n": (lambda: TransformerPrior(in_dim=512), "h512", 512, True),
}

# --- vbpm.fanout campaign protocol ---
OUTPUT = "features+activations"
SEQ_FIELDS = {"h512": 512, "h2": 2}
FLAT_FIELDS = ["s10", "C", "mask"]


def make_entry(song, crop, h_crop, t0):
    """Rich features + compressed activations (both fp16) + peak summary + emission stats."""
    features, acts = h_crop[:, :-2], h_crop[:, -2:]
    counts, mask = emission_counts(crop["y"], VALUES)
    return {"h512": features.astype(np.float16), "h2": acts.astype(np.float16),
            "s10": peak_summary(acts).numpy().astype(np.float32),
            "C": counts.astype(np.float32), "mask": mask.astype(np.float32),
            "m_true": crop["m_true"], "dataset": song.dataset, "fold": song.fold}


# ---- batching (length-bucketed, fp16 on the wire) -------------------------------------
def bucket_chunks(crops, batch_size):
    order = sorted(range(len(crops)), key=lambda i: len(crops[i]["h512"]))
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


def pad_features(crops, idx, field, width, device):
    Ts = [len(crops[i][field]) for i in idx]
    x16 = torch.zeros(len(idx), max(Ts), width, dtype=torch.float16)
    for j, i in enumerate(idx):
        x16[j, :Ts[j]] = torch.from_numpy(crops[i][field])
    return x16.to(device).float(), torch.tensor(Ts).to(device)


def stack_field(crops, idx, field, device):
    return torch.from_numpy(np.stack([crops[i][field] for i in idx])).to(device)


def normalize_per_crop(x, lengths):
    """Centre and scale each crop's channels over ITS OWN valid frames; pads stay zero.

    Statistics come from the crop, never from the corpus, so nothing about which
    checkpoint produced the features can leak in as a constant offset.
    """
    valid = (torch.arange(x.shape[1], device=x.device)[None, :]
             < lengths[:, None]).to(x.dtype)[..., None]
    n = lengths[:, None, None].to(x.dtype)
    mean = (x * valid).sum(1, keepdim=True) / n
    var = ((x - mean) * valid).pow(2).sum(1, keepdim=True) / n
    return ((x - mean) / (var.sqrt() + 1e-5)) * valid


def arm_logits(model, crops, idx, field, width, stats, device, normalize=False):
    """[B, K] prior logits for one arm on one index batch."""
    if width is None:                                   # linear arm reads z-scored s10
        mean, std = stats
        return model((stack_field(crops, idx, field, device) - mean) / std)
    x, lengths = pad_features(crops, idx, field, width, device)
    if normalize:
        x = normalize_per_crop(x, lengths)
    return model(x, lengths)


# ---- training -------------------------------------------------------------------------
def train_arm(name, train_crops, stats, device):
    """Joint ELBO ascent on theta + psi(this arm's model) + phi. Same budget for all arms."""
    torch.manual_seed(SEED)
    build, field, width, normalize = ARMS[name]
    model = build().to(device)
    scalars = {"alpha": torch.tensor(0.5, device=device, requires_grad=True),
               "beta": torch.tensor(-0.5, device=device, requires_grad=True),
               "c": torch.tensor(1.0, device=device, requires_grad=True)}
    opt = torch.optim.Adam([{"params": model.parameters(), "lr": 3e-3},
                            {"params": list(scalars.values()), "lr": 0.1}])
    log_m = torch.log(torch.tensor([float(m) for m in VALUES], device=device))
    chunks = bucket_chunks(train_crops, BATCH)
    rng = np.random.default_rng(SEED)

    for epoch in range(EPOCHS):
        total, n_batches = 0.0, 0
        rng.shuffle(chunks)
        for idx in chunks:
            logits = arm_logits(model, train_crops, idx, field, width, stats, device,
                                normalize)
            em = emission_logp_from_counts(stack_field(train_crops, idx, "C", device),
                                           stack_field(train_crops, idx, "mask", device),
                                           log_m, scalars["alpha"], scalars["beta"])
            loss = -elbo_mean_from(em, logits, scalars["c"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, n_batches = total - float(loss), n_batches + 1
        if epoch % 10 == 9 or epoch == 0:
            print(f"    [{name}] epoch {epoch + 1}: mean ELBO {total / n_batches:+.4f}",
                  flush=True)

    dead = [n for n, p in model.named_parameters() if p.grad is None]
    assert not dead, f"psi parameters with no gradient: {dead}"
    return model


@torch.no_grad()
def predict_with(name, model, crops, stats, device):
    _, field, width, normalize = ARMS[name]
    preds = []
    for i in range(0, len(crops), BATCH):
        idx = list(range(i, min(i + BATCH, len(crops))))
        logits = arm_logits(model, crops, idx, field, width, stats, device, normalize)
        preds += [VALUES[int(k)] for k in logits.argmax(-1).cpu()]
    return preds


def fit(arm, train_crops, device="cuda"):
    """vbpm.fanout protocol: one arm's fit on one train split (standardizer + model)."""
    stats = standardizer(train_crops, device)
    return (train_arm(arm, train_crops, stats, device), stats)


def predict(arm, fitted, crops, device="cuda"):
    """vbpm.fanout protocol: predictions for one held split."""
    return predict_with(arm, fitted[0], crops, fitted[1], device)


def standardizer(train_crops, device):
    s = np.stack([c["s10"] for c in train_crops])
    return (torch.tensor(s.mean(0), dtype=torch.float32, device=device),
            torch.tensor(s.std(0) + 1e-8, dtype=torch.float32, device=device))


def main():
    smoke = "--smoke" in sys.argv
    device = "cuda"
    crops, report = load_crops(limit_per_fold=6 if smoke else None, device=device,
                               output="features+activations", make_entry=make_entry)
    gib = sum(c["h512"].nbytes for c in crops) / 2**30
    print(f"crops: {len(crops)}  rejects: {report['rejects']}  features: {gib:.1f} GiB")

    cv = [c for c in crops if c["fold"] is not None]
    test = [c for c in crops if c["fold"] is None]

    for name in ARMS:
        pooled, preds, test_preds = cv_out_of_fold(
            cv, test, lambda cs, n=name: fit(n, cs, device),
            lambda fitted, cs, n=name: predict(n, fitted, cs, device),
            verbose=False)
        print(f"\n######## arm: {name} ########")
        score_per_dataset(pooled, preds, VALUES)
        if test:
            score("gtzan", test, test_preds, VALUES)


if __name__ == "__main__":
    main()
