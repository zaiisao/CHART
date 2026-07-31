"""End-to-end trained evidence head over frozen Beat This [T,512] features.

The rich-features negative (2026-07-31) convicted FIXED reductions, not the representation.
This trains the reduction itself: a temporal head (learned channel projection -> exact FFT
autocorrelation over lags -> conv over the lag axis -> psi logits), jointly with theta
{alpha, beta} and phi {c}, by maximizing the SAME exact-enumeration Stage-0 ELBO:

    ELBO = E_q[log p_theta(y|m)] - KL(q_phi || p_psi(head(h)))

The frontend stays frozen (SS6.1); the head is OUR evidence head, trained fold-honest.
The deployable path stays annotation-blind: the head reads features only; the labels
enter training solely through the emission and encoder terms of the ELBO.

Experiment-level departures from SS5 defaults, deliberate and noted: minibatch Adam
(18.9k crops x [T,512] cannot be full-batch), float32, GPU. The ELBO per crop is still an
exact 3-term enumeration - nothing is sampled; the minibatching only chunks the mean.

Run: CUDA_VISIBLE_DEVICES=1 /disk4/anaconda3/envs/vbpm/bin/python experiments/stage0_e2e_head.py
"""
import sys

import numpy as np
import torch


from vbpm.data import FPS, VALUES, load_crops
from vbpm.heads import AutocorrHead
from vbpm.fitting import (cv_out_of_fold, elbo_mean_from, emission_counts,
                             emission_logp_from_counts, score, score_per_dataset)

K = len(VALUES)
N_LAGS = int(5 * FPS)   # 5 s of lags; longest bars here are ~4 s
EPOCHS = 30
BATCH = 256
SEED = 0


def emission_stats(y):
    """float32 view of the shared emission linearisation (vbpm.fitting.emission_counts)."""
    C, mask = emission_counts(y, VALUES)
    return C.astype(np.float32), mask.astype(np.float32)


# --------------------------------------------------------------------------------------
# ELBO on a minibatch (exact 3-term enumeration per crop)
# --------------------------------------------------------------------------------------
def batch_elbo(head, scalars, log_m, x, lengths, Cs, masks):
    """The shared ELBO composition with the head's logits as the prior — no local re-derivation."""
    em = emission_logp_from_counts(Cs, masks, log_m, scalars["alpha"], scalars["beta"])
    return elbo_mean_from(em, head(x, lengths), scalars["c"])


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------
def make_entry(song, crop, h_crop, t0):
    C, mask = emission_stats(crop["y"])
    return {"h16": h_crop.astype(np.float16), "C": C, "mask": mask,
            "m_true": crop["m_true"], "dataset": song.dataset, "fold": song.fold}


def bucket_chunks(crops, batch_size):
    """Length-bucketed index chunks (sorted once; shuffle order per epoch, not contents)."""
    order = sorted(range(len(crops)), key=lambda i: len(crops[i]["h16"]))
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


def pad_features(crops, idx, device):
    """(x [B, Tmax, 512] float32-on-GPU, lengths): fp16 on the wire, cast after transfer."""
    Ts = [len(crops[i]["h16"]) for i in idx]
    x16 = torch.zeros(len(idx), max(Ts), 512, dtype=torch.float16)
    for j, i in enumerate(idx):
        x16[j, :Ts[j]] = torch.from_numpy(crops[i]["h16"])
    return x16.to(device).float(), torch.tensor(Ts).to(device)


def emission_batch(crops, idx, device):
    Cs = torch.from_numpy(np.stack([crops[i]["C"] for i in idx]))
    masks = torch.from_numpy(np.stack([crops[i]["mask"] for i in idx]))
    return Cs.to(device), masks.to(device)


# --------------------------------------------------------------------------------------
# train / eval per fold
# --------------------------------------------------------------------------------------
def train_head(train_crops, device):
    torch.manual_seed(SEED)
    head = AutocorrHead(n_lags=N_LAGS).to(device)
    # SS4.7 split: alpha/beta are theta (emission), c is phi (encoder); head is psi
    scalars = {"alpha": torch.tensor(0.5, device=device, requires_grad=True),
               "beta": torch.tensor(-0.5, device=device, requires_grad=True),
               "c": torch.tensor(1.0, device=device, requires_grad=True)}
    opt = torch.optim.Adam([{"params": head.parameters(), "lr": 3e-3},
                            {"params": list(scalars.values()), "lr": 0.1}])
    log_m = torch.log(torch.tensor([float(m) for m in VALUES], device=device))
    chunks = bucket_chunks(train_crops, BATCH)
    rng = np.random.default_rng(SEED)
    for epoch in range(EPOCHS):
        total, nb = 0.0, 0
        rng.shuffle(chunks)
        for idx in chunks:
            x, lengths = pad_features(train_crops, idx, device)
            Cs, masks = emission_batch(train_crops, idx, device)
            loss = -batch_elbo(head, scalars, log_m, x, lengths, Cs, masks)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, nb = total - float(loss), nb + 1
        if epoch % 10 == 9 or epoch == 0:
            print(f"    epoch {epoch + 1}: mean ELBO {total / nb:+.4f}", flush=True)
    # every parameter must have received gradient (SS4.7 discipline)
    dead = [n for n, p in head.named_parameters() if p.grad is None]
    assert not dead, f"head parameters with no gradient: {dead}"
    return head, scalars


@torch.no_grad()
def predict(head, crops, device):
    preds = []
    for i in range(0, len(crops), BATCH):
        idx = list(range(i, min(i + BATCH, len(crops))))
        x, lengths = pad_features(crops, idx, device)   # eval needs no emission stats
        preds += [VALUES[int(k)] for k in head(x, lengths).argmax(-1).cpu()]
    return preds


def main():
    smoke = "--smoke" in sys.argv
    device = "cuda"
    crops, report = load_crops(limit_per_fold=6 if smoke else None, device=device,
                               output="features", make_entry=make_entry)
    gb = sum(c["h16"].nbytes for c in crops) / 2**30
    print(f"crops: {len(crops)}  rejects: {report['rejects']}  features in RAM: {gb:.1f} GiB")

    cv = [c for c in crops if c["fold"] is not None]
    test = [c for c in crops if c["fold"] is None]

    pooled, preds, test_preds = cv_out_of_fold(
        cv, test,
        lambda train: train_head(train, device)[0],
        lambda head, cs: predict(head, cs, device))

    print("\n== pooled out-of-fold, per dataset ==")
    score_per_dataset(pooled, preds, VALUES)
    if test:
        score("gtzan", test, test_preds, VALUES)


if __name__ == "__main__":
    main()
