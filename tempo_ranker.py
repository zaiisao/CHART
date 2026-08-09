"""Can a LEARNED reranker close the gap the hand rules cannot?

The audit says the harmonic-sum ACF nominates the true bar period for 96.5% of gtzan
windows, and every hand-written decision rule -- smallest accepted peak (79.3%), plain
argmax (81.4%), comb contrast (80.5%), a train-fit log-normal tempo prior (worse at
every weight) -- picks it for at most ~81%. So the evidence is there and the RULE is
what is missing. This trains a small softmax reranker over the same nominees and reports
how much of the 15-point gap is reachable.

Supervision is the TRAIN split's annotated bar periods -- train labels, exactly like the
emission's targets. The deployed reader sees only folded activations and never a label.
Phase is rotated out of every feature, so it cannot smuggle the answer through position.
"""
from __future__ import annotations

import argparse
import collections

import numpy as np
import torch
from torch import nn

from phasevae.data.tempo import P_MAX_S, P_MIN_S, harmonic_score, pick_period
from tempo_lab import (TOL, acf_candidates, comb_contrast, fold_profile, label_ratio,
                       ladder, peakpick_period, smooth_zero_mean)

NBINS = 32


def features(down, beat, mask, fps, nominees):
    """[B,C] nominees -> [B,C,D] rotation-invariant evidence for each candidate period.

    Each candidate's folded downbeat profile is rotated to put its own maximum in bin 0
    and the BEAT profile is rotated by the SAME offset, so absolute phase is gone but the
    beat/downbeat alignment -- which is what says "these two combs are the same event" --
    survives.
    """
    frames = nominees * fps
    down_profile, occupancy = fold_profile(down, mask, frames, NBINS)
    beat_profile, _ = fold_profile(beat, mask, frames, NBINS)

    shift = down_profile.argmax(dim=2, keepdim=True)
    index = (torch.arange(NBINS, device=down.device)[None, None, :] + shift) % NBINS
    down_rot = down_profile.gather(2, index)
    beat_rot = beat_profile.gather(2, index)

    scale = down_rot.amax(dim=2, keepdim=True).clamp(min=1e-6)
    contrast = comb_contrast(down, mask, frames)
    extra = torch.stack([torch.log(nominees),
                         down_rot.mean(2), scale.squeeze(2),
                         torch.log(contrast["ratio"].clamp(min=1e-6)),
                         torch.log(contrast["isolation"].clamp(min=1e-6)),
                         (occupancy > 0).float().mean(2)], dim=2)
    return torch.cat([down_rot / scale, beat_rot / scale, extra], dim=2)


def build(blob, device, batch_size=64):
    """Cached activations -> (nominees, features, truth) for every window."""
    fps = float(blob["fps"])
    truth_all = blob["truth"]
    out_nom, out_feat = [], []
    for start in range(0, len(truth_all), batch_size):
        stop = min(start + batch_size, len(truth_all))
        act = torch.from_numpy(blob["act"][start:stop]).float().to(device)
        mask = torch.from_numpy(blob["mask"][start:stop]).float().to(device)
        down, beat = torch.sigmoid(act[..., 1]), torch.sigmoid(act[..., 0])
        score, grid = harmonic_score(smooth_zero_mean(down, mask), fps)
        pp = peakpick_period(down, mask, fps)
        pp = torch.where(pp.isnan(), grid[pick_period(score)], pp).clamp(P_MIN_S, P_MAX_S)
        nominees = torch.cat([ladder(acf_candidates(score, grid)),
                              pp[:, None], (2 * pp).clamp(P_MIN_S, P_MAX_S)[:, None]], 1)
        out_nom.append(nominees)
        out_feat.append(features(down, beat, mask, fps, nominees))
    return (torch.cat(out_nom), torch.cat(out_feat),
            torch.from_numpy(truth_all).float().to(device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-acts", default="tempo_acts_train.npz")
    ap.add_argument("--no-period-feature", action="store_true",
                    help="ablate log(period): if the ranker is only a tempo prior in "
                         "disguise, removing its one absolute-scale input kills it")
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"
    torch.manual_seed(args.seed)

    train_blob = np.load(args.train_acts, allow_pickle=True)
    eval_blob = np.load("tempo_acts.npz", allow_pickle=True)
    train_nom, train_feat, train_truth = build(train_blob, device)
    eval_nom, eval_feat, eval_truth = build(eval_blob, device)
    split_all, dataset_all = eval_blob["split"], eval_blob["dataset"]

    # The ladder emits near-duplicates (clamping collapses several factors onto the same
    # period), so ONE correct index is the wrong target -- it splits probability between
    # identical candidates and caps accuracy for free. Every nominee within TOL of the
    # truth counts, and the loss is -log of their SUMMED probability.
    error = (train_nom / train_truth[:, None] - 1.0).abs()
    correct = error <= TOL
    usable = correct.any(1)
    print(f"train windows {len(train_truth)}, nominee set contains the truth for "
          f"{100 * usable.float().mean():.1f}%")

    if args.no_period_feature:
        keep_dim = [i for i in range(train_feat.shape[2]) if i != 2 * NBINS]
        train_feat, eval_feat = train_feat[..., keep_dim], eval_feat[..., keep_dim]

    net = nn.Sequential(nn.Linear(train_feat.shape[2], 64), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

    # Held out BY SONG: an 8-window dump puts eight views of one song in the pool, so a
    # random split would score the ranker on windows of songs it trained on.
    stems = train_blob["stem"]
    held = np.isin(stems, np.unique(stems)[::5])
    fit = torch.from_numpy(~held).to(device) & usable
    check = torch.from_numpy(held).to(device) & usable
    x, y = train_feat[fit], correct[fit]
    print(f"  fit on {int(fit.sum())} windows, held-out {int(check.sum())} "
          f"({len(np.unique(stems[held]))} songs)")
    for epoch in range(args.epochs):
        order = torch.randperm(len(x), device=device)
        for start in range(0, len(x), 256):
            batch = order[start:start + 256]
            log_p = net(x[batch]).squeeze(-1).log_softmax(1)
            loss = -torch.logsumexp(log_p.masked_fill(~y[batch], -1e9), dim=1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 200 == 0 or epoch == args.epochs - 1:
            net.eval()
            with torch.no_grad():
                hit = y.gather(1, net(x).squeeze(-1).argmax(1)[:, None]).float().mean()
                out = net(train_feat[check]).squeeze(-1).argmax(1)
                held_hit = correct[check].gather(1, out[:, None]).float().mean()
            net.train()
            print(f"  epoch {epoch:4d}  loss {float(loss):.4f}  fit {float(hit):.3f}  "
                  f"held-out {float(held_hit):.3f}")

    net.eval()
    with torch.no_grad():
        chosen = eval_nom.gather(1, net(eval_feat).squeeze(-1).argmax(1, keepdim=True))
    est = chosen.squeeze(1).cpu().numpy()
    truth = eval_truth.cpu().numpy()
    ceiling = ((eval_nom / eval_truth[:, None] - 1.0).abs().min(1).values
               <= TOL).cpu().numpy()

    print("\n===== learned reranker =====")
    for split_name in ("val", "gtzan"):
        keep = split_all == split_name
        counts = collections.Counter(label_ratio(e / t)
                                     for e, t in zip(est[keep], truth[keep]))
        total = int(keep.sum())
        print(f"  {split_name:6s} n={total}  exact {100 * counts['1'] / total:5.1f}%  "
              f"(ceiling {100 * ceiling[keep].mean():5.1f}%)  "
              f"halved {100 * counts['1/2'] / total:4.1f}%  "
              f"doubled {100 * counts['2'] / total:4.1f}%  "
              f"est/ref {np.mean(truth[keep] / est[keep]):.3f}")
    by = collections.defaultdict(list)
    for e, t, d, s in zip(est, truth, dataset_all, split_all):
        if s == "val":
            by[d].append(abs(e / t - 1.0) <= TOL)
    print("  val by dataset: " + "  ".join(
        f"{k} {100 * np.mean(v):.0f}%" for k, v in sorted(by.items())))
    np.save("tempo_ranker_est.npy", est)


if __name__ == "__main__":
    main()
