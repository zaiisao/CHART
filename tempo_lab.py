"""Bench several bar-period decision rules on ONE frozen set of frontend activations.

The audit (octave_audit.log) showed the harmonic-sum ACF nominates the right period far
more often than it picks it: on gtzan's 206 wrong picks the truth is an accepted peak in
85 of them and loses only the "smallest accepted peak" race, and the winner outscores it
by a median of 3.6%. So the scoring is nearly right and the DECISION is wrong.

Candidate rules here, all reading the same cached activations:
  acf-smallest   what ships today: smallest ACF peak clearing ALPHA * max
  acf-argmax     same score, strongest peak instead of smallest
  comb           rank the ACF's nominees by COMB CONTRAST on the downbeat activation:
                 fold at the candidate period, compare mean activation on the best comb
                 against off it. Halving dilutes the comb (half its slots are not
                 downbeats); doubling splits the mass into two comb positions, so the
                 contrast peaks at the true period from both sides.
  comb+beat      same, but nominees also include beat-period multiples read off the
                 frontend's BEAT channel, which the shipped estimator ignores entirely.
"""
from __future__ import annotations

import argparse
import collections

import numpy as np
import torch

from phasevae.data.tempo import (ALPHA, N_HARM, P_MAX_S, P_MIN_S, SMOOTH,
                                 harmonic_score, pick_period)

RATIOS = [(1 / 3, "1/3"), (0.5, "1/2"), (2 / 3, "2/3"), (1.0, "1"),
          (1.5, "3/2"), (2.0, "2"), (3.0, "3")]
TOL = 0.05
NBINS = 64                      # folded-profile resolution
COMB_WIDTH = 3                  # bins counted as "on the comb" (of NBINS)


def label_ratio(r: float) -> str:
    for value, name in RATIOS:
        if abs(r / value - 1.0) <= TOL:
            return name
    return "other"


def smooth_zero_mean(activation, mask):
    p = (activation * mask).unsqueeze(1)
    kernel = torch.full((1, 1, SMOOTH), 1.0 / SMOOTH, device=p.device, dtype=p.dtype)
    p = torch.conv1d(p, kernel, padding=SMOOTH // 2).squeeze(1)
    valid = mask.sum(1, keepdim=True).clamp(min=1.0)
    return (p - (p * mask).sum(1, keepdim=True) / valid) * mask


def fold_profile(act, mask, periods_frames, nbins=NBINS):
    """[B,T] activation, [B,C] periods (frames) -> ([B,C,nbins] profile, occupancy).

    profile[b, c, j] is the MEAN activation of every frame whose phase at candidate c
    falls in bin j -- the shape a reader needs to tell a bar from half a bar: at twice
    the true period the mass splits into two equal bins, at half it is one bin diluted
    by the non-downbeats sharing it.
    """
    batch, length = act.shape
    n_cand = periods_frames.shape[1]
    time = torch.arange(length, device=act.device, dtype=torch.float32)
    phase = (time[None, None, :] % periods_frames[:, :, None]) / periods_frames[:, :, None]
    index = (phase * nbins).long().clamp(max=nbins - 1)
    weights = (act * mask)[:, None, :].expand(-1, n_cand, -1)
    counts = mask[:, None, :].expand(-1, n_cand, -1)
    profile = torch.zeros(batch, n_cand, nbins, device=act.device)
    occupancy = torch.zeros_like(profile)
    profile.scatter_add_(2, index, weights.float())
    occupancy.scatter_add_(2, index, counts.float())
    return profile / occupancy.clamp(min=1.0), occupancy


def comb_contrast(act, mask, periods_frames):
    """[B,T] activation, [B,C] candidate periods (frames) -> [B,C] contrast.

    Fold the activation onto NBINS phase bins at each candidate period and take the
    ratio of the mean activation inside the best COMB_WIDTH-bin window to the mean
    outside it. A candidate at HALF the true period keeps every downbeat in one bin but
    fills the same bin with as many non-downbeats, so its on-comb MEAN halves; a
    candidate at DOUBLE splits the downbeats across two windows, so its off-comb mean
    stays high. Both dilute the ratio, which is why it peaks at the truth.
    """
    profile, occupancy = fold_profile(act, mask, periods_frames)
    ring = torch.cat([profile, profile[:, :, :COMB_WIDTH - 1]], dim=2)
    window = ring.unfold(2, COMB_WIDTH, 1).mean(-1)                      # [B,C,NBINS]
    on, best = window.max(dim=2)

    # ISOLATION: how much the winning comb beats the best OTHER comb in the same fold.
    # At twice the true period the downbeats split into two equally strong combs, so the
    # runner-up matches the winner and the margin collapses -- the symmetric penalty the
    # on-vs-background ratio was missing (it only punished halving).
    bins = torch.arange(NBINS, device=act.device)
    apart = (bins[None, None, :] - best[:, :, None]).abs()
    apart = torch.minimum(apart, NBINS - apart)
    runner_up = torch.where(apart > COMB_WIDTH, window,
                            torch.full_like(window, -1e9)).max(dim=2).values

    total = profile.sum(dim=2)
    background = (total - on * COMB_WIDTH) / max(NBINS - COMB_WIDTH, 1)
    scores = {"ratio": on / background.clamp(min=1e-6),
              "margin": on - runner_up,
              "isolation": on / runner_up.clamp(min=1e-6),
              "margin_over_bg": (on - runner_up) / background.clamp(min=1e-6)}
    # a candidate shorter than NBINS frames leaves phase bins empty; judge it on the
    # bins it does fill rather than discarding it (that silently barred short periods).
    alive = occupancy.sum(dim=2) > 0
    return {k: torch.where(alive, v, torch.zeros_like(v)) for k, v in scores.items()}


def acf_candidates(score, grid, top_k=4):
    """[B,G] score -> [B,K] the K strongest ACF peaks (periods in seconds)."""
    interior = score[:, 1:-1]
    is_peak = (interior > score[:, :-2]) & (interior > score[:, 2:])
    masked = torch.where(is_peak, interior, torch.full_like(interior, -1e9))
    _, index = masked.topk(top_k, dim=1)
    return grid[index + 1]


def peakpick_period(act, mask, fps, threshold=0.5, min_gap_s=0.4):
    """[B,T] downbeat activation -> [B] median gap between its own peaks, seconds.

    The frontend is a TRAINED downbeat detector; autocorrelation second-guesses it.
    This asks it directly: keep local maxima above ``threshold``, at least ``min_gap_s``
    apart, and take the median spacing. Falls back to NaN when a window yields under two
    peaks, so the caller can defer to the ACF there.
    """
    gap = max(int(round(min_gap_s * fps)), 1)
    pooled = torch.nn.functional.max_pool1d(act[:, None], 2 * gap + 1, 1, gap)[:, 0]
    peaks = (act >= pooled) & (act > threshold) & (mask > 0)
    out = torch.full((act.shape[0],), float("nan"), device=act.device)
    for i in range(act.shape[0]):
        where = peaks[i].nonzero().flatten()
        if len(where) >= 3:
            out[i] = torch.median(where[1:] - where[:-1]).float() / fps
    return out


def ladder(periods, lo=P_MIN_S, hi=P_MAX_S):
    """[B,K] nominees -> [B,K*M] nominees plus their small-integer multiples/divisors."""
    factors = torch.tensor([0.5, 1.0, 2.0, 3.0, 4.0], device=periods.device)
    out = (periods[:, :, None] * factors[None, None, :]).flatten(1)
    return out.clamp(lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="tempo_acts.npz")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    blob = np.load(args.acts, allow_pickle=True)
    fps = float(blob["fps"])
    truth_all, split_all = blob["truth"], blob["split"]
    dataset_all = blob["dataset"]
    n = len(truth_all)
    device = f"cuda:{args.gpu}"

    # Log-normal bar-period prior FIT ON THE TRAIN SPLIT's annotations (1661 songs,
    # log-mean 0.604, log-sd 0.381 -> median 1.83 s). Train labels, not test ones: the
    # deployed estimator carries two constants, it does not read the test annotations.
    LOG_MU, LOG_SD = 0.6042, 0.3808

    def log_prior(periods):
        return -0.5 * ((torch.log(periods) - LOG_MU) / LOG_SD) ** 2

    picks = collections.defaultdict(list)
    for start in range(0, n, args.batch_size):
        stop = min(start + args.batch_size, n)
        act = torch.from_numpy(blob["act"][start:stop]).float().to(device)
        mask = torch.from_numpy(blob["mask"][start:stop]).float().to(device)
        down = torch.sigmoid(act[..., 1])
        beat = torch.sigmoid(act[..., 0])

        score, grid = harmonic_score(smooth_zero_mean(down, mask), fps)
        picks["acf-smallest"].append(grid[pick_period(score)].cpu().numpy())
        picks["acf-argmax"].append(grid[score.argmax(1)].cpu().numpy())

        nominees = ladder(acf_candidates(score, grid))
        for name, value in comb_contrast(down, mask, nominees * fps).items():
            picks[f"comb-{name}"].append(nominees.gather(
                1, value.argmax(1, keepdim=True)).squeeze(1).cpu().numpy())

        for weight in (0.5, 1.0, 2.0):
            posterior = torch.log(score.clamp(min=1e-9)) + weight * log_prior(grid)[None]
            picks[f"acf+prior{weight}"].append(grid[posterior.argmax(1)].cpu().numpy())

        pp = peakpick_period(down, mask, fps)
        acf_fallback = grid[pick_period(score)]
        picks["peakpick"].append(torch.where(pp.isnan(), acf_fallback, pp)
                                 .clamp(P_MIN_S, P_MAX_S).cpu().numpy())

        # ceiling: could ANY reranker over these nominees get it right?
        best = (nominees / torch.from_numpy(
            truth_all[start:stop]).float().to(device)[:, None] - 1.0).abs().min(1).values
        picks["oracle-over-nominees"].append(
            torch.where(best <= TOL,
                        torch.from_numpy(truth_all[start:stop]).float().to(device),
                        nominees[:, 0]).cpu().numpy())

        with_pp = torch.cat([nominees, torch.where(pp.isnan(), acf_fallback, pp)
                             .clamp(P_MIN_S, P_MAX_S)[:, None],
                             (torch.where(pp.isnan(), acf_fallback, pp) * 2)
                             .clamp(P_MIN_S, P_MAX_S)[:, None]], dim=1)
        for name, value in comb_contrast(down, mask, with_pp * fps).items():
            picks[f"pp+comb-{name}"].append(with_pp.gather(
                1, value.argmax(1, keepdim=True)).squeeze(1).cpu().numpy())

        contrast = comb_contrast(down, mask, with_pp * fps)["ratio"]
        for weight in (1.0, 2.0):
            posterior = torch.log(contrast.clamp(min=1e-9)) + weight * log_prior(with_pp)
            picks[f"comb+prior{weight}"].append(with_pp.gather(
                1, posterior.argmax(1, keepdim=True)).squeeze(1).cpu().numpy())

        beat_score, beat_grid = harmonic_score(smooth_zero_mean(beat, mask), fps)
        beat_period = beat_grid[beat_score.argmax(1)]
        multiples = torch.tensor([2.0, 3.0, 4.0, 6.0, 8.0], device=device)
        beat_nominees = (beat_period[:, None] * multiples[None, :]).clamp(P_MIN_S, P_MAX_S)
        both = torch.cat([nominees, beat_nominees], dim=1)
        for name, value in comb_contrast(down, mask, both * fps).items():
            picks[f"beat-{name}"].append(both.gather(
                1, value.argmax(1, keepdim=True)).squeeze(1).cpu().numpy())

    print(f"windows: {n}  (val {int((split_all == 'val').sum())}, "
          f"gtzan {int((split_all == 'gtzan').sum())})\n")
    order = [name for _, name in RATIOS] + ["other"]
    for rule, chunks in picks.items():
        est = np.concatenate(chunks)
        np.save(f"tempo_lab_{rule}.npy", est)
        print(f"--- {rule}")
        for split_name in ("val", "gtzan"):
            keep = split_all == split_name
            counts = collections.Counter(
                label_ratio(e / t) for e, t in zip(est[keep], truth_all[keep]))
            total = max(int(keep.sum()), 1)
            exact = 100 * counts["1"] / total
            # est/ref proxy: a window off by ratio r emits 1/r times as many downbeats
            ratio = est[keep] / truth_all[keep]
            print(f"  {split_name:6s} exact {exact:5.1f}%   " + "  ".join(
                f"{k}:{100 * counts[k] / total:4.1f}%" for k in order
                if counts[k] and k != "1") + f"   est/ref {np.mean(1 / ratio):.3f}")
        keep = split_all == "gtzan"
        print("   gtzan by-genre worst: ", end="")
        by = collections.defaultdict(list)
        for e, t, d, s in zip(est, truth_all, dataset_all, split_all):
            if s == "val":
                by[d].append(abs(e / t - 1.0) <= TOL)
        print("  ".join(f"{k} {100 * np.mean(v):.0f}%" for k, v in sorted(by.items())))


if __name__ == "__main__":
    main()
