"""Why the audio bar-period estimator halves: the evidence behind each wrong pick.

The oracle-removal commit measured the cost (est/ref 1.003 -> 1.161, CMLt 0.844 ->
0.710) but not the cause. This reads, per window, the harmonic-sum score curve the
estimator decides on, and asks of every wrong pick: was the TRUE period even a peak,
did it clear ALPHA * max, and did it simply lose the "smallest accepted peak" race?

Runs the frontend only -- no model, no training.
"""
from __future__ import annotations

import argparse
import collections

import numpy as np
import torch

from phasevae.data.dataset import split_songs
from phasevae.data.excerpts import ExcerptDataset, collate_excerpts
from phasevae.data.tempo import (ALPHA, P_MAX_S, P_MIN_S, SMOOTH,
                                 harmonic_score, pick_period)

RATIOS = [(1 / 3, "1/3"), (0.5, "1/2"), (2 / 3, "2/3"), (1.0, "1"),
          (1.5, "3/2"), (2.0, "2"), (3.0, "3")]
TOL = 0.05


def label_ratio(r: float) -> str:
    for value, name in RATIOS:
        if abs(r / value - 1.0) <= TOL:
            return name
    return "other"


def zero_mean(activation, mask):
    """The same preparation estimate_bar_period does before scoring."""
    p = (activation * mask).unsqueeze(1)
    kernel = torch.full((1, 1, SMOOTH), 1.0 / SMOOTH, device=p.device, dtype=p.dtype)
    p = torch.conv1d(p, kernel, padding=SMOOTH // 2).squeeze(1)
    valid = mask.sum(1, keepdim=True).clamp(min=1.0)
    return (p - (p * mask).sum(1, keepdim=True) / valid) * mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--excerpt-seconds", type=float, default=45.0)
    ap.add_argument("--limit-per-fold", type=int, default=None)
    args = ap.parse_args()

    import importlib
    device = f"cuda:{args.gpu}"
    frontend = importlib.import_module("phasevae.data.frontends.beat_this").FRONTEND(
        checkpoint="final0", device=device, output="features+activations")

    _, val_songs, test_songs = split_songs(args.limit_per_fold)
    splits = {"val": val_songs, "gtzan": test_songs}

    rows = []
    for split_name, songs in splits.items():
        data = ExcerptDataset(songs, frontend, args.excerpt_seconds, deterministic=True)
        loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,
                                             shuffle=False, num_workers=4,
                                             collate_fn=collate_excerpts)
        for raw in loader:
            with torch.no_grad():
                h = frontend.forward_features(raw["input"])
                mask = raw["mask"].to(device)
                act = torch.sigmoid(h[..., -1])
                p = zero_mean(act, mask)
                score, grid = harmonic_score(p, frontend.FPS)
                if score is None:
                    continue
                index = pick_period(score)

            grid_np = grid.cpu().numpy()
            score_np = score.cpu().numpy()
            index_np = index.cpu().numpy()
            truth = raw["bar_period"].numpy()
            for i in range(len(truth)):
                curve, chosen = score_np[i], int(index_np[i])
                truth_i = int(np.abs(grid_np - truth[i]).argmin())
                peak_floor = ALPHA * curve.max()
                is_peak = (curve[truth_i] > curve[truth_i - 1]
                           and curve[truth_i] > curve[truth_i + 1]
                           if 0 < truth_i < len(curve) - 1 else False)
                # a broad peak can be flat at the exact truth bin; also allow a local
                # window maximum within +-2% of the truth period as "truth is a peak"
                span = max(1, int(round(0.02 * truth_i)))
                lo, hi = max(0, truth_i - span), min(len(curve), truth_i + span + 1)
                local_max = int(lo + curve[lo:hi].argmax())
                is_peak = is_peak or (local_max not in (lo, hi - 1))
                rows.append(dict(
                    split=split_name, dataset=raw["dataset"][i],
                    truth=float(truth[i]), est=float(grid_np[chosen]),
                    in_range=bool(P_MIN_S <= truth[i] <= P_MAX_S),
                    score_truth=float(curve[local_max]),
                    score_est=float(curve[chosen]),
                    score_max=float(curve.max()),
                    truth_clears=bool(curve[local_max] >= peak_floor),
                    truth_is_peak=bool(is_peak),
                ))
        print(f"  {split_name}: {len(rows)} windows scored", flush=True)

    np.save("octave_audit.npy", np.array(rows, dtype=object), allow_pickle=True)

    print("\n===== ratio est/truth, by split =====")
    for split_name in splits:
        sub = [r for r in rows if r["split"] == split_name]
        counts = collections.Counter(label_ratio(r["est"] / r["truth"]) for r in sub)
        total = max(len(sub), 1)
        order = [n for _, n in RATIOS] + ["other"]
        print(f"{split_name}  n={len(sub)}  " + "  ".join(
            f"{n}:{100 * counts[n] / total:5.1f}%" for n in order if counts[n]))

    print("\n===== gtzan, by dataset-of-origin =====")
    by_ds = collections.defaultdict(list)
    for r in rows:
        by_ds[(r["split"], r["dataset"])].append(r)
    for key, sub in sorted(by_ds.items()):
        counts = collections.Counter(label_ratio(r["est"] / r["truth"]) for r in sub)
        correct = 100 * counts["1"] / len(sub)
        halved = 100 * counts["1/2"] / len(sub)
        print(f"  {key[0]:6s} {key[1]:12s} n={len(sub):4d}  exact {correct:5.1f}%  "
              f"halved {halved:5.1f}%  other {100 - correct - halved:5.1f}%")

    print("\n===== the wrong picks: was the truth even available? =====")
    for split_name in splits:
        wrong = [r for r in rows if r["split"] == split_name
                 and label_ratio(r["est"] / r["truth"]) != "1"]
        if not wrong:
            continue
        in_range = [r for r in wrong if r["in_range"]]
        peak = [r for r in in_range if r["truth_is_peak"]]
        clears = [r for r in peak if r["truth_clears"]]
        smaller = [r for r in clears if r["est"] < r["truth"]]
        print(f"{split_name}: {len(wrong)} wrong  |  truth in [{P_MIN_S},{P_MAX_S}]s: "
              f"{len(in_range)}  |  truth is a peak: {len(peak)}  |  peak clears "
              f"{ALPHA}*max: {len(clears)}  |  of those, estimate is SMALLER: {len(smaller)}")
        if clears:
            gap = np.array([r["score_est"] / max(r["score_truth"], 1e-9) for r in clears])
            print(f"    score(est)/score(truth) on those: median {np.median(gap):.3f}  "
                  f"mean {gap.mean():.3f}  frac>1 {float((gap > 1).mean()):.3f}")


if __name__ == "__main__":
    main()
