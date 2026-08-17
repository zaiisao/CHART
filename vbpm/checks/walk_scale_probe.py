"""Is the per-song tempo variability recoverable without the annotations?

The per-song walk scale is worth more than any tail refinement (+2.2 nats/step against
+0.9), and it is a stable property of a performance (split-half r = 0.81). It is only
usable, though, if it can be inferred at deployment. This asks the label-free question
directly: peak-pick the frozen frontend's downbeat activation, measure the variability
of the intervals it produces, and compare that against the variability of the annotated
downbeats on held-out songs.
"""
from __future__ import annotations

import argparse
import importlib

import numpy as np
import torch
from scipy import stats

from ..config import load_config
from ..data.dataset import split_songs
from ..data.excerpts import ExcerptDataset, collate_excerpts
from ..scoring.evaluation import peak_times, scoring_records


def variability(times):
    """Laplace scale of the per-bar log-tempo change: mean |d log interval|."""
    times = np.asarray(times, dtype=np.float64)
    if len(times) < 6:
        return None
    iv = np.diff(times)
    if iv.min() <= 0:
        return None
    return float(np.mean(np.abs(np.diff(np.log(iv)))))


def main():
    """Ask whether per-song tempo variability is recoverable without labels."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="vbpm/configs/rate_grid.yaml")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--limit-per-fold", type=int, default=None)
    p.add_argument("--split", default="both", choices=("train", "val", "both"))
    p.add_argument("--per-dataset", type=int, default=0,
                   help="sample this many songs from EACH corpus; limit-per-fold fills "
                        "its quota alphabetically, which silently returns one corpus")
    args = p.parse_args()

    cfg, _hooks = load_config(args.config, [])
    frontend = importlib.import_module(f"vbpm.data.frontends.{cfg.frontend}").FRONTEND(
        checkpoint=cfg.frontend_checkpoint, device=f"cuda:{args.gpu}", output="features")
    train_songs, val_songs, _t = split_songs(0, args.limit_per_fold)
    songs = {"train": train_songs, "val": val_songs,
             "both": train_songs + val_songs}[args.split]
    if args.per_dataset:
        by = {}
        for song in songs:
            by.setdefault(song.dataset, []).append(song)
        rng = np.random.default_rng(0)
        songs = [s for group in by.values()
                 for s in [group[i] for i in
                           rng.permutation(len(group))[:args.per_dataset]]]

    dataset = ExcerptDataset(songs, frontend, cfg.excerpt_seconds, deterministic=True,
                             target_tol_frames=getattr(cfg, "target_tol_frames", 0))
    loader = torch.utils.data.DataLoader(dataset, batch_size=8,
                                         collate_fn=collate_excerpts)
    rows = []
    with torch.no_grad():
        for raw in loader:
            h = frontend.forward_features(raw["input"])
            acts = frontend._audio2frames.model.task_heads(h)
            probs = torch.sigmoid(acts["downbeat"]).cpu().numpy()
            for i, crop in enumerate(scoring_records(raw)):
                if crop is None:
                    continue
                truth = np.asarray(crop["downbeat_times"])
                b_true = variability(truth)
                if b_true is None:
                    continue
                valid = int(raw["mask"][i].sum())
                period = float(np.median(np.diff(np.asarray(crop["anchors"], float))))
                pred = peak_times(probs[i, :valid], crop["fps"], period)
                b_hat = variability(pred)
                if b_hat is None:
                    continue
                rows.append((crop["dataset"], b_true, b_hat))

    ds = np.array([r[0] for r in rows])
    bt = np.log(np.array([r[1] for r in rows]) + 1e-4)
    bh = np.log(np.array([r[2] for r in rows]) + 1e-4)
    print(f"n = {len(rows)} songs\n")
    print("label-free estimate vs annotated, log scale:")
    print(f"   Pearson r  {np.corrcoef(bt, bh)[0, 1]:+.3f}")
    print(f"   Spearman   {stats.spearmanr(bt, bh).statistic:+.3f}")
    print(f"   R^2 of a linear map {np.corrcoef(bt, bh)[0, 1] ** 2:.3f}")
    lo, hi = np.percentile(bt, [33, 67])
    for name, sel in (("low third", bt <= lo), ("middle", (bt > lo) & (bt < hi)),
                      ("high third", bt >= hi)):
        print(f"   annotated {name:11} median estimate "
              f"{np.exp(np.median(bh[sel])):.4f}   (annotated "
              f"{np.exp(np.median(bt[sel])):.4f})")
    print("\nper corpus:")
    for d in sorted(set(ds)):
        m = ds == d
        if m.sum() < 5:
            continue
        print(f"   {d:12} n={int(m.sum()):4d}  r {np.corrcoef(bt[m], bh[m])[0, 1]:+.3f}   "
              f"annotated {np.exp(np.median(bt[m])):.4f}  estimated "
              f"{np.exp(np.median(bh[m])):.4f}")


if __name__ == "__main__":
    main()
