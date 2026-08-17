"""The frontend's own downbeat head, peak-picked, as the baseline to clear.

Scored on the same deterministic crops the overfit checks use, so the
numbers sit beside them directly.

The picker is given the ANNOTATED bar period for its minimum gap, which is an oracle
advantage the model's own read-outs do not get -- read it as an upper bound on what
peak-picking alone delivers.
"""
from __future__ import annotations

import argparse
import importlib

import numpy as np
import torch

from ..config import load_config
from ..data.dataset import load_catalog
from ..data.excerpts import ExcerptDataset, collate_excerpts
from ..scoring.evaluation import continuity_scores, f_measure, peak_times


def main():
    """Peak-pick the frontend's own head over the named songs."""
    p = argparse.ArgumentParser()
    p.add_argument("songs", nargs="+", help="dataset:index pairs")
    p.add_argument("--config", default="vbpm/configs/rate_grid.yaml")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--head", default="downbeat", choices=("downbeat", "beat"))
    args = p.parse_args()

    cfg, _hooks = load_config(args.config, [])
    frontend = importlib.import_module(f"vbpm.data.frontends.{cfg.frontend}").FRONTEND(
        checkpoint=cfg.frontend_checkpoint, device=f"cuda:{args.gpu}", output="features")

    rows = []
    for pair in args.songs:
        name, index = pair.split(":")
        song = sorted(sum(load_catalog([name]).values(), []),
                      key=lambda s: s.song_id)[int(index)]
        data = ExcerptDataset([song], frontend, cfg.excerpt_seconds, deterministic=True,
                              target_tol_frames=getattr(cfg, "target_tol_frames", 0))
        raw = collate_excerpts([data[0]])
        with torch.no_grad():
            h = frontend.forward_features(raw["input"])
            acts = frontend._audio2frames.model.task_heads(h)
            probs = torch.sigmoid(acts[args.head])[0].cpu().numpy()
        truth = np.asarray(raw["downbeat_times"][0])
        fps = float(raw["fps"][0])
        anchors = np.asarray(raw["anchors"][0], dtype=np.float64)
        period = float(np.median(np.diff(anchors)))
        valid = int(raw["mask"][0].sum())
        pred = peak_times(probs[:valid], fps, period) + float(raw["t0"][0])
        F = f_measure(pred, truth)[0]
        cmlt, amlt = continuity_scores(truth, pred)
        err = ([min(abs(d - pred)) * 1000 for d in truth] if len(pred) else [9e9])
        rows.append((pair, period, np.median(err),
                     float(np.mean(np.asarray(err) <= 70)), F, cmlt, amlt,
                     len(pred), len(truth)))

    print(f"{'song':>18} {'bar s':>6} {'err ms':>7} {'in-tol':>7} {'F':>6} "
          f"{'CMLt':>6} {'AMLt':>6} {'n/ref':>9}")
    for r in rows:
        print(f"{r[0]:>18} {r[1]:6.2f} {r[2]:7.0f} {r[3]:6.0%} {r[4]:6.3f} "
              f"{r[5]:6.3f} {r[6]:6.3f} {r[7]:5d}/{r[8]:3d}")
    a = np.array([[r[3], r[4], r[5], r[6]] for r in rows])
    print(f"\nmean over {len(rows)}: in-tol {a[:, 0].mean():.1%}  F {a[:, 1].mean():.3f}  "
          f"CMLt {a[:, 2].mean():.3f}  AMLt {a[:, 3].mean():.3f}")


if __name__ == "__main__":
    main()
