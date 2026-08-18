"""The 241 songs whose period is right and placement is not: systematic or not?

For every true downbeat, the signed distance to our nearest prediction, in
units of the bar. A spike away from zero is a gauge offset and is cheap to
fix; a flat spread is the placement objective and is not.
"""
from __future__ import annotations

import argparse
import importlib

import numpy as np
import torch

from ..config import load_config
from ..data.dataset import load_catalog
from ..data.excerpts import ExcerptDataset, collate_excerpts
from ..scoring.evaluation import (f_measure, rule_g_times, scoring_records,
                                  trajectory_period)


def main():
    """Dump per-song phase offsets and summarise their concentration."""
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = f"cuda:{args.gpu}"
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg, hooks = load_config(blob["config_path"], list(blob.get("overrides", [])))
    frontend = importlib.import_module(f"vbpm.data.frontends.{cfg.frontend}").FRONTEND(
        checkpoint=cfg.frontend_checkpoint, device=device, output="features")
    frontend._audio2frames.model.load_state_dict(blob["frontend"])
    model = hooks.build_model(cfg, frontend.num_channels).to(device)
    model.load_state_dict(blob["model"])
    model.eval()

    songs = sorted(sum(load_catalog(["gtzan"]).values(), []), key=lambda s: s.song_id)
    data = ExcerptDataset(songs, frontend, cfg.excerpt_seconds, deterministic=True,
                          target_tol_frames=getattr(cfg, "target_tol_frames", 0))
    loader = torch.utils.data.DataLoader(data, batch_size=cfg.batch_size,
                                         collate_fn=collate_excerpts)

    rows = []
    with torch.no_grad():
        for raw in loader:
            recs = scoring_records(raw)
            keep = [i for i, c in enumerate(recs) if c is not None]
            if not keep:
                continue
            crops = [recs[i] for i in keep]
            h = frontend.forward_features(raw["input"]).clone()
            mask = raw["mask"].to(device)
            mu = model.infer_phase(h, mask)[keep]
            times = rule_g_times(mu, mask[keep], crops)
            per = trajectory_period(mu, mask[keep], crops[0]["fps"])
            for i, c in enumerate(crops):
                truth = np.asarray(c["downbeat_times"])
                if len(truth) < 2 or not len(times[i]):
                    continue
                ref = float(np.median(np.diff(truth)))
                if not (0.9 <= float(per[i]) / ref <= 1.1):
                    continue
                d = np.array([times[i][np.argmin(np.abs(times[i] - t))] - t for t in truth])
                frac = d / ref
                frac = frac - np.round(frac)
                rows.append((c["song_id"], f_measure(times[i], truth)[0],
                             float(np.median(d)), float(np.median(frac)),
                             float(np.std(d)), ref))

    med = np.array([r[2] for r in rows])
    sd = np.array([r[4] for r in rows])
    lose = [r for r in rows if r[1] < 0.9]
    print(f"period-right songs: {len(rows)}  (of which F<0.9: {len(lose)})")
    print(f"\nper-song MEDIAN signed offset (s): mean {med.mean():+.4f}  "
          f"median {np.median(med):+.4f}  sd {med.std():.4f}")
    print(f"  |median offset| <= 70ms on {np.mean(np.abs(med) <= 0.07):.1%} of songs")
    print(f"per-song SD of offset within a song (s): median {np.median(sd):.4f}")
    print("\noffset histogram (fraction of a bar, folded to [-.5,.5]):")
    fr = np.array([r[3] for r in rows])
    h, edges = np.histogram(fr, bins=20, range=(-0.5, 0.5))
    for c, e in zip(h, edges):
        print(f"  {e:+.2f} {'#' * int(60 * c / max(h.max(), 1)):<60} {c}")
    lm = np.array([r[2] for r in lose])
    ls = np.array([r[4] for r in lose])
    print(f"\namong F<0.9: median offset {np.median(lm):+.4f}s  "
          f"within-song sd {np.median(ls):.4f}s")
    print(f"  systematic (|median|>70ms, sd<70ms): "
          f"{np.mean((np.abs(lm) > 0.07) & (ls < 0.07)):.1%}")
    print(f"  jittery    (sd>70ms):                {np.mean(ls > 0.07):.1%}")


if __name__ == "__main__":
    main()
