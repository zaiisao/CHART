"""Per-song comparison against vanilla Beat This on the same excerpts.

Scores our read-out and the frontend's own system, song by song, on the
identical deterministic crop, so the difference is the read-out and nothing
else. Beat This here is its OWN minimal postprocessor, the system its paper
reports, not a peak picker we wrote.
"""
from __future__ import annotations

import argparse
import csv
import importlib

import numpy as np
import torch

from beat_this.model.postprocessor import Postprocessor

from ..config import load_config
from ..data.dataset import load_catalog
from ..data.excerpts import ExcerptDataset, collate_excerpts
from ..scoring.evaluation import (continuity_scores, f_measure, rule_g_times,
                                  scoring_records, trajectory_period)


def main():
    """Score both systems per song and write the joined table."""
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--dataset", default="gtzan")
    p.add_argument("--out", default="/tmp/scratch/compare_bt.csv")
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

    songs = sorted(sum(load_catalog([args.dataset]).values(), []), key=lambda s: s.song_id)
    data = ExcerptDataset(songs, frontend, cfg.excerpt_seconds, deterministic=True,
                          target_tol_frames=getattr(cfg, "target_tol_frames", 0))
    loader = torch.utils.data.DataLoader(data, batch_size=cfg.batch_size,
                                         collate_fn=collate_excerpts)
    post = Postprocessor(type="minimal", fps=int(frontend.FPS))

    rows = []
    with torch.no_grad():
        for raw in loader:
            records = scoring_records(raw)
            keep = [i for i, c in enumerate(records) if c is not None]
            if not keep:
                continue
            crops = [records[i] for i in keep]
            h = frontend.forward_features(raw["input"]).clone()
            mask = raw["mask"].to(device, non_blocking=True)

            mu = model.infer_phase(h, mask)[keep]
            ours = rule_g_times(mu, mask[keep], crops)
            period = trajectory_period(mu, mask[keep], float(raw["fps"][0]))

            heads = frontend._audio2frames.model.task_heads(h)
            bt_beat, bt_db = post(heads["beat"][keep], heads["downbeat"][keep],
                                  mask[keep].bool())

            for i, crop in enumerate(crops):
                truth = np.asarray(crop["downbeat_times"])
                t0, fps = crop["t0"], crop["fps"]
                theirs = np.asarray(bt_db[i], dtype=np.float64) + t0
                theirs = theirs[theirs < t0 + len(crop["y"]) / fps]
                f_ours = f_measure(ours[i], truth)[0]
                f_bt = f_measure(theirs, truth)[0]
                c_ours, a_ours = continuity_scores(truth, ours[i])
                c_bt, a_bt = continuity_scores(truth, theirs)
                ref = float(np.median(np.diff(truth))) if len(truth) > 1 else float("nan")
                rows.append(dict(
                    song=f"{crop['dataset']}:{crop['song_id']}",
                    f_ours=f_ours, f_bt=f_bt, delta=f_ours - f_bt,
                    cmlt_ours=c_ours, cmlt_bt=c_bt, amlt_ours=a_ours, amlt_bt=a_bt,
                    n_ours=len(ours[i]), n_bt=len(theirs), n_ref=len(truth),
                    ref_bar=ref, est_bar=float(period[i]),
                    ratio_ours=len(ours[i]) / max(len(truth), 1),
                    ratio_bt=len(theirs) / max(len(truth), 1)))

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    d = np.array([r["delta"] for r in rows])
    print(f"n={len(rows)}  ours {np.mean([r['f_ours'] for r in rows]):.4f}  "
          f"beat-this {np.mean([r['f_bt'] for r in rows]):.4f}  delta {d.mean():+.4f}")
    print(f"we lose on {int((d < -0.05).sum())} songs, win on {int((d > 0.05).sum())}, "
          f"tie {int((abs(d) <= 0.05).sum())}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
