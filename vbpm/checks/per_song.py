"""Per-song scores for a trained checkpoint, beside each song's tempo variability."""
from __future__ import annotations

import argparse
import csv
import importlib

import numpy as np
import torch

from ..config import load_config
from ..variants.base import load_model_state
from ..data.dataset import load_catalog
from ..data.excerpts import ExcerptDataset, collate_excerpts
from ..scoring.evaluation import (continuity_scores, f_measure, rule_g_times,
                                  scoring_records, trajectory_period)


def tempo_cv(crop) -> float:
    """Coefficient of variation of the bar intervals this song was annotated with."""
    d = np.diff(np.asarray(crop["anchors"], dtype=float))
    d = d[(d > 0.2) & (d < 8.0)]
    return float(np.std(d) / np.mean(d)) if len(d) >= 4 else float("nan")


def main():
    """One row per song: F, CMLt, AMLt, est/ref and the song's tempo variability."""
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", default="gtzan")
    p.add_argument("--gpu", type=int, default=3)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = f"cuda:{args.gpu}"
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg, hooks = load_config(blob["config_path"], list(blob.get("overrides", [])))
    frontend = importlib.import_module(f"vbpm.data.frontends.{cfg.frontend}").FRONTEND(
        checkpoint=cfg.frontend_checkpoint, device=device, output="features")
    frontend._audio2frames.model.load_state_dict(blob["frontend"])
    model = hooks.build_model(cfg, frontend.num_channels).to(device)
    load_model_state(model, blob["model"])
    model.eval()

    songs = sorted(sum(load_catalog([args.dataset]).values(), []), key=lambda s: s.song_id)
    data = ExcerptDataset(songs, frontend, cfg.excerpt_seconds, deterministic=True,
                          target_tol_frames=getattr(cfg, "target_tol_frames", 0))
    loader = torch.utils.data.DataLoader(data, batch_size=cfg.batch_size,
                                         collate_fn=collate_excerpts)

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
            times = rule_g_times(mu, mask[keep], crops)
            period = trajectory_period(mu, mask[keep], float(frontend.FPS))
            for j, crop in enumerate(crops):
                ref = np.asarray(crop["downbeat_times"], dtype=float)
                est = np.asarray(times[j], dtype=float)
                cml, aml = continuity_scores(est, ref)
                rows.append({"song_id": crop["song_id"], "dataset": crop["dataset"],
                             "F": f_measure(est, ref), "CMLt": cml, "AMLt": aml,
                             "n_ref": len(ref), "n_est": len(est),
                             "est_over_ref": (len(est) / len(ref)) if len(ref) else float("nan"),
                             "period": float(period[j]), "tempo_cv": tempo_cv(crop)})

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    f = np.array([r["F"] for r in rows])
    print(f"{len(rows)} songs -> {args.out}   F mean {f.mean():.4f}  sd {f.std():.4f}")


if __name__ == "__main__":
    main()
