"""Is the wander the chain's own freedom? Asked at decode time.

chain_sigma was widened to let the tempo bend for rubato. This re-decodes a
trained checkpoint at other widths, so any change is inference alone.
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
    """Sweep the chain width and report F beside the within-song wander."""
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

    cache = []
    with torch.no_grad():
        for raw in loader:
            recs = scoring_records(raw)
            keep = [i for i, c in enumerate(recs) if c is not None]
            if keep:
                cache.append((frontend.forward_features(raw["input"]).clone(),
                              raw["mask"].to(device), keep, [recs[i] for i in keep]))

    trained = float(getattr(cfg, "chain_sigma", 0.03))
    print(f"trained at chain_sigma={trained}\n")
    print(f"{'sigma':>8} {'F all':>7} {'F bar<2.5':>10} {'F bar>3.0':>10} {'wander s':>9}")
    for sig in (0.003, 0.006, 0.01, 0.02, trained, 0.05):
        model.sigma = sig
        F, ref, wander = [], [], []
        with torch.no_grad():
            for h, mask, keep, crops in cache:
                mu = model.infer_phase(h, mask)[keep]
                times = rule_g_times(mu, mask[keep], crops)
                per = trajectory_period(mu, mask[keep], crops[0]["fps"])
                for i, c in enumerate(crops):
                    truth = np.asarray(c["downbeat_times"])
                    if len(truth) < 2:
                        continue
                    r = float(np.median(np.diff(truth)))
                    F.append(f_measure(times[i], truth)[0])
                    ref.append(r)
                    if len(times[i]) and 0.9 <= float(per[i]) / r <= 1.1:
                        d = [times[i][np.argmin(np.abs(times[i] - t))] - t for t in truth]
                        wander.append(float(np.std(d)))
        F, ref = np.array(F), np.array(ref)
        tag = " (trained)" if sig == trained else ""
        print(f"{sig:8.3f} {F.mean():7.4f} {F[ref < 2.5].mean():10.4f} "
              f"{F[ref > 3.0].mean():10.4f} {np.median(wander):9.4f}{tag}")


if __name__ == "__main__":
    main()
