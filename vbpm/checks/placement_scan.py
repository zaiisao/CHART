"""Where does the objective put the downbeat, and where is it actually?

For songs whose annotations are perfectly uniform the walk prior charges nothing, so a
low score cannot be blamed on the dynamics. This trains one song, then rotates the
converged path through a whole bar and reads two curves: the reconstruction term, and
the in-tolerance rate. If they peak at different rotations, the emission's optimum is
not the annotation -- a bias, not a search failure. If they peak together but the model
sits elsewhere, the search is at fault.
"""
from __future__ import annotations

import argparse
import importlib
import math

import numpy as np
import torch

from ..config import load_config
from ..data.dataset import load_catalog
from ..data.excerpts import ExcerptDataset, collate_excerpts


def main():
    """Train one song, then rotate the converged path through a whole bar."""
    p = argparse.ArgumentParser()
    p.add_argument("song", help="dataset:index")
    p.add_argument("--config", default="vbpm/configs/rate_grid.yaml")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--grid", type=int, default=240)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    cfg, hooks = load_config(args.config, args.set)
    cfg.epochs = args.epochs
    device = torch.device(f"cuda:{args.gpu}")
    name, index = args.song.split(":")
    frontend = importlib.import_module(f"vbpm.data.frontends.{cfg.frontend}").FRONTEND(
        checkpoint=cfg.frontend_checkpoint, device=f"cuda:{args.gpu}", output="features")
    song = sorted(sum(load_catalog([name]).values(), []),
                  key=lambda s: s.song_id)[int(index)]
    data = ExcerptDataset([song], frontend, cfg.excerpt_seconds, deterministic=True,
                          target_tol_frames=getattr(cfg, "target_tol_frames", 0))
    raw = collate_excerpts([data[0]])
    with torch.no_grad():
        h = frontend.forward_features(raw["input"])
    mask, y = raw["mask"].to(device), raw["y"].to(device)
    truth = np.asarray(raw["downbeat_times"][0])
    fps, t0 = float(raw["fps"][0]), float(raw["t0"][0])
    period = float(np.median(np.diff(np.asarray(raw["anchors"][0], dtype=np.float64))))

    torch.manual_seed(0)
    model = hooks.build_model(cfg, frontend.num_channels).to(device)
    opt, clip = hooks.optimizer(model, cfg)
    frames = mask.sum(1).clamp(min=1.0)
    for epoch in range(args.epochs):
        hooks.on_epoch(model, cfg, epoch)
        extra = {"raw": raw} if getattr(model, "wants_raw", False) else {}
        out = model(h, mask, y, samples=cfg.samples, pos_weight=cfg.pos_weight, **extra)
        loss = -(hooks.objective(out, 1.0, cfg) / frames).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(clip, cfg.clip)
        opt.step()

    model.eval()
    with torch.no_grad():
        phi = model.infer_phase(h, mask)[0]
        rows = []
        for shift in np.linspace(-math.pi, math.pi, args.grid):
            path = phi + float(shift)
            recon = float(model.recon_term(
                model.emission_logits(path[None], mask), y, mask, cfg.pos_weight))
            wraps = torch.nonzero(torch.diff(torch.floor(path / (2 * math.pi))) > 0)[:, 0]
            times = wraps.cpu().numpy() / fps + t0
            err = ([min(abs(d - times)) * 1000 for d in truth] if len(times) else [9e9])
            rows.append((float(shift), recon, float(np.mean(np.asarray(err) <= 70)),
                         float(np.median(err))))
    a = np.array(rows)
    ms = a[:, 0] / (2 * math.pi) * period * 1000.0
    i_recon, i_intol = int(a[:, 1].argmax()), int(a[:, 2].argmax())
    i_zero = int(np.abs(a[:, 0]).argmin())
    print(f"{song.song_id}  bar {period:.3f}s  {len(truth)} downbeats")
    print(f"   model sits at            shift    0 ms   recon {a[i_zero,1]:9.2f}  "
          f"in-tol {a[i_zero,2]:5.0%}  med|err| {a[i_zero,3]:4.0f} ms")
    print(f"   recon peaks at           shift {ms[i_recon]:+5.0f} ms   recon {a[i_recon,1]:9.2f}  "
          f"in-tol {a[i_recon,2]:5.0%}  med|err| {a[i_recon,3]:4.0f} ms")
    print(f"   in-tolerance peaks at    shift {ms[i_intol]:+5.0f} ms   recon {a[i_intol,1]:9.2f}  "
          f"in-tol {a[i_intol,2]:5.0%}  med|err| {a[i_intol,3]:4.0f} ms")
    print(f"   recon cost of moving to the best placement: "
          f"{a[i_recon,1] - a[i_intol,1]:.2f} nats")


if __name__ == "__main__":
    main()
