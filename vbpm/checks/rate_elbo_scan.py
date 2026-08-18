"""Does the ELBO prefer the wrong period, or did the optimiser miss the right one?

Trains one song, then re-scores it with the tempo posterior PINNED to each grid
rate in turn. If the ELBO peaks away from the true rate the objective is at
fault; if it peaks at truth the search is.
"""
from __future__ import annotations

import argparse
import importlib
import math

import numpy as np
import torch

from .. import run as run_mod
from ..config import load_config
from ..data.dataset import load_catalog
from ..data.excerpts import ExcerptDataset, collate_excerpts
from ..scoring.evaluation import f_measure, rule_g_times, scoring_records


def main():
    """Train one song, then sweep the ELBO over pinned tempo bins."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="vbpm/configs/tchain.yaml")
    p.add_argument("--dataset", default="asap")
    p.add_argument("--song", type=int, default=343)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    dev = f"cuda:{args.gpu}"
    cfg, hooks = load_config(args.config, [])
    fe = importlib.import_module(f"vbpm.data.frontends.{cfg.frontend}").FRONTEND(
        checkpoint=cfg.frontend_checkpoint, device=dev, output="features")
    songs = sorted(sum(load_catalog([args.dataset]).values(), []), key=lambda s: s.song_id)
    ds = ExcerptDataset([songs[args.song]], fe, cfg.excerpt_seconds, deterministic=True,
                        target_tol_frames=getattr(cfg, "target_tol_frames", 0))
    raw = collate_excerpts([ds[0]])
    with torch.no_grad():
        h = fe.forward_features(raw["input"]).detach()
    mask, y = raw["mask"].to(dev), raw["y"].to(dev)
    crops = scoring_records(raw)
    truth = np.asarray(crops[0]["downbeat_times"])
    true_bar = float(np.median(np.diff(truth)))
    fps = crops[0]["fps"]

    model = hooks.build_model(cfg, fe.num_channels).to(dev)
    opt, _ = hooks.optimizer(model, cfg)
    for ep in range(args.epochs):
        hooks.on_epoch(model, cfg, ep)
        out = model(h, mask, y, pos_weight=cfg.pos_weight)
        loss = -(hooks.objective(out, run_mod.beta_at(ep, cfg), cfg) / mask.sum()).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()

    saved = model.tempo_log_prior.clone()
    print(f"song {crops[0]['song_id']}   true bar {true_bar:.3f}s "
          f"({2 * math.pi / (true_bar * fps):.4f} rad/frame)")
    print(f"\n{'bin':>4} {'bar s':>7} {'ratio':>6} {'ELBO':>10} {'recon':>10} "
          f"{'KL':>8} {'F':>6}")
    rows = []
    for i in range(model.tempo_bins):
        lp = torch.full_like(saved, -1e9)
        lp[i] = 0.0
        model.tempo_log_prior.copy_(lp - torch.logsumexp(lp, 0))
        with torch.no_grad():
            o = model(h, mask, y, pos_weight=cfg.pos_weight)
            mu = model.infer_phase(h, mask)
            F = f_measure(rule_g_times(mu, mask, crops)[0], truth)[0]
        bar = 2 * math.pi / (float(model.rates[i]) * fps)
        rows.append((i, bar, bar / true_bar, float(o["elbo"]), float(o["recon"]),
                     float(o["kl"]), F))
        print(f"{i:4d} {bar:7.3f} {bar / true_bar:6.2f} {float(o['elbo']):10.2f} "
              f"{float(o['recon']):10.2f} {float(o['kl']):8.2f} {F:6.3f}")
    model.tempo_log_prior.copy_(saved)

    best_e = max(rows, key=lambda r: r[3])
    best_f = max(rows, key=lambda r: r[6])
    near = min(rows, key=lambda r: abs(r[2] - 1.0))
    print(f"\nELBO peaks at bin {best_e[0]} (bar {best_e[1]:.3f}s, ratio {best_e[2]:.2f}, "
          f"F {best_e[6]:.3f})")
    print(f"F    peaks at bin {best_f[0]} (bar {best_f[1]:.3f}s, ratio {best_f[2]:.2f}, "
          f"F {best_f[6]:.3f})")
    print(f"true-rate bin   {near[0]} (bar {near[1]:.3f}s, ratio {near[2]:.2f}, "
          f"F {near[6]:.3f}, ELBO {near[3]:.2f})")
    print(f"\nELBO cost of the truth: {best_e[3] - near[3]:+.2f} nats  "
          f"(positive => the objective PREFERS the wrong period)")


if __name__ == "__main__":
    main()
