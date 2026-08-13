"""Control: can the model fit ONE song, with everything else removed?"""
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
from ..model import downbeat_frames


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="phasevae/configs/baseline.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--dataset", default="ballroom")
    p.add_argument("--song", type=int, default=0, help="index into the sorted catalog")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--every", type=int, default=25)
    p.add_argument("--gpu", type=int, default=1, choices=(0, 1, 2, 3))
    p.add_argument("--plot", default="/tmp/overfit_emission.png",
                   help='PNG of emission vs ground truth at each snapshot; "" disables')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg, hooks = load_config(args.config, args.set)
    cfg.epochs = args.epochs
    device = torch.device(f"cuda:{args.gpu}")

    catalog = sorted(sum(load_catalog([args.dataset]).values(), []),
                     key=lambda s: s.song_id)
    song = catalog[args.song]

    frontend_module = importlib.import_module(f"phasevae.data.frontends.{cfg.frontend}")
    frontend = frontend_module.FRONTEND(checkpoint=cfg.frontend_checkpoint,
                                        device=f"cuda:{args.gpu}", output="features")

    # deterministic: the window must be IDENTICAL every epoch or this is not overfitting.
    dataset = ExcerptDataset([song], frontend, cfg.excerpt_seconds, deterministic=True)
    raw = collate_excerpts([dataset[0]])

    downbeats = np.asarray(raw["downbeat_times"][0])
    fps = float(raw["fps"][0])
    assert len(downbeats) >= 4, f"{song.song_id}: only {len(downbeats)} downbeats"

    k = np.arange(len(downbeats))
    period_s = np.polyfit(k, downbeats, 1)[0]          # slope = seconds per bar
    true_dotphi = 2.0 * math.pi / (period_s * fps)
    targets = (downbeats - float(raw["t0"][0])) * fps

    print(f"song {song.song_id}  {len(downbeats)} downbeats  "
          f"period {period_s:.3f}s  true tempo {true_dotphi:.4f} rad/frame",
          flush=True)

    torch.manual_seed(0)
    model = hooks.build_model(cfg, frontend.num_channels).to(device)
    opt, clip_params = hooks.optimizer(model, cfg)

    # ONCE: the frontend is frozen, so its features never change. Recomputing them per
    # epoch is pure waste and drags 20M parameters into the graph.
    with torch.no_grad():
        h = frontend.forward_features(raw["input"])
    mask = raw["mask"].to(device)
    y = raw["y"].to(device)
    frames = mask.sum(1).clamp(min=1.0)
    snapshots = []

    for epoch in range(args.epochs):
        model.train()
        hooks.on_epoch(model, cfg, epoch)
        out = model(h, mask, y, samples=cfg.samples, pos_weight=cfg.pos_weight)
        loss = -(hooks.objective(out, run_mod.beta_at(epoch, cfg), cfg) / frames).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(clip_params, cfg.clip)
        opt.step()

        if epoch % args.every and epoch != args.epochs - 1:
            continue

        model.eval()          # infer_phase asserts eval mode; it is the deployed path
        with torch.no_grad():
            mu_t = model.infer_phase(h, mask)
            # THE conversion (model.downbeat_frames): a downbeat is where phase crosses
            # ZERO. Hand-rolling this from the +-pi DISPLAY discontinuity puts every
            # wrap half a bar out -- the markers then land in the emission's troughs.
            wraps = np.flatnonzero(downbeat_frames(mu_t, mask)[0].cpu().numpy())
        errs = np.array([abs(w - targets[np.argmin(abs(targets - w))]) / fps * 1000.0
                         for w in wraps]) if len(wraps) else np.array([1e9])
        tempo = float((out["mu"][0, 1:] - out["mu"][0, :-1]).mean())
        # res = the normalised resultant of the phase-folded evidence, in [0, 1]: how
        # much the bars agree about where the downbeat is. On ONE song there is no
        # generalisation to fail, so if this does not grow the anchor mechanism itself
        # is broken rather than merely untrained.
        res = f"  res {float(out['resultant'].mean()):5.3f}" if "resultant" in out else ""
        if args.plot:
            phi_w = torch.atan2(torch.sin(mu_t[0]), torch.cos(mu_t[0]))
            prox = (1.0 - phi_w.abs() / math.pi).float().cpu().numpy()
            snapshots.append((epoch, prox, wraps, tempo / true_dotphi))

        print(f"  ep {epoch:4d}  recon {float(out['recon'].mean()):9.2f}  "
              f"kl {float(out['kl'].mean()):9.2f}  b {float(model.emission_b):5.2f}  "
              f"tempo {tempo:.4f} (ratio {tempo / true_dotphi:5.2f})  "
              f"med|err| {np.median(errs):6.0f}ms  in-tol {np.mean(errs < 70.0):4.0%}"
              f"{res}",
              flush=True)

    if args.plot:
        _render(args.plot, snapshots, y[0].cpu().numpy(), targets, song.song_id)


def _render(path, snapshots, y, targets, song_id):
    """One panel per snapshot: PHASE PROXIMITY 1 - |wrap(mu)|/pi vs the annotated downbeats."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(snapshots)
    fig, axes = plt.subplots(n, 1, figsize=(15, 1.5 * n), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (epoch, probs, wraps, ratio) in zip(axes, snapshots):
        ax.fill_between(np.arange(len(y)), 0, y, color="0.85", step="mid",
                        label="y (target)")
        ax.plot(probs, lw=.8, color="C0", label="phase proximity 1-|wrap(mu)|/pi")
        for t in targets:
            ax.axvline(t, color="r", lw=.7, alpha=.55)
        if len(wraps):
            ax.plot(wraps, np.full(len(wraps), probs.max() * .95), "kv", ms=3,
                    label="model wraps")
        ax.set_ylabel(f"ep {epoch}\nratio {ratio:.2f}", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0].legend(fontsize=6, loc="upper right", ncol=3)
    axes[-1].set_xlabel("frame")
    fig.suptitle(f"bar phase vs ground truth -- {song_id}  (red = annotated downbeats)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
