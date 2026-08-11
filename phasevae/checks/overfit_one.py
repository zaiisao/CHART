"""Control: can the model fit ONE song, with everything else removed?

The cheapest decisive test in the project. One song, one window pinned by
``deterministic=True``, one batch reused every step, frontend features computed once
because the frontend is frozen. Nothing here can fail for reasons of data volume,
crop noise, tempo spread or generalisation -- so if the numbers do not converge, the
failure is structural and no amount of data or objective redesign will reach it.

Read four numbers:
  * recon   -- if this does not fall on a SINGLE song, nothing is learning.
  * rate    -- against the song's own bar rate, printed as a ratio. 1.00 is the target.
                Sitting exactly on 0.0100 is the LOWER CLAMP in Encoder.heads, whose
                gradient is exactly zero: a railed rate can never return.
  * b       -- the emission amplitude. Falling means the likelihood is flattening; the
                broad-emission/imprecise-phase equilibrium (see emission_b_floor).
  * in-tol  -- wraps within +-70 ms of an annotated downbeat, i.e. F for this song.

First run of this check (2026-08-11, ballroom song 0, 400 epochs): recon -340.29 ->
-336.87, rate railed to the 0.0100 clamp by epoch 50 and pinned there with the KL frozen
at 3465.15 to six figures for 350 epochs, b decaying 1.31 -> 1.23, in-tol 0%. The model
could not fit one song.

anchor_k.yaml fails IDENTICALLY -- same clamp, same frozen KL to six figures, same b
trace -- because kl_k = log C + sum q log q is exactly 0 while q(k) is uniform, which it
was throughout. So the defect is in the SHARED core (Encoder.heads, BarPhaseVAE.forward),
not in any variant. baseline is the default here for that reason: it is the smallest
configuration that reproduces the failure.

Run: PYTHONPATH=. python -m phasevae.checks.overfit_one [--epochs 400] [--gpu 1]
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
    true_rate = 2.0 * math.pi / (period_s * fps)
    targets = (downbeats - float(raw["t0"][0])) * fps

    print(f"song {song.song_id}  {len(downbeats)} downbeats  "
          f"period {period_s:.3f}s  true rate {true_rate:.4f} rad/frame",
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
    tk = raw["targets"].to(device)
    tv = raw["valid"].to(device)
    frames = mask.sum(1).clamp(min=1.0)
    snapshots = []

    for epoch in range(args.epochs):
        model.train()
        out = model(h, mask, y, samples=cfg.samples, pos_weight=cfg.pos_weight,
                    targets=tk, valid=tv,
                    anchor_penalty=cfg.anchor_penalty,
                    anchor_kappa=cfg.anchor_kappa)
        loss = -(hooks.objective(out, cfg.beta_end, cfg) / frames).mean()

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
        rate = float((out["mu"][0, 1:] - out["mu"][0, :-1]).mean())
        # res = the normalised resultant of the phase-folded evidence, in [0, 1]: how
        # much the bars agree about where the downbeat is. On ONE song there is no
        # generalisation to fail, so if this does not grow the anchor mechanism itself
        # is broken rather than merely untrained.
        res = f"  res {float(out['resultant'].mean()):5.3f}" if "resultant" in out else ""
        if args.plot:
            # Phase proximity, NOT emission_probs. The tent is affine in (a, b), so
            # normalising to [0, 1] deletes them -- which is what we want: under the
            # alignment objective nothing updates a or b, so sigmoid(a + b tent) is two
            # frozen init values (peak 0.156, trough 0.013) dressed up as a prediction,
            # and the sigmoid's convexity over that range rounds the tent's corner into
            # a scoop. 1 - |wrap(mu)|/pi is the same information, parameter-free, and
            # actually pointy at the downbeat.
            phi_w = torch.atan2(torch.sin(mu_t[0]), torch.cos(mu_t[0]))
            prox = (1.0 - phi_w.abs() / math.pi).float().cpu().numpy()
            snapshots.append((epoch, prox, wraps, rate / true_rate))

        print(f"  ep {epoch:4d}  recon {float(out['recon'].mean()):9.2f}  "
              f"kl {float(out['kl'].mean()):9.2f}  b {float(model.emission_b):5.2f}  "
              f"rate {rate:.4f} (ratio {rate / true_rate:5.2f})  "
              f"med|err| {np.median(errs):6.0f}ms  in-tol {np.mean(errs < 70.0):4.0%}"
              f"{res}",
              flush=True)


    if args.plot:
        _render(args.plot, snapshots, y[0].cpu().numpy(), targets, song.song_id)


def _render(path, snapshots, y, targets, song_id):
    """One panel per snapshot: PHASE PROXIMITY 1 - |wrap(mu)|/pi vs the annotated downbeats.

    What to look for: a train of cusps whose spacing matches the red lines, moving onto
    them over epochs. Cusps at the wrong spacing = the rate is wrong. Cusps EVENLY spaced
    on a song with rubato = the trajectory has been straightened, which is what the KL
    does once kappa rails and it starts to dominate the alignment term.
    """
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
