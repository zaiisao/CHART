"""Control: can the model fit ONE song, with everything else removed?"""
from __future__ import annotations

import argparse
import importlib
import math
import pathlib

import numpy as np
import torch

from .. import run as run_mod
from ..config import load_config
from ..variants.base import emission_of
from ..data.dataset import load_catalog
from ..data.excerpts import ExcerptDataset, collate_excerpts
from ..readout import downbeat_times
from ..scoring.evaluation import continuity_scores, f_measure


def parse_args():
    """The command line for this check."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="vbpm/configs/baseline.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--dataset", default="ballroom")
    p.add_argument("--song", type=int, default=0, help="index into the sorted catalog")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--every", type=int, default=25)
    p.add_argument("--gpu", type=int, default=1, choices=(0, 1, 2, 3))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grad-log", default="")
    p.add_argument("--kl-only", action="store_true")
    p.add_argument("--pin-gain", action="store_true")
    p.add_argument("--lr-drop", type=int, default=0)
    p.add_argument("--lr-anneal", type=float, default=0.0)
    p.add_argument("--acf-init", action="store_true")
    p.add_argument("--truncate-sec", type=float, default=0.0,
                   help="keep only the first N seconds of the crop: mask the rest and "
                        "drop its annotations, so a suspect passage can be excluded "
                        "without moving the window's start")
    p.add_argument("--thaw", type=float, default=0.0,
                   help="lr scale for the frontend; 0 keeps it frozen (the default)")
    p.add_argument("--save", default="", help="path to write the trained state_dict")
    p.add_argument("--plot", default=None,
                   help="PNG of phase vs ground truth at each snapshot. A plot is "
                        "ALWAYS written -- pass a path to choose where, or omit for an "
                        "auto-named one under logs/. It cannot be disabled: the numbers "
                        "alone have repeatedly hidden what the panels make obvious "
                        "(rate-vs-offset failures look identical in med|err|).")
    return p.parse_args()


def resolve_plot_path(args) -> str:
    """A plot is mandatory. Empty/none resolves to an auto-named path under logs/."""
    if args.plot:
        return args.plot
    stamp = f"{args.dataset}{args.song}_s{args.seed}"
    out = pathlib.Path("logs/overfit") / f"{stamp}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    return str(out)


def main() -> None:
    """Fit ONE song and report what the trajectory does."""
    args = parse_args()
    args.plot = resolve_plot_path(args)
    cfg, hooks = load_config(args.config, args.set)
    assert not getattr(cfg, "clip_per_group", False), \
        "overfit_one clips pooled only; clip_per_group configs are not mirrored here"
    assert getattr(cfg, "frontend_lr_scale", 0.0) == 0.0 or args.thaw > 0.0, \
        "overfit_one freezes the frontend; frontend_lr_scale configs are not mirrored here"
    shipped_epochs = cfg.epochs
    cfg.epochs = args.epochs
    if hasattr(cfg, "dec_warmup") and args.epochs != shipped_epochs:
        cfg.dec_warmup = round(cfg.dec_warmup * args.epochs / shipped_epochs)
    device = torch.device(f"cuda:{args.gpu}")

    catalog = sorted(sum(load_catalog([args.dataset]).values(), []),
                     key=lambda s: s.song_id)
    song = catalog[args.song]

    frontend_module = importlib.import_module(f"vbpm.data.frontends.{cfg.frontend}")
    frontend = frontend_module.FRONTEND(checkpoint=cfg.frontend_checkpoint,
                                        device=f"cuda:{args.gpu}", output="features")

    # deterministic: the window must be IDENTICAL every epoch or this is not overfitting.
    dataset = ExcerptDataset([song], frontend, cfg.excerpt_seconds, deterministic=True,
                             target_tol_frames=getattr(cfg, "target_tol_frames", 0))
    raw = collate_excerpts([dataset[0]])

    if args.truncate_sec > 0.0:
        fps_ = float(raw["fps"][0])
        keep = int(round(args.truncate_sec * fps_))
        raw["mask"][0, keep:] = 0.0
        raw["y"][0, keep:] = 0.0
        cut = float(raw["t0"][0]) + args.truncate_sec
        for field in ("downbeat_times", "anchors"):
            kept = np.asarray(raw[field][0])
            raw[field][0] = kept[kept <= cut]
        print(f"truncated to {args.truncate_sec:.2f}s: "
              f"{len(raw['downbeat_times'][0])} downbeats kept", flush=True)

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

    torch.manual_seed(args.seed)
    model = hooks.build_model(cfg, frontend.num_channels).to(device)
    assert not (args.pin_gain and getattr(model, "wants_raw", False)), \
        "--pin-gain pins the Bernoulli emission gain, which the interval recipe never reads"

    opt, clip_params = hooks.optimizer(model, cfg)
    if args.pin_gain:
        with torch.no_grad():
            emission_of(model).b_raw.fill_(2.0)
        emission_of(model).a.requires_grad_(False)
        emission_of(model).b_raw.requires_grad_(False)
    # ONCE: the frontend is frozen, so its features never change. Recomputing them per
    # epoch is pure waste and drags 20M parameters into the graph. --thaw gives that up:
    # the features are then a function of parameters under training, so they move.
    fe_params = list(frontend._audio2frames.model.parameters())
    if args.thaw > 0.0:
        opt.add_param_group({"params": fe_params, "lr": cfg.lr * args.thaw})
        print(f"frontend THAWED: {sum(q.numel() for q in fe_params)/1e6:.1f}M params "
              f"at lr {cfg.lr * args.thaw:.2e}", flush=True)
    with torch.no_grad():
        h = frontend.forward_features(raw["input"])
    mask = raw["mask"].to(device)
    if args.acf_init:
        from ..data.tempo import estimate_bar_period
        with torch.no_grad():
            acts = frontend._audio2frames.model.task_heads(h)
            act = torch.sigmoid(acts["downbeat"])
            period_est = float(estimate_bar_period(act, mask, fps)[0])
            assert hasattr(model, "init_rate_prior"), \
                f"{cfg.variant} has no init_rate_prior: --acf-init cannot be honoured"
            model.init_rate_prior(2.0 * math.pi / (period_est * fps))
        print(f"acf-init: bar period {period_est:.3f}s "
              f"(birth ratio {period_s / period_est:.2f})", flush=True)
    y = raw["y"].to(device)
    frames = mask.sum(1).clamp(min=1.0)
    snapshots = []

    for epoch in range(args.epochs):
        model.train()
        hooks.on_epoch(model, cfg, epoch)
        if args.lr_drop and epoch == args.lr_drop:
            for g in opt.param_groups:
                g["lr"] *= 0.1
        if args.lr_anneal and epoch == cfg.sharpness_warmup:
            for g in opt.param_groups:
                g["lr"] *= args.lr_anneal
        if args.pin_gain:
            em = emission_of(model)
            em.a.data.fill_(2.2 - float(em.b))
        if args.thaw > 0.0:
            h = frontend.forward_features(raw["input"])
        extra = {"raw": raw} if getattr(model, "wants_raw", False) else {}
        out = model(h, mask, y, pos_weight=cfg.pos_weight, **extra)
        if args.kl_only:
            loss = (out["kl"] / frames).mean()
        else:
            loss = -(hooks.objective(out, run_mod.beta_at(epoch, cfg), cfg) / frames).mean()

        opt.zero_grad()
        loss.backward()

        if args.grad_log:
            groups = {"evidence": "posterior_model.evidence_head",
                      "rate_head": "posterior_model.rate_head",
                      "emission": "emission_model",
                      "enc_trunk": "posterior_model.encoder.blocks"}
            norms = {}
            for tag, prefix in groups.items():
                tot = 0.0
                for name, prm in model.named_parameters():
                    if prm.grad is not None and name.startswith(prefix):
                        tot += float(prm.grad.pow(2).sum())
                norms[tag] = tot ** 0.5
            with open(args.grad_log, "a") as fh:
                if epoch == 0:
                    fh.write("epoch,loss,recon,kl,d_absmean,d_sd,"
                             + ",".join(f"g_{k}" for k in groups) + "\n")
                fh.write(f"{epoch},{float(loss):.6f},"
                         f"{float(out['recon'].mean()):.4f},{float(out['kl'].mean()):.4f},"
                         f"{float(out.get('delta', torch.zeros(1)).abs().mean()):.6f},"
                         f"{float(out.get('delta', torch.zeros(1)).std()):.6f},"
                         + ",".join(f"{norms[k]:.6e}" for k in groups) + "\n")

        torch.nn.utils.clip_grad_norm_(clip_params, cfg.clip)
        if args.thaw > 0.0:
            torch.nn.utils.clip_grad_norm_(fe_params, cfg.clip)
        opt.step()

        if epoch % args.every and epoch != args.epochs - 1:
            continue

        model.eval()          # infer_phase asserts eval mode; it is the deployed path
        with torch.no_grad():
            mu_t = model.infer_phase(h, mask)
            # THE conversion (model.downbeat_times): a downbeat is where phase crosses
            # ZERO, at the interpolated sub-frame instant. Hand-rolling this from the
            # +-pi DISPLAY discontinuity puts every wrap half a bar out.
            wraps = downbeat_times(mu_t, mask)[0].cpu().numpy()
        errs = np.array([abs(w - targets[np.argmin(abs(targets - w))]) / fps * 1000.0
                         for w in wraps]) if len(wraps) else np.array([1e9])
        fsc = f_measure(wraps / fps, targets / fps)[0] if len(wraps) else 0.0
        cmlt, amlt = continuity_scores(targets / fps, wraps / fps)
        inc = mu_t[0, 1:] - mu_t[0, :-1]
        step_ok = ((mask[0, 1:] > 0) & (mask[0, :-1] > 0)).to(inc.dtype)
        tempo = float((inc * step_ok).sum() / step_ok.sum().clamp(min=1.0))
        # res = the normalised resultant of the phase-folded evidence, in [0, 1]: how
        # much the bars agree about where the downbeat is. On ONE song there is no
        # generalisation to fail, so if this does not grow the anchor mechanism itself
        # is broken rather than merely untrained.
        res = f"  res {float(out['resultant'].mean()):5.3f}" if "resultant" in out else ""
        bw = ""
        if True:
            phi_w = torch.atan2(torch.sin(mu_t[0]), torch.cos(mu_t[0]))
            prox = (1.0 - phi_w.abs() / math.pi).float().cpu().numpy()
            snapshots.append((epoch, prox, wraps, tempo / true_dotphi))

        print(f"  ep {epoch:4d}  recon {float(out['recon'].mean()):9.2f}  "
              f"kl {float(out['kl'].mean()):9.2f}  "
              f"b {float(emission_of(model).b):5.2f}  "
              f"tempo {tempo:.4f} (ratio {tempo / true_dotphi:5.2f})  "
              f"med|err| {np.median(errs):6.0f}ms  in-tol {np.mean(errs < 70.0):4.0%}"
              f"  F {fsc:.3f} CMLt {cmlt:.3f} AMLt {amlt:.3f}  n{len(wraps)}/{len(targets)}"
              f"{bw}{res}",
              flush=True)

    if args.save:
        torch.save(model.state_dict(), args.save)
        print(f"wrote {args.save}", flush=True)
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
