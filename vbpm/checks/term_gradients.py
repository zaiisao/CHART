"""Which term pushes the trajectory where.

Every loss term is a function of the same path phi. Perturb that path along the two
directions the metrics care about,

    phi'(theta, k) = k (phi - phi_1) + phi_1 + theta

and read each term's derivative at theta = 0, log k = 0. d/dtheta is the force on
PLACEMENT in nats per radian of bar phase; d/dlog k is the force on RATE. Signs are
reported as gradients of the OBJECTIVE (higher is better), so a term that wants the
path rotated later has positive d/dtheta.

The point is attribution: when the objective improves while placement degrades, this
says which term paid for it.
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
from ..observation import (annotation_frames, count_loglik, gauss_time_loglik,
                           interval_loglik)


def term_forces(model, phi, raw, mask, device, y=None, pos_weight: float = 1.0):
    """{term: (d/dtheta, d/dlogk)} in nats, evaluated at the model's own path."""
    ann_f, ann_valid = annotation_frames(raw, device)
    fps = float(raw["fps"][0])
    theta = torch.zeros((), device=device, requires_grad=True)
    logk = torch.zeros((), device=device, requires_grad=True)
    base = phi[:, :1].detach()
    path = torch.exp(logk) * (phi - base) + base + theta

    terms = {}
    if not getattr(model, "wants_raw", False):
        # the Bernoulli family: the emission is read at the deformed path directly
        terms["bernoulli_recon"] = model.recon_term(
            model.emission_logits(path, mask), y, mask, pos_weight)
        out = {}
        for name, value in terms.items():
            g_theta, g_logk = torch.autograd.grad(value.sum(), [theta, logk],
                                                  retain_graph=True, allow_unused=True)
            out[name] = (float(g_theta or 0.0), float(g_logk or 0.0))
        return out
    terms["gauss_time"] = gauss_time_loglik(path, ann_f, ann_valid, model.b_ratio,
                                            fps=fps, phase_half=model.phase_half)
    terms["count"] = count_loglik(path, ann_valid, mask)
    em = interval_loglik(path, ann_f, ann_valid, model.kappa_place, model.b_ratio,
                         model.phase_half, "huber", None, 0.0, "first", 0.0, mask)
    terms["interval:place"] = em["place"]
    terms["interval:ruler"] = em["interval"]

    out = {}
    for name, value in terms.items():
        g_theta, g_logk = torch.autograd.grad(value.sum(), [theta, logk],
                                              retain_graph=True, allow_unused=True)
        out[name] = (float(g_theta or 0.0), float(g_logk or 0.0))
    return out


OBJECTIVE_TERMS = ("recon", "kl", "kl_phase", "kl_rate", "tempo_prior",
                   "tempo_entropy", "kl_delta")


def parameter_forces(out, model):
    """Each objective term's gradient NORM on the parameters that actually get updated.

    The objective is recon - kl_phase + tempo_prior + tempo_entropy (signs as the model
    assembles them), so these norms say which term is writing the step, not merely which
    term is large in value.
    """
    # Per-head channels used to be broken out from encoder.out; that five-channel
    # amortized head was deleted with the rest of the regression posterior, so the
    # breakdown is now by the heads the variants actually own.
    heads = ("posterior_model.evidence_head", "posterior_model.rate_head",
             "emission_model")
    rows = {}
    for name in OBJECTIVE_TERMS:
        value = out.get(name)
        if value is None or not torch.is_tensor(value) or not value.requires_grad:
            continue
        params = [q for q in model.parameters() if q.requires_grad]
        grads = torch.autograd.grad(value.mean(), params,
                                    retain_graph=True, allow_unused=True)
        total = math.sqrt(sum(float((g ** 2).sum()) for g in grads if g is not None))
        per_channel = {}
        for tag in heads:
            sel = [q for n, q in model.named_parameters()
                   if n.startswith(tag) and q.requires_grad]
            if not sel:
                continue
            g = torch.autograd.grad(value.mean(), sel,
                                    retain_graph=True, allow_unused=True)
            per_channel[tag] = math.sqrt(
                sum(float((q ** 2).sum()) for q in g if q is not None))
        rows[name] = (total, per_channel)
    return rows


def main():
    """Report each objective term's force on placement and on rate."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="vbpm/configs/interval.yaml")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--dataset", default="ballroom")
    p.add_argument("--song", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--every", type=int, default=25)
    args = p.parse_args()

    cfg, hooks = load_config(args.config, args.set)
    cfg.epochs = args.epochs
    device = torch.device(f"cuda:{args.gpu}")
    frontend = importlib.import_module(f"vbpm.data.frontends.{cfg.frontend}").FRONTEND(
        checkpoint=cfg.frontend_checkpoint, device=f"cuda:{args.gpu}", output="features")
    song = sorted(sum(load_catalog([args.dataset]).values(), []),
                  key=lambda s: s.song_id)[args.song]
    data = ExcerptDataset([song], frontend, cfg.excerpt_seconds, deterministic=True,
                          target_tol_frames=getattr(cfg, "target_tol_frames", 0))
    raw = collate_excerpts([data[0]])
    with torch.no_grad():
        h = frontend.forward_features(raw["input"])
    mask, y = raw["mask"].to(device), raw["y"].to(device)
    dbs = np.asarray(raw["downbeat_times"][0])
    fps = float(raw["fps"][0])
    period = float(np.polyfit(np.arange(len(dbs)), dbs, 1)[0])

    torch.manual_seed(0)
    model = hooks.build_model(cfg, frontend.num_channels).to(device)
    opt, clip = hooks.optimizer(model, cfg)

    print(f"{song.song_id}  {len(dbs)} downbeats  period {period:.3f}s")
    print(f"{'ep':>4} {'in-tol':>7} {'mederr':>7} | term: d/dtheta  d/dlogk ...")
    for epoch in range(args.epochs + 1):
        hooks.on_epoch(model, cfg, epoch)
        extra = {"raw": raw} if getattr(model, "wants_raw", False) else {}
        out = model(h, mask, y, pos_weight=cfg.pos_weight, **extra)
        if epoch % args.every == 0:
            phi = out["phi"].detach().requires_grad_(True)
            forces = term_forces(model, phi, raw, mask, device, y, cfg.pos_weight)
            times = (torch.nonzero(torch.diff(
                torch.floor(out["phi"][0].detach() / (2 * math.pi))) > 0)[:, 0]
                .cpu().numpy() / fps + float(raw["t0"][0]))
            err = ([min(abs(d - times)) * 1000 for d in dbs] if len(times) else [9e9])
            cells = " ".join(f"{n}: {v[0]:+9.2f}{v[1]:+11.2f}"
                             for n, v in forces.items())
            print(f"{epoch:4d} {np.mean(np.asarray(err) <= 70):7.0%} "
                  f"{np.median(err):7.0f} | {cells}", flush=True)
            pf = parameter_forces(out, model)
            for name, (total, per_channel) in pf.items():
                bits = "  ".join(f"{c.replace('_logit','').replace('phase_','ph_')}"
                                 f" {v:.2e}" for c, v in per_channel.items())
                print(f"       |grad {name:14s}| total {total:.3e}   {bits}", flush=True)
        loss = -(hooks.objective(out, 1.0, cfg) / mask.sum(1).clamp(min=1.0)).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(clip, cfg.clip)
        opt.step()


if __name__ == "__main__":
    main()
