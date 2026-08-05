"""Warm-start the encoder from supervision, then train the ELBO. Does it hold or slide?

The landscape sweep showed a clean path from random init to a working encoder -- the ELBO
improves by 88 nats/frame along it and val F rises 0.38 -> 0.79 -- while the gradient at
initialisation is orthogonal to that path (cosine 0.0006, versus 0.0013 for a random
direction in 561k dimensions). So the objective prefers the right answer and cannot find
it from a cold start.

This asks the follow-up. Put the encoder AT the good solution and hand it to the ELBO:

  * if F HOLDS, the objective is sound and the whole failure was initialisation. A
    supervised warm start plus ELBO fine-tuning is then the working recipe.
  * if F SLIDES BACK toward the ~0.07 the cold runs reach, the ELBO is destroying a
    solution it also scores 88 nats/frame better -- a far stranger and more serious
    statement about the objective than anything measured so far.

The emission is fitted to the warm-started trajectory before the ELBO starts, so the model
begins self-consistent rather than handing the optimiser a large first-step correction.
``emission_b`` is reported every epoch: it decays 1.35 -> 0.91 in the cold runs, and a flat
emission is the regime where a constant-rate metronome beats the true phase.

    PYTHONPATH=. python -m phasevae.check_warm_start --gpu 1
"""
from __future__ import annotations

import argparse

import mir_eval
import numpy as np
import torch

from vbpm.data import FPS

from .check_encoder_supervised import collate_targets, phase_targets, wrap_readout

from .model import BarPhaseVAE
from .run import VAL_FOLD, Batches, load_or_build


def val_f(model, loader):
    """(F, CMLt, AMLt, est/ref count ratio) on validation, deployment path (h only).

    AMLt is a DIAGNOSTIC, never a target: it accepts the metrical variants (double, half,
    offbeat), so F low with AMLt high means the model is making a genuine metrical mistake
    -- right rate, wrong phase -- while both low means it is producing noise. The count
    ratio separates those further: ~1.0 says the rate is right whatever the phase does.
    """
    model.eval()
    rows = []
    with torch.no_grad():
        for raw, batch in loader():
            wraps = (wrap_readout(model.infer_phase(batch["h"]))
                     & (batch["mask"][:, 1:] > 0)).cpu().numpy()
            for i, crop in enumerate(raw):
                t = len(crop["y"])
                est = (np.flatnonzero(wraps[i, :t - 1]) + 1) / FPS + crop["t0"]
                ref = np.asarray(crop["downbeat_times"])
                if len(ref) < 2 or len(est) < 2:
                    rows.append((0.0, 0.0, 0.0, len(est) / max(len(ref), 1)))
                    continue
                _cc, _ac, cmlt, amlt = mir_eval.beat.continuity(ref, est)
                rows.append((mir_eval.beat.f_measure(ref, est, f_measure_threshold=0.07),
                             cmlt, amlt, len(est) / len(ref)))
    return tuple(np.array(rows).mean(0))


def supervise(model, loader, rng, epochs, lr):
    """Fit the encoder to the true phase. Circular loss; never a squared angle error."""
    opt = torch.optim.Adam(model.encoder.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for raw, batch in loader(shuffle=True, rng=rng):
            phi, ok = collate_targets(raw, batch["h"].device)
            ok = ok * batch["mask"]
            mu, _kappa = model.encoder(batch["h"])
            loss = ((1.0 - torch.cos(mu - phi)) * ok).sum() / ok.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 5.0)
            opt.step()
    return model


def fit_emission(model, loader, steps, lr):
    """Fit (a, b) to the warm-started trajectory, so the ELBO starts self-consistent."""
    opt = torch.optim.Adam([model.emission_a, model.emission_b_raw], lr=lr)
    batches = list(loader())
    model.eval()
    for step in range(steps):
        _raw, batch = batches[step % len(batches)]
        with torch.no_grad():
            mu = model.encoder(batch["h"])[0]
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(
            model.emission_logits(mu), batch["y"], reduction="none")
            * batch["mask"]).sum() / batch["mask"].sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def main() -> None:
    """Supervise, fit the emission, then train the ELBO and watch F every epoch."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1, choices=(1, 3))
    ap.add_argument("--sup-epochs", type=int, default=25)
    ap.add_argument("--elbo-epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--crop-cache", default="/disk4/jaehoon/phasevae_dedup.pkl")
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    crops, _rejects = load_or_build(args.crop_cache, None)
    assert len({(c["stem"], round(c["t0"], 3)) for c in crops}) == len(crops), "duplicates"
    crops = phase_targets(crops)
    train = [c for c in crops if c["fold"] not in (None, VAL_FOLD)]
    val = [c for c in crops if c["fold"] == VAL_FOLD]
    print(f"train {len(train)} / val {len(val)}", flush=True)

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = BarPhaseVAE(train[0]["h"].shape[1]).to(device)
    loader = Batches(train, args.batch_size, device)
    val_loader = Batches(val, args.batch_size, device)

    for tag, step in (("cold start", None), ("after supervision", "sup"),
                      ("after fitting the emission", "emis")):
        if step == "sup":
            supervise(model, loader, rng, args.sup_epochs, args.lr)
        elif step == "emis":
            fit_emission(model, loader, 400, 0.05)
        f, cmlt, amlt, ratio = val_f(model, val_loader)
        print(f"{tag:28s} F {f:.3f}  CMLt {cmlt:.3f}  AMLt {amlt:.3f}  "
              f"est/ref {ratio:.2f}")
    print(f"   emission a {float(model.emission_a):+.3f}  b {float(model.emission_b):.3f}\n")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"{'epoch':>6s} {'elbo/fr':>9s} {'kl/fr':>8s} {'emis b':>7s} "
          f"{'F':>7s} {'CMLt':>7s} {'AMLt':>7s} {'est/ref':>8s}")
    for epoch in range(args.elbo_epochs):
        model.train()
        totals, steps = np.zeros(3), 0
        for _raw, batch in loader(shuffle=True, rng=rng):
            out = model(batch["h"], batch["delta"], batch["mask"], batch["y"])
            frames = batch["mask"].sum(1)
            loss = -((out["recon"] - out["kl"]) / frames).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            totals += [float(-loss), float((out["recon"] / frames).mean()),
                       float((out["kl"] / frames).mean())]
            steps += 1
        f, cmlt, amlt, ratio = val_f(model, val_loader)
        print(f"{epoch:6d} {totals[0] / steps:9.4f} {totals[2] / steps:8.4f} "
              f"{float(model.emission_b):7.3f} {f:7.3f} {cmlt:7.3f} {amlt:7.3f} "
              f"{ratio:8.2f}", flush=True)


if __name__ == "__main__":
    main()
