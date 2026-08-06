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
The ELBO stage runs through run.train with a hooks object that injects the warm model and
appends val (F, CMLt, AMLt, est/ref) to every epoch line -- one training loop, one scorer.

    PYTHONPATH=. python -m phasevae.checks.warm_start --gpu 1
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import torch

from ..variants import base
from .encoder_supervised import phase_targets, supervise
from ..scoring.controls import assert_no_duplicate_crops
from ..scoring.evaluation import evaluate_pooled
from ..data.dataset import Batches, load_or_build, split_folds
from ..model import BarPhaseVAE
from ..run import COMMON, train as elbo_train


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


class WarmHooks:
    """Base hooks with the warm model injected and val metrics per epoch.

    build_model returns the ALREADY-WARM model (run.train's fresh init is exactly
    what this check skips), and every epoch line carries the pooled val readout.
    """

    def __init__(self, model, val_crops, device, batch_size):
        self.model = model
        self.val = val_crops
        self.device = device
        self.batch_size = batch_size

    def build_model(self, cfg, input_dim):
        """The warm-started model; run.train's fresh-init is exactly what we skip."""
        return self.model

    optimizer = staticmethod(base.optimizer)
    objective = staticmethod(base.objective)
    on_epoch = staticmethod(base.on_epoch)

    def epoch_note(self, model, probe):
        """Pooled val readout per epoch: F sliding back is this check's whole verdict."""
        f, cmlt, amlt, ratio = evaluate_pooled(model, self.val, self.device,
                                               self.batch_size)
        return f"  F {f:.3f}  CMLt {cmlt:.3f}  AMLt {amlt:.3f}  est/ref {ratio:.2f}"


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
    assert_no_duplicate_crops(crops)
    crops = phase_targets(crops)

    train_crops, val_crops, _test = split_folds(crops)
    print(f"train {len(train_crops)} / val {len(val_crops)}", flush=True)

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = BarPhaseVAE(train_crops[0]["h"].shape[1]).to(device)
    loader = Batches(train_crops, args.batch_size, device)

    for tag, step in (("cold start", None), ("after supervision", "sup"),
                      ("after fitting the emission", "emis")):
        if step == "sup":
            supervise(model, loader, rng, args.sup_epochs, args.lr, log_every=0)
        elif step == "emis":
            fit_emission(model, loader, 400, 0.05)
        f, cmlt, amlt, ratio = evaluate_pooled(model, val_crops, device,
                                               args.batch_size)
        print(f"{tag:28s} F {f:.3f}  CMLt {cmlt:.3f}  AMLt {amlt:.3f}  "
              f"est/ref {ratio:.2f}")
    print(f"   emission a {float(model.emission_a):+.3f}  b {float(model.emission_b):.3f}\n")

    # beta fixed at 1 from epoch 0 (no anneal), as the original hold-or-slide protocol
    cfg = SimpleNamespace(**(COMMON | dict(epochs=args.elbo_epochs, lr=args.lr,
                                           batch_size=args.batch_size, beta_warmup=0)))
    hooks = WarmHooks(model, val_crops, device, args.batch_size)
    elbo_train(train_crops, device, cfg, hooks, seed=0)


if __name__ == "__main__":
    main()
