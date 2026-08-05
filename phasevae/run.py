"""Train and evaluate the bar-phase VAE: one continuous latent, no meter, no beat grid.

The latent is a phase turning once per bar; the bar PERIOD is given (one constant per
crop, from annotations) and the model finds the bar's PHASE from audio. Architecture is
the tutorial's §7 configuration: encoder-only deployment, fixed physical prior, no psi.
The rationale and measurements behind every flag live in docs/phasevae_decisions.md.

Usage:
    PYTHONPATH=. python -m phasevae.run --gpu 1
    PYTHONPATH=. python -m phasevae.run --gpu 3 --shuffle-h     # leak control
"""
from __future__ import annotations

import argparse
import pathlib
from collections import defaultdict

import numpy as np
import torch

from vbpm.data import FPS

from .controls import (assert_encoder_is_target_blind,          # noqa: F401  (re-export)
                       assert_readout_recovers_oracle, gradient_audit, preflight)
from .crops import MAX_CROP_SECONDS, MIN_CROP_SECONDS
from .evaluation import evaluate, print_table
from .loading import (MAX_CROPS, VAL_FOLD, Batches, collate,    # noqa: F401  (re-export)
                      load_dataset, load_gtzan_through, load_or_build, to_device)
from .model import BarPhaseVAE


def beta_at(epoch: int, args) -> float:
    """Linear KL annealing from ``beta_start`` to ``beta_end`` over ``beta_warmup``."""
    if args.beta_warmup <= 0:
        return args.beta_end
    fraction = min(1.0, epoch / args.beta_warmup)
    return args.beta_start + fraction * (args.beta_end - args.beta_start)


def train(crops, device, args, seed: int):
    """One seed: run the controls, then fit the ELBO."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = BarPhaseVAE(crops[0]["h"].shape[1], emission=args.emission,
                        emission_layers=args.emission_layers,
                        emission_positional=args.emission_positional,
                        drift_bound=args.drift_bound).to(device)

    loader = Batches(crops, args.batch_size, device)
    _raw, probe = next(iter(loader()))
    dead = gradient_audit(model, probe)
    print(f"  CONTROL gradient audit: {len(dead)} dead parameters"
          + (f" -> {dead[:4]}" if dead else ""))
    assert not dead, "dead parameters at initialisation"
    assert_encoder_is_target_blind(model, probe)
    print("  CONTROL encoder is target-blind by signature and behaviour")
    gap = model.phase_ablation_gap(model.infer_phase(probe["h"], probe["delta"]),
                                   probe["mask"])
    print(f"  CONTROL emission depends on phase: mean |logit shift| {gap:.4f} "
          f"when phi is frozen")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        beta = beta_at(epoch, args)
        if args.emission_sharpness > 0 and model.emission_net is None:
            ramp = min(1.0, epoch / max(args.sharpness_warmup, 1))
            model.emission_b_floor.fill_(args.emission_sharpness * ramp)
        totals, steps = np.zeros(3), 0
        for _raw, batch in loader(shuffle=True, rng=rng):
            out = model(batch["h"], batch["delta"], batch["mask"], batch["y"],
                        samples=args.samples, pos_weight=args.pos_weight)
            # per-frame normalisation and beta-annealed loss; reported elbo is beta=1
            # (see docs/phasevae_decisions.md for both decisions)
            frames = batch["mask"].sum(1)
            loss = -((out["recon"] - beta * out["kl"]) / frames).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            totals += [float(out["elbo"].mean()), float(out["recon"].mean()),
                       float(out["kl"].mean())]
            steps += 1
        b_note = ("" if model.emission_net is not None
                  else f"  b {float(model.emission_b):5.2f}")
        print(f"  epoch {epoch:2d}  beta {beta:5.3f}  elbo {totals[0] / steps:9.2f}  "
              f"recon {totals[1] / steps:8.2f}  kl {totals[2] / steps:9.2f}{b_note}",
              flush=True)
    return model


def parse_args():
    """CLI. One-line summaries here; measured rationale in docs/phasevae_decisions.md."""
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=1, choices=(1, 3))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--drift-bound", type=float, default=0.0,
                   help="structured q: mu = offset + cumsum(delta + eps*tanh(g)); "
                        "0 = free per-frame mu")
    p.add_argument("--emission", default="cosine",
                   choices=("cosine", "transformer", "triangle"))
    p.add_argument("--emission-layers", type=int, default=2)
    p.add_argument("--emission-positional", action="store_true",
                   help="positional encoding in the transformer emission (shortcut risk)")
    p.add_argument("--pos-weight", type=float, default=1.0,
                   help="downbeat-frame weight; anything but 1 is a surrogate, not an ELBO")
    p.add_argument("--beta-start", type=float, default=0.0)
    p.add_argument("--beta-end", type=float, default=1.0)
    p.add_argument("--beta-warmup", type=int, default=4,
                   help="epochs to ramp beta; 0 disables annealing")
    p.add_argument("--emission-sharpness", type=float, default=0.0,
                   help="scheduled floor on the emission amplitude b; 0 = free")
    p.add_argument("--sharpness-warmup", type=int, default=30)
    p.add_argument("--limit-per-fold", type=int, default=None)
    p.add_argument("--save-dir", default=None,
                   help="save each seed's model to <save-dir>/seed<k>.pt")
    p.add_argument("--gtzan-checkpoint", default=None,
                   help="rebuild gtzan crops through this fold checkpoint (not final0)")
    p.add_argument("--crop-cache", default=None)
    p.add_argument("--shuffle-h", action="store_true",
                   help="LEAK CONTROL: shuffle features along time; above-null = leak")
    return p.parse_args()


def main() -> None:
    """Load crops, run the controls, train every seed, print the per-dataset table."""
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    print("loading crops (fold-honest features, random time windows)...", flush=True)
    crops, rejects = load_or_build(args.crop_cache, args.limit_per_fold)
    print(f"\ncrops: {len(crops)}   rejects: {dict(rejects)}")

    if args.shuffle_h:
        shuffle_rng = np.random.default_rng(12345)
        for crop in crops:
            crop["h"] = crop["h"][shuffle_rng.permutation(len(crop["h"]))]
        print("LEAK CONTROL: features shuffled along time. Above-null here is a leak.")

    preflight(crops, MAX_CROPS)

    train_crops = [c for c in crops if c["fold"] not in (None, VAL_FOLD)]
    val_crops = [c for c in crops if c["fold"] == VAL_FOLD]
    test_crops = [c for c in crops if c["fold"] is None]
    if args.gtzan_checkpoint:
        test_crops, gt_rejects = load_gtzan_through(args.gtzan_checkpoint)
        print(f"  gtzan rebuilt through {args.gtzan_checkpoint}: {len(test_crops)} "
              f"crops, rejects {dict(gt_rejects)}")
    print(f"\ntrain {len(train_crops)} / val {len(val_crops)} / "
          f"gtzan-test {len(test_crops)}")

    per_seed: dict = defaultdict(dict)
    for seed in args.seeds:
        print(f"\n===== seed {seed} =====", flush=True)
        model = train(train_crops, device, args, seed)
        if args.save_dir:
            save_dir = pathlib.Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / f"seed{seed}.pt")
        for split, name in ((val_crops, "val"), (test_crops, "gtzan")):
            if not split:
                continue
            for dataset, modes in evaluate(model, split, device, args.batch_size,
                                           seed=seed).items():
                for mode, (value, count) in modes.items():
                    per_seed[(name, dataset, mode)][seed] = (value, count)

    print_table(per_seed)
    print(f"\nfps={FPS}  crop={MIN_CROP_SECONDS}-{MAX_CROP_SECONDS}s (per song)  "
          f"no meter, no beat grid, no offset")


if __name__ == "__main__":
    main()
