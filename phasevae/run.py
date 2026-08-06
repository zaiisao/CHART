"""Train and evaluate the bar-phase VAE: one continuous latent, no meter, no beat grid.

The latent is a phase turning once per bar; the bar PERIOD is given (one constant per
crop, from annotations) and the model finds the bar's PHASE from audio.

The RECIPE (model + training) lives in a YAML config; the CLI carries only run
mechanics (device, seeds, paths). The config's ``variant:`` key names the hooks
module that owns the model -- ``base`` is tutorial §7 (encoder-deployed, no
conditional prior) and is itself just the default variant. run.py drives whichever
module is named through five hooks (build_model / optimizer / objective / on_epoch /
epoch_note) and never branches on which one it is: a new variant is a new module and
a new config, never a flag here.

Usage:
    PYTHONPATH=. python -m phasevae.run --config phasevae/configs/baseline.yaml --gpu 1
    PYTHONPATH=. python -m phasevae.run --config phasevae/configs/psi.yaml --gpu 3
    ... --set epochs=2 --set emission=cosine        # ad-hoc override, still one run
"""
from __future__ import annotations

import argparse
import importlib
import pathlib
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from .data.features import FPS

from .scoring.controls import (assert_encoder_is_target_blind,  # noqa: F401  (re-export)
                               assert_readout_recovers_oracle, gradient_audit,
                               preflight)
from .scoring.evaluation import evaluate, print_table
from .data.dataset import (MAX_CROP_SECONDS, MAX_CROPS,        # noqa: F401  (re-export)
                           MIN_CROP_SECONDS, VAL_FOLD, Batches, collate,
                           load_dataset, load_gtzan_through, load_or_build,
                           split_folds, to_device)

# Every config key the MAINLINE understands, with its default. Variant-specific keys
# live in the variant module's own DEFAULTS; anything else in a config is an error.
COMMON = dict(
    variant="base",
    emission="cosine",            # cosine | triangle | transformer
    emission_layers=2,
    emission_positional=False,
    drift_bound=0.0,              # structured q: mu = offset + cumsum(delta + eps*tanh(g))
    bar_rate=False,               # ONE drift per bar-length segment (needs drift_bound)
    epochs=12,
    batch_size=32,
    lr=3e-4,
    clip=5.0,
    samples=1,
    pos_weight=1.0,               # anything but 1 is a surrogate, not an ELBO
    beta_start=0.0,
    beta_end=1.0,
    beta_warmup=4,                # epochs to ramp beta; 0 disables annealing
    emission_sharpness=0.0,       # scheduled floor on emission amplitude b; 0 = free
    sharpness_warmup=30,
)


def load_config(path: str, overrides: list[str]):
    """YAML + --set overrides -> (cfg namespace, hooks module). Unknown keys refuse."""
    recipe = yaml.safe_load(pathlib.Path(path).read_text()) or {}
    for item in overrides:
        key, _, value = item.partition("=")
        recipe[key.strip().replace("-", "_")] = yaml.safe_load(value)

    hooks = importlib.import_module(
        f"phasevae.variants.{recipe.get('variant', COMMON['variant'])}")
    known = COMMON | getattr(hooks, "DEFAULTS", {})
    unknown = set(recipe) - set(known)
    assert not unknown, f"unknown config keys {sorted(unknown)} (typo? variant key?)"
    return SimpleNamespace(**(known | recipe)), hooks


def beta_at(epoch: int, cfg) -> float:
    """Linear KL annealing from ``beta_start`` to ``beta_end`` over ``beta_warmup``."""
    if cfg.beta_warmup <= 0:
        return cfg.beta_end
    fraction = min(1.0, epoch / cfg.beta_warmup)
    return cfg.beta_start + fraction * (cfg.beta_end - cfg.beta_start)


def train(crops, device, cfg, hooks, seed: int):
    """One seed: run the controls, then fit the objective the hooks define."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = hooks.build_model(cfg, crops[0]["h"].shape[1]).to(device)

    loader = Batches(crops, cfg.batch_size, device)
    _raw, probe = next(iter(loader()))

    dead = gradient_audit(model, probe)
    print(f"  CONTROL gradient audit: {len(dead)} dead parameters"
          + (f" -> {dead[:4]}" if dead else ""))
    assert not dead, "dead parameters at initialisation"
    assert_encoder_is_target_blind(model, probe)
    print("  CONTROL deployed inference is target-blind by signature and behaviour")
    gap = model.phase_ablation_gap(model.infer_phase(probe["h"], probe["delta"]),
                                   probe["mask"])
    print(f"  CONTROL emission depends on phase: mean |logit shift| {gap:.4f} "
          f"when phi is frozen")

    opt, clip_params = hooks.optimizer(model, cfg)
    for epoch in range(cfg.epochs):
        model.train()
        beta = beta_at(epoch, cfg)
        hooks.on_epoch(model, cfg, epoch)

        totals, steps = np.zeros(3), 0
        for _raw, batch in loader(shuffle=True, rng=rng):
            out = model(batch["h"], batch["delta"], batch["mask"], batch["y"],
                        samples=cfg.samples, pos_weight=cfg.pos_weight)

            # per-frame normalisation and beta-annealed loss; reported elbo is beta=1
            frames = batch["mask"].sum(1)
            loss = -(hooks.objective(out, beta, cfg) / frames).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clip_params, cfg.clip)
            opt.step()

            totals += [float(out["elbo"].mean()), float(out["recon"].mean()),
                       float(out["kl"].mean())]
            steps += 1

        b_note = ("" if model.emission_net is not None
                  else f"  b {float(model.emission_b):5.2f}")
        b_note += hooks.epoch_note(model, probe)
        print(f"  epoch {epoch:2d}  beta {beta:5.3f}  elbo {totals[0] / steps:9.2f}  "
              f"recon {totals[1] / steps:8.2f}  kl {totals[2] / steps:9.2f}{b_note}",
              flush=True)

    return model


def parse_args():
    """Run mechanics ONLY -- the recipe is the config's business."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="phasevae/configs/baseline.yaml",
                   help="YAML recipe; its variant: key names the hooks module")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config key for this run (repeatable)")
    p.add_argument("--gpu", type=int, default=1, choices=(1, 3))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--limit-per-fold", type=int, default=None)
    p.add_argument("--save-dir", default=None,
                   help="save each seed's model to <save-dir>/seed<k>.pt")
    p.add_argument("--gtzan-checkpoint", default=None,
                   help="rebuild gtzan crops through this fold checkpoint (not final0)")
    p.add_argument("--crop-cache", default=None)
    return p.parse_args()


def main() -> None:
    """Load crops, run the controls, train every seed, print the per-dataset table."""
    args = parse_args()
    cfg, hooks = load_config(args.config, args.set)
    device = torch.device(f"cuda:{args.gpu}")
    print(f"config {args.config}  ->  {vars(cfg)}", flush=True)

    print("loading crops (fold-honest features, random time windows)...", flush=True)
    crops, rejects = load_or_build(args.crop_cache, args.limit_per_fold)
    print(f"\ncrops: {len(crops)}   rejects: {dict(rejects)}")

    preflight(crops, MAX_CROPS)

    train_crops, val_crops, test_crops = split_folds(crops)
    if args.gtzan_checkpoint:
        test_crops, gt_rejects = load_gtzan_through(args.gtzan_checkpoint)
        print(f"  gtzan rebuilt through {args.gtzan_checkpoint}: {len(test_crops)} "
              f"crops, rejects {dict(gt_rejects)}")
    print(f"\ntrain {len(train_crops)} / val {len(val_crops)} / "
          f"gtzan-test {len(test_crops)}")

    per_seed: dict = defaultdict(dict)
    for seed in args.seeds:
        print(f"\n===== seed {seed} =====", flush=True)
        model = train(train_crops, device, cfg, hooks, seed)
        if args.save_dir:
            save_dir = pathlib.Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / f"seed{seed}.pt")

        for split, name in ((val_crops, "val"), (test_crops, "gtzan")):
            if not split:
                continue
            for dataset, modes in evaluate(model, split, device, cfg.batch_size,
                                           seed=seed).items():
                for mode, (value, count) in modes.items():
                    per_seed[(name, dataset, mode)][seed] = (value, count)

    print_table(per_seed)
    print(f"\nfps={FPS}  crop={MIN_CROP_SECONDS}-{MAX_CROP_SECONDS}s (per song)  "
          f"no meter, no beat grid, no offset")


if __name__ == "__main__":
    main()
