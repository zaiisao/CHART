"""Train and evaluate the bar-phase VAE: one continuous latent, no meter, no beat grid.

The latent is a phase turning once per bar; the bar PERIOD is given (one constant per
window, from annotations) and the model finds the bar's PHASE from audio.

The RECIPE (model + training) lives in a YAML config; the CLI carries only run
mechanics (device, seed, paths). The config's ``variant:`` key names the hooks
module that owns the model -- ``base`` is tutorial §7 (encoder-deployed, no
conditional prior) and is itself just the default variant. run.py drives whichever
module is named through five hooks (build_model / optimizer / objective / on_epoch /
epoch_note) and never branches on which one it is: a new variant is a new module and
a new config, never a flag here.

DATA (2026-08-07): Beat This-style excerpts over the plug-and-play Frontend contract.
Each song's model INPUT (the frontend's frozen preprocessing, e.g. its log-mel) is
cached once per frontend; every epoch draws a FRESH random window per song; the frozen
frontend turns windows into features INSIDE the training loop -- features never touch
disk. The config's ``frontend:``/``frontend_checkpoint:`` keys pick the frontend
(default beat_this/final0); ONE checkpoint serves train/val/test alike: fold-honesty
is phase-gated
(gtzan was held out of EVERY checkpoint, so the decision metric stays clean; CV-dataset
train/val scores are optimistic diagnostics; 8-fold routing returns before any
baseline-comparable claim). Val/gtzan are scored straight off their
deterministic excerpt datasets -- same frontend call as training, valid frames only.

Usage:
    PYTHONPATH=. python -m phasevae.run --config phasevae/configs/baseline.yaml --gpu 1
    PYTHONPATH=. python -m phasevae.run --config phasevae/configs/psi.yaml --gpu 3
    ... --set epochs=2 --set emission=cosine        # ad-hoc override, still one run
"""
from __future__ import annotations

import argparse
import importlib
import pathlib
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from .scoring.controls import (assert_encoder_is_target_blind,  # noqa: F401  (re-export)
                               assert_readout_recovers_oracle, gradient_audit,
                               preflight)
from .scoring.evaluation import evaluate, print_table
from .data.dataset import split_songs
from .data.excerpts import (ExcerptDataset, collate_excerpts,
                            to_model_batch)

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
    excerpt_seconds=45.0,         # window length; below ~45 s the ELBO cannot separate
                                  # tracking the truth from coasting (see MAX_CROP_SECONDS)
    kappa_physical=2000.0,        # physical prior increment concentration; the DRIFT TAX.
                                  # Increment law says real wobble is heavy-tailed -- lowering
                                  # this cuts the KL price of expressing it.
    frontend="beat_this",         # frontends.<name> module; the frontend is part of the
    frontend_checkpoint="final0",  # model (see phasevae-checkpoint-artifact). final0 is
                                  # Beat This's; beat_transformer wants e.g. fold_0.
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


def _seed_worker(_worker_id: int) -> None:
    """Reseed np.random per DataLoader worker.

    Fork copies ONE numpy state into every worker, which would otherwise make them all
    draw IDENTICAL windows.
    """
    np.random.seed(torch.initial_seed() % 2**32)


def train(dataset, frontend, device, cfg, hooks, seed: int, workers: int):
    """One seed: run the controls, then fit the objective the hooks define.

    Windows come fresh from the DataLoader every epoch; the frontend is frozen, so its
    forward runs under no_grad and only the VAE trains.
    """
    torch.manual_seed(seed)
    model = hooks.build_model(cfg, frontend.num_channels).to(device)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=workers,
        collate_fn=collate_excerpts, pin_memory=True, worker_init_fn=_seed_worker,
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed))
    probe = to_model_batch(next(iter(loader)), frontend, device)

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
        for raw in loader:
            batch = to_model_batch(raw, frontend, device)
            out = model(batch["h"], batch["delta"], batch["mask"], batch["y"],
                        samples=cfg.samples, pos_weight=cfg.pos_weight)

            # per-frame normalisation and beta-annealed loss; reported elbo is beta=1.
            # clamp: a backstop item (fully-masked window) must cost 0, not produce nan.
            frames = batch["mask"].sum(1).clamp(min=1.0)
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
    p.add_argument("--config", default="phasevae/configs/anchor_k.yaml",
                   help="YAML recipe; its variant: key names the hooks module")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config key for this run (repeatable)")
    p.add_argument("--gpu", type=int, default=1, choices=(0, 1, 2, 3))
    p.add_argument("--seed", type=int, default=0,
                   help="one run = one seed; sweep seeds with an outer script")
    p.add_argument("--limit-per-fold", type=int, default=None)
    p.add_argument("--workers", type=int, default=4,
                   help="DataLoader workers (window draws + mmap reads)")
    p.add_argument("--save-dir", default=None,
                   help="save the model to <save-dir>/seed<k>.pt")
    return p.parse_args()


def main() -> None:
    """Catalog, train, evaluate, print the per-dataset table."""
    args = parse_args()
    cfg, hooks = load_config(args.config, args.set)
    device = torch.device(f"cuda:{args.gpu}")
    print(f"config {args.config}  seed {args.seed}  ->  {vars(cfg)}", flush=True)

    train_songs, val_songs, test_songs = split_songs(args.limit_per_fold)
    frontend_class = importlib.import_module(
        f"phasevae.data.frontends.{cfg.frontend}").FRONTEND
    frontend = frontend_class(checkpoint=cfg.frontend_checkpoint,
                              device=f"cuda:{args.gpu}", output="features+activations")

    train_set = ExcerptDataset(train_songs, frontend, cfg.excerpt_seconds)
    val_set = ExcerptDataset(val_songs, frontend, cfg.excerpt_seconds, deterministic=True)
    test_set = ExcerptDataset(test_songs, frontend, cfg.excerpt_seconds, deterministic=True)

    print(f"songs: train {len(train_songs)} / val {len(val_songs)} / "
          f"gtzan-test {len(test_songs)}")
    print(f"train: {len(train_set)} songs, fresh {cfg.excerpt_seconds:.0f}s window "
          f"per epoch, rejects {len(train_set.rejects)}")
    preflight(val_set, test_set)

    model = train(train_set, frontend, device, cfg, hooks, args.seed, args.workers)
    if args.save_dir:
        save_dir = pathlib.Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / f"seed{args.seed}.pt")

    results = {name: evaluate(model, split, frontend, device, cfg.batch_size,
                              seed=args.seed)
               for split, name in ((val_set, "val"), (test_set, "gtzan")) if len(split)}
    print_table(results)
    print(f"\nfps={frontend.FPS}  excerpt={cfg.excerpt_seconds}s (fresh window per epoch)  "
          f"frontend={frontend.name}/{cfg.frontend_checkpoint}  "
          f"no meter, no beat grid, no offset")


if __name__ == "__main__":
    main()
