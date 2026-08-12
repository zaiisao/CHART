"""Train and evaluate the bar-phase VAE: one continuous latent, no meter, no beat grid."""
from __future__ import annotations

import argparse
import importlib
import pathlib

import numpy as np
import torch

from .config import load_config
from .scoring.evaluation import (evaluate, print_table, scoring_records,
                                 trajectory_health)
from .data.dataset import split_songs
from .data.excerpts import (ExcerptDataset, collate_excerpts)


# JA: For naive training, we always choose fold 7 to serve as the validation fold.
VAL_FOLD = 7


def beta_at(epoch: int, cfg) -> float:
    """Linear KL annealing from ``beta_start`` to ``beta_end`` over ``beta_warmup``."""
    if cfg.beta_warmup <= 0:
        return cfg.beta_end
    fraction = min(1.0, epoch / cfg.beta_warmup)
    return cfg.beta_start + fraction * (cfg.beta_end - cfg.beta_start)


def _seed_worker(_worker_id: int) -> None:
    """Reseed np.random per DataLoader worker."""
    np.random.seed(torch.initial_seed() % 2**32)


def train(dataset, frontend, device, cfg, hooks, seed: int, workers: int):
    """One seed: run the controls, then fit the objective the hooks define."""
    torch.manual_seed(seed)
    model = hooks.build_model(cfg, frontend.num_channels).to(device)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=workers,
        collate_fn=collate_excerpts, pin_memory=True, worker_init_fn=_seed_worker,
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed))

    opt, clip_params = hooks.optimizer(model, cfg)

    if cfg.frontend_lr_scale > 0:
        fe = list(frontend._audio2frames.model.parameters())
        opt.add_param_group({"params": fe, "lr": cfg.lr * cfg.frontend_lr_scale})

    probe_raw = collate_excerpts([dataset[i] for i in
                                  range(min(cfg.batch_size, len(dataset)))])

    for epoch in range(cfg.epochs):
        model.train()
        beta = beta_at(epoch, cfg)
        hooks.on_epoch(model, cfg, epoch)

        totals, steps = np.zeros(3), 0
        health = np.zeros(4)
        gnorm = 0.0
        anchor = np.zeros(2)
        for raw in loader:
            with torch.set_grad_enabled(cfg.frontend_lr_scale > 0):
                h = frontend.forward_features(raw["input"])

            mask = raw["mask"].to(device, non_blocking=True)
            y = raw["y"].to(device, non_blocking=True)

            out = model(h, mask, y, samples=cfg.samples, pos_weight=cfg.pos_weight,
                        targets=raw["targets"].to(device),
                        valid=raw["valid"].to(device),
                        anchor_penalty=cfg.anchor_penalty,
                        anchor_kappa=cfg.anchor_kappa)

            # per-frame normalisation and beta-annealed loss; reported elbo is beta=1.
            # clamp: a backstop item (fully-masked window) must cost 0, not produce nan.
            frames = mask.sum(1).clamp(min=1.0)
            loss = -(hooks.objective(out, beta, cfg) / frames).mean()

            opt.zero_grad()
            loss.backward()

            # clip_per_group: per-TENSOR clip budgets (pooled norm ran 93-1330 vs clip 5.0
            # all run long; evidence: search-read-out-verdict). |g| logged = pooled norm
            # when off, LARGEST single-tensor norm when on.
            if cfg.clip_per_group:
                gnorm += max(float(torch.nn.utils.clip_grad_norm_([p], cfg.clip))
                             for p in clip_params if p.grad is not None)
            else:
                gnorm += float(torch.nn.utils.clip_grad_norm_(clip_params, cfg.clip))
            if cfg.frontend_lr_scale > 0:
                torch.nn.utils.clip_grad_norm_(fe, cfg.clip)

            opt.step()

            totals += [float(out["elbo"].mean()),
                       float(out["recon"].mean()),
                       float(out["kl"].mean())]
            if "resultant" in out:
                anchor += [float(out["resultant"].mean()),
                           float(out["kl_offset"].mean())]
            # What the ELBO cannot tell you: whether mu is a bar pointer or an arbitrary
            # path that crosses zero in the right places. Read off the batch already
            # computed -- no extra forward. See trajectory_health for the thresholds.
            with torch.no_grad():
                records = [c for c in scoring_records(raw) if c is not None]
                if records:
                    keep = [i for i, c in enumerate(scoring_records(raw))
                            if c is not None]
                    health += trajectory_health(out["mu"][keep].detach(),
                                                out["kappa"][keep].detach(),
                                                mask[keep], records)
            steps += 1

        b_note = ("" if model.emission_net is not None
                  else f"  b {float(model.emission_b):5.2f}")
        adv, kap, perr, cov = health / steps
        # The variant's own diagnostic, on the frozen probe. no_grad covers the frontend
        # forward too, which is NOT redundant once frontend_lr_scale > 0 -- without it the
        # probe would build a graph over 20M thawed parameters purely to print a number.
        with torch.no_grad():
            note = hooks.epoch_note(model, {
                "h": frontend.forward_features(probe_raw["input"]),
                "mask": probe_raw["mask"].to(device),
                "y": probe_raw["y"].to(device)})
        res, kloff = anchor / steps
        a_note = "" if anchor[0] == 0.0 else f"  res {res:5.3f}  kl_off {kloff:6.2f}"
        print(f"  epoch {epoch:2d}  beta {beta:5.3f}  elbo {totals[0] / steps:9.2f}  "
              f"recon {totals[1] / steps:8.2f}  kl {totals[2] / steps:9.2f}{b_note}\n"
              f"            advance {adv:7.4f} (true 0.025-0.063)  kappa {kap:9.1f}  "
              f"phase_err {perr:5.3f} (chance 1.571)  circle {cov:5.1%}  "
              f"|g| {gnorm / steps:8.2f}{a_note}"
              f"{note}",
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

    train_songs, val_songs, test_songs = split_songs(VAL_FOLD, args.limit_per_fold)

    frontend_name = f"phasevae.data.frontends.{cfg.frontend}"
    frontend_class = importlib.import_module(frontend_name).FRONTEND
    frontend = frontend_class(
        checkpoint=cfg.frontend_checkpoint,
        device=f"cuda:{args.gpu}",
        output="features"
    )

    train_set = ExcerptDataset(train_songs, frontend, cfg.excerpt_seconds)
    val_set = ExcerptDataset(val_songs, frontend, cfg.excerpt_seconds, deterministic=True)
    test_set = ExcerptDataset(test_songs, frontend, cfg.excerpt_seconds, deterministic=True)

    print(f"songs: train {len(train_songs)} / val {len(val_songs)} / "
          f"gtzan-test {len(test_songs)}")
    print(f"train: {len(train_set)} songs, fresh {cfg.excerpt_seconds:.0f}s window "
          f"per epoch, rejects {len(train_set.rejects)}")

    model = train(train_set, frontend, device, cfg, hooks, args.seed, args.workers)

    if args.save_dir:
        save_dir = pathlib.Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(),
                    "frontend": frontend._audio2frames.model.state_dict()},
                   save_dir / f"seed{args.seed}.pt")

    results = {name: evaluate(model, split, frontend, device, cfg.batch_size,
                              seed=args.seed)
               for split, name in ((val_set, "val"), (test_set, "gtzan")) if len(split)}

    print_table(results)
    print(f"\nfps={frontend.FPS}  excerpt={cfg.excerpt_seconds}s (fresh window per epoch)  "
          f"frontend={frontend.name}/{cfg.frontend_checkpoint}  "
          f"no meter, no beat grid, no offset")


if __name__ == "__main__":
    main()
