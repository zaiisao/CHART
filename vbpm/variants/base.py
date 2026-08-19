"""The hooks every variant shares: how to build the specs, optimise, and score.

There is no model here any more. The factorized-posterior VBPM this module used
to construct was deleted on 2026-08-19 -- it scored at the null floor on gtzan
(0.074 against nulls of 0.074/0.075, n=993) and could not overfit a metronomic
single song. It is recoverable from git history at 644182b if a control run is
ever needed.
"""
from __future__ import annotations

import torch

from ..specs import EmissionSpec, WalkSpec


COMMON_KEYS = ("emission", "emission_layers", "emission_positional", "emission_recon",
               "kappa_physical")


def common_kwargs(cfg) -> dict:
    """The spec objects every variant is built with."""
    return {"emission": EmissionSpec(kind=cfg.emission, layers=cfg.emission_layers,
                                     positional=cfg.emission_positional,
                                     bump_kappa=cfg.emission_bump_kappa,
                                     recon=getattr(cfg, "emission_recon", "event"),
                                     subdiv=getattr(cfg, "ar_beat_subdiv", 4)),
            "walk": WalkSpec(kappa_physical=cfg.kappa_physical,
                             tempo_mu=cfg.tempo_prior_mu,
                             tempo_sigma=cfg.tempo_prior_sigma,
                             walk_sigma=cfg.walk_sigma)}


def optimizer(model, cfg):
    """(optimizer, params-to-clip). One Adam group; everything clipped."""
    params = list(model.parameters())
    return torch.optim.Adam(params, lr=cfg.lr), params


def objective(out, beta: float, cfg):
    """Per-crop training objective [B]: the (beta-annealed) ELBO, tutorial 7.7."""
    return out["recon"] - beta * out["kl"]


def on_epoch(model, cfg, epoch: int) -> None:
    """Scheduled emission-sharpness floor, where the emission has a gain to raise."""
    if cfg.emission_sharpness > 0 and getattr(model, "emission_net", None) is None \
            and hasattr(model, "emission_b_floor"):
        ramp = min(1.0, epoch / max(cfg.sharpness_warmup, 1))
        model.emission_b_floor.fill_(cfg.emission_sharpness * ramp)


def epoch_note(model, probe) -> str:
    """Extra per-epoch log fields; the base recipe has none beyond run.py's."""
    return ""
