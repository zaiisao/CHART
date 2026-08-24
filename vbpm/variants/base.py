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


class _InlineEmission:
    """Adapter for the variants that still keep a, b_raw and b_floor flat.

    Delete this, and emission_of(), once schain/tchain/vmchain hold an
    nets.EmissionModel of their own; every caller below already speaks that shape.
    """

    def __init__(self, model):
        self._m = model

    a = property(lambda self: self._m.emission_a)
    b_raw = property(lambda self: self._m.emission_b_raw)
    b = property(lambda self: self._m.emission_b)
    b_floor = property(lambda self: getattr(self._m, "emission_b_floor", None))


def emission_of(model):
    """The object owning the emission: the submodule where the variant has one."""
    em = getattr(model, "emission_model", None)
    return em if isinstance(em, torch.nn.Module) else _InlineEmission(model)


def common_kwargs(cfg) -> dict:
    """The spec objects every variant is built with."""
    return {"emission": EmissionSpec(kind=cfg.emission, layers=cfg.emission_layers,
                                     positional=cfg.emission_positional,
                                     bump_kappa=cfg.emission_bump_kappa,
                                     fit_init=cfg.emission_fit_init,
                                     frozen=cfg.emission_frozen,
                                     floor=cfg.emission_floor),
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
    if cfg.emission_sharpness <= 0:
        return
    floor = emission_of(model).b_floor
    if floor is not None:
        ramp = min(1.0, epoch / max(cfg.sharpness_warmup, 1))
        floor.fill_(cfg.emission_sharpness * ramp)
