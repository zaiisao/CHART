"""The base hooks: tutorial §7, encoder-deployed, no conditional prior."""
from __future__ import annotations

import torch

from ..model import BarPhaseVAE


COMMON_KEYS = ("emission", "emission_layers", "emission_positional", "kappa_physical")

DEFAULTS_IF_UNSUPPORTED = {}


def common_kwargs(cfg) -> dict:
    """The kwargs every BarPhaseVAE subclass should be built with."""
    return {k: getattr(cfg, k) for k in COMMON_KEYS}


def refuse_unsupported(cfg, variant: str, supported=()) -> None:
    """Fail loudly if the config asks for a key this variant does not forward."""
    for key, default in DEFAULTS_IF_UNSUPPORTED.items():
        if key in supported:
            continue
        got = getattr(cfg, key, default)
        assert got == default, (
            f"variant {variant!r} does not forward {key}={got!r}; it would train "
            f"{key}={default!r} and the result would be a null by construction")


def build_model(cfg, input_dim: int) -> BarPhaseVAE:
    """The §7 model: encoder + fixed physical prior + latent-only emission."""
    return BarPhaseVAE(input_dim, **common_kwargs(cfg))


def optimizer(model, cfg):
    """(optimizer, params-to-clip). One Adam group; everything clipped."""
    params = list(model.parameters())
    return torch.optim.Adam(params, lr=cfg.lr), params


def objective(out, beta: float, cfg):
    """Per-crop training objective [B]: the (beta-annealed) ELBO, tutorial 7.7."""
    return out["recon"] - beta * out["kl"]


def on_epoch(model, cfg, epoch: int) -> None:
    """Scheduled emission-sharpness floor, and the tempo posterior's annealed width."""
    if cfg.emission_sharpness > 0 and model.emission_net is None:
        ramp = min(1.0, epoch / max(cfg.sharpness_warmup, 1))
        model.emission_b_floor.fill_(cfg.emission_sharpness * ramp)


def epoch_note(model, probe) -> str:
    """Extra per-epoch log fields; the base recipe has none beyond run.py's."""
    return ""
