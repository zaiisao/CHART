"""The base hooks: tutorial §7, encoder-deployed, no conditional prior."""
from __future__ import annotations

import torch

from ..model import BarPhaseVAE


def build_model(cfg, input_dim: int) -> BarPhaseVAE:
    """The §7 model: encoder + fixed physical prior + latent-only emission."""
    return BarPhaseVAE(input_dim, emission=cfg.emission,
                       emission_layers=cfg.emission_layers,
                       emission_positional=cfg.emission_positional,
                       kappa_physical=cfg.kappa_physical,
                       evidence=cfg.evidence,
                       detector_layers=cfg.detector_layers,
                       rate_init_seconds=cfg.rate_init_seconds,
                       rate_bias_learnable=cfg.rate_bias_learnable,
                       readout=cfg.readout)


def optimizer(model, cfg):
    """(optimizer, params-to-clip). One Adam group; everything clipped."""
    params = list(model.parameters())
    return torch.optim.Adam(params, lr=cfg.lr), params


def objective(out, beta: float, cfg):
    """Per-crop training objective [B]: the (beta-annealed) ELBO, tutorial 7.7."""
    return out["recon"] - beta * out["kl"]


def on_epoch(model, cfg, epoch: int) -> None:
    """Scheduled emission-sharpness floor (0 = free); see model.emission_b_floor."""
    if cfg.emission_sharpness > 0 and model.emission_net is None:
        ramp = min(1.0, epoch / max(cfg.sharpness_warmup, 1))
        model.emission_b_floor.fill_(cfg.emission_sharpness * ramp)


def epoch_note(model, probe) -> str:
    """Extra per-epoch log fields; the base recipe has none beyond run.py's."""
    return ""
