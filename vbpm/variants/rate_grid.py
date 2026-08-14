"""Categorical q over a log-spaced rate grid: rate is EVALUATED, never descended."""
from __future__ import annotations

import math

import torch
from torch import nn

from .base import common_kwargs, epoch_note, objective, on_epoch, optimizer  # noqa: F401
from ..model import (TEMPO_HI, TEMPO_LO, TEMPO_PRIOR_EPS, TEMPO_PRIOR_MU,
                     TEMPO_PRIOR_SIGMA, VBPM, Encoder, bounded_kappa,
                     event_recon, sample_vonmises)


DEFAULTS = {"rate_grid_size": 1024, "rate_score_scale": 20.0}


class RateGridEncoder(Encoder):
    def __init__(self, input_dim: int, d_model: int = 128,
                 grid_size: int = 1024, score_scale: float = 20.0, **kw):
        super().__init__(input_dim, d_model, **kw)
        self.register_buffer("log_rates", torch.linspace(TEMPO_LO, TEMPO_HI, grid_size))
        self.score_scale_raw = nn.Parameter(torch.tensor(math.log(score_scale)))

    def rate_log_prior(self):
        z = (self.log_rates - TEMPO_PRIOR_MU) / TEMPO_PRIOR_SIGMA
        log_gauss = -0.5 * z ** 2 - math.log(TEMPO_PRIOR_SIGMA) \
            - 0.5 * math.log(2.0 * math.pi)
        log_unif = -math.log(TEMPO_HI - TEMPO_LO)
        floor = torch.full_like(log_gauss, math.log(TEMPO_PRIOR_EPS) + log_unif)
        init = torch.logaddexp(math.log(1.0 - TEMPO_PRIOR_EPS) + log_gauss, floor)
        return init - torch.logsumexp(init, 0)

    def heads(self, trunk, mask=None, h=None):
        channels = self.output_channels(trunk)
        kappa = bounded_kappa(
            torch.exp(channels["phase_log_kappa"] + self.log_phi_kappa_bias) + 1e-3)

        w = torch.ones(trunk.shape[:2], device=trunk.device, dtype=trunk.dtype) \
            if mask is None else mask
        a = torch.sigmoid(channels["rotation_weight_logit"]) * w

        t = torch.arange(trunk.shape[1], device=trunk.device, dtype=trunk.dtype)
        phase0 = torch.exp(self.log_rates)[:, None] * t[None, :]
        real = a @ torch.cos(phase0).T
        imag = a @ torch.sin(phase0).T
        norm2 = real ** 2 + imag ** 2
        offset = -torch.atan2(imag, torch.where(norm2 > 1e-12, real, torch.ones_like(real)))
        resultant = torch.sqrt(norm2 + 1e-12) / a.sum(1, keepdim=True).clamp(min=1e-6)

        log_prior = self.rate_log_prior()
        scores = torch.exp(self.score_scale_raw) * resultant + log_prior
        log_q = torch.log_softmax(scores, dim=1)

        mu = phase0[None] + offset[..., None]

        return {"phase": {"mu": mu, "kappa": kappa},
                "log_q": log_q, "log_prior": log_prior,
                           "resultant": resultant}


class RateGridVAE(VBPM):
    def __init__(self, input_dim: int, d_model: int = 128,
                 grid_size: int = 1024, score_scale: float = 20.0, **kw):
        super().__init__(input_dim, d_model=d_model, **kw)
        self.encoder = RateGridEncoder(input_dim, d_model, grid_size=grid_size,
                                       score_scale=score_scale,
                                       kappa_physical=self.kappa_physical)

    def forward(self, h, mask, y, samples: int = 1, pos_weight: float = 1.0):
        post, _ = self.encoder(h, mask)
        mu, kappa, aux = post["phase"]["mu"], post["phase"]["kappa"], post
        b, r, t = mu.shape
        w = torch.ones_like(y) if mask is None else mask
        log_q = aux["log_q"]
        q = log_q.exp()

        kl_rate = (q * (log_q - aux["log_prior"][None])).sum(1)
        kl = self.kl_jitter(torch.zeros_like(kappa), kappa, w) + kl_rate

        y_r = y[:, None, :].expand(b, r, t).reshape(b * r, t)
        w_r = w[:, None, :].expand(b, r, t).reshape(b * r, t)
        recon = 0.0
        for _ in range(samples):
            phi = mu + sample_vonmises(kappa)[:, None, :]
            logits = self.emission_logits(phi.reshape(b * r, t))
            recon_r = event_recon(logits, y_r, w_r, pos_weight).reshape(b, r)
            recon = recon + (q * recon_r).sum(1)
        recon = recon / samples

        idx = log_q.argmax(1)
        mu_best = mu.gather(1, idx[:, None, None].expand(b, 1, t)).squeeze(1)
        res_best = aux["resultant"].gather(1, idx[:, None]).squeeze(1)

        return {"elbo": recon - kl, "recon": recon, "kl": kl, "phi": mu_best,
                "kappa": kappa, "resultant": res_best, "log_q": log_q,
                "kl_rate": kl_rate, "kl_offset": kl_rate}

    @torch.no_grad()
    def infer_phase(self, h, mask=None):
        assert not self.training, "deployment path must run in eval mode"
        post, _ = self.encoder(h, mask)
        mu, aux = post["phase"]["mu"], post
        idx = aux["log_q"].argmax(1)
        return mu.gather(1, idx[:, None, None].expand(mu.shape[0], 1, mu.shape[2])).squeeze(1)


def build_model(cfg, input_dim: int) -> RateGridVAE:
    return RateGridVAE(input_dim, grid_size=cfg.rate_grid_size,
                       score_scale=cfg.rate_score_scale, **common_kwargs(cfg))
