"""VBPM: a downbeat chain whose tempo is fixed between downbeats."""
from __future__ import annotations

import torch
from torch import nn

from . import base
from ..nets import EmissionModel, PosteriorModel, PriorModel
from ..specs import EmissionSpec, RateSpec, WalkSpec

DEFAULTS = {"chain_rate_grid": 36, "ar_rate_lo": 0.012, "ar_rate_hi": 0.200,
            "meters": [], "meter_prior": "uniform"}


class VBPM(nn.Module):
    """The tutorial's generative model with a bar-gated tempo chain posterior."""

    def __init__(self, input_dim: int, d_model: int = 128,
                 emission: EmissionSpec | None = None,
                 walk: WalkSpec | None = None,
                 rate: RateSpec | None = None,
                 encoder_pe: bool = False):
        super().__init__()

        self.emission_spec = emission or EmissionSpec()
        self.walk_spec = walk or WalkSpec()
        self.rate_spec = rate or RateSpec()

        self.emission_model = EmissionModel(self.emission_spec)
        self.prior_model = PriorModel(self.rate_spec, self.walk_spec)
        self.posterior_model = PosteriorModel(input_dim, d_model, self.prior_model,
                                              encoder_pe=encoder_pe)

    @property
    def deployed_net(self):
        """The network that runs at test time; it reads audio only."""
        return self.posterior_model.encoder

    @torch.no_grad()
    def init_rate_prior(self, rate: float):
        """Centre the prior's rate distribution on ``rate`` (used by --acf-init)."""
        return self.prior_model.init_log_prior(rate, self.walk_spec.tempo_sigma)

    def forward(self, h, mask, y, pos_weight: float = 1.0, cls=None):
        """The ELBO and the trajectory diagnostics for one batch."""
        assert pos_weight == 1.0, \
            "pos_weight != 1 is a weighted surrogate, not an ELBO; this model has no such term"

        (evidence, log_q_rate0, log_q_meter,
         q_phase, q_rate, q_meter, log_z) = self.posterior_model(h, mask, self.prior_model)
        meters = getattr(self.prior_model, "meter_values", None)

        emission_ll = self.emission_model.loglik(
            y, mask, self.prior_model.grid, meters=meters, cls=cls)
        if q_meter is None:
            recon = torch.einsum("btn,btn->b", q_phase, emission_ll)
            phase_marginal = q_phase
        else:
            recon = torch.einsum("btmn,btmn->b", q_phase, emission_ll)
            phase_marginal = q_phase.sum(2)

        expected_evidence = torch.einsum("btn,btn->b", phase_marginal, evidence) \
            + torch.einsum("bc,bc->b", q_rate[:, 0], log_q_rate0)
        if log_q_meter is not None:
            expected_evidence = expected_evidence + torch.einsum(
                "bm,bm->b", q_meter[:, 0], log_q_meter)

        kl = expected_evidence - log_z
        elbo = recon - kl

        phi, cos_sum, sin_sum = self.posterior_model.unwrap(
            phase_marginal, self.prior_model.grid)
        resultant = (cos_sum ** 2 + sin_sum ** 2).sqrt().clamp(1e-6, 1 - 1e-6)
        kappa = resultant * (2 - resultant ** 2) / (1 - resultant ** 2)

        rate_traj = (q_rate * self.prior_model.rates[None, None, :]).sum(-1)

        return {"elbo": elbo, "recon": recon, "kl": kl, "phi": phi, "kappa": kappa,
                "rate_traj": rate_traj, "rate": rate_traj.mean(-1),
                "q_phase": phase_marginal, "q_rate": q_rate, "q_meter": q_meter}

    @torch.no_grad()
    def infer_phase(self, h, mask=None):
        """Label-free deployment: the potentials are functions of x only."""
        assert not self.training
        if mask is None:
            mask = torch.ones(h.shape[:2], device=h.device, dtype=h.dtype)
        out = self.posterior_model(h, mask, self.prior_model)
        q_phase = out[3] if out[5] is None else out[3].sum(2)
        return self.posterior_model.unwrap(q_phase, self.prior_model.grid)[0]

    @torch.no_grad()
    def emission_probs(self, h, mask=None):
        """Per-frame downbeat probability at the inferred phase."""
        return torch.sigmoid(self.emission_model(self.infer_phase(h, mask)))


def on_epoch(model, cfg, epoch: int) -> None:
    """Base's sharpness floor."""
    base.on_epoch(model, cfg, epoch)


def build_model(cfg, input_dim: int) -> VBPM:
    """One VBPM from a config: two specs, no loose floats."""
    rate_spec = RateSpec(grid=cfg.chain_rate_grid, lo=cfg.ar_rate_lo,
                         hi=cfg.ar_rate_hi, posterior="categorical", resid=0.0,
                         per_bar=cfg.tempo_per_bar, meters=tuple(cfg.meters),
                         meter_prior=cfg.meter_prior)
    vbpm = VBPM(input_dim, rate=rate_spec, **base.common_kwargs(cfg))
    return vbpm
