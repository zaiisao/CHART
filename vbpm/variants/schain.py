"""Structured posterior: q(phi) proportional to p(phi) times per-frame potentials.

The generative model is the tutorial's own (section 7): a fixed physical prior over the
bar-phase chain and an emission that reads phi alone. What changes is the variational
family. Instead of independent per-frame factors, q inherits the prior's chain and is
reweighted by potentials the encoder emits from x:

    q(phi | x)  =  p(phi) prod_t psi_t(phi_t ; x) / Z

Phase is discretised into ``bins`` cells so the chain admits exact inference. One
forward-backward sweep returns the marginals, the pairwise marginals and log Z, which
gives every ELBO term in closed form:

    E_q[log p(b | phi)]  =  sum_t sum_k gamma_t(k) log p(b_t | phi_t = k)
    KL(q || p)           =  E_q[sum_t log psi_t] - log Z

No sampling appears anywhere, so the reparameterised von Mises estimator and its bias
leave the objective with it. Rate remains enumerated: one chain per candidate, weighted
by q(rate) proportional to exp(log Z_r + log p(r)), so the trajectory's own evidence
scores the rate rather than a separate heuristic.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from ..constants import TWO_PI
from ..nets import Encoder
from ..specs import EmissionSpec, WalkSpec
from ..vonmises import log_i0


DEFAULTS = {"chain_bins": 96, "chain_rate_grid": 24, "chain_posterior": "chain"}


class ChainVBPM(nn.Module):
    """The tutorial's generative model with a chain-structured posterior."""

    wants_raw = False

    def __init__(self, input_dim: int, d_model: int = 128, bins: int = 96,
                 rate_grid: int = 24, rate_lo: float = 0.020, rate_hi: float = 0.200,
                 emission: EmissionSpec | None = None,
                 walk: WalkSpec | None = None,
                 tempo_prior_mu: float = -2.6827, tempo_prior_sigma: float = 0.3903,
                 posterior: str = "chain", encoder_pe: bool = False):
        super().__init__()
        self.walk = walk or WalkSpec()
        self.bins = int(bins)
        self.posterior = posterior
        emission = emission or EmissionSpec(kind="triangle")
        self.emission_kind = emission.kind
        self.bump_kappa = float(emission.bump_kappa)

        self.encoder = Encoder(input_dim, d_model, use_pe=encoder_pe)
        self.psi_head = nn.Linear(d_model, self.bins)
        nn.init.zeros_(self.psi_head.weight)
        nn.init.zeros_(self.psi_head.bias)

        self.emission_a = nn.Parameter(torch.tensor(-3.0))
        self.emission_b_raw = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("emission_b_floor", torch.tensor(0.0))

        rates = torch.exp(torch.linspace(math.log(rate_lo), math.log(rate_hi), rate_grid))
        self.register_buffer("rates", rates)
        self.register_buffer("theta", torch.arange(self.bins) * (TWO_PI / self.bins))
        z = (torch.log(rates) - tempo_prior_mu) / tempo_prior_sigma
        lp = -0.5 * z ** 2
        self.register_buffer("rate_log_prior", lp - torch.logsumexp(lp, 0))

    @property
    def emission_b(self):
        """Amplitude, positive by softplus, never below the scheduled floor."""
        return self.emission_b_floor + nn.functional.softplus(self.emission_b_raw)

    @property
    def deployed_net(self):
        """The inference network read at test time; controls assert ITS target-blindness."""
        return self.encoder

    def emission_logits_at_bins(self):
        """log-odds of a downbeat at each phase bin: p(b | phi), tabulated."""
        phi = self.theta
        if self.emission_kind == "triangle":
            wrapped = torch.atan2(torch.sin(phi), torch.cos(phi))
            shape = 1.0 - 2.0 * wrapped.abs() / math.pi
        elif self.emission_kind == "bump":
            shape = 2.0 * torch.exp(self.bump_kappa * (torch.cos(phi) - 1.0)) - 1.0
        else:
            shape = torch.cos(phi)
        return self.emission_a + self.emission_b * shape

    def emission_logits(self, phi, mask=None):
        """The same emission read at arbitrary phases rather than at bin centres."""
        if self.emission_kind == "triangle":
            wrapped = torch.atan2(torch.sin(phi), torch.cos(phi))
            return self.emission_a + self.emission_b * (1.0 - 2.0 * wrapped.abs() / math.pi)
        if self.emission_kind == "bump":
            peak = torch.exp(self.bump_kappa * (torch.cos(phi) - 1.0))
            return self.emission_a + self.emission_b * (2.0 * peak - 1.0)
        return self.emission_a + self.emission_b * torch.cos(phi)

    def log_transition(self):
        """Log p(phi_t | phi_t-1) per rate candidate: a von Mises shift, rows normalised."""
        diff = self.theta[None, :] - self.theta[:, None]
        d = diff[None] - self.rates[:, None, None]
        kappa = torch.as_tensor(self.walk.kappa_physical, device=d.device, dtype=d.dtype)
        lt = kappa * torch.cos(d) - math.log(TWO_PI) - log_i0(kappa)
        return lt - torch.logsumexp(lt, dim=2, keepdim=True)

    def forward_backward(self, log_psi, log_T, mask):
        """One sweep each way: log marginals and log Z for every rate candidate."""
        b, t, k = log_psi.shape
        r = log_T.shape[0]
        log_a = torch.full((b, r, k), -math.log(k), device=log_psi.device,
                           dtype=log_psi.dtype)
        alphas = []
        logZ = torch.zeros(b, r, device=log_psi.device, dtype=log_psi.dtype)
        for i in range(t):
            if i > 0:
                log_a = torch.logsumexp(log_a[:, :, :, None] + log_T[None], dim=2)
            log_a = log_a + mask[:, i][:, None, None] * log_psi[:, i][:, None, :]
            m = log_a.logsumexp(-1, keepdim=True)
            log_a = log_a - m
            logZ = logZ + m[..., 0]
            alphas.append(log_a)
        log_b = torch.zeros(b, r, k, device=log_psi.device, dtype=log_psi.dtype)
        betas = [None] * t
        betas[t - 1] = log_b
        for i in range(t - 1, 0, -1):
            msg = log_b + mask[:, i][:, None, None] * log_psi[:, i][:, None, :]
            log_b = torch.logsumexp(log_T[None] + msg[:, :, None, :], dim=3)
            log_b = log_b - log_b.logsumexp(-1, keepdim=True)
            betas[i - 1] = log_b
        log_g = torch.stack(alphas, 2) + torch.stack(betas, 2)
        return log_g - log_g.logsumexp(-1, keepdim=True), logZ

    def posterior_marginals(self, h, mask):
        """The encoder pass plus inference, for whichever posterior the spec names."""
        feats = self.encoder.features(h, mask)
        log_psi = torch.log_softmax(self.psi_head(feats), dim=-1)
        log_T = self.log_transition()
        if self.posterior == "chain":
            log_g, logZ = self.forward_backward(log_psi, log_T, mask)
            return log_psi, log_T, log_g, logZ
        log_g = log_psi[:, None].expand(-1, log_T.shape[0], -1, -1)
        return log_psi, log_T, log_g, None

    def mean_path(self, gamma):
        """Circular mean of the marginals, unwrapped: the discrete analogue of mu(x)."""
        re = (gamma * torch.cos(self.theta)[None, None]).sum(-1)
        im = (gamma * torch.sin(self.theta)[None, None]).sum(-1)
        wrapped = torch.atan2(im, re)
        step = torch.diff(wrapped, dim=-1)
        step = torch.atan2(torch.sin(step), torch.cos(step))
        path = torch.cat([wrapped[:, :1], wrapped[:, :1] + torch.cumsum(step, -1)], -1)
        return path, torch.sqrt(re ** 2 + im ** 2 + 1e-12)

    def forward(self, h, mask, y, pos_weight: float = 1.0):
        """One ELBO evaluation: exact expectations under the structured posterior."""
        log_psi, log_T, log_g, logZ = self.posterior_marginals(h, mask)
        gamma = log_g.exp()

        e = self.emission_logits_at_bins()
        ll_pos = -nn.functional.softplus(-e)
        ll_neg = -nn.functional.softplus(e)
        per_bin = (pos_weight * y[:, None, :, None] * ll_pos
                   + (1.0 - y)[:, None, :, None] * ll_neg)
        recon_r = ((gamma * per_bin).sum(-1) * mask[:, None]).sum(-1)

        if self.posterior == "chain":
            kl_r = ((gamma * log_psi[:, None]).sum(-1) * mask[:, None]).sum(-1) - logZ
            score = logZ
        else:
            ent = -((gamma * log_g).sum(-1) * mask[:, None]).sum(-1)
            pair = mask[:, 1:] * mask[:, :-1]
            cross = torch.einsum("brtj,rjk,brtk->brt",
                                 gamma[:, :, :-1], log_T, gamma[:, :, 1:])
            e_logp = (cross * pair[:, None]).sum(-1) - math.log(self.bins)
            kl_r = -ent - e_logp
            score = recon_r - kl_r

        log_qr = torch.log_softmax(score + self.rate_log_prior[None], dim=1)
        qr = log_qr.exp()
        kl_rate = (qr * (log_qr - self.rate_log_prior[None])).sum(1)

        best = log_qr.argmax(1)
        idx = best[:, None, None, None].expand(-1, 1, gamma.shape[2], gamma.shape[3])
        gamma_best = gamma.gather(1, idx).squeeze(1)
        phi, resultant = self.mean_path(gamma_best)

        recon = (qr * recon_r).sum(1)
        kl = (qr * kl_r).sum(1) + kl_rate
        return {"elbo": recon - kl, "recon": recon, "kl": kl, "phi": phi,
                "kappa": resultant * float(self.walk.kappa_physical),
                "resultant": resultant.mean(1), "log_qr": log_qr,
                "rate": self.rates[best]}

    @torch.no_grad()
    def infer_phase(self, h, mask=None):
        """Deployment: the mean phase path from x alone."""
        assert not self.training, "deployment path must run in eval mode"
        if mask is None:
            mask = torch.ones(h.shape[:2], device=h.device, dtype=h.dtype)
        _psi, _T, log_g, logZ = self.posterior_marginals(h, mask)
        score = logZ if logZ is not None else log_g.sum((2, 3))
        best = torch.log_softmax(score + self.rate_log_prior[None], dim=1).argmax(1)
        gamma = log_g.exp()
        idx = best[:, None, None, None].expand(-1, 1, gamma.shape[2], gamma.shape[3])
        return self.mean_path(gamma.gather(1, idx).squeeze(1))[0]

    @torch.no_grad()
    def emission_probs(self, h, mask=None):
        """Alternative D on the marginals: the emission read through q's belief."""
        if mask is None:
            mask = torch.ones(h.shape[:2], device=h.device, dtype=h.dtype)
        _psi, _T, log_g, logZ = self.posterior_marginals(h, mask)
        score = logZ if logZ is not None else log_g.sum((2, 3))
        best = torch.log_softmax(score + self.rate_log_prior[None], dim=1).argmax(1)
        gamma = log_g.exp()
        idx = best[:, None, None, None].expand(-1, 1, gamma.shape[2], gamma.shape[3])
        gamma_best = gamma.gather(1, idx).squeeze(1)
        return (gamma_best * torch.sigmoid(self.emission_logits_at_bins())).sum(-1)


def build_model(cfg, input_dim: int) -> ChainVBPM:
    """The hooks entry point: one ChainVBPM from a config."""
    return ChainVBPM(input_dim,
                     bins=cfg.chain_bins,
                     rate_grid=cfg.chain_rate_grid,
                     emission=EmissionSpec(kind=cfg.emission,
                                           bump_kappa=cfg.emission_bump_kappa),
                     walk=WalkSpec(kappa_physical=cfg.kappa_physical),
                     posterior=cfg.chain_posterior)
