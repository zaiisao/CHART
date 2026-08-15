"""Mixture posterior on the phase innovation: the one-variable version.

The corrector cannot learn to repair states training never visits (measured: the knot
decoder converges to a constant, and a forced phase kick decays not at all). A denoiser
needs corrupted samples. This variant supplies them from INSIDE q, at no cost to the
ELBO's form: at bar crossings the innovation is drawn from a two-component von Mises
mixture, tight with probability 1 - mix_eps and wide (KAPPA_INTER) otherwise, so a
window visits a displaced phase roughly mix_eps times per bar while the emission is
still scoring it.

    q(dphi_t) = (1 - eps) vM(kappa_q) + eps vM(KAPPA_INTER)   at crossings
              =           vM(kappa_q)                          elsewhere

The bound stays exact -- lq is the mixture's own log-density evaluated at the realised
sample, lp the prior's -- but the phase KL becomes a one-sample estimator where the
mainline uses the closed form, so mix_eps = 0 reduces to the mainline IN EXPECTATION
and not per draw. That is the price of the mixture and the reason this is a variant.

The PRIOR is untouched: the crossing gate on p belongs to kappa_gate, which stays
independently switchable. An earlier probe conflated the two and its results were not
attributable to either.
"""
from __future__ import annotations

import math

import torch

from .base import common_kwargs, epoch_note, objective, optimizer  # noqa: F401
from .interval import DEFAULTS as INTERVAL_DEFAULTS
from .interval import on_epoch  # noqa: F401
from ..constants import KAPPA_INTER, TWO_PI
from ..model import IntervalVAE
from ..observation import annotation_frames, interval_loglik
from ..specs import DecoderSpec, PlacementSpec
from ..vonmises import log_i0, sample_vonmises

DEFAULTS = {**INTERVAL_DEFAULTS, "mix_eps": 0.3, "mix_kappa": 17.0}


def vonmises_logpdf(z, kappa):
    return kappa * torch.cos(z) - math.log(TWO_PI) - log_i0(kappa)


class MixtureQVAE(IntervalVAE):

    def __init__(self, input_dim: int, mix_eps: float = 0.3,
                 mix_kappa: float = KAPPA_INTER, **kw):
        super().__init__(input_dim, **kw)
        self.mix_eps = float(mix_eps)
        self.mix_kappa = float(mix_kappa)

    def sample_innovation(self, kappa_q, crossing):
        """Draw the phase innovation and return (sample, log q) under the mixture."""
        tight = sample_vonmises(kappa_q)
        lq_tight = vonmises_logpdf(tight, kappa_q)
        if self.mix_eps <= 0.0 or crossing is None:
            return tight, lq_tight
        kappa_w = torch.full_like(kappa_q, self.mix_kappa)
        wide = sample_vonmises(kappa_w)
        take = crossing & (torch.rand_like(kappa_q) < self.mix_eps)
        z = torch.where(take, wide, tight)
        lq = torch.where(
            crossing,
            torch.logaddexp(math.log(1.0 - self.mix_eps) + vonmises_logpdf(z, kappa_q),
                            math.log(self.mix_eps) + vonmises_logpdf(z, kappa_w)),
            vonmises_logpdf(z, kappa_q))
        return z, lq

    def phase_kl_sampled(self, z, lq, kappa_q, mask, crossing):
        kappa_p = torch.full_like(kappa_q, self.kappa_physical)
        if crossing is not None and self.kappa_gate:
            kappa_p = torch.where(crossing,
                                  torch.full_like(kappa_q, self.mix_kappa), kappa_p)
        return ((lq - vonmises_logpdf(z, kappa_p)) * mask).sum(1)

    def forward(self, h, mask, y, samples: int = 1, pos_weight: float = 1.0, raw=None):
        assert raw is not None, "the interval emission needs the batch's downbeat_times"
        post, memory = self.encoder(h, mask)
        phase_kappa = post["phase"]["kappa"]
        tempo_mu, tempo_sigma = post["tempo"]["mu"], post["tempo"]["sigma"]
        rotation_weight = post["rotation"]["weight"]

        w = torch.ones_like(phase_kappa) if mask is None else mask
        pair_w = ((w[:, 1:] > 0) & (w[:, :-1] > 0)).to(w.dtype)
        ann_f, ann_valid = annotation_frames(raw, phase_kappa.device)

        h_tempo = ((0.5 * math.log(2.0 * math.pi * math.e)
                    + torch.log(tempo_sigma)) * w).sum(1)

        mean_ramp = torch.cumsum(tempo_mu, dim=1) - tempo_mu[:, :1]
        theta, _ = self.encoder._anchor(mean_ramp.detach(), rotation_weight)
        crossing = None
        if self.walk_kind == "gated" or self.kappa_gate or self.mix_eps > 0.0:
            mean_phi = (mean_ramp + theta[:, None]).detach()
            crossing = torch.div(mean_phi[:, 1:], TWO_PI, rounding_mode="floor") \
                != torch.div(mean_phi[:, :-1], TWO_PI, rounding_mode="floor")

        recon = 0.0
        logp_tempo = 0.0
        kl_phase = 0.0
        resultant = 0.0
        corr_abs = 0.0
        wide_rate = 0.0
        phi = None
        corr_full = None
        knots = None

        for _ in range(samples):
            dotphi = tempo_mu * torch.exp(tempo_sigma * torch.randn_like(tempo_sigma))
            jitter, lq = self.sample_innovation(phase_kappa[:, 1:], crossing)
            phi, corr_full, knots, lift, _d, _kd = self._scan(dotphi, jitter, memory,
                                                     theta, pair_w)
            dot_eff = dotphi * torch.exp(corr_full)
            phi_place = None if self.place_attach else (
                phi.detach() + (theta - theta.detach())[:, None]
                + (lift - lift.detach()) * self.place_lift)

            logp_tempo = logp_tempo + self.walk_log_prior(dot_eff, w, crossing)
            kl_phase = kl_phase + self.phase_kl_sampled(
                jitter, lq, phase_kappa[:, 1:], pair_w, crossing)
            em = interval_loglik(phi, ann_f, ann_valid, self.kappa_place,
                                 self.b_ratio, self.phase_half,
                                 self.interval_kind, phi_place, self.disp_weight,
                                 self.place_coord)
            recon = recon + em["loglik"]
            resultant = resultant + em["resultant"]
            corr_abs = corr_abs + corr_full.abs().mean()
            if crossing is not None:
                wide_rate = wide_rate + (jitter.abs() > 0.5).to(w.dtype).mean()

        recon = recon / samples
        logp_tempo = logp_tempo / samples
        kl_phase = kl_phase / samples
        kl = kl_phase - h_tempo - logp_tempo

        return {"elbo": recon - kl, "recon": recon, "kl": kl,
                "phi": phi, "kappa": phase_kappa,
                "tempo_prior": logp_tempo, "tempo_entropy": h_tempo,
                "kl_phase": kl_phase, "resultant": resultant / samples,
                "corr": corr_full, "corr_nodes": tuple(knots),
                "corr_abs": corr_abs / samples,
                "wide_rate": wide_rate / samples}


def build_model(cfg, input_dim: int) -> MixtureQVAE:
    return MixtureQVAE(input_dim, mix_eps=cfg.mix_eps, mix_kappa=cfg.mix_kappa, b_ratio=cfg.b_ratio,
                       kappa_place=cfg.kappa_place, phase_half=cfg.phase_half,
                       interval_kind=cfg.interval_kind, disp_weight=cfg.disp_weight,
                       decoder=DecoderSpec(dim=cfg.dec_dim,
                                           knot_stride=cfg.knot_stride),
                       placement=PlacementSpec(coord=cfg.place_coord,
                                               lift=cfg.place_lift,
                                               attach=cfg.place_attach),
                       encoder_pe=cfg.encoder_pe,
                       **common_kwargs(cfg))
