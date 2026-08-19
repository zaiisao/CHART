"""Form 1 of the phase note, run as written: a continuous autoregressive posterior.

The chain family (schain/tchain) answers the q question with exact discrete
inference. The note's own prescription for a continuous q is its Form 1: the
posterior keeps the prior's chain factorisation,

    q(phi_0 | c_0) prod_t q(phi_t | phi_{t-1}, c_t),

each factor a von Mises whose parameters one shared head reads off the SAMPLED
previous phase and the bidirectional context. The head sees where the path
actually is, so it can steer back toward the evidence -- the feedback wire the
per-frame regression families never had. At zero init every step factor equals
the prior transition, so training starts exactly at KL = 0 and every departure
is learned.

What an autoregressive unimodal chain cannot represent is the global mode:
which octave, which alignment. Both are carried by enumeration, marginalised
exactly: a categorical q(rate | x) over the schain grid, and (ar_phi0_grid > 0)
a categorical q(phi_0) over a uniform circle grid -- the base case is where all
the posterior's mode mass concentrates, so it is the one factor that must not
be unimodal. ``lognormal`` replaces the categorical rate with a regressed
lognormal: the arm the multimodality verdict predicts must fail on octaves,
kept as its own control.

Steering authority is bounded the way the prior prices it: ar_delta_rel caps
the per-step correction at a fraction OF THE ADVANCE (a log-rate trim), so no
rate bin can finance stopping the clock; ar_delta_max is the older absolute cap.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .base import epoch_note as _base_note, on_epoch, optimizer  # noqa: F401
from ..constants import KAPPA_Q_MIN, TWO_PI
from ..nets import Encoder, inverse_softplus
from ..specs import EmissionSpec, WalkSpec
from ..observation import class_recon
from ..vonmises import kl_vonmises, log_i0, mean_resultant, sample_vonmises_icdf


DEFAULTS = {"chain_rate_grid": 24, "rate_posterior": "categorical",
            "ar_delta_max": 3.1416, "ar_delta_rel": 0.0, "ar_phi0_grid": 0,
            "ar_phi0_anchor": False, "ar_rate_resid": 0.0}


def objective(out, beta: float, cfg):
    return out["recon"] - beta * out["kl"]


class ARChainVBPM(nn.Module):
    """The tutorial's generative model with the note's Form 1 posterior."""

    wants_raw = False
    emission_net = None

    def __init__(self, input_dim: int, d_model: int = 128,
                 rate_grid: int = 24, rate_lo: float = 0.020, rate_hi: float = 0.200,
                 emission: EmissionSpec | str = "triangle",
                 walk: WalkSpec | None = None,
                 tempo_prior_mu: float = -2.6827, tempo_prior_sigma: float = 0.3903,
                 rate_posterior: str = "categorical", encoder_pe: bool = False,
                 delta_max: float = 3.1416, delta_rel: float = 0.0,
                 phi0_grid: int = 0, phi0_anchor: bool = False,
                 rate_resid: float = 0.0):
        super().__init__()
        emission = EmissionSpec.coerce(emission)
        self.walk = walk or WalkSpec()
        self.emission_kind = emission.kind
        self.bump_kappa = float(emission.bump_kappa)
        self.recon_kind = emission.recon
        self.harmonics = int(emission.harmonics)
        if self.recon_kind == "class":
            self.wants_raw = True
            coef = torch.zeros(3, 2 * self.harmonics)
            coef[2, 0] = 1.0
            self.emission_coef = nn.Parameter(coef)
            self.emission_bias = nn.Parameter(torch.tensor([0.0, -2.5, -3.6]))
        self.rate_posterior = rate_posterior
        self.phi0_anchor = bool(phi0_anchor)
        self.rate_resid = float(rate_resid)
        self.rate_resid_head = None
        if self.rate_resid > 0.0 and rate_posterior == "categorical":
            self.rate_resid_head = nn.Linear(d_model, rate_grid)
            nn.init.zeros_(self.rate_resid_head.weight)
            nn.init.zeros_(self.rate_resid_head.bias)
        self.tempo_prior_mu = float(tempo_prior_mu)
        self.tempo_prior_sigma = float(tempo_prior_sigma)
        self.delta_max = float(delta_max)
        self.delta_rel = float(delta_rel)
        self.phi0_grid = int(phi0_grid)

        self.encoder = Encoder(input_dim, d_model,
                               kappa_physical=self.walk.kappa_physical,
                               use_pe=encoder_pe)

        assert not (self.phi0_anchor and self.phi0_grid > 0), \
            "ar_phi0_anchor replaces the phi0 grid; enable one or the other"
        if self.phi0_anchor:
            self.evidence_head = nn.Linear(d_model, 1)
            nn.init.zeros_(self.evidence_head.weight)
            nn.init.zeros_(self.evidence_head.bias)
            self.kappa0_raw = nn.Parameter(torch.tensor(5.0))
        elif self.phi0_grid > 0:
            self.phi0_grid_head = nn.Linear(d_model, self.phi0_grid)
            nn.init.zeros_(self.phi0_grid_head.weight)
            nn.init.zeros_(self.phi0_grid_head.bias)
            self.register_buffer(
                "phi0_vals", torch.arange(self.phi0_grid) * (TWO_PI / self.phi0_grid))
        else:
            self.phi0_head = nn.Linear(d_model, 3)
            nn.init.zeros_(self.phi0_head.weight)
            with torch.no_grad():
                self.phi0_head.bias.copy_(torch.tensor([1.0, 0.0, 0.0]))

        self.step_head = nn.Sequential(nn.Linear(d_model + 2, d_model), nn.ReLU(),
                                       nn.Linear(d_model, 3))
        last = self.step_head[-1]
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias.copy_(torch.tensor(
                [1.0, 0.0, inverse_softplus(self.walk.kappa_physical)]))

        if rate_posterior == "categorical":
            self.rate_head = nn.Linear(d_model, rate_grid)
            nn.init.zeros_(self.rate_head.weight)
            nn.init.zeros_(self.rate_head.bias)
        else:
            self.rate_head = nn.Linear(d_model, 2)
            nn.init.zeros_(self.rate_head.weight)
            with torch.no_grad():
                self.rate_head.bias.copy_(torch.tensor(
                    [self.tempo_prior_mu, inverse_softplus(0.05)]))

        if self.recon_kind != "class":
            self.emission_a = nn.Parameter(torch.tensor(-3.0))
            self.emission_b_raw = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("emission_b_floor", torch.tensor(0.0))

        rates = torch.exp(torch.linspace(math.log(rate_lo), math.log(rate_hi),
                                         rate_grid))
        self.register_buffer("rates", rates)
        z = (torch.log(rates) - tempo_prior_mu) / tempo_prior_sigma
        lp = -0.5 * z ** 2
        self.register_buffer("rate_log_prior", lp - torch.logsumexp(lp, 0))

    @property
    def emission_b(self):
        return self.emission_b_floor + nn.functional.softplus(self.emission_b_raw)

    @property
    def deployed_net(self):
        return self.encoder

    def class_logits(self, phi):
        """[..., 3] logits over (non-beat, beat, downbeat) at the sampled phase.

        The mainline's three-way emission verbatim: truncated Fourier series per
        class, only the downbeat class seeded (first cosine), because phi = 0 is
        the downbeat by the coordinate's definition and everything else is
        learned. This is the vocabulary that prices a metrical level: under the
        2x mode half the claimed downbeats sit on annotated beats and one shared
        shape must split its mass -- ~log 2 per conflicted event.
        """
        j = torch.arange(1, self.harmonics + 1, device=phi.device, dtype=phi.dtype)
        angle = phi[..., None] * j
        basis = torch.cat([angle.cos(), angle.sin()], dim=-1)
        return self.emission_bias + basis @ self.emission_coef.T

    def emission_logits(self, phi, mask=None):
        if self.recon_kind == "class":
            lp = torch.log_softmax(self.class_logits(phi), dim=-1)
            return lp[..., 2] - torch.logsumexp(lp[..., :2], dim=-1)
        if self.emission_kind == "triangle":
            wrapped = torch.atan2(torch.sin(phi), torch.cos(phi))
            return self.emission_a + self.emission_b * (1.0 - 2.0 * wrapped.abs() / math.pi)
        if self.emission_kind == "bump":
            peak = torch.exp(self.bump_kappa * (torch.cos(phi) - 1.0))
            return self.emission_a + self.emission_b * (2.0 * peak - 1.0)
        return self.emission_a + self.emission_b * torch.cos(phi)

    def _phi0_amortized(self, feats, sample):
        a1, a2, u = self.phi0_head(feats[:, 0]).unbind(-1)
        mu0 = torch.atan2(a2, a1)
        k0 = nn.functional.softplus(u) + KAPPA_Q_MIN
        phi0 = mu0 + (sample_vonmises_icdf(k0) if sample else torch.zeros_like(k0))
        kl0 = k0 * mean_resultant(k0) - log_i0(k0)
        return phi0, kl0

    def _anchor_phi0(self, feats, mask, rates_c, sample):
        """phi_0*(x, r): the closed-form anchor under each candidate ramp.

        The alignment is neither regressed nor enumerated: for a given rate it
        is the circular mean of the evidence phases folded under that ramp,
        recomputed every forward pass. The only learned objects are the
        per-frame evidence weight -- a local function of the audio -- and one
        posterior concentration around the anchored direction.
        """
        a = torch.sigmoid(self.evidence_head(feats)[..., 0]) * mask
        t = torch.arange(feats.shape[1], device=feats.device, dtype=feats.dtype)
        ramp = rates_c[..., None] * t
        re = (a[:, None, :] * torch.cos(ramp)).sum(-1)
        im = (a[:, None, :] * torch.sin(ramp)).sum(-1)
        mu0 = -torch.atan2(im, re)
        k0 = nn.functional.softplus(self.kappa0_raw) + KAPPA_Q_MIN
        if sample:
            eps = sample_vonmises_icdf(k0.expand(feats.shape[0]))
            mu0 = mu0 + eps[:, None]
        kl0 = (k0 * mean_resultant(k0) - log_i0(k0)).expand(feats.shape[0])
        return mu0, kl0

    def _rollout(self, feats, mask, rates, phi0, sample=True):
        """Sequential Form 1 chain per component.

        feats [B,T,D]; rates, phi0 [B,C] per-component columns. Returns the
        sampled (or mean) path [B,C,T], the summed per-step KL [B,C], and the
        step kappas [B,C,T-1].
        """
        B, T, D = feats.shape
        C = rates.shape[1]
        kp = torch.as_tensor(self.walk.kappa_physical, device=feats.device,
                             dtype=feats.dtype)
        phi = phi0
        phis = [phi]
        kls = feats.new_zeros(B, C)
        kappas = []
        for t in range(1, T):
            inp = torch.cat([phi.cos()[..., None], phi.sin()[..., None],
                             feats[:, t][:, None, :].expand(B, C, D)], dim=-1)
            d1, d2, u = self.step_head(inp).unbind(-1)
            delta = torch.atan2(d2, d1)
            if self.delta_rel > 0.0:
                cap = self.delta_rel * rates
                delta = cap * torch.tanh(delta / cap)
            elif self.delta_max < math.pi:
                delta = self.delta_max * torch.tanh(delta / self.delta_max)
            kq = nn.functional.softplus(u) + KAPPA_Q_MIN
            mu = phi + rates + delta
            eps = sample_vonmises_icdf(kq) if sample else torch.zeros_like(kq)
            phi = mu + eps
            pair = (mask[:, t] * mask[:, t - 1])[:, None]
            kls = kls + kl_vonmises(delta, kq, torch.zeros_like(delta), kp) * pair
            phis.append(phi)
            kappas.append(kq)
        return torch.stack(phis, 2), kls, torch.stack(kappas, 2)

    def _recon(self, phi, mask, y, pos_weight, cls=None):
        if self.recon_kind == "class":
            C = phi.shape[1]
            lp = torch.log_softmax(self.class_logits(phi), dim=-1)
            picked = lp.gather(-1, cls[:, None, :, None].expand(-1, C, -1, -1)
                               ).squeeze(-1)
            return (picked * mask[:, None, :]).sum(-1)
        e = self.emission_logits(phi)
        ll = (pos_weight * y[:, None, :] * -nn.functional.softplus(-e)
              + (1.0 - y)[:, None, :] * -nn.functional.softplus(e))
        return (ll * mask[:, None, :]).sum(-1)

    def _pooled(self, feats, mask):
        w = mask[..., None]
        return (feats * w).sum(1) / w.sum(1).clamp(min=1.0)

    def _components(self, feats, mask, pooled, sample):
        """(rates_c, phi0_c, log_prior_c, logits_c, kl0) for the categorical mixture."""
        B = feats.shape[0]
        R = self.rates.shape[0]
        rate_logits = self.rate_head(pooled)
        base = self.rates[None, :].expand(B, -1)
        if self.rate_resid_head is not None:
            resid = self.rate_resid * torch.tanh(self.rate_resid_head(pooled))
            base = base * torch.exp(resid)
        if self.phi0_anchor:
            phi0_c, kl0 = self._anchor_phi0(feats, mask, base, sample)
            return base, phi0_c, self.rate_log_prior, rate_logits, kl0
        if self.phi0_grid > 0:
            N = self.phi0_grid
            p0_logits = self.phi0_grid_head(pooled)
            logits = (rate_logits[:, :, None] + p0_logits[:, None, :]).reshape(B, R * N)
            log_prior = (self.rate_log_prior[:, None].expand(R, N)
                         - math.log(N)).reshape(-1)
            rates_c = base[:, :, None].expand(B, R, N).reshape(B, R * N)
            phi0_c = self.phi0_vals[None, :].expand(R, N).reshape(-1)[None].expand(B, -1)
            kl0 = feats.new_zeros(B)
        else:
            logits = rate_logits
            log_prior = self.rate_log_prior
            rates_c = base
            p0, kl0 = self._phi0_amortized(feats, sample)
            phi0_c = p0[:, None].expand(-1, rates_c.shape[1])
        return rates_c, phi0_c, log_prior, logits, kl0

    def forward(self, h, mask, y, samples: int = 1, pos_weight: float = 1.0,
                raw=None):
        cls = None
        if self.recon_kind == "class":
            assert raw is not None, "the three-way emission needs the batch's class targets"
            cls = raw["cls"].to(h.device)
        feats = self.encoder.features(h, mask)
        pooled = self._pooled(feats, mask)
        B = feats.shape[0]

        if self.rate_posterior == "categorical":
            rates_c, phi0_c, log_prior, logits, kl0 = self._components(
                feats, mask, pooled, sample=True)
            phi, kl_chain, kq = self._rollout(feats, mask, rates_c, phi0_c)
            recon_c = self._recon(phi, mask, y, pos_weight, cls)
            log_qc = torch.log_softmax(logits + log_prior[None], dim=-1)
            qc = log_qc.exp()
            kl_mix = (qc * (log_qc - log_prior[None])).sum(-1)
            recon = (qc * recon_c).sum(-1)
            kl = (qc * kl_chain).sum(-1) + kl0 + kl_mix
            best = log_qc.argmax(-1)
            rate_best = rates_c.gather(1, best[:, None])[:, 0]
        else:
            mu_lr, s_raw = self.rate_head(pooled).unbind(-1)
            sigma = nn.functional.softplus(s_raw) + 1e-4
            log_r = mu_lr + sigma * torch.randn_like(sigma)
            rates_c = torch.exp(log_r)[:, None]
            p0, kl0 = self._phi0_amortized(feats, sample=True)
            phi, kl_chain, kq = self._rollout(feats, mask, rates_c, p0[:, None])
            recon = self._recon(phi, mask, y, pos_weight, cls)[:, 0]
            kl_mix = (torch.log(torch.tensor(self.tempo_prior_sigma))
                      - torch.log(sigma)
                      + (sigma ** 2 + (mu_lr - self.tempo_prior_mu) ** 2)
                      / (2.0 * self.tempo_prior_sigma ** 2) - 0.5)
            kl = kl_chain[:, 0] + kl0 + kl_mix
            best = torch.zeros(B, dtype=torch.long, device=feats.device)
            rate_best = torch.exp(mu_lr)

        idx = best[:, None, None]
        phi_best = phi.gather(1, idx.expand(-1, 1, phi.shape[2])).squeeze(1)
        kq_best = kq.gather(1, idx.expand(-1, 1, kq.shape[2])).squeeze(1)
        kappa = torch.cat([kq_best[:, :1], kq_best], dim=1)

        return {"elbo": recon - kl, "recon": recon, "kl": kl,
                "phi": phi_best, "kappa": kappa, "rate": rate_best,
                "kl_mix": kl_mix.mean(), "kl0": kl0.mean()}

    @torch.no_grad()
    def infer_phase(self, h, mask=None):
        assert not self.training, "deployment path must run in eval mode"
        if mask is None:
            mask = torch.ones(h.shape[:2], device=h.device, dtype=h.dtype)
        feats = self.encoder.features(h, mask)
        pooled = self._pooled(feats, mask)
        if self.rate_posterior == "categorical":
            rates_c, phi0_c, log_prior, logits, _kl0 = self._components(
                feats, mask, pooled, sample=False)
            best = torch.log_softmax(logits + log_prior[None], dim=-1).argmax(-1)
            idx = best[:, None]
            rates = rates_c.gather(1, idx)
            phi0 = phi0_c.gather(1, idx)
        else:
            mu_lr, _ = self.rate_head(pooled).unbind(-1)
            rates = torch.exp(mu_lr)[:, None]
            phi0 = self._phi0_amortized(feats, sample=False)[0][:, None]
        phi, _kl, _kq = self._rollout(feats, mask, rates, phi0, sample=False)
        return phi[:, 0]

    @torch.no_grad()
    def emission_probs(self, h, mask=None):
        return torch.sigmoid(self.emission_logits(self.infer_phase(h, mask), mask))


def build_model(cfg, input_dim: int) -> ARChainVBPM:
    return ARChainVBPM(input_dim,
                       rate_grid=cfg.chain_rate_grid,
                       emission=EmissionSpec(kind=cfg.emission,
                                             bump_kappa=cfg.emission_bump_kappa,
                                             recon=getattr(cfg, "emission_recon",
                                                           "event")),
                       walk=WalkSpec(kappa_physical=cfg.kappa_physical),
                       tempo_prior_mu=cfg.tempo_prior_mu,
                       tempo_prior_sigma=cfg.tempo_prior_sigma,
                       rate_posterior=cfg.rate_posterior,
                       delta_max=cfg.ar_delta_max,
                       delta_rel=cfg.ar_delta_rel,
                       phi0_grid=cfg.ar_phi0_grid,
                       rate_resid=cfg.ar_rate_resid,
                       phi0_anchor=cfg.ar_phi0_anchor)


def epoch_note(model, probe) -> str:
    if getattr(model, "wants_raw", False):
        return ""
    out = model(probe["h"], probe["mask"], probe["y"])
    return (f"  rate {float(out['rate'].mean()):.4f}"
            f"  kl_mix {float(out['kl_mix']):.2f}  kl0 {float(out['kl0']):.2f}")
