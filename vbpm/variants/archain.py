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
from ..specs import ChainSpec, EmissionSpec, RateSpec, TempoWalkSpec, WalkSpec
from ..observation import class_recon
from ..vonmises import kl_vonmises, log_i0, mean_resultant, sample_vonmises_icdf


DEFAULTS = {"chain_rate_grid": 24, "rate_posterior": "categorical",
            "ar_delta_max": 3.1416, "ar_delta_rel": 0.0, "ar_phi0_grid": 0,
            "ar_phi0_anchor": False, "ar_rate_resid": 0.0, "ar_stride": 1,
            "ar_rate_lo": 0.020, "ar_rate_hi": 0.200,
            "ar_phase_kernel": "vonmises", "ar_tempo_walk": False,
            "ar_tempo_kernel": "cauchy", "ar_beat_subdiv": 4}


def objective(out, beta: float, cfg):
    return out["recon"] - beta * out["kl"]


class VBPM(nn.Module):
    """The tutorial's generative model with the note's Form 1 posterior."""

    wants_raw = False
    emission_net = None

    def __init__(self, input_dim: int, d_model: int = 128,
                 emission: EmissionSpec | str = "triangle",
                 walk: WalkSpec | None = None,
                 rate: RateSpec | None = None,
                 chain: ChainSpec | None = None,
                 tempo_walk: TempoWalkSpec | None = None,
                 encoder_pe: bool = False):
        super().__init__()
        # the specs ARE the configuration: nothing below copies a field out of
        # one, so there is exactly one place each knob is written down.
        self.emission = EmissionSpec.coerce(emission)
        self.walk = walk or WalkSpec()
        self.rate = rate or RateSpec()
        self.chain = chain or ChainSpec()
        self.tempo = tempo_walk or TempoWalkSpec()

        assert not (self.rate.resid > 0.0 and self.rate.posterior != "categorical"), \
            "a rate residual only acts on the categorical rate posterior"
        assert not (self.chain.phi0 == "anchor" and self.chain.phi0_grid > 0), \
            "the anchor replaces the phi0 grid; enable one or the other"

        if self.recon_kind in ("class", "tied"):
            self.wants_raw = True
        if self.recon_kind == "tied":
            self.tied_a = nn.Parameter(torch.tensor([-3.0, -3.0]))
            self.tied_b_raw = nn.Parameter(torch.tensor([1.0, 1.0]))
        if self.recon_kind == "class":
            coef = torch.zeros(3, 2 * self.emission.harmonics)
            coef[2, 0] = 1.0
            self.emission_coef = nn.Parameter(coef)
            self.emission_bias = nn.Parameter(torch.tensor([0.0, -2.5, -3.6]))
        self.rate_resid_head = None
        if self.rate.resid > 0.0 and self.rate.posterior == "categorical":
            self.rate_resid_head = nn.Linear(d_model, self.rate.grid)
            nn.init.zeros_(self.rate_resid_head.weight)
            nn.init.zeros_(self.rate_resid_head.bias)

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

        n_out = 5 if self.tempo_walk else 3
        self.step_head = nn.Sequential(nn.Linear(d_model + 2, d_model), nn.ReLU(),
                                       nn.Linear(d_model, n_out))
        last = self.step_head[-1]
        nn.init.zeros_(last.weight)
        kappa_step = self.walk.kappa_physical
        if self.stride > 1:
            from .vmchain import a_ratio, inv_a
            kp = torch.tensor(float(self.walk.kappa_physical))
            kappa_step = float(inv_a(a_ratio(kp) ** self.stride))
        self.kappa_p_stride = float(kappa_step)
        self.gamma_stride = self.stride / math.sqrt(self.walk.kappa_physical)
        bias = [1.0, 0.0, inverse_softplus(kappa_step)]
        if self.tempo_walk:
            walk_scale = (self.walk_sigma * self.stride if self.tempo_kernel == "cauchy"
                          else self.walk_sigma * math.sqrt(self.stride))
            bias += [0.0, inverse_softplus(walk_scale)]
        with torch.no_grad():
            last.bias.copy_(torch.tensor(bias))

        if self.rate.posterior == "categorical":
            self.rate_head = nn.Linear(d_model, self.rate.grid)
            nn.init.zeros_(self.rate_head.weight)
            nn.init.zeros_(self.rate_head.bias)
        else:
            self.rate_head = nn.Linear(d_model, 2)
            nn.init.zeros_(self.rate_head.weight)
            with torch.no_grad():
                self.rate_head.bias.copy_(torch.tensor(
                    [self.tempo_prior_mu, inverse_softplus(0.05)]))

        if self.recon_kind not in ("class", "tied"):
            self.emission_a = nn.Parameter(torch.tensor(-3.0))
            self.emission_b_raw = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("emission_b_floor", torch.tensor(0.0))

        self.log_rate_lo, self.log_rate_hi = math.log(self.rate.lo), math.log(self.rate.hi)
        # the residual moves a candidate off its bin, so the walk's support must
        # admit the shifted rate or a positive residual on the top bin is
        # discarded at the first step while the anchor still used it.
        self.log_rate_lo_eff = self.log_rate_lo - float(self.rate.resid)
        self.log_rate_hi_eff = self.log_rate_hi + float(self.rate.resid)
        rates = torch.exp(torch.linspace(math.log(self.rate.lo), math.log(self.rate.hi),
                                         self.rate.grid))
        self.register_buffer("rates", rates)
        z = (torch.log(rates) - self.tempo_prior_mu) / self.tempo_prior_sigma
        lp = -0.5 * z ** 2
        self.register_buffer("rate_log_prior", lp - torch.logsumexp(lp, 0))

    # ---- read-throughs: the spec is the field, these are just short names -----
    @property
    def recon_kind(self):
        return self.emission.recon

    @property
    def beat_subdiv(self):
        """The beat channel's subdivision. Meter stays deferred as a LATENT; this
        is a named constant, not a hidden hardcode, and becomes the enumerated M
        the day meter returns."""
        return self.emission.subdiv

    @property
    def stride(self):
        return self.chain.stride

    @property
    def phase_kernel(self):
        return self.chain.phase_kernel

    @property
    def delta_max(self):
        return self.chain.delta_max

    @property
    def delta_rel(self):
        return self.chain.delta_rel

    @property
    def phi0_anchor(self):
        return self.chain.phi0 == "anchor"

    @property
    def phi0_grid(self):
        return self.chain.phi0_grid

    @property
    def rate_posterior(self):
        return self.rate.posterior

    @property
    def rate_resid(self):
        return self.rate.resid

    @property
    def tempo_walk(self):
        return self.tempo.enabled

    @property
    def tempo_kernel(self):
        return self.tempo.kernel

    @property
    def walk_sigma(self):
        return self.walk.walk_sigma

    @property
    def tempo_prior_mu(self):
        return self.walk.tempo_mu

    @property
    def tempo_prior_sigma(self):
        return self.walk.tempo_sigma

    @property
    def emission_kind(self):
        return self.emission.kind

    @property
    def bump_kappa(self):
        return self.emission.bump_kappa

    @property
    def harmonics(self):
        return self.emission.harmonics

    @property
    def emission_b(self):
        return self.emission_b_floor + nn.functional.softplus(self.emission_b_raw)

    @property
    def deployed_net(self):
        return self.encoder

    def tied_logits(self, phi):
        """(downbeat, beat) log-odds from ONE frozen triangle read twice.

        The downbeat channel is the triangle at the bar phase; the beat channel
        is the SAME shape at the subdivided phase. Only two gains are learnable,
        so the optimizer cannot reshape its way out of a metrical level -- the
        failure the free Fourier shapes showed when they co-adapted until 2x
        paid better than the truth. Measured training-free on ten windows: this
        pairing carries 145.9 nats of phase information against the free
        shapes' 30.5, and resists prior-scale diffusion by 69.2 nats against
        7.5.
        """
        b = nn.functional.softplus(self.tied_b_raw)
        wrapped = torch.atan2(torch.sin(phi), torch.cos(phi))
        tri_db = 1.0 - 2.0 * wrapped.abs() / math.pi
        sub_phi = self.beat_subdiv * phi
        wrapped_b = torch.atan2(torch.sin(sub_phi), torch.cos(sub_phi))
        tri_bt = 1.0 - 2.0 * wrapped_b.abs() / math.pi
        return (self.tied_a[0] + b[0] * tri_db,
                self.tied_a[1] + b[1] * tri_bt)

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
        if self.recon_kind == "tied":
            return self.tied_logits(phi)[0]
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

    def step_kl(self, delta, kq, kp, x):
        """KL( q(step) || p(step) ) for the step whose realised value is x.

        Under a von Mises prior this is the closed form. Under the wrapped
        Cauchy the prior has a tight core and a heavy tail -- the shape the
        corpus prefers by 0.77 nats/bar, and the one a single concentration
        cannot express: phase follows tempo almost exactly, yet a fermata or a
        tempo break stays affordable. Its density is closed form and
        normalised, so only the cross term needs the sample:

            KL = -H(q) - E_q[log p]  ~=  -H(q) - log p(x),   x ~ q

        H(q) is exact, so the estimator is unbiased with the variance of one
        term rather than of the whole divergence. The core is matched to the
        von Mises it replaces by CORE width (gamma = 1/sqrt(kappa) is HWHM
        matching, not resultant matching -- the two differ by an order of
        magnitude, so swapping the kernel at fixed kappa_physical is not a
        neutral tail change and the two arms are not like-for-like).
        """
        if self.phase_kernel == "vonmises":
            return kl_vonmises(delta, kq, torch.zeros_like(delta), kp)
        # S independent wrapped Cauchy steps compose to a wrapped Cauchy of
        # scale S*gamma, NOT sqrt(S)*gamma: reading gamma off the von Mises
        # stride concentration mixed the two conventions and made the prior a
        # factor sqrt(S) too tight. tempo_step_kl already had this right.
        gamma = self.gamma_stride
        rho = torch.exp(-torch.as_tensor(gamma, device=delta.device,
                                         dtype=delta.dtype)).clamp(1e-4, 1 - 1e-6)
        denom = 1.0 + rho ** 2 - 2.0 * rho * torch.cos(x)
        log_p = torch.log1p(-rho ** 2) - torch.log(denom) - math.log(TWO_PI)
        entropy = math.log(TWO_PI) + log_i0(kq) - kq * mean_resultant(kq)
        return -entropy - log_p

    def _bound_log_rate(self, u):
        """Keep the walk inside the rate support WITHOUT an absorbing rail.

        torch.clamp has exactly zero gradient past the bound, so a walk that
        reaches the rail stops receiving any likelihood gradient and only the
        KL pushes back -- the absorbing-clamp pathology this project already
        recorded once. A tanh saturation is bounded by the same interval and
        differentiable everywhere, so a state at the edge still learns which
        way to move.
        """
        mid = 0.5 * (self.log_rate_lo_eff + self.log_rate_hi_eff)
        half = 0.5 * (self.log_rate_hi_eff - self.log_rate_lo_eff)
        return mid + half * torch.tanh((u - mid) / half)

    def tempo_step_kl(self, w, s_w):
        """KL of one tempo increment: exact Gaussian entropy, sampled cross term.

        The increment law is where the heavy tail belongs. A ritardando is a
        PERSISTENT change, so the phase channel cannot carry it -- an i.i.d.
        innovation pays for the same deviation at every step and never learns
        that the tempo moved. Priced on the corpus's own rubato: the same tempo
        path costs 17490 nats under the shipped Gaussian walk and 33 under a
        Cauchy of the SAME scale, while a steady window still prefers the
        steady path under either.
        """
        S = max(self.stride, 1)
        if self.tempo_kernel == "cauchy":
            scale = self.walk_sigma * S
            log_p = -math.log(math.pi * scale) - torch.log1p((w / scale) ** 2)
        elif self.tempo_kernel == "laplace":
            scale = self.walk_sigma * math.sqrt(S)
            log_p = -w.abs() / scale - math.log(2.0 * scale)
        else:
            scale = self.walk_sigma * math.sqrt(S)
            log_p = -0.5 * (w / scale) ** 2 - math.log(scale) \
                - 0.5 * math.log(TWO_PI)
        entropy = 0.5 * math.log(TWO_PI * math.e) + torch.log(s_w)
        return -entropy - log_p

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
        kl0 = (k0 * mean_resultant(k0) - log_i0(k0)).expand_as(mu0)
        return mu0, kl0

    def _rollout(self, feats, mask, rates, phi0, sample=True):
        """Sequential Form 1 chain per component.

        feats [B,T,D]; rates, phi0 [B,C] per-component columns. Returns the
        sampled (or mean) path [B,C,T], the summed per-step KL [B,C], and the
        step kappas [B,C,T-1].
        """
        B, T, D = feats.shape
        C = rates.shape[1]
        assert not (self.tempo_walk and self.stride <= 1), \
            "the tempo walk is a per-stride latent; run it with ar_stride > 1"
        if self.stride > 1:
            return self._rollout_strided(feats, mask, rates, phi0, sample)
        kp = torch.as_tensor(self.walk.kappa_physical, device=feats.device,
                             dtype=feats.dtype)
        phi = phi0
        phis = [phi]
        kls = feats.new_zeros(B, C)
        kappas = []
        lin1, act, lin2 = self.step_head[0], self.step_head[1], self.step_head[2]
        proj = feats @ lin1.weight[:, 2:].T + lin1.bias
        w_phi = lin1.weight[:, :2].T
        for t in range(1, T):
            trig = torch.stack([phi.cos(), phi.sin()], dim=-1)
            d1, d2, u = lin2(act(trig @ w_phi + proj[:, t][:, None, :])).unbind(-1)
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
            kls = kls + self.step_kl(delta, kq, kp, delta + eps) * pair
            phis.append(phi)
            kappas.append(kq)
        return torch.stack(phis, 2), kls, torch.stack(kappas, 2)

    def _rollout_strided(self, feats, mask, rates, phi0, sample=True):
        """Form 1 at stride S: decisions every S frames, evidence at every frame.

        This is a DIFFERENT generative model, not a coarser reading of the same
        one, and the difference is worth stating because the audit caught the
        earlier docstring claiming otherwise. Two approximations: the S-step
        kernel is a von Mises whose RESULTANT matches the S-fold convolution
        (first Fourier coefficient only -- 9e-6 nats per step at kappa 383,
        S 8, harmless but not exact), and the within-stride path is the
        deterministic interpolation rather than the Brownian bridge the fine
        chain implies (bridge sd peaks at sqrt(S/4/kappa)). Every frame still
        pays its own emission evidence. Consequence: a stride > 1 ELBO is NOT
        comparable term-for-term with a stride = 1 one.
        """
        B, T, D = feats.shape
        C = rates.shape[1]
        S = self.stride
        kp = torch.as_tensor(self.kappa_p_stride, device=feats.device,
                             dtype=feats.dtype)
        lin1, act, lin2 = self.step_head[0], self.step_head[1], self.step_head[2]
        proj = feats @ lin1.weight[:, 2:].T + lin1.bias
        w_phi = lin1.weight[:, :2].T
        phi = phi0
        knots = [phi]
        kls = feats.new_zeros(B, C)
        kappas = []
        log_rate = torch.log(rates)
        steps = list(range(S, T, S))
        for t in steps:
            trig = torch.stack([phi.cos(), phi.sin()], dim=-1)
            out = lin2(act(trig @ w_phi + proj[:, t][:, None, :]))
            if self.tempo_walk:
                d1, d2, u, mu_w, s_raw = out.unbind(-1)
                s_w = nn.functional.softplus(s_raw) + 1e-6
                eps_w = torch.randn_like(s_w) if sample else torch.zeros_like(s_w)
                w = mu_w + s_w * eps_w
                # The walk owns an UNBOUNDED state, and a heavy tail supplies
                # almost no restoring force, so an unclamped log-rate runs away
                # (measured: nan within one run). The model's own rate support
                # is the physical bound, and reverting the step toward the
                # corpus mean is what tchain's kernel does for the same reason:
                # a driftless walk has a uniform stationary law and expresses no
                # preference among tempi.
                log_rate = self._bound_log_rate(log_rate + w)
                rate_t = log_rate.exp()
                kls = kls + self.tempo_step_kl(w, s_w) \
                    * (mask[:, t] * mask[:, t - S])[:, None]
            else:
                d1, d2, u = out.unbind(-1)
                rate_t = rates
            delta = torch.atan2(d2, d1)
            if self.delta_rel > 0.0:
                cap = self.delta_rel * rate_t * S
                delta = cap * torch.tanh(delta / cap)
            elif self.delta_max < math.pi:
                delta = self.delta_max * torch.tanh(delta / self.delta_max)
            kq = nn.functional.softplus(u) + KAPPA_Q_MIN
            mu = phi + rate_t * S + delta
            eps = sample_vonmises_icdf(kq) if sample else torch.zeros_like(kq)
            phi = mu + eps
            pair = (mask[:, t] * mask[:, t - S])[:, None]
            kls = kls + self.step_kl(delta, kq, kp, delta + eps) * pair
            knots.append(phi)
            kappas.append(kq)
        del log_rate
        knot = torch.stack(knots, 2)
        seg = (knot[..., 1:] - knot[..., :-1]) / S
        frac = torch.arange(S, device=feats.device, dtype=feats.dtype)
        inner = knot[..., :-1, None] + seg[..., None] * frac
        path = inner.reshape(B, C, -1)
        tail = T - path.shape[-1]
        if tail > 0:
            last = path[..., -1:]
            ramp = rate_t[..., None] * torch.arange(
                1, tail + 1, device=feats.device, dtype=feats.dtype)
            path = torch.cat([path, last + ramp], dim=-1)
        kq_full = torch.stack(kappas, 2).repeat_interleave(S, dim=-1)
        kq_full = torch.cat([kq_full,
                             kq_full[..., -1:].expand(B, C, T - kq_full.shape[-1])],
                            dim=-1)[..., :T - 1]
        return path[..., :T], kls, kq_full

    def _recon(self, phi, mask, y, pos_weight, cls=None):
        if self.recon_kind == "tied":
            e_db, e_bt = self.tied_logits(phi)
            is_db = (cls == 2).to(phi.dtype)[:, None, :]
            # a downbeat IS a beat: the targets are written downbeat-last, so
            # cls == 2 at a downbeat and the beat channel must still fire there.
            # Testing cls == 1 told the beat channel "no beat" at exactly the
            # frames its own triangle peaks, so the two channels fought each
            # other at every bar line.
            is_bt = (cls >= 1).to(phi.dtype)[:, None, :]
            ll = (is_db * nn.functional.logsigmoid(e_db)
                  + (1.0 - is_db) * nn.functional.logsigmoid(-e_db)
                  + is_bt * nn.functional.logsigmoid(e_bt)
                  + (1.0 - is_bt) * nn.functional.logsigmoid(-e_bt))
            return (ll * mask[:, None, :]).sum(-1)
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
        log_prior = self.rate_log_prior[None].expand(B, -1)
        if self.rate_resid_head is not None:
            resid = self.rate_resid * torch.tanh(self.rate_resid_head(pooled))
            base = base * torch.exp(resid)
            # the residual moves each candidate off its bin centre, so pricing
            # it at the unshifted centre stops being a KL on a common support.
            # This is a CONDITIONAL model, so an x-dependent p(c | x) is
            # legitimate; what is not legitimate is scoring one rate and
            # charging for another.
            z = (torch.log(base) - self.tempo_prior_mu) / self.tempo_prior_sigma
            log_prior = torch.log_softmax(-0.5 * z ** 2, dim=-1)
        if self.phi0_anchor:
            phi0_c, kl0 = self._anchor_phi0(feats, mask, base, sample)
            return base, phi0_c, log_prior, rate_logits, kl0
        if self.phi0_grid > 0:
            N = self.phi0_grid
            p0_logits = self.phi0_grid_head(pooled)
            logits = (rate_logits[:, :, None] + p0_logits[:, None, :]).reshape(B, R * N)
            log_prior = (log_prior[:, :, None].expand(B, R, N)
                         - math.log(N)).reshape(B, R * N)
            rates_c = base[:, :, None].expand(B, R, N).reshape(B, R * N)
            phi0_c = self.phi0_vals[None, :].expand(R, N).reshape(-1)[None].expand(B, -1)
            kl0 = feats.new_zeros(B, R * N)
        else:
            logits = rate_logits
            rates_c = base
            p0, kl0 = self._phi0_amortized(feats, sample)
            phi0_c = p0[:, None].expand(-1, rates_c.shape[1])
            kl0 = kl0[:, None].expand(-1, rates_c.shape[1])
        return rates_c, phi0_c, log_prior, logits, kl0

    def forward(self, h, mask, y, samples: int = 1, pos_weight: float = 1.0,
                raw=None):
        cls = None
        if self.recon_kind in ("class", "tied"):
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
            log_qc = torch.log_softmax(logits + log_prior, dim=-1)
            qc = log_qc.exp()
            kl_mix = (qc * (log_qc - log_prior)).sum(-1)
            recon = (qc * recon_c).sum(-1)
            kl = (qc * (kl_chain + kl0)).sum(-1) + kl_mix
            best = log_qc.argmax(-1)
            rate_best = rates_c.gather(1, best[:, None])[:, 0]
        else:
            mu_lr, s_raw = self.rate_head(pooled).unbind(-1)
            sigma = nn.functional.softplus(s_raw) + 1e-4
            log_r = mu_lr + sigma * torch.randn_like(sigma)
            rates_c = torch.exp(log_r)[:, None]
            p0, kl0 = self._phi0_amortized(feats, sample=True)
            kl0 = kl0[:, None]
            phi, kl_chain, kq = self._rollout(feats, mask, rates_c, p0[:, None])
            recon = self._recon(phi, mask, y, pos_weight, cls)[:, 0]
            kl_mix = (torch.log(torch.tensor(self.tempo_prior_sigma))
                      - torch.log(sigma)
                      + (sigma ** 2 + (mu_lr - self.tempo_prior_mu) ** 2)
                      / (2.0 * self.tempo_prior_sigma ** 2) - 0.5)
            kl = kl_chain[:, 0] + kl0[:, 0] + kl_mix
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
            best = torch.log_softmax(logits + log_prior, dim=-1).argmax(-1)
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


def build_model(cfg, input_dim: int) -> VBPM:
    """One VBPM from a config: five specs, no loose floats."""
    from .base import common_kwargs
    return VBPM(input_dim,
                rate=RateSpec(grid=cfg.chain_rate_grid, lo=cfg.ar_rate_lo,
                              hi=cfg.ar_rate_hi, posterior=cfg.rate_posterior,
                              resid=cfg.ar_rate_resid),
                chain=ChainSpec(stride=cfg.ar_stride,
                                phase_kernel=cfg.ar_phase_kernel,
                                delta_max=cfg.ar_delta_max,
                                delta_rel=cfg.ar_delta_rel,
                                phi0="anchor" if cfg.ar_phi0_anchor else "amortized",
                                phi0_grid=cfg.ar_phi0_grid),
                tempo_walk=TempoWalkSpec(enabled=cfg.ar_tempo_walk,
                                         kernel=cfg.ar_tempo_kernel),
                **common_kwargs(cfg))


def epoch_note(model, probe) -> str:
    if getattr(model, "wants_raw", False):
        return ""
    out = model(probe["h"], probe["mask"], probe["y"])
    return (f"  rate {float(out['rate'].mean()):.4f}"
            f"  kl_mix {float(out['kl_mix']):.2f}  kl0 {float(out['kl0']):.2f}")
