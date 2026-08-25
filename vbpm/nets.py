"""The encoder trunk, the emission, the prior chain, and the posterior."""
from __future__ import annotations

import math
import torch
from torch import nn

from .constants import (EMISSION_FIT_A, EMISSION_FIT_B, EMISSION_FIT_TAU,
                        EMISSION_FIT_TAU_BACK, FPS, TEMPO_PRIOR_MU,
                        TOLERANCE_SECONDS, TWO_PI)
from .specs import EmissionSpec, RateSpec, WalkSpec

N_HARM = 12             # band limit of the recognition potentials
GAMMA = 0.0363          # corpus median per-bar |dlog rate|, the Cauchy scale
N_GRID = 128            # quadrature nodes on the phase circle
METER_STAY = 0.999      # mass a meter keeps across a bar line


def sinusoidal_encoding(length: int, dim: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding [length, dim]."""
    pos = torch.arange(length, dtype=torch.float32)[:, None]
    scale = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32)
                      * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim)
    pe[:, 0::2] = torch.sin(pos * scale)
    pe[:, 1::2] = torch.cos(pos * scale)
    return pe


class Encoder(nn.Module):
    """The shared trunk (tutorial 9.2), reading AUDIO ONLY.

    It produces context, not posterior parameters: the variants build their own
    heads on top of features(). forward() is features(), so the deployed-net
    target-blindness control still has a signature to inspect.
    """

    def __init__(self, input_dim: int, d_model: int = 128, heads: int = 4, layers: int = 2,
                 max_len: int = 4096, use_pe: bool = False):
        super().__init__()
        self.d_model = d_model
        self.use_pe = use_pe

        self.proj = nn.Linear(input_dim, d_model)
        self.in_drop = nn.Dropout1d(0.1)
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=4 * d_model,
                                           dropout=0.0, activation="relu",
                                           batch_first=True, norm_first=False)
        self.blocks = nn.TransformerEncoder(layer, layers)

        if use_pe:
            self.register_buffer("pe", sinusoidal_encoding(max_len, d_model))

    def features(self, h, mask=None):
        """[B, T, D] -> [B, T, d_model]: the trunk shared by every head."""
        pad = None if mask is None else (mask <= 0)

        h = self.proj(h) * math.sqrt(self.d_model)
        if self.use_pe:
            h = h + self.pe[:h.shape[1]]
        h = self.in_drop(h.transpose(1, 2)).transpose(1, 2)

        return self.blocks(h, src_key_padding_mask=pad)

    def forward(self, h, mask=None):
        """The trunk output; the encoder has no head of its own."""
        return self.features(h, mask)


USE_COMPILE = False


def _fwd_step(a, kernel, e, floor):
    """One forward propagation, tilt, and renormalisation."""
    a = (a.reshape(a.shape[0], -1) @ kernel).view_as(a) * e
    s = a.sum(dim=(1, 2), keepdim=True).clamp_min(floor)
    return a / s, s.squeeze(-1).squeeze(-1).log()


def _bwd_step(b, kernel, e, floor):
    """One backward propagation and renormalisation."""
    b = ((b * e).reshape(b.shape[0], -1) @ kernel.T).view_as(b)
    return b / b.sum(dim=(1, 2), keepdim=True).clamp_min(floor)


_STEPS = None


def chain_steps():
    """(fwd_step, bwd_step), compiled once if the backend accepts them."""
    global _STEPS
    if _STEPS is None:
        if USE_COMPILE:
            try:
                _STEPS = (torch.compile(_fwd_step, dynamic=False),
                          torch.compile(_bwd_step, dynamic=False))
            except Exception:
                _STEPS = (_fwd_step, _bwd_step)
        else:
            _STEPS = (_fwd_step, _bwd_step)
    return _STEPS


class PosteriorModel(nn.Module):
    """q(path | x) = p_prior(path) exp(sum_t evidence_t(phi_t) + head(c_0)) / Z.

    Marginal carriage: the chain prior tilted by per-frame recognition potentials
    read locally from the trunk, combined by exact forward-backward over the
    prior's quadrature. Nothing is sampled, so C_commit = 0 and

        KL(q || p) = E_q[sum_t evidence_t + head] - log Z        (exact, >= 0).

    The prior is an ARGUMENT of smooth(), not a submodule: it appears in q's
    definition, but it belongs to the generative model and must not be
    registered twice.
    """

    def __init__(self, input_dim: int, d_model: int, prior: "PriorModel",
                 n_harm: int = N_HARM, encoder_pe: bool = False):
        super().__init__()
        # The prior is READ here and passed back to smooth(); it is deliberately
        # not stored. Assigning it would register it as a child of this module
        # as well as of the model, and its buffers would then appear twice in
        # every state_dict.
        self.encoder = Encoder(input_dim, d_model, use_pe=encoder_pe)
        self.rate_head = nn.Linear(d_model, prior.n_rates)
        nn.init.zeros_(self.rate_head.weight)
        nn.init.zeros_(self.rate_head.bias)
        self.meter_head = (nn.Linear(d_model, len(prior.meters))
                           if prior.meters else None)
        if self.meter_head is not None:
            nn.init.zeros_(self.meter_head.weight)
            nn.init.zeros_(self.meter_head.bias)
        self.evidence_head = nn.Linear(d_model, 2 * n_harm)
        nn.init.zeros_(self.evidence_head.weight)
        nn.init.zeros_(self.evidence_head.bias)
        j = torch.arange(1, n_harm + 1, dtype=torch.float32)
        self.register_buffer("evidence_basis",
                             torch.cat([torch.cos(prior.grid[:, None] * j),
                                        torch.sin(prior.grid[:, None] * j)], dim=1))

    def forward(self, h, mask, prior):
        """(evidence, rate/meter logits, marginals, log_z) for one window."""
        evidence, log_q_rate0, log_q_meter = self.potentials(h, mask)
        marginals = self.smooth(evidence, log_q_rate0, prior, log_q_meter)
        return (evidence, log_q_rate0, log_q_meter) + marginals

    def potentials(self, h, mask):
        """(evidence [B,T,N], head [B,C]): the local evidence, and the initial rate."""
        feats = self.encoder.features(h, mask)
        pooled = self._pooled(feats, mask)
        meter = (torch.log_softmax(self.meter_head(pooled), dim=-1)
                 if self.meter_head is not None else None)
        return (self._evidence(feats, mask),
                torch.log_softmax(self.rate_head(pooled), dim=-1), meter)

    def _pooled(self, feats, mask):
        w = mask[..., None]
        return (feats * w).sum(1) / w.sum(1).clamp(min=1.0)

    def _evidence(self, feats, mask):
        """[B,T,N]: per-frame recognition potentials, band-limited on the circle.

        Shifted to a per-frame max of 0. _smooth_joint runs the recursions in
        the LINEAR domain, so exp(evidence) overflows float32 once any potential
        passes ~88 -- which is reachable, and the whole window goes NaN at once.
        The ELBO is invariant to a per-frame constant in evidence: it multiplies q by
        exp(sum_t c_t), which cancels in the normalisation, and shifts E_q[sum
        evidence] and log Z by the same sum_t c_t, so kl = expected_evidence - log_z is unchanged.
        """
        evidence = self.evidence_head(feats) @ self.evidence_basis.T
        evidence = evidence - evidence.max(dim=-1, keepdim=True).values
        return evidence * mask[..., None]

    def smooth(self, evidence, log_q_rate0, prior, log_q_meter=None):
        """(q_phase, q_rate, q_meter, log_z): marginals only, never the joint."""
        B, T, N = evidence.shape
        M = len(prior.meters)
        FL = 1e-30
        E = evidence.exp()
        red = (1, 2, 3) if M else (1, 2)
        tilt = (slice(None), None, None, slice(None)) if M else (slice(None), None, slice(None))
        a = torch.softmax(prior.rate_log_prior, 0)[None, :, None] \
            * log_q_rate0.exp()[:, :, None] * E[:, 0][:, None, :] / N
        if M:
            a = a[:, None] * torch.softmax(prior.meter_log_prior, 0)[None, :, None, None]
            if log_q_meter is not None:
                a = a * log_q_meter.exp()[:, :, None, None]
        s = a.sum(dim=red, keepdim=True).clamp_min(FL)
        logz = s.reshape(B).log()
        a = a / s
        A = [a]
        for t in range(1, T):
            a = prior.fwd(a) * E[:, t][tilt]
            s = a.sum(dim=red, keepdim=True).clamp_min(FL)
            logz = logz + s.reshape(B).log()
            a = a / s
            A.append(a)
        b = torch.ones_like(a) / a[0].numel()
        Bl = [b]
        for t in range(T - 1, 0, -1):
            b = prior.bwd(b * E[:, t][tilt])
            b = b / b.sum(dim=red, keepdim=True).clamp_min(FL)
            Bl.append(b)
        Bl.reverse()
        phase, rate, meter = [], [], []
        for t in range(T):
            post = A[t] * Bl[t]
            post = post / post.sum(dim=red, keepdim=True).clamp_min(FL)
            if M:
                phase.append(post.sum(2))
                rate.append(post.sum((1, 3)))
                meter.append(post.sum((2, 3)))
            else:
                phase.append(post.sum(1))
                rate.append(post.sum(2))
        return (torch.stack(phase, 1), torch.stack(rate, 1),
                torch.stack(meter, 1) if M else None, logz)

    def unwrap(self, qn, grid):
        """[B,T,N] marginals -> [B,T] unwrapped circular-mean phase."""
        re = (qn * grid.cos()).sum(-1)
        im = (qn * grid.sin()).sum(-1)
        ang = torch.atan2(im, re)
        d = torch.remainder(ang[:, 1:] - ang[:, :-1] + math.pi, TWO_PI) - math.pi
        phi = torch.cat([ang[:, :1], ang[:, :1] + torch.cumsum(d, 1)], 1)
        return phi, re, im


class EmissionModel(nn.Module):
    """p(y_t | phi_t): one fixed shape, two scalars, one downbeat channel.

    The shape is frozen and only a baseline and a gain are learnable, so the
    optimizer cannot grow a second peak and make a wrong metrical level pay as
    well as the truth.
    """

    def __init__(self, spec: EmissionSpec):
        super().__init__()
        self.spec = spec

        fit = spec.fit_init
        self.a = nn.Parameter(torch.tensor(EMISSION_FIT_A if fit else -3.0))
        self.b_raw = nn.Parameter(torch.tensor(EMISSION_FIT_B if fit else 1.0))

        self.register_buffer("b_floor", torch.tensor(0.0))

        # The decay length and the band's width are the same quantity -- how far
        # past the downbeat the emission still fires -- fixed as a DURATION at the
        # scoring tolerance and converted to phase at the prior-mean rate.
        tau = TOLERANCE_SECONDS * FPS * math.exp(TEMPO_PRIOR_MU)
        self.log_tau = nn.Parameter(torch.tensor(
            math.log(EMISSION_FIT_TAU if fit else tau)))
        # `alaplace` only: the reach BACKWARD from the onset. Starting it equal to
        # the forward reach starts the shape symmetric, so the asymmetry has to be
        # earned; a one-sided shape is the tau_back -> 0 corner of the same family.
        self.log_tau_back = nn.Parameter(torch.tensor(
            math.log(EMISSION_FIT_TAU_BACK if fit else tau)))
        if spec.frozen:
            for p in (self.a, self.b_raw, self.log_tau, self.log_tau_back):
                p.requires_grad_(False)

    @property
    def kind(self) -> str:
        """The emission shape's name."""
        return self.spec.kind

    @property
    def b(self):
        """The emission gain, floored by the sharpness schedule."""
        return self.b_floor + nn.functional.softplus(self.b_raw)

    def forward(self, phi):
        """Downbeat log-odds at phase ``phi``."""
        if self.spec.kind == "laplace":
            # The band's asymmetry -- decay runs FORWARD from the onset only --
            # with a tail instead of a cliff, so every phase in the bar still
            # carries a gradient, and with a continuous scale in place of the
            # band's integer width.
            forward = torch.remainder(phi, TWO_PI)
            decay = torch.exp(-forward / self.log_tau.exp().clamp(1e-3, math.pi))
            return self.a + self.b * (2.0 * decay - 1.0)
        if self.spec.kind == "alaplace":
            # `laplace` is biased late: with mass only ahead of the onset, the
            # circular mean the read-out takes sits a first moment past the peak
            # (49 ms predicted at a 1.94 s bar, against 0 for the symmetric tent).
            # Giving the shape its own backward reach removes that bias without
            # giving up the asymmetry -- and both endpoints, the tent and the
            # one-sided band, are corners of this one family.
            wrapped = torch.atan2(phi.sin(), phi.cos())
            decay = torch.exp(-wrapped.clamp(min=0.0) / self.log_tau.exp().clamp(1e-3, math.pi)
                              - (-wrapped).clamp(min=0.0)
                              / self.log_tau_back.exp().clamp(1e-3, math.pi))
            return self.a + self.b * (2.0 * decay - 1.0)
        if self.spec.kind == "triangle":
            wrapped = torch.atan2(torch.sin(phi), torch.cos(phi))
            return self.a + self.b * (1.0 - 2.0 * wrapped.abs() / math.pi)
        if self.spec.kind == "bump":
            peak = torch.exp(self.spec.bump_kappa * (torch.cos(phi) - 1.0))
            return self.a + self.b * (2.0 * peak - 1.0)
        assert self.spec.kind == "cosine", f"unknown emission kind {self.spec.kind!r}"
        return self.a + self.b * torch.cos(phi)

    def channel_loglik(self, y, mask, phi):
        """[B,T,...]: log p(y_t | phi) for one Bernoulli channel at phases ``phi``."""
        e = self(phi)
        log_hit, log_miss = nn.functional.logsigmoid(e), nn.functional.logsigmoid(-e)
        if self.spec.floor > 0.0:
            keep = math.log1p(-self.spec.floor)
            log_hit = torch.logaddexp(torch.full_like(log_hit,
                                                      math.log(self.spec.floor)),
                                      keep + log_hit)
            log_miss = keep + log_miss
        extra = (1,) * (e.dim())
        ll = y.reshape(y.shape + extra) * log_hit \
            + (1.0 - y).reshape(y.shape + extra) * log_miss
        return ll * mask.reshape(mask.shape + extra)

    def loglik(self, y, mask, grid, meters=None, cls=None):
        """[B,T,N] downbeats, or [B,T,M,N] with the beat channel added per meter."""
        down = self.channel_loglik(y, mask, grid)
        if meters is None or not self.spec.beat_channel:
            return down
        m = meters.reshape(-1, 1).to(grid.dtype)
        beat_phase = torch.remainder(m * grid.reshape(1, -1), TWO_PI)
        beat_y = (cls >= 1).to(y.dtype)
        beat = self.channel_loglik(beat_y, mask, beat_phase)
        return down.unsqueeze(2) + beat


class PriorModel(nn.Module):
    """p(phi_t, c_t | phi_{t-1}, c_{t-1}): the bar-gated tempo chain.

    Two factors on one quadrature state space. Phase advances by a von Mises
    kernel of concentration kappa_physical around the current rate. The rate is
    part of the state and may change ONLY through a wrapping phase transition --
    a downbeat -- paying a wrapped-Cauchy cost on log-rate at the corpus per-bar
    scale GAMMA. Restricting changes to bar boundaries is what makes them
    affordable (~20 charged events per window instead of ~1500), and the heavy
    tail belongs here rather than in the phase channel, where it would hand back
    the octave.

    The nodes are QUADRATURE points over continuous phase and log-rate, not a
    state tiling: refining 36 -> 288 rate nodes leaves F, CMLt and rho unchanged
    at decode time, so the represented densities are continuous.
    """

    def __init__(self, rate: RateSpec, walk: WalkSpec, n_grid: int = N_GRID):
        super().__init__()
        rates = torch.exp(torch.linspace(math.log(rate.lo), math.log(rate.hi),
                                         rate.grid))
        self.register_buffer("rates", rates)
        z = (torch.log(rates) - walk.tempo_mu) / walk.tempo_sigma
        lp = -0.5 * z ** 2
        self.register_buffer("rate_log_prior", lp - torch.logsumexp(lp, 0))

        grid = torch.arange(n_grid) * (TWO_PI / n_grid)
        self.register_buffer("grid", grid)
        diff = grid[None, :, None] - grid[None, None, :] + rates[:, None, None]
        logk = walk.kappa_physical * (torch.cos(diff) - 1.0)
        # A von Mises transition is strictly positive everywhere, but at
        # kappa=383 exp() underflows float32 for ~76% of the grid pairs, so the
        # kernel arrives hard-sparse. That is an artefact, not the model, and it
        # is not harmless: the forward and backward messages can then end up with
        # DISJOINT support at a frame, whose posterior row sums to exactly zero.
        # The linear-domain recursions guard that with clamp_min, which keeps the
        # forward finite while the backward divides by 1e-30 and returns NaN.
        # Flooring restores strict positivity; measured, it moves no entry by
        # more than 1.2e-07 absolute, which is float32 epsilon.
        logk = logk.clamp_min(-70.0)
        logk = logk - torch.logsumexp(logk, dim=2, keepdim=True)
        k_lin = logk.exp()

        lr = torch.log(rates)
        sw = 1.0 / (1.0 + ((lr[None, :] - lr[:, None]) / GAMMA) ** 2)
        sw = sw / sw.sum(dim=1, keepdim=True)
        self.register_buffer("switch", sw if rate.per_bar
                             else torch.eye(rates.shape[0], dtype=sw.dtype))
        cell = TWO_PI / n_grid
        wrap = ((grid[None, :] + cell - (TWO_PI - rates[:, None])) / cell).clamp(0.0, 1.0)
        k_wrap = k_lin * wrap[:, :, None]
        k_stay = k_lin - k_wrap
        self.register_buffer("k_wrap", k_wrap)
        self.register_buffer("k_stay", k_stay)

        c, g = rates.shape[0], n_grid
        eye = torch.eye(c, dtype=k_lin.dtype)
        stay = torch.einsum("cmn,cd->cmdn", k_stay, eye).reshape(c * g, c * g)
        moved = torch.einsum("cmn,cd->cmdn", k_wrap, self.switch).reshape(c * g, c * g)
        self.register_buffer("kernel_stay", stay.contiguous(), persistent=False)
        self.register_buffer("kernel_wrap", moved.contiguous(), persistent=False)
        self.register_buffer("kernel", (stay + moved).contiguous(), persistent=False)

        self.meters = tuple(int(v) for v in rate.meters)
        if self.meters:
            M = len(self.meters)
            self.register_buffer("meter_values",
                                 torch.tensor(self.meters, dtype=k_lin.dtype))
            move = torch.full((M, M), (1.0 - METER_STAY) / (M - 1), dtype=k_lin.dtype)
            move.fill_diagonal_(METER_STAY)
            self.register_buffer("meter_switch", move)
            self.register_buffer("meter_log_prior",
                                 torch.full((M,), -math.log(M), dtype=k_lin.dtype))

    @property
    def n_rates(self) -> int:
        """How many rate candidates the chain carries."""
        return self.rates.shape[0]

    def fwd(self, a):
        """Propagate forward. Rate and meter may change only through a wrap."""
        if not self.meters:
            return (a.reshape(a.shape[0], -1) @ self.kernel).view_as(a)
        flat = a.reshape(-1, self.kernel.shape[0])
        stay = (flat @ self.kernel_stay).view_as(a)
        moved = (flat @ self.kernel_wrap).view_as(a)
        return stay + torch.einsum("bmcn,mp->bpcn", moved, self.meter_switch)

    def bwd(self, b):
        """Propagate backward, the transpose of fwd."""
        if not self.meters:
            return (b.reshape(b.shape[0], -1) @ self.kernel.T).view_as(b)
        stay = (b.reshape(-1, self.kernel.shape[0]) @ self.kernel_stay.T).view_as(b)
        pre = torch.einsum("bpcn,mp->bmcn", b, self.meter_switch)
        moved = (pre.reshape(-1, self.kernel.shape[0]) @ self.kernel_wrap.T).view_as(b)
        return stay + moved

    @torch.no_grad()
    def init_log_prior(self, rate: float, tempo_sigma: float):
        """Centre the rate prior on ``rate``, leaving it learnable downstream."""
        z = (torch.log(self.rates) - math.log(rate)) / tempo_sigma
        lp = -0.5 * z ** 2
        self.rate_log_prior.copy_(lp - torch.logsumexp(lp, 0))
        return int((self.rates - rate).abs().argmin())
