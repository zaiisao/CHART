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
        self.evidence_head = nn.Linear(d_model, 2 * n_harm)
        nn.init.zeros_(self.evidence_head.weight)
        nn.init.zeros_(self.evidence_head.bias)
        j = torch.arange(1, n_harm + 1, dtype=torch.float32)
        self.register_buffer("evidence_basis",
                             torch.cat([torch.cos(prior.grid[:, None] * j),
                                        torch.sin(prior.grid[:, None] * j)], dim=1))

    def forward(self, h, mask, prior):
        """(evidence, log_q_rate0, q_joint, log_z) for one window."""
        evidence, log_q_rate0 = self.potentials(h, mask)
        q_joint, log_z = self.smooth(evidence, log_q_rate0, prior)
        return evidence, log_q_rate0, q_joint, log_z

    def potentials(self, h, mask):
        """(evidence [B,T,N], head [B,C]): the local evidence, and the initial rate."""
        feats = self.encoder.features(h, mask)
        return self._evidence(feats, mask), \
            torch.log_softmax(self.rate_head(self._pooled(feats, mask)), dim=-1)

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

    def smooth(self, evidence, log_q_rate0, prior):
        """Exact forward-backward over the prior's quadrature."""
        B, T, N = evidence.shape
        C = prior.rates.shape[0]
        FL = 1e-30
        E = evidence.exp()
        p0 = torch.softmax(prior.rate_log_prior, 0)[None, :, None]
        a = p0 * log_q_rate0.exp()[:, :, None] * E[:, 0][:, None, :] / N
        s = a.sum(dim=(1, 2), keepdim=True).clamp_min(FL)
        logz = s.squeeze(-1).squeeze(-1).log()
        a = a / s
        A = [a]
        for t in range(1, T):
            a = prior.fwd(a) * E[:, t][:, None, :]
            s = a.sum(dim=(1, 2), keepdim=True).clamp_min(FL)
            logz = logz + s.squeeze(-1).squeeze(-1).log()
            a = a / s
            A.append(a)
        b = torch.ones_like(a) / (C * N)
        Bl = [b]
        for t in range(T - 1, 0, -1):
            b = prior.bwd(b * E[:, t][:, None, :])
            b = b / b.sum(dim=(1, 2), keepdim=True).clamp_min(FL)
            Bl.append(b)
        Bl.reverse()
        post = torch.stack(A, 1) * torch.stack(Bl, 1)               # [B,T,C,N]
        post = post / post.sum(dim=(2, 3), keepdim=True).clamp_min(FL)
        return post, logz

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
    well as the truth. `band` is NeuralDBN 8.2's own rectangular beat window and
    `laplace` is that window with an exponential tail in place of its far edge;
    both reach only forward from the onset, and both take their extent from the
    scoring tolerance rather than from a fitted integer.
    """

    def __init__(self, spec: EmissionSpec, n_grid: int = N_GRID):
        super().__init__()
        self.spec = spec
        self.n_grid = n_grid

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
        self.register_buffer("band_w", torch.tensor(round(tau * n_grid / TWO_PI)))

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
        if self.spec.kind == "band":
            # 8.2: the first w positions AFTER the beat point, not a window
            # centred on it -- the activation is a bump that follows the onset.
            inside = torch.remainder(phi, TWO_PI) < float(self.band_w) * (TWO_PI / self.n_grid)
            return self.a + self.b * (2.0 * inside.to(phi.dtype) - 1.0)
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

    def loglik(self, y, mask, grid):
        """[B,T,N]: log p(y_t | phi) for the DOWNBEAT target, at every grid phase."""
        e = self(grid)
        log_hit, log_miss = nn.functional.logsigmoid(e), nn.functional.logsigmoid(-e)
        if self.spec.floor > 0.0:
            keep = math.log1p(-self.spec.floor)
            log_hit = torch.logaddexp(torch.full_like(log_hit,
                                                      math.log(self.spec.floor)),
                                      keep + log_hit)
            log_miss = keep + log_miss
        ll = y[..., None] * log_hit + (1.0 - y)[..., None] * log_miss
        return ll * mask[..., None]


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
        self.register_buffer("k_wrap", k_lin * wrap[:, :, None])
        self.register_buffer("k_stay", k_lin - k_lin * wrap[:, :, None])

    @property
    def n_rates(self) -> int:
        """How many rate candidates the chain carries."""
        return self.rates.shape[0]

    def fwd(self, a):
        """[B,C,N] -> propagated forward. Rate may change only through a wrap."""
        stay = torch.einsum("bcm,cmn->bcn", a, self.k_stay)
        wm = torch.einsum("bcm,cmn->bcn", a, self.k_wrap)
        return stay + torch.einsum("bdn,dc->bcn", wm, self.switch)

    def bwd(self, b):
        """[B,C,N] -> propagated backward, the transpose of fwd."""
        stay = torch.einsum("bcn,cmn->bcm", b, self.k_stay)
        sw = torch.einsum("bcn,dc->bdn", b, self.switch)
        return stay + torch.einsum("bdn,dmn->bdm", sw, self.k_wrap)

    @torch.no_grad()
    def init_log_prior(self, rate: float, tempo_sigma: float):
        """Centre the rate prior on ``rate``, leaving it learnable downstream."""
        z = (torch.log(self.rates) - math.log(rate)) / tempo_sigma
        lp = -0.5 * z ** 2
        self.rate_log_prior.copy_(lp - torch.logsumexp(lp, 0))
        return int((self.rates - rate).abs().argmin())
