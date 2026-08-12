"""The bar-pointer CVAE of the tutorial's section 7, with a periodic latent (section 9.9)."""
from __future__ import annotations

import math

import torch
from torch import nn

from .vonmises import (kl_vonmises, log_i0, mean_resultant, sample_vonmises,
                       second_resultant)


TWO_PI = 2.0 * math.pi

# The physical prior's concentration: how much per-frame phase drift the dynamics allow.
# 1/sqrt(kappa) is the per-frame sd, so 2000 gives 0.022 rad at 50 fps against a bar
# advance of roughly 0.06 rad. A much softer prior (kappa ~ 20, sd 0.22 rad) is a random
# walk whose phase is uniform within two seconds and regularises nothing.
KAPPA_PHYSICAL = 383.0
SEARCH_LO, SEARCH_HI, SEARCH_STEP = 0.015, 0.250, 0.015
"""Candidate bar rates for readout='search': log-spaced from SEARCH_LO to SEARCH_HI in
steps of SEARCH_STEP nats, i.e. 1.5% apart, K = 188. Rationale in Encoder.search_phase."""

MAX_ANCHOR_KAPPA = 200.0   # q(offset)'s concentration is MAX * (normalised resultant),
                           # so perfect agreement across bars gives sd 1/sqrt(200) = 0.071
                           # rad -- at a 2.5 s bar that is 28 ms, comfortably inside the
                           # +-70 ms tolerance, and finite, so KL( q || Uniform ) stays
                           # bounded. An unbounded map would let a confident window buy
                           # arbitrary KL, which is the mechanism that railed kappa.
MAX_KAPPA = 1.0e5     # smooth ceiling on the encoder's concentration; an unbounded
                      # softplus inside a KL term can and did run away. Bounded via
                      # MAX * tanh(x / MAX), which is the identity for x << MAX and never
                      # passes exactly zero gradient -- unlike a hard clamp.


def bounded_kappa(raw):
    """Smoothly bound a concentration to (0, MAX_KAPPA); see MAX_KAPPA."""
    return MAX_KAPPA * torch.tanh(raw / MAX_KAPPA)


def inverse_softplus(value: float) -> float:
    """Pre-activation giving softplus(x) = value; linear above 30 where expm1 overflows."""
    return value if value > 30.0 else math.log(math.expm1(value))


def vonmises_log_density(z, mu, kappa):
    """Per-element log vM(z; mu, kappa): kappa cos(z - mu) - log(2 pi I0(kappa))."""
    return kappa * torch.cos(z - mu) - math.log(TWO_PI) - log_i0(kappa)


def vonmises_entropy(kappa):
    """H(vM(mu, kappa)) = log(2 pi I0(kappa)) - kappa A(kappa). Independent of mu."""
    return math.log(TWO_PI) + log_i0(kappa) - kappa * mean_resultant(kappa)


class Encoder(nn.Module):
    """q_phi(phi_t | x) = vM(mu_t(x), kappa_t(x)), per frame, reading AUDIO ONLY."""

    def __init__(self, input_dim: int, d_model: int = 128, heads: int = 4, layers: int = 2,
                 kappa_physical: float = KAPPA_PHYSICAL, pool_span: int = 150,
                 max_len: int = 4096, evidence: str = "head",
                 detector_layers: int = 0, rate_init_seconds: float = 3.0,
                 rate_bias_learnable: bool = False):
        super().__init__()
        self.evidence = evidence

        self.proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=4 * d_model,
                                        dropout=0.0, activation="gelu",
                                        batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(d_model, 4)
        self.pool_span = int(pool_span)

        # dedicated a_t head (default: ONE 128-param row of `out`); a_t is the measured
        # bottleneck -- own score F 0.359 vs oracle score 0.767 at a fixed grid.
        self.detector = None
        if detector_layers > 0:
            mods, d = [], d_model
            for _ in range(detector_layers):
                mods += [nn.Linear(d, d_model), nn.GELU()]
                d = d_model
            mods += [nn.Linear(d, 1)]
            self.detector = nn.Sequential(*mods)

        nn.init.normal_(self.out.weight, std=1e-2)
        nn.init.zeros_(self.out.bias)

        with torch.no_grad():
            self.out.bias[1] = 1.0

        # search_phase's candidate rates; grid rationale: search-read-out-verdict.
        self.register_buffer(
            "search_rates",
            torch.exp(torch.arange(math.log(SEARCH_LO), math.log(SEARCH_HI), SEARCH_STEP)),
            persistent=False)

        self.register_buffer("pe", EmissionTransformer._sinusoidal(max_len, d_model))
        self.register_buffer("log_kappa_bias",
                     torch.tensor(math.log(kappa_physical)), persistent=False)
        # rate start (3.0 s bar = corpus p10; geo-mean is 1.92 s) and whether the global
        # scale is learnable. Evidence + a PE-contamination retraction: search-read-out-verdict.
        log_rate_bias = torch.tensor(math.log(TWO_PI / (rate_init_seconds * 50.0)))
        if rate_bias_learnable:
            self.log_rate_bias = nn.Parameter(log_rate_bias)
        else:
            self.register_buffer("log_rate_bias", log_rate_bias, persistent=False)

    def features(self, h, mask=None):
        """[B, T, D] -> [B, T, d_model]: the trunk shared by every head (tutorial 9.2)."""
        pad = None if mask is None else (mask <= 0)
        h = self.proj(h) #+ self.pe[:h.shape[1]]
        return self.blocks(h, src_key_padding_mask=pad)

    def evidence_scores(self, trunk, out, w, h=None):
        """[B, T] a_t, the per-frame downbeat-ness the anchor folds.

        ONE implementation shared by ``heads`` and ``search_phase`` -- a second copy is
        how the delta-dropping and half-bar bugs happened.
        """
        if self.detector is not None:
            return torch.sigmoid(self.detector(trunk).squeeze(-1)) * w
        if self.evidence == "frontend":
            # DIAGNOSTIC arm (peak-pick baseline, not an input this model may consume).
            # CAVEAT 2026-08-12: correct only under a features+activations frontend, and
            # h[..., -2:] maxes the BEAT channel into a BAR anchor -- both fail silently.
            assert h is not None, "evidence='frontend' needs the raw features"
            return torch.sigmoid(h[..., -2:]).max(-1).values * w
        return torch.sigmoid(out[..., 0]) * w

    @torch.no_grad()
    def search_phase(self, h, mask=None):
        """SEARCH read-out: rate from the candidate grid, offset closed-form. [B, T]."""
        trunk = self.features(h, mask)
        out = self.out(trunk)
        w = torch.ones(h.shape[:2], device=h.device, dtype=h.dtype) if mask is None else mask
        a = self.evidence_scores(trunk, out, w, h)
        rates = self.search_rates.to(a.device)
        B, T = a.shape
        t = torch.arange(T, device=a.device, dtype=torch.float32)
        mu0 = rates[:, None] * t[None, :]                          # [K, T], batch-free
        aa = a[:, None, :]
        R = torch.complex((aa * torch.cos(mu0)).sum(-1), (aa * torch.sin(mu0)).sum(-1))
        k = (R.abs() / aa.sum(-1).clamp(min=1e-6)).argmax(-1)      # [B]
        mu0_best = torch.gather(mu0.expand(B, -1, -1), 1,
                                k[:, None, None].expand(-1, 1, T)).squeeze(1)
        return mu0_best + (-torch.angle(torch.gather(R, 1, k[:, None]).squeeze(1)))[:, None]

    def heads(self, trunk, mask=None, h=None):
        """Trunk -> (mu [B, T], kappa [B, T], anchor): the parameters of q, not a sample."""
        out = self.out(trunk)
        kappa = bounded_kappa(torch.exp(out[..., 2] + self.log_kappa_bias) + 1e-3)

        log_rate = self._pool(out[..., 3] + self.log_rate_bias, self.pool_span)
        rate = torch.exp(log_rate.clamp(math.log(0.01), math.log(0.2)))
        mu0 = torch.cumsum(rate, dim=1) - rate[:, :1]          # anchor-free: mu0[0] = 0

        w = torch.ones_like(mu0) if mask is None else mask
        a = self.evidence_scores(trunk, out, w, h)
        R = torch.complex(a * torch.cos(mu0), a * torch.sin(mu0)).sum(1)
        shift = -torch.angle(R)
        resultant = R.abs() / a.sum(1).clamp(min=1e-6)

        mu = mu0 + shift.unsqueeze(1)
        return mu, kappa, {"shift": shift, "resultant": resultant, "evidence": a}

    @staticmethod
    def _pool(x, span):
        """Mean over fixed blocks of `span` frames, broadcast back.

        Deletes the degrees of freedom rather than taxing them: within-span increment
        variance becomes exactly 0.
        """
        b, t = x.shape
        pad = (-t) % span
        xp = torch.nn.functional.pad(x, (0, pad))
        means = xp.reshape(b, -1, span).mean(-1, keepdim=True)
        return means.expand(-1, -1, span).reshape(b, -1)[:, :t]

    def forward(self, h, mask=None):
        """[B, T, D] -> (mu, kappa). Per-frame and free: see kl_to_physical_prior."""
        return self.heads(self.features(h, mask), mask, h)


class EmissionTransformer(nn.Module):
    """The tutorial's section 9.6 emission: a Transformer over the LATENT sequence."""

    def __init__(self, d_model: int = 64, layers: int = 2, heads: int = 4,
                 use_positional: bool = False, max_len: int = 4096):
        super().__init__()
        self.proj = nn.Linear(2, d_model)            # (cos phi, sin phi) -- latent ONLY
        self.use_positional = use_positional
        if use_positional:
            self.register_buffer("pe", self._sinusoidal(max_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=4 * d_model,
                                           dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(d_model, 1)

        nn.init.normal_(self.out.weight, std=1e-2)
        nn.init.constant_(self.out.bias, -3.4)       # ~= the 3.2% base rate, so training
                                                     # starts calibrated instead of spending
                                                     # its first epochs finding the prior

    @staticmethod
    def _sinusoidal(length, dim):
        """Standard sinusoidal positional encoding [length, dim]."""
        pos = torch.arange(length, dtype=torch.float32)[:, None]
        rate = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32)
                         * (-math.log(10000.0) / dim))
        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(pos * rate)
        pe[:, 1::2] = torch.cos(pos * rate)
        return pe

    def forward(self, phi, mask=None):
        """[B, T] phase -> [B, T] downbeat logits. Reads the LATENT only, never h."""
        x = self.proj(torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1))
        if self.use_positional:
            x = x + self.pe[:x.shape[1]]
        pad = None if mask is None else (mask <= 0)
        return self.out(self.blocks(x, src_key_padding_mask=pad)).squeeze(-1)


class BarPhaseVAE(nn.Module):
    """Encoder + physical prior + latent-only emission. Learnable: theta and phi only."""

    def __init__(self, input_dim: int, d_model: int = 128, emission: str = "cosine",
                 emission_layers: int = 2, emission_dim: int = 64,
                 emission_positional: bool = False, kappa_physical: float = KAPPA_PHYSICAL,
                 evidence: str = "head", detector_layers: int = 0,
                 rate_init_seconds: float = 3.0, rate_bias_learnable: bool = False,
                 readout: str = "head"):
        super().__init__()
        self.kappa_physical = float(kappa_physical)
        # deployment only: forward() never consults it, so one checkpoint scores under both.
        assert readout in ("head", "search"), f"unknown readout {readout!r}"
        self.readout = readout
        self.encoder = Encoder(input_dim, d_model, kappa_physical=kappa_physical,
                               evidence=evidence, detector_layers=detector_layers,
                               rate_init_seconds=rate_init_seconds,
                               rate_bias_learnable=rate_bias_learnable)

        self.emission_kind = emission
        # ONLY the arm in use gets parameters. Registering both left the unused pair with
        # no gradient, which the audit correctly refused -- and an audit that has to be
        # weakened to accommodate dead parameters stops being an audit.
        self.emission_net = None
        if emission == "transformer":
            self.emission_net = EmissionTransformer(emission_dim, emission_layers,
                                                    use_positional=emission_positional)
        else:
            # the two-scalar cosine: the BASELINE arm, not the default story. It is my
            # invention rather than the spec's, and every number before 2026-08-04 was
            # measured with it.
            self.emission_a = nn.Parameter(torch.tensor(-3.0))
            self.emission_b_raw = nn.Parameter(torch.tensor(1.0))

        self.register_buffer("emission_b_floor", torch.tensor(0.0))

    @property
    def emission_b(self):
        """Amplitude, positive by softplus, never below the scheduled floor."""
        return self.emission_b_floor + nn.functional.softplus(self.emission_b_raw)

    def emission_logits(self, phi, mask=None):
        """Downbeat logits from the LATENT alone (Point 1) -- never from h."""
        if self.emission_net is not None:
            return self.emission_net(phi, mask)
        if self.emission_kind == "triangle":
            wrapped = torch.atan2(torch.sin(phi), torch.cos(phi))   # (-pi, pi]
            return self.emission_a + self.emission_b * (1.0 - 2.0 * wrapped.abs() / math.pi)
        return self.emission_a + self.emission_b * torch.cos(phi)

    @torch.no_grad()
    def phase_ablation_gap(self, phi, mask=None):
        """How much the emission actually depends on phi. Near 0 means it has cheated."""
        live = self.emission_logits(phi, mask)
        frozen = self.emission_logits(torch.zeros_like(phi), mask)
        weight = torch.ones_like(live) if mask is None else mask
        return float(((live - frozen).abs() * weight).sum() / weight.sum())

    def kl_to_physical_prior(self, mu, kappa, mask):
        """KL( q_phi(phi_{1:T} | x) || p(phi_{1:T}) ), closed form -- never sampled."""
        kp = torch.as_tensor(self.kappa_physical, device=mu.device)
        a1, a2 = mean_resultant(kappa), second_resultant(kappa)

        inc = mu[:, 1:] - mu[:, :-1]
        accel = inc[:, 1:] - inc[:, :-1]                        # [B, T-2]

        # E_q[log p(phi_t | phi_{t-1}, phi_{t-2})] for t >= 3
        cross = (kp * a1[:, 2:] * a2[:, 1:-1] * a1[:, :-2] * torch.cos(accel)
                 - math.log(TWO_PI) - log_i0(kp))

        neg_entropy = -(vonmises_entropy(kappa) * mask).sum(1)
        log_p = (-math.log(TWO_PI) * (mask[:, 0] + mask[:, 1])
                 + (cross * mask[:, 2:]).sum(1))
        return neg_entropy - log_p

    def forward(self, h, mask, y, samples: int = 1, pos_weight: float = 1.0,
                targets=None, valid=None, sigma_frames: float = 3.5,
                anchor_penalty: str = "wrap", anchor_kappa: float = 65.0):
        """One ELBO evaluation on a padded batch (tutorial 7.7)."""
        assert targets is not None and valid is not None, \
            "the alignment objective needs collate_excerpts' targets/valid"

        mu, kappa, anchor = self.encoder(h, mask)
        kl = self.kl_to_physical_prior(mu, kappa, mask)

        kappa_off = MAX_ANCHOR_KAPPA * anchor["resultant"]
        kl_offset = kl_vonmises(anchor["shift"], kappa_off,
                                torch.zeros_like(anchor["shift"]),
                                torch.zeros_like(kappa_off))

        recon = 0.0
        for _ in range(samples):
            # Two independent draws: the per-frame phase noise, and ONE anchor sample per
            # window that rotates the whole trajectory. The anchor's variance is therefore
            # window-level, which is why it needs its own sample rather than riding on the
            # per-frame one.
            phi = (mu + sample_vonmises(kappa)
                   + sample_vonmises(kappa_off).unsqueeze(1))
            # velocity=mu: q is per-frame factorised, so the SAMPLE has no usable
            # derivative and it appears here as a divisor. See align_log_prob.
            recon = recon + align_log_prob(phi, targets, valid, sigma_frames,
                                           velocity=mu,
                                           anchor_penalty=anchor_penalty,
                                           anchor_kappa=anchor_kappa)

        recon = recon / samples
        kl = kl + kl_offset

        return {"elbo": recon - kl, "recon": recon, "kl": kl, "mu": mu, "kappa": kappa,
                "resultant": anchor["resultant"], "kl_offset": kl_offset}

    @property
    def deployed_net(self):
        """The inference network read at test time; controls assert ITS target-blindness."""
        return self.encoder

    @torch.no_grad()
    def infer_phase(self, h, mask=None):
        """Deployment (8.1.2): zhat = mu_phi(x). Reads audio only; returns [B, T]."""
        assert not self.training, "deployment path must run in eval mode"
        if self.readout == "search":
            return self.encoder.search_phase(h, mask)
        return self.encoder(h, mask)[0]

    @torch.no_grad()
    def emission_probs(self, h, mask=None):
        """Alternative D (8.3.4): the emission evaluated at the deployed mean path."""
        return torch.sigmoid(self.emission_logits(self.infer_phase(h, mask), mask))


def align_log_prob(mu, targets, valid, sigma_frames: float = 3.5, velocity=None,
                   anchor_penalty: str = "wrap", anchor_kappa: float = 65.0):
    """Log p(D | phi): the annotated downbeat TIMES as the observation. [B, T], [B, K] -> [B]."""
    B, T = mu.shape
    idx = targets.clamp(0.0, T - 2.0)
    lo = idx.floor()
    frac = idx - lo
    lo = lo.long()

    mu_lo = torch.gather(mu, 1, lo)
    mu_hi = torch.gather(mu, 1, lo + 1)
    mu_at = mu_lo + frac * (mu_hi - mu_lo)          # phase at t_k, sub-frame

    src = mu if velocity is None else velocity
    v_lo = torch.gather(src, 1, lo)
    v_hi = torch.gather(src, 1, lo + 1)
    rate_at = (v_hi - v_lo).clamp(min=1e-6)         # local phase velocity, rad/frame

    k = torch.cumsum(valid, dim=1) - 1.0
    n = valid.sum(1, keepdim=True).clamp(min=1.0)

    raw = mu_at - TWO_PI * k
    c = (raw * valid).sum(1, keepdim=True) / n          # the window's mean offset
    # Split it: the INTEGER part of c is a free gauge (which absolute bar the window
    # starts on is arbitrary), the SUB-BAR part is not (it is where the bar line sits).
    # Centring alone deletes both and the anchor loses its only signal -- measured, the
    # rate converged faster and in-tol fell to 0%. So (raw - c) carries the SHAPE (the
    # rate learns from it) and c carries the ANCHOR, scored separately below.
    shape_err = raw - c

    if anchor_penalty == "cos":
        dt = shape_err / rate_at
        shape_ll = (-0.5 * (dt / sigma_frames) ** 2
                    - math.log(sigma_frames) - 0.5 * math.log(TWO_PI))
        anchor_ll = anchor_kappa * (torch.cos(c) - 1.0) - math.log(TWO_PI) \
            - log_i0(torch.as_tensor(anchor_kappa, device=c.device, dtype=c.dtype))
        return (shape_ll * valid).sum(1) + anchor_ll.squeeze(1)

    phase_err = shape_err + torch.atan2(torch.sin(c), torch.cos(c))
    dt = phase_err / rate_at                        # phase residual -> TIME residual
    log_density = (-0.5 * (dt / sigma_frames) ** 2
                   - math.log(sigma_frames) - 0.5 * math.log(TWO_PI))
    return (log_density * valid).sum(1)


def downbeat_frames(mu, mask=None):
    """Rule g (8.1.2): a downbeat is where the phase crosses ZERO. Deterministic."""
    zero_to_two_pi = torch.remainder(mu, 2.0 * math.pi)
    crossing = torch.diff(zero_to_two_pi, dim=-1) < -math.pi
    if mask is not None:
        crossing = crossing & (mask[:, 1:] > 0)
    return crossing
