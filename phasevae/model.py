"""The bar-pointer CVAE of the tutorial's section 7, with a periodic latent (section 9.9)."""
from __future__ import annotations

import math

import torch
from torch import nn

from .vonmises import kl_vonmises, log_i0, mean_resultant, sample_vonmises


TWO_PI = 2.0 * math.pi

# The physical prior's concentration: how much per-frame phase drift the dynamics allow.
# 1/sqrt(kappa) is the per-frame sd, so 2000 gives 0.022 rad at 50 fps against a bar
# advance of roughly 0.06 rad. A much softer prior (kappa ~ 20, sd 0.22 rad) is a random
# walk whose phase is uniform within two seconds and regularises nothing.
KAPPA_PHYSICAL = 383.0
TEMPO_BOUND_MARGIN = 0.35

TEMPO_LO, TEMPO_HI = math.log(0.01), math.log(0.2)

TEMPO_PRIOR_MU = math.log(TWO_PI / (1.83 * 50.0))
TEMPO_PRIOR_SIGMA = 0.381
TEMPO_PRIOR_EPS = 0.08
TEMPO_WALK_B = 0.04

TEMPO_SIGMA_CEIL = 0.25
TEMPO_SIGMA_INIT = 0.15

BAR_POOL_ITERS = 8

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
                 max_len: int = 4096):
        super().__init__()
        self.pool_span = pool_span

        self.proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=4 * d_model,
                                        dropout=0.0, activation="gelu",
                                        batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(d_model, 4)

        nn.init.normal_(self.out.weight, std=1e-2)
        nn.init.zeros_(self.out.bias)

        self.register_buffer("pe", EmissionTransformer._sinusoidal(max_len, d_model))
        self.register_buffer("log_phi_kappa_bias",
                     torch.tensor(math.log(kappa_physical)), persistent=False)

    def output_channels(self, trunk):
        """[B, T, d_model] -> {channel: [B, T]}, one named single-row head each."""
        out = self.out(trunk)
        result = {"downbeat_logit": out[..., 0], "log_phi_kappa": out[..., 1],
                  "log_dotphi": out[..., 2], "log_sigma_dotphi": out[..., 3]}

        return result

    def features(self, h, mask=None):
        """[B, T, D] -> [B, T, d_model]: the trunk shared by every head (tutorial 9.2)."""
        pad = None if mask is None else (mask <= 0)
        h = self.proj(h) #+ self.pe[:h.shape[1]]
        return self.blocks(h, src_key_padding_mask=pad)

    @staticmethod
    def _ramp(log_dotphi):
        """log-dotphi -> (tempo, anchor-free phase ramp). mu0[0] = 0 by construction."""
        lo, hi = TEMPO_LO, TEMPO_HI
        m = TEMPO_BOUND_MARGIN
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        d = log_dotphi - mid
        flat = half - m
        over = (d.abs() - flat).clamp(min=0.0)
        squashed = flat + m * torch.tanh(over / m)
        bounded = mid + torch.sign(d) * torch.minimum(d.abs(), squashed)
        dotphi = torch.exp(bounded)
        return dotphi, torch.cumsum(dotphi, dim=1) - dotphi[:, :1]

    def _anchor(self, mu0, a):
        """The window's single rotation: circular mean of the evidence folded on mu0."""
        R = torch.complex(a * torch.cos(mu0), a * torch.sin(mu0)).sum(1)
        return -torch.angle(R), R.abs() / a.sum(1).clamp(min=1e-6)

    def _bar_seg(self, log_dotphi_raw, w):
        _d, mu0_i = self._ramp(log_dotphi_raw)
        seg = self._bar_index(mu0_i, w)
        log_dotphi = self._pool_by_bar(log_dotphi_raw, seg, w)
        _d, mu0_i = self._ramp(log_dotphi)
        seg = self._bar_index(mu0_i, w)
        log_dotphi = self._pool_by_bar(log_dotphi_raw, seg, w)
        return log_dotphi, seg

    def heads(self, trunk, mask=None, h=None):
        """Trunk -> (mu [B, T], kappa [B, T], anchor): the parameters of q, not a sample."""
        channels = self.output_channels(trunk)

        # JA: log_phi_kappa_bias is a constant that is added to the encoder's log_kappa output.
        # As kappa initializes to a small value (around 1), it would otherwise take a long
        # time before reaching a reasonable value.
        kappa = bounded_kappa(torch.exp(channels["log_phi_kappa"] + self.log_phi_kappa_bias) + 1e-3)

        log_dotphi_raw = channels["log_dotphi"]
        w = torch.ones(trunk.shape[:2], device=trunk.device, dtype=trunk.dtype) \
            if mask is None else mask

        log_dotphi, seg = self._bar_seg(log_dotphi_raw, w)

        log_dotphi, tempo_entropy = self._sample_learned_sigma(
            channels["log_sigma_dotphi"], log_dotphi, seg, w)

        dotphi, mu0 = self._ramp(log_dotphi)
        mu = mu0

        tempo_prior = self._tempo_log_prior(torch.log(dotphi), seg, w)

        return mu, kappa, {"tempo_prior": tempo_prior,
                           "tempo_entropy": tempo_entropy}

    def _sample_learned_sigma(self, sigma_raw, log_dotphi, seg, w):
        """(noised log_dotphi, q's tempo entropy [B]): one draw and one H term per bar."""
        bias = math.log(TEMPO_SIGMA_INIT / (TEMPO_SIGMA_CEIL - TEMPO_SIGMA_INIT))
        if seg is not None:
            sigma_raw = self._pool_by_bar(sigma_raw, seg, w)
        sigma = TEMPO_SIGMA_CEIL * torch.sigmoid(sigma_raw + bias)

        if seg is not None:
            n_seg = int(seg.max().item()) + 1
            counts = sigma.new_zeros(sigma.shape[0], n_seg).scatter_add(1, seg, w)
            eps = torch.gather(torch.randn(sigma.shape[0], n_seg, device=sigma.device,
                                           dtype=sigma.dtype), 1, seg)
            per_bar_weight = w / torch.gather(counts, 1, seg).clamp(min=1.0)
        else:
            eps = torch.randn_like(sigma)
            per_bar_weight = w

        if self.training:
            log_dotphi = log_dotphi + sigma * eps
        entropy = ((0.5 * math.log(2.0 * math.pi * math.e) + torch.log(sigma))
                   * per_bar_weight).sum(1)
        return log_dotphi, entropy

    def _tempo_log_prior(self, log_dotphi, seg, w):
        """log p(dotphi_1) + sum_k log p(dotphi_k | dotphi_{k-1}) per window. [B]."""
        z = (log_dotphi[:, 0] - TEMPO_PRIOR_MU) / TEMPO_PRIOR_SIGMA
        log_gauss = -0.5 * z ** 2 - math.log(TEMPO_PRIOR_SIGMA) - 0.5 * math.log(2.0 * math.pi)
        log_unif = -math.log(TEMPO_HI - TEMPO_LO)
        floor = torch.full_like(log_gauss, math.log(TEMPO_PRIOR_EPS) + log_unif)
        init = torch.logaddexp(math.log(1.0 - TEMPO_PRIOR_EPS) + log_gauss, floor)

        if seg is None or TEMPO_WALK_B <= 0.0:
            return init

        step = log_dotphi[:, 1:] - log_dotphi[:, :-1]
        crossing = (seg[:, 1:] != seg[:, :-1]) & (w[:, 1:] > 0) & (w[:, :-1] > 0)
        walk = -(step.abs() / TEMPO_WALK_B + math.log(2.0 * TEMPO_WALK_B)) * crossing
        return init + walk.sum(1)

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

    @staticmethod
    @torch.no_grad()
    def _bar_index(mu, w):
        """[B, T] long: which bar each frame is in, from mu's own 2pi wraps, 0-based."""
        real = w > 0
        seg = torch.floor(mu / TWO_PI).long()
        lo = seg.masked_fill(~real, torch.iinfo(torch.long).max).min(dim=1, keepdim=True)
        hi = seg.masked_fill(~real, torch.iinfo(torch.long).min).max(dim=1, keepdim=True)
        seg = (seg - lo.values).clamp(min=0, max=None)
        return torch.minimum(seg, (hi.values - lo.values).clamp(min=0))

    @staticmethod
    def _resolve_cycle(seg_a, seg_b):
        """Deterministically pick one member of a period-2 segmentation cycle."""
        na, nb = int(seg_a.max().item()), int(seg_b.max().item())
        if na != nb:
            return seg_a if na < nb else seg_b
        diff = torch.nonzero(seg_a != seg_b)
        if diff.numel() == 0:
            return seg_a
        return seg_a if int(seg_a[tuple(diff[0])]) < int(seg_b[tuple(diff[0])]) else seg_b

    @staticmethod
    def _pool_by_bar(log_dotphi, seg, w):
        """Mask-weighted mean log-dotphi within each bar of the segmentation `seg`."""
        n_seg = int(seg.max().item()) + 1
        zeros = log_dotphi.new_zeros(log_dotphi.shape[0], n_seg)
        sums = zeros.scatter_add(1, seg, log_dotphi * w)
        counts = zeros.scatter_add(1, seg, w)
        means = sums / counts.clamp(min=1e-6)
        pooled = torch.gather(means, 1, seg)
        empty = torch.gather(counts <= 0, 1, seg)
        return torch.where(empty, log_dotphi, pooled)

    def forward(self, h, mask=None):
        """[B, T, D] -> (mu, kappa). Per-frame and free: see kl_jitter."""
        features = self.features(h, mask)
        heads = self.heads(features, mask, h)

        return heads


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
        nn.init.constant_(self.out.bias, -3.4)       # ~= the 3.2% base tempo, so training
                                                     # starts calibrated instead of spending
                                                     # its first epochs finding the prior

    @staticmethod
    def _sinusoidal(length, dim):
        """Standard sinusoidal positional encoding [length, dim]."""
        pos = torch.arange(length, dtype=torch.float32)[:, None]
        tempo = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32)
                         * (-math.log(10000.0) / dim))
        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(pos * tempo)
        pe[:, 1::2] = torch.cos(pos * tempo)
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
                 detector_layers: int = 0):
        super().__init__()
        self.kappa_physical = float(kappa_physical)
        self.encoder = Encoder(input_dim, d_model, kappa_physical=kappa_physical)

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

    def kl_jitter(self, mu, kappa, mask):
        """Per-frame KL( vM(mu,kappa) || vM(mu,kappa_physical) ): prices concentration only."""
        kp = torch.as_tensor(self.kappa_physical, device=mu.device, dtype=kappa.dtype)
        return (kl_vonmises(mu, kappa, mu, kp) * mask).sum(1)

    def forward(self, h, mask, y, samples: int = 1, pos_weight: float = 1.0):
        """One ELBO evaluation on a padded batch (tutorial 7.7)."""
        mu, kappa, aux = self.encoder(h, mask)
        kl = (self.kl_jitter(mu, kappa, mask) - aux["tempo_prior"]
              - aux["tempo_entropy"])

        recon = 0.0
        for _ in range(samples):
            phi = mu + sample_vonmises(kappa)
            w = torch.ones_like(y) if mask is None else mask
            pw = torch.where(y > 0.5, pos_weight, 1.0) * w
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                self.emission_logits(phi, mask), y, reduction="none")
            recon = recon - (bce * pw).sum(1)

        recon = recon / samples

        return {"elbo": recon - kl, "recon": recon, "kl": kl, "mu": mu, "kappa": kappa,
                "tempo_prior": aux["tempo_prior"],
                "tempo_entropy": aux["tempo_entropy"]}

    @property
    def deployed_net(self):
        """The inference network read at test time; controls assert ITS target-blindness."""
        return self.encoder

    @torch.no_grad()
    def infer_phase(self, h, mask=None):
        """Deployment (8.1.2): zhat = mu_phi(x). Reads audio only; returns [B, T]."""
        assert not self.training, "deployment path must run in eval mode"
        return self.encoder(h, mask)[0]

    @torch.no_grad()
    def emission_probs(self, h, mask=None):
        """Alternative D (8.3.4): the emission evaluated at the deployed mean path."""
        return torch.sigmoid(self.emission_logits(self.infer_phase(h, mask), mask))


def downbeat_frames(mu, mask=None):
    """Rule g (8.1.2): a downbeat is where the phase crosses ZERO. Deterministic."""
    zero_to_two_pi = torch.remainder(mu, 2.0 * math.pi)
    crossing = torch.diff(zero_to_two_pi, dim=-1) < -math.pi
    if mask is not None:
        crossing = crossing & (mask[:, 1:] > 0)
    return crossing
