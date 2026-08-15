"""The bar-pointer CVAE of the tutorial's section 7, with a periodic latent (section 9.9)."""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch
from torch import nn

from .constants import (CORRECTION_MAX, DELTA_MAX, KAPPA_INTER, KAPPA_PHYSICAL, MAX_KAPPA,
                        TEMPO_PRIOR_MU, TEMPO_PRIOR_SIGMA, TEMPO_WALK_SIGMA, TWO_PI,
                        WALK_INTER_SIGMA, WALK_INTER_W, WALK_INTRA_SIGMA, WALK_MIX_SIGMA,
                        WALK_MIX_W)
from .constants import (BAR_POOL_ITERS, TEMPO_BOUND_MARGIN, TEMPO_HI,  # noqa: F401
                        TEMPO_LO,  # noqa: F401
                        TEMPO_PRIOR_EPS, TEMPO_SIGMA_CEIL, TEMPO_SIGMA_INIT)  # noqa: F401
from .vonmises import kl_vonmises, log_i0, mean_resultant, sample_vonmises


@dataclasses.dataclass
class EmissionSpec:
    """p(y_t | phi_t): which shape reads the latent, and how big it is."""

    kind: str = "cosine"
    layers: int = 2
    dim: int = 64
    positional: bool = False

    @classmethod
    def coerce(cls, value):
        """Accept either a spec or the bare kind string the configs still pass."""
        return value if isinstance(value, cls) else cls(kind=value)


@dataclasses.dataclass
class WalkSpec:
    """p(phi_t | phi_t-1): the tempo walk's law and the phase prior's tightness."""

    kind: str = "gauss"
    kappa_physical: float = KAPPA_PHYSICAL
    kappa_gate: bool = False


@dataclasses.dataclass
class PlacementSpec:
    """How the placement factor reads phase, and which paths reach it."""

    coord: str = "first"
    lift: float = 0.0
    attach: bool = False


@dataclasses.dataclass
class UpdateSpec:
    """q(phi_t)'s free mean: the update half of the filter."""

    delta_on: bool = False
    gate_cond: bool = True


@dataclasses.dataclass
class DecoderSpec:
    """The knot decoder that emits the rate correction and delta."""

    dim: int = 32
    knot_stride: int = 25


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
                 kappa_physical: float = KAPPA_PHYSICAL, max_len: int = 4096,
                 use_pe: bool = False):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.d_model = d_model
        self.use_pe = use_pe
        self.in_drop = nn.Dropout1d(0.1)
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=4 * d_model,
                                        dropout=0.0, activation="relu",
                                        batch_first=True, norm_first=False)
        self.blocks = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(d_model, 4)

        nn.init.normal_(self.out.weight, std=1e-2)
        nn.init.zeros_(self.out.bias)
        with torch.no_grad():
            self.out.bias[2] = TEMPO_PRIOR_MU
            self.out.bias[3] = inverse_softplus(0.0005)

        self.register_buffer("pe", EmissionTransformer._sinusoidal(max_len, d_model))
        self.register_buffer("log_phi_kappa_bias",
                     torch.tensor(math.log(kappa_physical)), persistent=False)

    def output_channels(self, trunk):
        """[B, T, d_model] -> {channel: [B, T]}, one named single-row head each."""
        out = self.out(trunk)
        result = {"rotation_weight_logit": out[..., 0], "phase_log_kappa": out[..., 1],
                    "tempo_log_mu": out[..., 2], "tempo_sigma_logit": out[..., 3]}
        return result

    def features(self, h, mask=None):
        """[B, T, D] -> [B, T, d_model]: the trunk shared by every head (tutorial 9.2)."""
        pad = None if mask is None else (mask <= 0)
        h = self.proj(h) * math.sqrt(self.d_model)
        if self.use_pe:
            h = h + self.pe[:h.shape[1]]
        h = self.in_drop(h.transpose(1, 2)).transpose(1, 2)
        return self.blocks(h, src_key_padding_mask=pad)

    @staticmethod
    def _ramp(log_dotphi):
        dotphi = torch.exp(log_dotphi)
        return dotphi, torch.cumsum(dotphi, dim=1) - dotphi[:, :1]

    def _anchor(self, ramp, a):
        R = torch.complex(a * torch.cos(ramp), a * torch.sin(ramp)).sum(1)
        return -torch.angle(R), R.abs() / a.sum(1).clamp(min=1e-6)

    def heads(self, trunk, mask=None):
        channels = self.output_channels(trunk)
        w = torch.ones(trunk.shape[:2], device=trunk.device, dtype=trunk.dtype) \
            if mask is None else mask

        # JA: log_phi_kappa_bias is a constant that is added to the encoder's log_kappa output.
        # As kappa initializes to a small value (around 1), it would otherwise take a long
        # time before reaching a reasonable value.
        phase_kappa = bounded_kappa(
            torch.exp(channels["phase_log_kappa"] + self.log_phi_kappa_bias) + 1e-3)
        tempo_mu = torch.exp(channels["tempo_log_mu"])
        tempo_sigma = nn.functional.softplus(channels["tempo_sigma_logit"])
        rotation_weight = torch.sigmoid(channels["rotation_weight_logit"]) * w

        return {
            "phase": {"kappa": phase_kappa},
            "tempo": {"mu": tempo_mu, "sigma": tempo_sigma},
            "rotation": {"weight": rotation_weight},
        }

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
        """[B, T, D] -> (posterior param dict, trunk memory [B, T, d_model])."""
        features = self.features(h, mask)
        heads = self.heads(features)

        return heads, features


class EmissionTransformer(nn.Module):
    """The tutorial's section 9.6 emission: a Transformer over the LATENT sequence."""

    def __init__(self, d_model: int = 64, layers: int = 2, heads: int = 4,
                 use_positional: bool = False, max_len: int = 4096):
        super().__init__()
        self.proj = nn.Linear(2, d_model)
        self.use_positional = use_positional
        if use_positional:
            self.register_buffer("pe", self._sinusoidal(max_len, d_model))

        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=4 * d_model,
                                           dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(d_model, 1)

        nn.init.normal_(self.out.weight, std=1e-2)
        nn.init.constant_(self.out.bias, -3.4)

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






def interp_phase(phi, frames_f):
    """phi [B, T] sampled at integer frames -> its interpolation at frames_f [B, N]."""
    length = phi.shape[1]
    f = frames_f.clamp(min=0.0, max=float(length - 1) - 1e-4)
    i0 = f.floor().long()
    i1 = (i0 + 1).clamp(max=length - 1)
    frac = f - i0.to(f.dtype)
    return torch.gather(phi, 1, i0) * (1.0 - frac) + torch.gather(phi, 1, i1) * frac


def path_dotphi(phi, half: int = 25):
    """dphi/dt from the sampled path over +-half frames, so jitter averages out."""
    pad = torch.nn.functional.pad(phi[:, None, :], (half, half), mode="replicate")[:, 0]
    return ((pad[:, 2 * half:] - pad[:, :-2 * half]) / (2.0 * half)).clamp(min=1e-8)


def smooth_phase(phi, half: int):
    """Boxcar the phase before reading it at an annotation; the ramp is locally linear,
    so this is unbiased and cuts the jitter by sqrt(2*half+1)."""
    if half <= 0:
        return phi
    pad = torch.nn.functional.pad(phi[:, None, :], (half, half), mode="replicate")
    return torch.nn.functional.avg_pool1d(pad, 2 * half + 1, stride=1)[:, 0]


def interval_penalty(log_ratio, b_ratio: float, kind: str = "laplace", delta: float = 0.05):
    """-log density of log r. Laplace votes +-1/b at any size; Huber's force is
    proportional to the error inside delta, so its mean survives averaging."""
    if kind == "laplace":
        return log_ratio.abs() / b_ratio + math.log(2.0 * b_ratio)
    size = log_ratio.abs()
    shaped = torch.where(size <= delta, 0.5 * size ** 2 / delta, size - 0.5 * delta)
    return shaped / b_ratio + math.log(2.0 * b_ratio)


def annotation_frames(raw, device):
    """[B, Nmax] annotated downbeat positions in frames, and their 0/1 validity."""
    fps = raw["fps"].numpy()
    t0 = raw["t0"].numpy()
    per_item = [(np.asarray(t, dtype=np.float64) - t0[i]) * fps[i]
                for i, t in enumerate(raw["downbeat_times"])]
    width = max(1, max(len(a) for a in per_item))
    frames = np.zeros((len(per_item), width), dtype=np.float32)
    valid = np.zeros((len(per_item), width), dtype=np.float32)
    for i, a in enumerate(per_item):
        frames[i, :len(a)] = a
        valid[i, :len(a)] = 1.0
    return torch.from_numpy(frames).to(device), torch.from_numpy(valid).to(device)


def interval_loglik(phi, ann_f, ann_valid, kappa_place: float, b_ratio: float,
                    phase_half: int = 0, kind: str = "laplace", phi_place=None,
                    disp_weight: float = 0.0, place_coord: str = "first"):
    """log p(annotation times | phi) [B].

    ``phi_place`` carries the placement factor's phase. It exists because the two
    factors own different degrees of freedom -- placement the rotation, interval the
    rate -- and the split is not automatic: phi_1 = mu0(t_1) + theta, so
    d(kappa cos phi_1)/d log k = -kappa sin(phi_1) * mu0(t_1), and with the first
    annotation 42 frames in that leaks +-476 nats per unit log-rate into the rate
    channel, against the interval ruler's +-110. Measured, it reversed the net gradient
    at k = 1.25 and k = 2.65 -- the harmonics runs actually park on. Detaching the
    ramp inside this factor keeps its gradient to theta (which trains the evidence
    head) and returns the rate to the term that measures it.
    """
    kappa = torch.as_tensor(kappa_place, device=phi.device, dtype=phi.dtype)
    at = interp_phase(smooth_phase(phi, phase_half), ann_f)
    first = ann_valid.cumsum(1).eq(1.0).to(phi.dtype) * ann_valid

    at_place = at if phi_place is None else interp_phase(smooth_phase(phi_place, phase_half), ann_f)
    if place_coord == "mean":
        n_ann = ann_valid.sum(1).clamp(min=1.0)
        bar = torch.atan2((torch.sin(at_place) * ann_valid).sum(1) / n_ann,
                          (torch.cos(at_place) * ann_valid).sum(1) / n_ann)
        place = kappa * torch.cos(bar) - math.log(TWO_PI) - log_i0(kappa)
    else:
        place = ((kappa * torch.cos(at_place) - math.log(TWO_PI) - log_i0(kappa))
                 * first).sum(1)
    pair = ann_valid[:, 1:] * ann_valid[:, :-1]
    ratio = ((at[:, 1:] - at[:, :-1]) / TWO_PI).clamp(min=1e-6)
    interval = (-interval_penalty(torch.log(ratio), b_ratio, kind) * pair).sum(1)

    d_at = interp_phase(path_dotphi(phi), ann_f)
    jac = (torch.log(d_at) * ann_valid).sum(1) - (torch.log(TWO_PI * ratio) * pair).sum(1)

    n = ann_valid.sum(1).clamp(min=1.0)
    re = (torch.cos(at) * ann_valid).sum(1) / n
    im = (torch.sin(at) * ann_valid).sum(1) / n
    resultant = torch.sqrt(re ** 2 + im ** 2 + 1e-12)
    disp = kappa * disp_weight * resultant

    return {"loglik": place + interval + jac + disp, "place": place,
            "interval": interval, "resultant": resultant}



class ZDecoder(nn.Module):

    def __init__(self, feat_dim: int, d: int = 64, layers: int = 2, heads: int = 4,
                 max_knots: int = 512):
        super().__init__()
        self.proj = nn.Linear(feat_dim + 2, d)
        self.register_buffer("pe", EmissionTransformer._sinusoidal(max_knots, d))
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d,
                                           dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(d, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def next_correction(self, tokens):
        x = self.proj(tokens) + self.pe[:tokens.shape[1]]
        return self.out(self.blocks(x)[:, -1])[:, 0]


class VBPM(nn.Module):
    """Encoder + physical prior + latent-only emission. Learnable: theta and phi only."""

    def __init__(self, input_dim: int, d_model: int = 128,
                 emission: EmissionSpec | str = "cosine",
                 walk: WalkSpec | None = None,
                 placement: PlacementSpec | None = None,
                 update: UpdateSpec | None = None,
                 decoder: DecoderSpec | None = None,
                 encoder_pe: bool = False):
        super().__init__()
        emission = EmissionSpec.coerce(emission)
        self.walk = walk or WalkSpec()
        self.placement = placement or PlacementSpec()
        self.update = update or UpdateSpec()
        self.decoder_spec = decoder or DecoderSpec()

        self.kappa_physical = float(self.walk.kappa_physical)
        self.walk_kind = self.walk.kind
        self.kappa_gate = bool(self.walk.kappa_gate)
        self.place_coord = self.placement.coord
        self.place_lift = float(self.placement.lift)
        self.place_attach = bool(self.placement.attach)
        self.delta_on = bool(self.update.delta_on)
        self.gate_cond = bool(self.update.gate_cond)
        self.knot_stride = int(self.decoder_spec.knot_stride)

        self.encoder = Encoder(input_dim, d_model,
                               kappa_physical=self.kappa_physical, use_pe=encoder_pe)

        self.emission_kind = emission.kind
        # ONLY the arm in use gets parameters. Registering both left the unused pair with
        # no gradient, which the audit correctly refused -- and an audit that has to be
        # weakened to accommodate dead parameters stops being an audit.
        self.emission_net = None
        if emission.kind == "transformer":
            self.emission_net = EmissionTransformer(emission.dim, emission.layers,
                                                    use_positional=emission.positional)
        else:
            # the two-scalar cosine: the BASELINE arm, not the default story. It is my
            # invention rather than the spec's, and every number before 2026-08-04 was
            # measured with it.
            self.emission_a = nn.Parameter(torch.tensor(-3.0))
            self.emission_b_raw = nn.Parameter(torch.tensor(1.0))

        self.register_buffer("emission_b_floor", torch.tensor(0.0))

        self.zdec = ZDecoder(d_model, d=self.decoder_spec.dim)
        if self.delta_on:
            self.delta_head = nn.Linear(
                self.zdec.out.in_features + (1 if self.gate_cond else 0), 1)
            nn.init.zeros_(self.delta_head.weight)
            nn.init.zeros_(self.delta_head.bias)

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

    def _token(self, memory_frame, phase):
        return torch.cat([memory_frame, torch.cos(phase)[:, None],
                          torch.sin(phase)[:, None]], dim=-1)

    def walk_log_prior(self, dot_eff, w, crossing=None):
        if self.walk_kind == "gauss":
            return self.tempo_log_prior(dot_eff, w)

        log_dotphi = torch.log(dot_eff)
        z = (log_dotphi[:, 0] - TEMPO_PRIOR_MU) / TEMPO_PRIOR_SIGMA
        init = -0.5 * z ** 2 - math.log(TEMPO_PRIOR_SIGMA) - 0.5 * math.log(2.0 * math.pi)
        step = log_dotphi[:, 1:] - log_dotphi[:, :-1]
        pair = (w[:, 1:] > 0) & (w[:, :-1] > 0)
        if self.walk_kind == "gated":
            intra = -0.5 * (step / WALK_INTRA_SIGMA) ** 2 - math.log(WALK_INTRA_SIGMA) \
                - 0.5 * math.log(2.0 * math.pi)
            comps = [math.log(wt) - 0.5 * (step / sg) ** 2 - math.log(sg)
                     - 0.5 * math.log(2.0 * math.pi)
                     for wt, sg in zip(WALK_INTER_W, WALK_INTER_SIGMA)]
            calibration = -math.log(WALK_INTRA_SIGMA) \
                - math.log(sum(wt / sg for wt, sg in zip(WALK_INTER_W, WALK_INTER_SIGMA)))
            inter = torch.logsumexp(torch.stack(comps), dim=0) + calibration
            lp = torch.where(crossing, inter, intra)
        else:
            comps = [math.log(wt) - 0.5 * (step / sg) ** 2 - math.log(sg)
                     - 0.5 * math.log(2.0 * math.pi)
                     for wt, sg in zip(WALK_MIX_W, WALK_MIX_SIGMA)]
            lp = torch.logsumexp(torch.stack(comps), dim=0)
        return init + (lp * pair).sum(1)

    def _knot_delta(self, tokens, on=None):
        x = self.zdec.proj(tokens) + self.zdec.pe[:tokens.shape[1]]
        head_in = self.zdec.blocks(x)[:, -1]
        if self.gate_cond:
            g = (torch.zeros_like(head_in[:, :1]) if on is None
                 else on.to(head_in.dtype)[:, None])
            head_in = torch.cat([head_in, g], dim=1)
        raw = self.delta_head(head_in)[:, 0]
        return DELTA_MAX * torch.tanh(raw / DELTA_MAX)

    def _scan(self, dotphi, jitter, memory, theta, pair_w, sample_noise=True,
              crossing=None, kappa_q=None):
        T = dotphi.shape[1]
        stride = self.knot_stride
        phase = theta
        segments = [phase[:, None]]
        corr_frames = []
        knots = []
        tokens = []
        deltas = []
        kl_terms = []
        shift = torch.zeros_like(theta)
        shifts = [shift[:, None]]
        start = 1
        while start < T:
            stop = min(start + stride, T)

            token = self._token(memory[:, start - 1], phase)
            tokens.append(token)
            stack = torch.stack(tokens, dim=1)
            correction = self.zdec.next_correction(stack)
            correction = CORRECTION_MAX * torch.tanh(correction / CORRECTION_MAX)
            knots.append(correction)

            on = None
            if crossing is not None and self.kappa_gate:
                on = crossing[:, start - 1:stop - 1].any(1)
            d_k = (self._knot_delta(stack, on) if self.delta_on
                   else torch.zeros_like(correction))
            deltas.append(d_k)
            if self.delta_on and kappa_q is not None:
                kp = torch.full_like(d_k, self.kappa_physical)
                if on is not None:
                    kp = torch.where(on, torch.full_like(d_k, KAPPA_INTER), kp)
                kl_terms.append(mean_resultant(kappa_q[:, start - 1]) * kp
                                * (1.0 - torch.cos(d_k)))

            steps = dotphi[:, start - 1:stop - 1] * torch.exp(correction)[:, None]
            if sample_noise:
                steps = steps + jitter[:, start - 1:stop - 1]
            if self.delta_on:
                steps = torch.cat([steps[:, :1] + d_k[:, None], steps[:, 1:]], dim=1)

            segment = phase[:, None] + torch.cumsum(
                steps * pair_w[:, start - 1:stop - 1], dim=1)
            segments.append(segment)
            corr_frames.append(correction[:, None].expand(-1, stop - start))

            lift = dotphi[:, start - 1:stop - 1].detach() \
                * (torch.exp(correction) - 1.0)[:, None]
            span = shift[:, None] + torch.cumsum(
                lift * pair_w[:, start - 1:stop - 1], dim=1)
            shifts.append(span)
            shift = span[:, -1]

            phase = segment[:, -1]
            start = stop

        corr = torch.cat(corr_frames, dim=1)
        corr_full = torch.cat([corr, corr[:, -1:]], dim=1)

        kl_delta = (torch.stack(kl_terms, dim=1).sum(1) if kl_terms
                    else torch.zeros_like(theta))
        return (torch.cat(segments, dim=1), corr_full, knots,
                torch.cat(shifts, dim=1), torch.stack(deltas, dim=1), kl_delta)

    def kl_jitter(self, mu, kappa, mask, crossing=None):
        """Per-frame KL( vM(mu,kappa) || vM(mu,kappa_p) ): prices concentration only.
        crossing prices those steps against KAPPA_INTER instead (the kappa-gate:
        one cheap phase jump per bar line, the phase-side twin of the walk gate)."""
        kp = torch.full_like(kappa, self.kappa_physical)
        if crossing is not None:
            kp = torch.where(crossing, torch.full_like(kappa, KAPPA_INTER), kp)
        return (kl_vonmises(mu, kappa, mu, kp) * mask).sum(1)

    def tempo_log_prior(self, dotphi, w):
        log_dotphi = torch.log(dotphi)

        z = (log_dotphi[:, 0] - TEMPO_PRIOR_MU) / TEMPO_PRIOR_SIGMA
        init = -0.5 * z ** 2 - math.log(TEMPO_PRIOR_SIGMA) - 0.5 * math.log(2.0 * math.pi)

        step = log_dotphi[:, 1:] - log_dotphi[:, :-1]
        pair = (w[:, 1:] > 0) & (w[:, :-1] > 0)

        lp = -0.5 * (step / TEMPO_WALK_SIGMA) ** 2 - math.log(TEMPO_WALK_SIGMA) \
            - 0.5 * math.log(TWO_PI)

        return init + (lp * pair).sum(1)

    def phase_log_prior(self, phi, dotphi, w, kappa_p):
        pair = (w[:, 1:] > 0) & (w[:, :-1] > 0)
        predicted = phi[:, :-1] + dotphi[:, :-1]

        return (vonmises_log_density(phi[:, 1:], predicted, kappa_p) * pair).sum(1)

    def forward(self, h, mask, y, samples: int = 1, pos_weight: float = 1.0):
        post, memory = self.encoder(h, mask)
        phase, tempo, rotation = post["phase"], post["tempo"], post["rotation"]
        
        phase_kappa = phase["kappa"]
        tempo_mu, tempo_sigma = tempo["mu"], tempo["sigma"]
        rotation_weight = rotation["weight"]

        w = torch.ones_like(phase_kappa) if mask is None else mask
        pair_w = ((w[:, 1:] > 0) & (w[:, :-1] > 0)).to(w.dtype)

        entropy_norm = 0.5 * math.log(2.0 * math.pi * math.e)
        h_tempo = ((entropy_norm + torch.log(tempo_sigma)) * w).sum(1)

        mean_ramp = torch.cumsum(tempo_mu, dim=1) - tempo_mu[:, :1]
        theta, _ = self.encoder._anchor(mean_ramp.detach(), rotation_weight)
        crossing = None
        if self.walk_kind == "gated" or self.kappa_gate:
            mean_phi = (mean_ramp + theta[:, None]).detach()
            crossing = torch.div(mean_phi[:, 1:], TWO_PI, rounding_mode="floor") \
                != torch.div(mean_phi[:, :-1], TWO_PI, rounding_mode="floor")

        recon = 0.0
        logp_tempo = 0.0
        kl_phase = 0.0
        phi = None

        for _ in range(samples):
            dotphi = tempo_mu * torch.exp(tempo_sigma * torch.randn_like(tempo_sigma))
            jitter = sample_vonmises(phase_kappa[:, 1:])
            phi, corr_full, _knots, _lift, _d, _kd = self._scan(
                dotphi, jitter, memory, theta, pair_w)
            dot_eff = dotphi * torch.exp(corr_full)

            logp_tempo = logp_tempo + self.walk_log_prior(dot_eff, w, crossing)
            kl_phase = kl_phase + self.kl_jitter(
                phi[:, 1:], phase_kappa[:, 1:], pair_w,
                crossing if self.kappa_gate else None)
            recon = recon + event_recon(self.emission_logits(phi, mask), y, w, pos_weight)

        recon = recon / samples
        logp_tempo = logp_tempo / samples

        kl_phase = kl_phase / samples
        kl_tempo = h_tempo + logp_tempo
        kl = kl_phase - kl_tempo

        return {"elbo": recon - kl, "recon": recon, "kl": kl,
                "phi": phi, "kappa": phase_kappa,
                "tempo_prior": logp_tempo, "tempo_entropy": h_tempo,
                "kl_phase": kl_phase}

    @property
    def deployed_net(self):
        """The inference network read at test time; controls assert ITS target-blindness."""
        return self.encoder

    @torch.no_grad()
    def infer_phase(self, h, mask=None):
        assert not self.training, "deployment path must run in eval mode"
        post, memory = self.encoder(h, mask)
        dotphi = post["tempo"]["mu"]
        mean_ramp = torch.cumsum(dotphi, dim=1) - dotphi[:, :1]
        theta, _ = self.encoder._anchor(mean_ramp, post["rotation"]["weight"])
        T = dotphi.shape[1]
        stride = self.knot_stride
        phase = theta
        segments = [phase[:, None]]
        tokens = []
        start = 1
        while start < T:
            stop = min(start + stride, T)
            tokens.append(self._token(memory[:, start - 1], phase))
            correction = self.zdec.next_correction(torch.stack(tokens, dim=1))
            correction = CORRECTION_MAX * torch.tanh(correction / CORRECTION_MAX)
            segment = phase[:, None] + torch.cumsum(
                dotphi[:, start - 1:stop - 1] * torch.exp(correction)[:, None], dim=1)
            segments.append(segment)
            phase = segment[:, -1]
            start = stop
        return torch.cat(segments, dim=1)

    @torch.no_grad()
    def emission_probs(self, h, mask=None):
        """Alternative D (8.3.4): the emission evaluated at the deployed mean path."""
        return torch.sigmoid(self.emission_logits(self.infer_phase(h, mask), mask))


class IntervalVAE(VBPM):
    """VBPM with the time/interval observation model in place of the Bernoulli."""

    wants_raw = True

    def __init__(self, input_dim: int, b_ratio: float = 0.1, kappa_place: float = 100.0,
                 phase_half: int = 0, interval_kind: str = "laplace",
                 disp_weight: float = 0.0, **kw):
        super().__init__(input_dim, **kw)
        self.b_ratio = float(b_ratio)
        self.kappa_place = float(kappa_place)
        self.phase_half = int(phase_half)
        self.interval_kind = interval_kind
        self.disp_weight = float(disp_weight)

    def forward(self, h, mask, y, samples: int = 1, pos_weight: float = 1.0, raw=None):
        assert raw is not None, "the interval emission needs the batch's downbeat_times"
        post, memory = self.encoder(h, mask)
        phase_kappa = post["phase"]["kappa"]
        tempo_mu, tempo_sigma = post["tempo"]["mu"], post["tempo"]["sigma"]
        rotation_weight = post["rotation"]["weight"]

        w = torch.ones_like(phase_kappa) if mask is None else mask
        pair_mask = (w[:, 1:] > 0) & (w[:, :-1] > 0)
        pair_w = pair_mask.to(w.dtype)
        ann_f, ann_valid = annotation_frames(raw, phase_kappa.device)

        h_tempo = ((0.5 * math.log(2.0 * math.pi * math.e)
                    + torch.log(tempo_sigma)) * w).sum(1)

        recon = 0.0
        logp_tempo = 0.0
        kl_phase = 0.0
        resultant = 0.0
        phi = None

        mean_ramp = torch.cumsum(tempo_mu, dim=1) - tempo_mu[:, :1]
        theta, _ = self.encoder._anchor(mean_ramp.detach(), rotation_weight)
        crossing = None
        if self.walk_kind == "gated" or self.kappa_gate:
            mean_phi = (mean_ramp + theta[:, None]).detach()
            crossing = torch.div(mean_phi[:, 1:], TWO_PI, rounding_mode="floor") \
                != torch.div(mean_phi[:, :-1], TWO_PI, rounding_mode="floor")
        corr_full = None
        knots = None
        corr_abs = 0.0

        for _ in range(samples):
            dotphi = tempo_mu * torch.exp(tempo_sigma * torch.randn_like(tempo_sigma))
            jitter = sample_vonmises(phase_kappa[:, 1:])
            phi, corr_full, knots, lift, deltas, kl_delta = self._scan(
                dotphi, jitter, memory, theta, pair_w, crossing=crossing,
                kappa_q=phase_kappa)
            dot_eff = dotphi * torch.exp(corr_full)
            phi_place = None if self.place_attach else (
                phi.detach() + (theta - theta.detach())[:, None]
                + (lift - lift.detach()) * self.place_lift)

            logp_tempo = logp_tempo + self.walk_log_prior(dot_eff, w, crossing)
            kl_phase = kl_phase + self.kl_jitter(
                phi[:, 1:], phase_kappa[:, 1:], pair_w,
                crossing if self.kappa_gate else None) + kl_delta
            em = interval_loglik(phi, ann_f, ann_valid, self.kappa_place,
                                 self.b_ratio, self.phase_half,
                                 self.interval_kind, phi_place, self.disp_weight,
                                 self.place_coord)
            recon = recon + em["loglik"]
            resultant = resultant + em["resultant"]
            corr_abs = corr_abs + corr_full.abs().mean()

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
                "delta": deltas, "kl_delta": kl_delta}

def event_recon(logits, y, w, pos_weight: float = 1.0):
    pos = (y > 0.5) & (w > 0)
    log_miss = -torch.nn.functional.softplus(logits)
    start = pos & ~torch.nn.functional.pad(pos, (1, 0))[:, :-1]
    run_id = torch.cumsum(start.long(), dim=1) * pos.long()
    n_run = int(run_id.max().item()) + 1
    miss_sum = logits.new_zeros(logits.shape[0], n_run).scatter_add(
        1, run_id, log_miss * pos.to(log_miss.dtype))
    counts = logits.new_zeros(logits.shape[0], n_run).scatter_add(
        1, run_id, pos.to(log_miss.dtype))
    hit = torch.log1p(-torch.exp(miss_sum).clamp(max=1.0 - 1e-6))
    event_ll = (hit[:, 1:] * (counts[:, 1:] > 0)).sum(1)
    neg_ll = (log_miss * (~pos).to(log_miss.dtype) * w).sum(1)
    return pos_weight * event_ll + neg_ll


def downbeat_times(mu, mask=None):
    """Fractional frame times where the phase crosses multiples of 2 pi, per item.
    Linear interpolation between frames removes downbeat_frames' up-to-one-frame
    early bias."""
    r = torch.remainder(mu, 2.0 * math.pi)
    drop = torch.diff(r, dim=-1) < -math.pi
    if mask is not None:
        drop = drop & (mask[:, 1:] > 0)
    out = []
    for b in range(mu.shape[0]):
        idx = torch.nonzero(drop[b], as_tuple=False)[:, 0]
        r0 = r[b, idx]
        r1 = r[b, idx + 1] + 2.0 * math.pi
        frac = (2.0 * math.pi - r0) / (r1 - r0).clamp(min=1e-9)
        out.append(idx.to(mu.dtype) + frac)
    return out


def downbeat_frames(mu, mask=None):
    """Rule g (8.1.2): a downbeat is where the phase crosses ZERO. Deterministic."""
    zero_to_two_pi = torch.remainder(mu, 2.0 * math.pi)
    crossing = torch.diff(zero_to_two_pi, dim=-1) < -math.pi
    if mask is not None:
        crossing = crossing & (mask[:, 1:] > 0)
    return crossing
