"""The three networks -- encoder, emission transformer, knot decoder -- and the
von Mises helpers they are written in terms of."""
from __future__ import annotations

import math

import torch
from torch import nn

from .constants import (KAPPA_PHYSICAL, MAX_KAPPA, TEMPO_PRIOR_MU, TWO_PI)
from .vonmises import log_i0, mean_resultant


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
        result = {"phase_log_kappa": out[..., 0], "tempo_log_mu": out[..., 1],
                  "tempo_sigma_logit": out[..., 2], "phase_mu_offset": out[..., 3]}
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

        phase_mu_offset = math.pi * torch.tanh(channels["phase_mu_offset"])
        # JA: log_phi_kappa_bias is a constant that is added to the encoder's log_kappa output.
        # As kappa initializes to a small value (around 1), it would otherwise take a long
        # time before reaching a reasonable value.
        phase_kappa = bounded_kappa(
            torch.exp(channels["phase_log_kappa"] + self.log_phi_kappa_bias) + 1e-3)
        tempo_mu = torch.exp(channels["tempo_log_mu"])
        tempo_sigma = nn.functional.softplus(channels["tempo_sigma_logit"])
        return {
            "phase": {"kappa": phase_kappa, "mu_offset": phase_mu_offset},
            "tempo": {"mu": tempo_mu, "sigma": tempo_sigma},
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
