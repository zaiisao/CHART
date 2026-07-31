"""Trained evidence heads over frozen frontend features.

The frontend is never trained by us; what we own is the head that reads its features.

AutocorrHead is the 2026-07-31 e2e winner (ALL-CV 0.595): learned channel projection ->
exact masked FFT autocorrelation over lags -> conv over the lag axis -> prior logits.
``extra_dim`` appends precomputed per-crop features (e.g. the 10-dim peak summary) to the
final MLP input, for combined-evidence variants.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AutocorrHead(nn.Module):
    """[B, T, in_dim] zero-padded features (+ optional [B, extra_dim]) -> [B, K] logits."""

    def __init__(self, in_dim: int = 512, channels: int = 16, n_lags: int = 250,
                 n_classes: int = 3, extra_dim: int = 0):
        super().__init__()
        self.n_lags = n_lags
        self.extra_dim = extra_dim
        self.proj = nn.Linear(in_dim, channels, bias=False)
        self.conv = nn.Sequential(
            nn.Conv1d(channels, 32, 9, stride=2, padding=4), nn.ReLU(),
            nn.Conv1d(32, 32, 9, stride=2, padding=4), nn.ReLU(),
            nn.Conv1d(32, 32, 9, stride=2, padding=4), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8))
        self.out = nn.Sequential(nn.Linear(32 * 8 + extra_dim, 64), nn.ReLU(),
                                 nn.Linear(64, n_classes))

    def forward(self, x, lengths, extra=None):
        """Exact masked autocorrelation via FFT; pads contribute zero by construction."""
        B, T, _ = x.shape
        valid = (torch.arange(T, device=x.device)[None, :]
                 < lengths[:, None]).to(x.dtype)                      # [B, T]
        u = self.proj(x) * valid[..., None]                           # [B, T, C]
        mean = u.sum(1) / lengths[:, None].to(x.dtype)
        u = (u - mean[:, None, :]) * valid[..., None]                 # centred over valid
        u = u.transpose(1, 2)                                         # [B, C, T]

        n_fft = 2 * T
        spectrum = torch.fft.rfft(u, n=n_fft)
        r = torch.fft.irfft(spectrum.abs() ** 2, n=n_fft)[..., :self.n_lags + 1]
        # dividing by (T_i - lag) and the lag-0 variance makes this the exact
        # per-crop autocorrelation despite the padding
        counts = (lengths[:, None].to(x.dtype)
                  - torch.arange(self.n_lags + 1, device=x.device)[None, :]).clamp(min=1.0)
        r = r / counts[:, None, :]
        r = r[..., 1:] / (r[..., :1] + 1e-8)                          # [B, C, n_lags]

        pooled = self.conv(r).flatten(1)                              # [B, 32*8]
        if self.extra_dim:
            pooled = torch.cat([pooled, extra], dim=-1)
        return self.out(pooled)


class TransformerPrior(nn.Module):
    """[B, T, in_dim] zero-padded features -> [B, K] meter logits, by attention over time.

    The tutorial's conditional-prior transformer (its section 9.3) specialized to Stage 0's
    single categorical latent: per-frame projection + sinusoidal positional encoding, full
    self-attention across all frames, and one learned query token whose attended state is
    the crop's meter evidence. No pooled summary statistic anywhere — every frame is
    consulted through attention. Kept deliberately small: the measured failure mode of
    trained priors is overfitting, not undercapacity.
    """

    def __init__(self, in_dim: int = 512, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, n_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2 * d_model, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers)
        self.out = nn.Linear(d_model, n_classes)

    @staticmethod
    def _positional_encoding(T, d_model, device, dtype):
        position = torch.arange(T, device=device, dtype=dtype)[:, None]
        rate = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=dtype)
                         * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(T, d_model, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * rate)
        pe[:, 1::2] = torch.cos(position * rate)
        return pe

    def forward(self, x, lengths, extra=None):
        """Meter logits from the query token's attended state; pads are masked out."""
        B, T, _ = x.shape
        tokens = self.proj(x) + self._positional_encoding(T, self.proj.out_features,
                                                          x.device, x.dtype)
        tokens = torch.cat([self.query.expand(B, 1, -1), tokens], dim=1)   # [B, 1+T, d]
        pad = (torch.arange(T, device=x.device)[None, :] >= lengths[:, None])
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), pad], dim=1)
        attended = self.blocks(tokens, src_key_padding_mask=pad)
        return self.out(attended[:, 0])
