"""Trained evidence heads over frozen frontend features (§6.1: our head, their features).

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
