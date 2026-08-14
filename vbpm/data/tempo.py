"""Bar period FROM AUDIO: the estimator that removes the last oracle on the model's path."""
from __future__ import annotations

import math

import torch

P_MIN_S, P_MAX_S = 1.0, 6.0        # bar-period search range, seconds
P_STEP_S = 0.005                   # grid resolution, seconds
N_HARM = 4                         # harmonic-sum depth
SMOOTH = 3                         # box-filter width, frames
L_MIN = 40                         # ACF lags start here, frames
ALPHA = 0.75                       # accept peaks at >= ALPHA * max score
TWO_PI = 2.0 * math.pi


def _autocorrelation(p: torch.Tensor, n_lags: int) -> torch.Tensor:
    """[B, T] zero-mean signal -> [B, n_lags+1] linear (not circular) autocorrelation."""
    length = p.shape[1]
    size = 1 << (2 * length - 1).bit_length()
    spectrum = torch.fft.rfft(p, n=size)
    r = torch.fft.irfft(spectrum * spectrum.conj(), n=size)[:, :n_lags + 1]
    overlap = (length - torch.arange(n_lags + 1, device=p.device)).clamp(min=1)
    return r / overlap


def estimate_bar_period(activation: torch.Tensor, mask: torch.Tensor,
                        fps: float) -> torch.Tensor:
    """[B, T] downbeat activation in [0, 1], [B, T] validity -> [B] bar period, seconds."""
    device = activation.device
    p = (activation * mask).unsqueeze(1)
    kernel = torch.full((1, 1, SMOOTH), 1.0 / SMOOTH, device=device, dtype=p.dtype)
    p = torch.conv1d(p, kernel, padding=SMOOTH // 2).squeeze(1)

    valid = mask.sum(1, keepdim=True).clamp(min=1.0)
    p = (p - (p * mask).sum(1, keepdim=True) / valid) * mask     # zero-mean on real frames

    score, grid = harmonic_score(p, fps)
    if score is None:                                           # window too short to try
        return torch.full((activation.shape[0],), 0.5 * (P_MIN_S + P_MAX_S), device=device)
    return grid[pick_period(score)]


def harmonic_score(p: torch.Tensor, fps: float):
    """Zero-meaned [B, T] activation -> ([B, G] harmonic-sum score, [G] period grid, s)."""
    device = p.device
    lag_max = int(min(N_HARM * P_MAX_S * fps, p.shape[1] - 50))
    if lag_max <= L_MIN + 2:
        return None, None
    r = _autocorrelation(p, lag_max)
    r[:, :L_MIN] = 0.0

    grid = torch.arange(P_MIN_S, P_MAX_S + 1e-9, P_STEP_S, device=device) * fps   # [G]
    harmonics = torch.arange(1, N_HARM + 1, device=device, dtype=grid.dtype)
    lags = harmonics[:, None] * grid[None, :]                                     # [K, G]
    usable = lags <= lag_max
    lo = lags.clamp(max=lag_max - 1).floor().long()
    frac = (lags - lo).clamp(0.0, 1.0)
    vals = (r[:, lo] * (1.0 - frac) + r[:, lo + 1] * frac).clamp(min=0.0)         # [B,K,G]
    score = (vals * usable).sum(1) / usable.sum(0).clamp(min=1)                   # [B, G]
    return score, grid / fps


def pick_period(score: torch.Tensor) -> torch.Tensor:
    """[B, G] score -> [B] index of the SMALLEST peak clearing ALPHA * max."""
    peak_floor = ALPHA * score.max(dim=1, keepdim=True).values
    interior = score[:, 1:-1]
    is_peak = ((interior > score[:, :-2]) & (interior > score[:, 2:])
               & (interior >= peak_floor))
    return torch.where(is_peak.any(1), is_peak.float().argmax(1) + 1, score.argmax(1))


def delta_from_audio(activation: torch.Tensor, mask: torch.Tensor,
                     fps: float) -> torch.Tensor:
    """[B] per-frame phase advance 2*pi/(period*fps), estimated from audio alone."""
    period = estimate_bar_period(activation, mask, fps).clamp(min=P_MIN_S, max=P_MAX_S)
    return TWO_PI / (period * fps)
