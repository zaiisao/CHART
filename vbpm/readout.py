"""Rule g: the deployment read-out that turns a phase path into downbeat times.

A downbeat is where the bar phase crosses a multiple of 2 pi. This is the
model's whole output contract, so it lives on its own rather than inside
whichever model happens to be current.
"""
from __future__ import annotations

import math

import torch


def downbeat_times(mu, mask=None):
    """Fractional frame times where the phase crosses multiples of 2 pi, per item.

    Linear interpolation between frames removes downbeat_frames' up-to-one-frame
    early bias.
    """
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
