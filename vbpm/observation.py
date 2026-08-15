"""p(annotations | phi): the interval/placement factor, and the per-frame Bernoulli."""
from __future__ import annotations

import math

import numpy as np
import torch

from .constants import TWO_PI
from .vonmises import log_i0


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
