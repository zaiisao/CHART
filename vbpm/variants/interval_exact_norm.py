"""The interval emission, normalised for the window's truncation and additively guarded.

`interval.py` writes down a density over the annotated downbeat times and never divides
by its own mass. It cannot: the annotations are observed ONLY inside the excerpt, so the
sample space is the ordered N-tuples that fit in the window, and the mass the model puts
there depends on the latent. Measured on ballroom 0, log Z is +0.29 at the true rate and
climbs 3.35 nats across k in [1, 4] -- a rate-dependent tilt sitting inside what is
supposed to be a likelihood, i.e. an unpaid bill of the same kind the widened-target
audit found in the Bernoulli.

The normaliser is exact and cheap. Push the observation through the model's own phase
map, psi_i = phi(t_i); the Jacobian prod dotphi(t_i) is exactly the one the likelihood
already carries, so integrating the density over t in the window is integrating

    Z = int_{lo <= psi_1 < psi_2 < ... < psi_N <= hi}
            vM(psi_1; 0, kappa) prod_{i<N} K(psi_i, psi_{i+1})  dpsi
    K(a, b) = Laplace(log((b - a) / 2pi); 0, b_ratio) / (b - a)

over phase, with lo = phi(first frame) and hi = phi(last frame). K depends on b - a
alone, so the transfer operator is a CONVOLUTION: put f_1(u) = vM(lo + u) on a grid over
u in [0, L], apply f_{i+1} = f_i * K once per annotation gap, integrate. N-1 causal
convolutions, done by FFT, differentiable in lo and hi. Truncation needs no separate
enforcement: K is supported on b > a, so mass that leaves [0, L] can only travel further
right and never returns.

What that buys and what it costs, both measured (docs/interval_exact_norm_report):
the normalised density integrates to 1 at every rate (against 1.34 at truth and 38 at
k = 4 for the raw form), and the doubling side keeps its slope. The SLOW side pays:
conditioning on N annotations having been observed inside the window is exactly the
statement that they had to fit, so the normaliser refunds most of the interval ruler's
case against a too-slow rate -- and the refund is not monotone, so k in (0, 1) grows
spurious local maxima. Truth stays the global maximum on both songs measured.

Guards here are ADDITIVE. `clamp(min=eps)` returns EXACTLY zero gradient for every value
below eps -- the same dead zone that froze the tempo channel for hours -- while x + eps
stays bounded and never stops voting.

The rotation surgery of `interval.py` is inherited unchanged: the placement factor scores
a detached ramp. Z needs no such surgery, because `_ramp` pins mu0[0] = 0, so lo is the
rotation alone and L = hi - lo is the rate alone.
"""
from __future__ import annotations

import math

import torch

from .base import common_kwargs, epoch_note, objective, on_epoch as _base_on_epoch  # noqa: F401
from .base import optimizer  # noqa: F401
from .interval import (IntervalVAE, annotation_frames, interp_phase, interval_penalty,
                       smooth_phase)
from ..model import TWO_PI, log_i0, sample_vonmises

DEFAULTS = {"b_ratio": 0.1, "kappa_place": 100.0, "kappa_anneal": "3,300,0.7",
            "sigma_ceil": 0.01, "phase_half": 0, "interval_kind": "laplace",
            "norm_grid": 2048, "normalise": True}

DOTPHI_EPS = 1e-8
RATIO_EPS = 1e-6


def additive_floor(x, eps: float):
    """x + eps, the guard that keeps voting.

    `clamp(min=eps)` is flat below eps, so d/dx is EXACTLY 0 there and a value that has
    fallen through the floor can never climb back by gradient. `x + eps` is bounded above
    by 1/eps in the log and nonzero everywhere. Softplus enters only so that a negative x
    -- which phase jitter can produce between two close annotations -- stays positive
    instead of turning the log into a NaN; for x above roughly 20 eps it returns x + eps
    to the last bit of the float, so this is `x + eps` everywhere it is ever evaluated.
    """
    return eps * torch.nn.functional.softplus(x / eps + 1.0)


def path_dotphi(phi, half: int = 25):
    """dphi/dt from the sampled path over +-half frames, additively floored."""
    pad = torch.nn.functional.pad(phi[:, None, :], (half, half), mode="replicate")[:, 0]
    slope = (pad[:, 2 * half:] - pad[:, :-2 * half]) / (2.0 * half)
    return additive_floor(slope, DOTPHI_EPS)


def log_partition(lo, hi, n_ann, kappa_place, b_ratio: float, grid: int = 2048):
    """log Z(lo, hi, N) [B]: the truncated interval likelihood's exact normaliser.

    The transfer-operator recursion described in the module docstring, on `grid` points
    spanning [0, hi - lo] per window. Everything runs in float64 and every step is
    rescaled to unit peak with the log of the scale carried alongside, so the N-fold
    attenuation cannot underflow. `grid` is the only approximation: halving it moves the
    answer by less than 0.01 nats at the default (measured, N = 12 and N = 27).

    The convolution is done DIRECTLY (grouped conv1d), not by FFT. Every term in the sum
    is nonnegative, so direct summation is accurate to G*eps RELATIVE, whatever the
    dynamic range; the FFT is not, and on the slow side -- where the surviving density
    spans e^-200 across the grid and one step attenuates by 1e-9 -- its roundoff floor
    swamped the answer and produced negative "densities" and NaNs from k = 0.7 down.
    Direct summation returns exactly zero negatives at every k measured.
    """
    device = lo.device
    lo64 = lo.double()
    span = hi.double() - lo64
    step = span / float(grid - 1)

    axis = torch.linspace(0.0, 1.0, grid, dtype=torch.float64, device=device)
    u = span[:, None] * axis

    kappa = torch.as_tensor(float(kappa_place), dtype=torch.float64, device=device)
    log_f = kappa * torch.cos(lo64[:, None] + u) - math.log(TWO_PI) - log_i0(kappa)
    scale = log_f.max(dim=1).values
    f = torch.exp(log_f - scale[:, None])

    u_safe = torch.where(u > 0, u, torch.ones_like(u))
    log_k = (-torch.log(u_safe / TWO_PI).abs() / b_ratio
             - math.log(2.0 * b_ratio) - torch.log(u_safe))
    kern = torch.where(u > 0, torch.exp(log_k), torch.zeros_like(u))

    weight = kern.flip(-1)[:, None, :]

    left = torch.ones(grid, dtype=torch.float64, device=device)
    left[0] = 0.5
    trapz = left.clone()
    trapz[-1] = 0.5

    def mass(density):
        return (density * trapz).sum(1) * step

    log_z = torch.log(mass(f)) + scale
    counts = n_ann.to(device)
    for gap in range(1, int(counts.max().item())):
        padded = torch.nn.functional.pad(f * left, (grid - 1, 0))[None]
        moved = torch.nn.functional.conv1d(padded, weight, groups=lo.shape[0])[0]
        f = moved * step[:, None]
        peak = f.max(dim=1).values
        f = f / peak[:, None]
        scale = scale + torch.log(peak)
        log_z = torch.where(counts == gap + 1, torch.log(mass(f)) + scale, log_z)
    return log_z


def interval_loglik(phi, ann_f, ann_valid, kappa_place: float, b_ratio: float,
                    phase_half: int = 0, kind: str = "laplace", phi_place=None,
                    last_frame=None, norm_grid: int = 2048, normalise: bool = True):
    """log p(annotation times | phi, window) [B], normalised over the window.

    The unnormalised part is `interval.interval_loglik` with additive guards; the new
    term is -log Z, which prices the fact that N annotations were seen INSIDE the excerpt.
    `last_frame` [B] long indexes each window's final valid frame; lo and hi are read from
    phi there, which is the same map the change of variables uses.
    """
    kappa = torch.as_tensor(kappa_place, device=phi.device, dtype=phi.dtype)
    at = interp_phase(smooth_phase(phi, phase_half), ann_f)
    first = ann_valid.cumsum(1).eq(1.0).to(phi.dtype) * ann_valid

    at_place = at if phi_place is None else interp_phase(smooth_phase(phi_place, phase_half), ann_f)
    place = ((kappa * torch.cos(at_place) - math.log(TWO_PI) - log_i0(kappa)) * first).sum(1)
    pair = ann_valid[:, 1:] * ann_valid[:, :-1]
    ratio = additive_floor((at[:, 1:] - at[:, :-1]) / TWO_PI, RATIO_EPS)
    interval = (-interval_penalty(torch.log(ratio), b_ratio, kind) * pair).sum(1)

    d_at = interp_phase(path_dotphi(phi), ann_f)
    jac = (torch.log(d_at) * ann_valid).sum(1) - (torch.log(TWO_PI * ratio) * pair).sum(1)

    if not normalise:
        zero = torch.zeros_like(place)
        return {"loglik": place + interval + jac, "place": place, "interval": interval,
                "log_z": zero}

    if last_frame is None:
        last_frame = torch.full_like(ann_valid[:, 0], phi.shape[1] - 1).long()
    lo = phi[:, 0]
    hi = phi.gather(1, last_frame[:, None].clamp(min=0))[:, 0]
    n_ann = ann_valid.sum(1).long().clamp(min=1)
    log_z = log_partition(lo, hi, n_ann, kappa_place, b_ratio, norm_grid).to(phi.dtype)
    return {"loglik": place + interval + jac - log_z, "place": place,
            "interval": interval, "log_z": log_z}


class IntervalExactNormVAE(IntervalVAE):
    """IntervalVAE whose emission is a genuine density over the window's annotations."""

    wants_raw = True

    def __init__(self, input_dim: int, norm_grid: int = 2048, normalise: bool = True, **kw):
        super().__init__(input_dim, **kw)
        self.norm_grid = int(norm_grid)
        self.normalise = bool(normalise)

    def forward(self, h, mask, y, samples: int = 1, pos_weight: float = 1.0, raw=None):
        assert raw is not None, "the interval emission needs the batch's downbeat_times"
        post, _ = self.encoder(h, mask)
        mu, kappa = post["phase"]["mu"], post["phase"]["kappa"]
        tempo = post["tempo"]
        kl = (self.kl_jitter(mu, kappa, mask) - tempo["log_prior"]
              - tempo["entropy"])
        ann_f, ann_valid = annotation_frames(raw, mu.device)
        last_frame = (mask.sum(1).long() - 1).clamp(min=0)

        offset = post["offset"]["mu"][:, None]
        recon, log_z = 0.0, 0.0
        for _ in range(samples):
            jitter = sample_vonmises(kappa)
            phi = mu + jitter
            phi_place = (mu - offset).detach() + offset + jitter
            scored = interval_loglik(phi, ann_f, ann_valid, self.kappa_place,
                                     self.b_ratio, self.phase_half, self.interval_kind,
                                     phi_place, last_frame, self.norm_grid,
                                     self.normalise)
            recon = recon + scored["loglik"]
            log_z = log_z + scored["log_z"]
        recon = recon / samples

        return {"elbo": recon - kl, "recon": recon, "kl": kl, "phi": mu, "kappa": kappa,
                "log_z": log_z / samples,
                "tempo_prior": tempo["log_prior"], "tempo_entropy": tempo["entropy"]}


def build_model(cfg, input_dim: int) -> IntervalExactNormVAE:
    return IntervalExactNormVAE(input_dim, norm_grid=cfg.norm_grid,
                                normalise=cfg.normalise, b_ratio=cfg.b_ratio,
                                kappa_place=cfg.kappa_place, phase_half=cfg.phase_half,
                                interval_kind=cfg.interval_kind,
                                **common_kwargs(cfg))


def on_epoch(model, cfg, epoch: int) -> None:
    """Placement precision anneals; the interval ruler carries the rate meanwhile."""
    _base_on_epoch(model, cfg, epoch)
    if not cfg.kappa_anneal:
        return
    lo, hi, frac = (float(v) for v in cfg.kappa_anneal.split(","))
    ramp = min(1.0, epoch / max(1.0, frac * cfg.epochs))
    model.kappa_place = math.exp(math.log(lo) + ramp * (math.log(hi) - math.log(lo)))
