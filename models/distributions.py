"""Distribution utilities for the variational bar pointer model.

Provides closed-form KL divergences and reparameterized samplers for:
- Categorical (meter) with Gumbel-Softmax relaxation
- von Mises (phase) with Best-Fisher rejection sampling + implicit reparam
- Log-Normal (tempo) via Gaussian reparameterization in log-space
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_KAPPA_MAX = 700.0  # clamp to avoid Bessel overflow
_KAPPA_MIN = 1e-6
TWO_PI = 2.0 * math.pi

# ---------------------------------------------------------------------------
# Bessel helpers (numerically stable via exponentially-scaled versions)
# ---------------------------------------------------------------------------

def _log_i0(kappa: Tensor) -> Tensor:
    """Compute log I_0(kappa) in a numerically stable way."""
    return torch.log(torch.special.i0e(kappa)) + kappa.abs()


def _A(kappa: Tensor) -> Tensor:
    """Mean resultant length A(kappa) = I_1(kappa) / I_0(kappa)."""
    return torch.special.i1e(kappa) / torch.special.i0e(kappa)


# ---------------------------------------------------------------------------
# Categorical (meter)
# ---------------------------------------------------------------------------

def categorical_kl(logits_q: Tensor, logits_p: Tensor) -> Tensor:
    """KL(Cat(q) || Cat(p)) from unnormalized logits.

    Args:
        logits_q: Posterior logits, shape ``(..., K)``.
        logits_p: Prior logits, shape ``(..., K)``.

    Returns:
        KL divergence per element, shape ``(...)``.
    """
    log_q = F.log_softmax(logits_q, dim=-1)
    log_p = F.log_softmax(logits_p, dim=-1)
    q = log_q.exp()
    return (q * (log_q - log_p)).sum(dim=-1)


def gumbel_softmax_sample(
    logits: Tensor,
    temperature: float,
    hard: bool = False,
) -> Tensor:
    """Sample from Gumbel-Softmax relaxation.

    Args:
        logits: Unnormalized log-probabilities, shape ``(..., K)``.
        temperature: Softmax temperature (tau > 0).
        hard: If True, use straight-through estimator.

    Returns:
        Soft (or hard) sample, shape ``(..., K)``.
    """
    return F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)


# ---------------------------------------------------------------------------
# Von Mises (phase)
# ---------------------------------------------------------------------------

class _VonMisesSampleFn(torch.autograd.Function):
    """Von Mises sampler with implicit reparameterization gradients.

    Forward: Best-Fisher rejection algorithm (Algorithm 2 in paper).
    Backward: implicit reparameterization (Figurnov et al. 2018).
    """

    @staticmethod
    def forward(ctx, mu: Tensor, kappa: Tensor) -> Tensor:  # type: ignore[override]
        # Best-Fisher rejection sampler
        kappa_safe = kappa.clamp(min=_KAPPA_MIN, max=_KAPPA_MAX)

        tau = 1.0 + torch.sqrt(1.0 + 4.0 * kappa_safe ** 2)
        rho = (tau - torch.sqrt(2.0 * tau)) / (2.0 * kappa_safe)
        r = (1.0 + rho ** 2) / (2.0 * rho)

        shape = mu.shape
        device = mu.device
        dtype = mu.dtype

        # Batched rejection sampling
        done = torch.zeros(shape, dtype=torch.bool, device=device)
        f_accepted = torch.zeros(shape, dtype=dtype, device=device)
        max_iter = 1000

        for _ in range(max_iter):
            if done.all():
                break
            u1 = torch.rand(shape, dtype=dtype, device=device)
            u2 = torch.rand(shape, dtype=dtype, device=device)

            c = torch.cos(math.pi * u1)
            f = (1.0 + r * c) / (r + c)

            accept = (kappa_safe * (r - f) + torch.log(f) - torch.log(r)) >= torch.log(u2)
            newly_accepted = accept & ~done
            f_accepted = torch.where(newly_accepted, f, f_accepted)
            done = done | accept

        # If some samples did not converge, use the last f (extremely rare)
        f_accepted = torch.where(done, f_accepted, f)

        u3 = torch.rand(shape, dtype=dtype, device=device)
        z = torch.where(u3 > 0.5, torch.acos(f_accepted), -torch.acos(f_accepted))
        sample = mu + z

        ctx.save_for_backward(z, kappa_safe)
        return sample

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        z, kappa = ctx.saved_tensors

        # Implicit reparameterization gradients:
        # d(sample)/d(mu) = 1
        grad_mu = grad_output

        # d(sample)/d(kappa) = cos(z) / (kappa * sin(z))
        # Guard against sin(z) = 0
        sin_z = torch.sin(z)
        sin_z_safe = torch.where(sin_z.abs() < 1e-8, torch.ones_like(sin_z) * 1e-8, sin_z)
        dz_dkappa = torch.cos(z) / (kappa * sin_z_safe)
        grad_kappa = grad_output * dz_dkappa

        return grad_mu, grad_kappa


def von_mises_sample(mu: Tensor, kappa: Tensor) -> Tensor:
    """Draw reparameterized samples from vM(mu, kappa).

    Args:
        mu: Mean direction, shape ``(...)``.
        kappa: Concentration parameter, shape ``(...)``.

    Returns:
        Samples on the circle, shape ``(...)``.
    """
    return _VonMisesSampleFn.apply(mu, kappa)


def von_mises_kl(
    mu_q: Tensor,
    kappa_q: Tensor,
    mu_p: Tensor,
    kappa_p: Tensor,
) -> Tensor:
    """KL(vM(mu_q, kappa_q) || vM(mu_p, kappa_p)).

    Args:
        mu_q: Posterior mean direction.
        kappa_q: Posterior concentration.
        mu_p: Prior mean direction.
        kappa_p: Prior concentration.

    Returns:
        KL divergence, same shape as inputs.
    """
    kappa_q = kappa_q.clamp(min=_KAPPA_MIN, max=_KAPPA_MAX)
    kappa_p = kappa_p.clamp(min=_KAPPA_MIN, max=_KAPPA_MAX)

    return (
        _log_i0(kappa_p) - _log_i0(kappa_q)
        + _A(kappa_q) * (kappa_q - kappa_p * torch.cos(mu_q - mu_p))
    )


# ---------------------------------------------------------------------------
# Log-Normal (tempo) — implemented as Gaussian KL in log-space
# ---------------------------------------------------------------------------

def lognormal_kl(
    mu_q: Tensor,
    sigma_q: Tensor,
    mu_p: Tensor,
    sigma_p: Tensor,
) -> Tensor:
    """KL(LogN(mu_q, sigma_q^2) || LogN(mu_p, sigma_p^2)).

    All parameters are in log-space (i.e. mu and sigma of the underlying
    Gaussian), so this reduces to the standard Gaussian KL.

    Args:
        mu_q: Posterior mean in log-space.
        sigma_q: Posterior std in log-space (> 0).
        mu_p: Prior mean in log-space.
        sigma_p: Prior std in log-space (> 0).

    Returns:
        KL divergence, same shape as inputs.
    """
    sigma_q = sigma_q.clamp(min=1e-8)
    sigma_p = sigma_p.clamp(min=1e-8)

    return (
        torch.log(sigma_p / sigma_q)
        + (sigma_q ** 2 + (mu_q - mu_p) ** 2) / (2.0 * sigma_p ** 2)
        - 0.5
    )


def lognormal_sample(mu: Tensor, sigma: Tensor) -> Tensor:
    """Reparameterized sample from LogNormal(mu, sigma^2).

    Returns the sample in the *original* (positive) space: exp(mu + sigma * eps).

    Args:
        mu: Mean in log-space.
        sigma: Std in log-space (> 0).

    Returns:
        Positive-valued sample, same shape as inputs.
    """
    eps = torch.randn_like(mu)
    return torch.exp(mu + sigma * eps)


def lognormal_sample_logspace(mu: Tensor, sigma: Tensor) -> Tensor:
    """Reparameterized sample, returned in log-space: mu + sigma * eps.

    Args:
        mu: Mean in log-space.
        sigma: Std in log-space (> 0).

    Returns:
        Sample in log-space, same shape as inputs.
    """
    eps = torch.randn_like(mu)
    return mu + sigma * eps
