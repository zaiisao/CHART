"""von Mises: closed-form KL and reparameterised Best-Fisher sampling."""
from __future__ import annotations

import math

import torch

SMALL_KAPPA = 1e-4   # below this the von Mises is uniform to float64 precision, and the
                     # Best-Fisher rho = (tau - sqrt(2 tau)) / (2 kappa) is 0/0


def log_i0(kappa: torch.Tensor) -> torch.Tensor:
    """Return log I0(kappa), stable for large kappa via the exponentially scaled Bessel."""
    return torch.log(torch.special.i0e(kappa)) + kappa


ASYMPTOTIC_KAPPA = 50.0   # above this A'(kappa) is taken from the series: the exact form
                          # 1 - A/k - A^2 cancels two float32 quantities agreeing to ~5e-6
                          # at kappa 1e5, which flips the sign of every phase-kappa gradient


class _MeanResultant(torch.autograd.Function):
    """A(kappa) with a derivative that survives float32 at the operating concentration."""

    @staticmethod
    def forward(ctx, kappa):
        a = torch.special.i1e(kappa) / torch.special.i0e(kappa)
        ctx.save_for_backward(kappa, a)
        return a

    @staticmethod
    def backward(ctx, grad_out):
        kappa, a = ctx.saved_tensors
        k = kappa.double().clamp_min(1e-12)
        ad = a.double()
        exact = 1.0 - ad / k - ad * ad
        asym = 0.5 / k ** 2 + 0.25 / k ** 3 + 0.375 / k ** 4
        deriv = torch.where(k > ASYMPTOTIC_KAPPA, asym, exact)
        return grad_out * deriv.to(grad_out.dtype)


def mean_resultant(kappa: torch.Tensor) -> torch.Tensor:
    """A(kappa) = I1(kappa) / I0(kappa) = E[cos x] under vM(0, kappa)."""
    return _MeanResultant.apply(kappa)


def kl_vonmises(mu1, kappa1, mu2, kappa2):
    """KL( vM(mu1, kappa1) || vM(mu2, kappa2) ), closed form, elementwise."""
    return (log_i0(kappa2) - log_i0(kappa1)
            + mean_resultant(kappa1) * (kappa1 - kappa2 * torch.cos(mu1 - mu2)))


def _best_fisher_rho(kappa: torch.Tensor) -> torch.Tensor:
    """Best-Fisher's rho, differentiable in kappa (kappa clamped away from 0)."""
    safe = kappa.clamp_min(SMALL_KAPPA)
    tau = 1.0 + torch.sqrt(1.0 + 4.0 * safe * safe)
    return (tau - torch.sqrt(2.0 * tau)) / (2.0 * safe)


def sample_vonmises(kappa: torch.Tensor, max_rounds: int = 64) -> torch.Tensor:
    """Reparameterised sample from vM(0, kappa), elementwise, shape of ``kappa``."""
    work = kappa.double()
    shape = work.shape
    with torch.no_grad():
        rho = _best_fisher_rho(work)
        r = (1.0 + rho * rho) / (2.0 * rho)
        accepted = torch.zeros(shape, dtype=torch.bool, device=work.device)
        u1 = torch.zeros(shape, dtype=torch.float64, device=work.device)
        u3 = torch.zeros(shape, dtype=torch.float64, device=work.device)

        for _ in range(max_rounds):
            if bool(accepted.all()):
                break
            p1 = torch.rand(shape, dtype=torch.float64, device=work.device)
            p2 = torch.rand(shape, dtype=torch.float64, device=work.device)
            p3 = torch.rand(shape, dtype=torch.float64, device=work.device)
            z = torch.cos(math.pi * p1)
            f = (1.0 + r * z) / (r + z)
            c = work * (r - f)
            ok = ((c * (2.0 - c) - p2) > 0) | ((torch.log(c / p2.clamp_min(1e-300))
                                                + 1.0 - c) >= 0)
            take = ok & (~accepted)
            u1 = torch.where(take, p1, u1)
            u3 = torch.where(take, p3, u3)
            accepted = accepted | ok

        # kappa below SMALL_KAPPA is uniform: accept any draw, the f-path is degenerate
        tiny = work < SMALL_KAPPA

    rho = _best_fisher_rho(work)
    r = (1.0 + rho * rho) / (2.0 * rho)

    # u1 = 0 exactly (torch.rand includes it) drives 1 - f to 0; clamping costs 1e-7 of
    # the support and keeps every derivative below finite
    u1 = u1.clamp(1e-7, 1.0 - 1e-7)
    z = torch.cos(math.pi * u1)

    one_minus_f = ((r - 1.0) * (1.0 - z) / (r + z)).clamp(0.0, 2.0)
    sign = torch.where(u3 > 0.5, 1.0, -1.0)
    angle = 2.0 * sign * torch.asin(
        torch.sqrt(one_minus_f / 2.0 + 1e-24).clamp(max=1.0))

    uniform = (2.0 * u1 - 1.0) * math.pi
    angle = torch.where(tiny | (~accepted), uniform, angle)
    return angle.to(kappa.dtype)


def second_resultant(kappa: torch.Tensor) -> torch.Tensor:
    """A_2(kappa) = I_2/I_0 = E[cos 2(phi - mu)] under vM(mu, kappa)."""
    a1 = mean_resultant(kappa)
    return torch.where(kappa < 0.1,
                       kappa * kappa / 8.0,
                       1.0 - (2.0 / kappa.clamp(min=1e-12)) * a1)
