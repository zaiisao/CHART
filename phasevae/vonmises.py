"""von Mises: closed-form KL and reparameterised Best-Fisher sampling.

Both halves are the ones the ELBO needs and nothing else. The KL is analytic (never
sampled -- the reconstruction term is the only Monte Carlo in this model), and the
sampler is reparameterised so that d(sample)/d(kappa) exists.

Numerics: every Bessel call goes through ``torch.special.i0e``/``i1e`` (exponentially
scaled), so log I0(k) = log i0e(k) + k stays finite for k in the hundreds. The sampler
runs in float64 internally.
"""
from __future__ import annotations

import math

import torch

SMALL_KAPPA = 1e-4   # below this the von Mises is uniform to float64 precision, and the
                     # Best-Fisher rho = (tau - sqrt(2 tau)) / (2 kappa) is 0/0


def log_i0(kappa: torch.Tensor) -> torch.Tensor:
    """Return log I0(kappa), stable for large kappa via the exponentially scaled Bessel."""
    return torch.log(torch.special.i0e(kappa)) + kappa


def mean_resultant(kappa: torch.Tensor) -> torch.Tensor:
    """A(kappa) = I1(kappa) / I0(kappa) = E[cos x] under vM(0, kappa)."""
    return torch.special.i1e(kappa) / torch.special.i0e(kappa)


def kl_vonmises(mu1, kappa1, mu2, kappa2):
    """KL( vM(mu1, kappa1) || vM(mu2, kappa2) ), closed form, elementwise.

    log(I0(k2)/I0(k1)) + A(k1) * (k1 - k2 cos(mu1 - mu2)).

    Args:
        mu1: posterior mean direction (radians).
        kappa1: posterior concentration, >= 0.
        mu2: prior mean direction (radians).
        kappa2: prior concentration, >= 0. kappa2 = 0 gives the KL to Uniform[0, 2pi).

    Returns:
        Elementwise KL, same shape as the broadcast of the arguments.
    """
    return (log_i0(kappa2) - log_i0(kappa1)
            + mean_resultant(kappa1) * (kappa1 - kappa2 * torch.cos(mu1 - mu2)))


def _best_fisher_rho(kappa: torch.Tensor) -> torch.Tensor:
    """Best-Fisher's rho, differentiable in kappa (kappa clamped away from 0)."""
    safe = kappa.clamp_min(SMALL_KAPPA)
    tau = 1.0 + torch.sqrt(1.0 + 4.0 * safe * safe)
    return (tau - torch.sqrt(2.0 * tau)) / (2.0 * safe)


def sample_vonmises(kappa: torch.Tensor, max_rounds: int = 64) -> torch.Tensor:
    """Reparameterised sample from vM(0, kappa), elementwise, shape of ``kappa``.

    Best-Fisher (1979) rejection sampling. The rejection loop runs under no_grad and
    only decides WHICH uniform draws are used; the returned angle is then recomputed
    from those frozen draws as a differentiable function of kappa, which is the
    Naesseth et al. (2017) reparameterised-rejection path.

    Caveat, stated rather than hidden: this is the pathwise term only. The exact
    gradient also carries a score-function correction for the acceptance probability's
    own dependence on kappa; it is omitted here (standard RSVI practice), so gradients
    w.r.t. kappa are slightly biased while gradients w.r.t. the mean direction -- which
    enters additively outside this function -- are exact.

    Args:
        kappa: non-negative concentrations, any shape.
        max_rounds: proposal rounds before falling back to uniform on the stragglers.

    Returns:
        Angles in (-pi, pi].
    """
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

    # acos(f) directly is a NaN factory here: at kappa ~ 10^3 the accepted f sits at
    # 1 - 1e-7, d acos/df = -1/sqrt(1 - f^2) blows up, and f rounding to exactly 1.0
    # makes it infinite -- which is how the first full training run went NaN in one
    # epoch. 1 - f is instead computed in closed form (no cancellation) and
    # acos(f) = 2 asin(sqrt((1 - f) / 2)) is stable at both ends. sqrt(x + eps), not
    # sqrt(clamp(x)): d sqrt / dx is infinite at 0 and a clamp passes zero gradient
    # there, so a hard floor gives inf * 0 = NaN on the rare draw that lands on it.
    one_minus_f = ((r - 1.0) * (1.0 - z) / (r + z)).clamp(0.0, 2.0)
    sign = torch.where(u3 > 0.5, 1.0, -1.0)
    angle = 2.0 * sign * torch.asin(
        torch.sqrt(one_minus_f / 2.0 + 1e-24).clamp(max=1.0))

    uniform = (2.0 * u1 - 1.0) * math.pi
    angle = torch.where(tiny | (~accepted), uniform, angle)
    return angle.to(kappa.dtype)


def second_resultant(kappa: torch.Tensor) -> torch.Tensor:
    """A_2(kappa) = I_2/I_0 = E[cos 2(phi - mu)] under vM(mu, kappa).

    Exact via the Bessel recurrence I_0 - I_2 = (2/kappa) I_1, so A_2 = 1 - (2/kappa) A_1.
    Needed because the constant-rate prior's mean 2*phi_{t-1} - phi_{t-2} puts coefficient
    2 on the middle frame, and E[e^{i n phi}] = A_n(kappa) e^{i n mu}. Substituting A_1^2
    is wrong by 61% at kappa = 2.

    Below kappa ~ 0.1 the closed form is 1 - (something ~ 1) and cancels catastrophically;
    the series limit A_2 -> kappa^2/8 is used there (matches scipy to 2e-7 at 0.1).
    """
    a1 = mean_resultant(kappa)
    return torch.where(kappa < 0.1,
                       kappa * kappa / 8.0,
                       1.0 - (2.0 / kappa.clamp(min=1e-12)) * a1)
