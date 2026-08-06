"""Control 1: the von Mises sampler against scipy, before any training happens.

A previous implementation in this project returned E[cos x] ~ 0.8 regardless of kappa,
and every number taken through it was worthless. Three independent checks:
  * mean resultant length E[cos x] vs the analytic A(kappa) = I1/I0,
  * a tail comparison: P(|x| > pi/2) vs scipy's survival function,
  * a two-sample Kolmogorov-Smirnov test against scipy's own sampler,
plus a finite-difference check that d E[cos x] / d kappa flows (the sample is
reparameterised, not detached).

Run: PYTHONPATH=. python -m phasevae.checks.sampler
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import stats

from ..vonmises import mean_resultant, sample_vonmises

KAPPAS = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0, 2000.0, 10000.0)


def check(n: int = 200_000, seed: int = 0) -> list[dict]:
    """Sample at each kappa and compare to scipy/analytic. Returns one row per kappa."""
    torch.manual_seed(seed)
    rows = []

    for kappa in KAPPAS:
        k = torch.full((n,), kappa, dtype=torch.float64)
        x = sample_vonmises(k).numpy()
        analytic = float(mean_resultant(torch.tensor(kappa, dtype=torch.float64)))
        empirical = float(np.cos(x).mean())
        tail_emp = float((np.abs(x) > np.pi / 2).mean())
        tail_ref = float(2 * stats.vonmises.sf(np.pi / 2, kappa))
        reference = stats.vonmises.rvs(kappa, size=n, random_state=seed + 1)
        ks = stats.ks_2samp(x, reference)
        rows.append({"kappa": kappa, "E[cos] sampled": empirical, "A(kappa)": analytic,
                     "abs err": abs(empirical - analytic), "tail sampled": tail_emp,
                     "tail scipy": tail_ref, "KS D": float(ks.statistic),
                     "KS p": float(ks.pvalue)})
    return rows


def check_gradient(kappa: float = 3.0, n: int = 400_000) -> tuple[float, float]:
    """(pathwise, finite-difference) d E[cos x] / d kappa -- the reparam path must flow."""
    torch.manual_seed(7)
    k = torch.full((n,), kappa, dtype=torch.float64, requires_grad=True)
    torch.cos(sample_vonmises(k)).mean().backward()
    pathwise = float(k.grad.sum())

    eps = 1e-3
    lo = float(mean_resultant(torch.tensor(kappa - eps, dtype=torch.float64)))
    hi = float(mean_resultant(torch.tensor(kappa + eps, dtype=torch.float64)))
    return pathwise, (hi - lo) / (2 * eps)


def main() -> None:
    """Print the comparison table and the gradient check; assert the hard bounds."""
    rows = check()
    header = ("kappa", "E[cos] sampled", "A(kappa)", "abs err", "tail sampled",
              "tail scipy", "KS D", "KS p")
    print(" ".join(f"{h:>14s}" for h in header))
    for row in rows:
        print(" ".join(f"{row[h]:>14.6f}" for h in header))

    worst = max(r["abs err"] for r in rows)
    worst_ks = max(r["KS D"] for r in rows)
    min_p = min(r["KS p"] for r in rows)
    print(f"\nworst |E[cos] - A(kappa)| = {worst:.2e}   worst KS D = {worst_ks:.4f} "
          f"(min p = {min_p:.3f})")

    pathwise, finite = check_gradient()
    print(f"d E[cos]/d kappa: pathwise {pathwise:.5f} vs finite-difference {finite:.5f}")

    assert worst < 5e-3, "sampler does not match A(kappa)"
    assert min_p > 1e-3, "sampler distribution differs from scipy (KS)"
    print("\nSAMPLER OK")


if __name__ == "__main__":
    main()
