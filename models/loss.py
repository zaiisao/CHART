"""ELBO loss for the variational bar pointer model.

Implements the complete loss function from Section 5.5 of the paper:

    L = BCE(reconstruction)
      + KL_meter(Categorical)
      + KL_phase(von Mises)
      + KL_tempo(Log-Normal)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from models.distributions import categorical_kl, lognormal_kl, von_mises_kl


def compute_elbo_loss(
    beat_logits: Tensor,
    beat_targets: Tensor,
    posterior: dict[str, Tensor],
    prior: dict[str, Tensor],
    beta: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the negative ELBO loss.

    Args:
        beat_logits: ``[B, T, 1]`` decoder output (pre-sigmoid).
        beat_targets: ``[B, T]`` binary beat indicators.
        posterior: Dict with keys ``meter_logits``, ``phase_mu``,
            ``phase_log_kappa``, ``tempo_mu``, ``tempo_log_sigma``.
        prior: Dict with keys ``meter_logits``, ``phase_mu``,
            ``phase_kappa``, ``tempo_mu``, ``tempo_sigma``.
        beta: KL weight for annealing (0 = pure reconstruction, 1 = full ELBO).

    Returns:
        Tuple of (total_loss, component_dict) where component_dict has
        keys ``bce``, ``kl_meter``, ``kl_phase``, ``kl_tempo``.
    """
    # --- Reconstruction: Binary Cross-Entropy ---
    bce = F.binary_cross_entropy_with_logits(
        beat_logits.squeeze(-1), beat_targets, reduction="mean",
    )

    # --- KL: Meter (Categorical) ---
    kl_m = categorical_kl(
        posterior["meter_logits"],
        prior["meter_logits"],
    ).mean()

    # --- KL: Phase (von Mises) ---
    kappa_q = posterior["phase_log_kappa"].exp()
    kl_phi = von_mises_kl(
        posterior["phase_mu"],
        kappa_q,
        prior["phase_mu"],
        prior["phase_kappa"],
    ).mean()

    # --- KL: Tempo (Log-Normal / Gaussian in log-space) ---
    sigma_q = posterior["tempo_log_sigma"].exp()
    kl_tempo = lognormal_kl(
        posterior["tempo_mu"],
        sigma_q,
        prior["tempo_mu"],
        prior["tempo_sigma"],
    ).mean()

    # --- Total ---
    total = bce + beta * (kl_m + kl_phi + kl_tempo)

    components = {
        "bce": bce.detach(),
        "kl_meter": kl_m.detach(),
        "kl_phase": kl_phi.detach(),
        "kl_tempo": kl_tempo.detach(),
    }

    return total, components
