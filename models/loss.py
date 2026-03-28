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
    pos_weight: float = 20.0,
    free_bits: float = 0.0,
    free_bits_meter: float | None = None,
    free_bits_phase: float | None = None,
    free_bits_tempo: float | None = None,
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
        pos_weight: Weight for positive beat frames in BCE loss. Compensates for
            class imbalance (~1% positive rate). Default: 20.0.
        free_bits: Default minimum KL per latent per sample (nats). Used for any
            latent that doesn't have a per-latent override. Default: 0.0.
        free_bits_meter: Override free_bits for meter KL. Default: None (use free_bits).
        free_bits_phase: Override free_bits for phase KL. Default: None (use free_bits).
        free_bits_tempo: Override free_bits for tempo KL. Default: None (use free_bits).

    Returns:
        Tuple of (total_loss, component_dict) where component_dict has
        keys ``bce``, ``kl_meter``, ``kl_phase``, ``kl_tempo``.
    """
    fb_meter = free_bits_meter if free_bits_meter is not None else free_bits
    fb_phase = free_bits_phase if free_bits_phase is not None else free_bits
    fb_tempo = free_bits_tempo if free_bits_tempo is not None else free_bits

    # --- Reconstruction: Binary Cross-Entropy ---
    pw = torch.tensor([pos_weight], device=beat_logits.device, dtype=beat_logits.dtype)
    bce = F.binary_cross_entropy_with_logits(
        beat_logits.squeeze(-1), beat_targets, pos_weight=pw, reduction="mean",
    )

    def _kl_with_free_bits(kl: Tensor, fb: float) -> Tensor:
        # kl: [B, T] — mean over T per sample, clamp, then mean over B
        if fb > 0.0:
            return kl.mean(dim=-1).clamp(min=fb).mean()
        return kl.mean()

    # --- KL: Meter (Categorical) ---
    kl_m = _kl_with_free_bits(categorical_kl(
        posterior["meter_logits"],
        prior["meter_logits"],
    ), fb_meter)

    # --- KL: Phase (von Mises) ---
    kappa_q = posterior["phase_log_kappa"].exp()
    kl_phi = _kl_with_free_bits(von_mises_kl(
        posterior["phase_mu"],
        kappa_q,
        prior["phase_mu"],
        prior["phase_kappa"],
    ), fb_phase)

    # --- KL: Tempo (Log-Normal / Gaussian in log-space) ---
    sigma_q = posterior["tempo_log_sigma"].exp()
    kl_tempo = _kl_with_free_bits(lognormal_kl(
        posterior["tempo_mu"],
        sigma_q,
        prior["tempo_mu"],
        prior["tempo_sigma"],
    ), fb_tempo)

    # --- Total ---
    total = bce + beta * (kl_m + kl_phi + kl_tempo)

    components = {
        "bce": bce.detach(),
        "kl_meter": kl_m.detach(),
        "kl_phase": kl_phi.detach(),
        "kl_tempo": kl_tempo.detach(),
    }

    return total, components
