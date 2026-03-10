"""Loss utilities for the CHART SVT model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import math

def compute_prior_energy_loss(z_t: torch.Tensor, z_t_minus_1: torch.Tensor) -> torch.Tensor:
    """Compute continuous cosine-energy prior penalties.

    This placeholder represents the prior energy term for continuous rhythm
    states, including penalties related to bar-pointer progression consistency
    and tempo stability across adjacent latent states.

    Args:
        z_t: Current latent phase variables.
        z_t_minus_1: Previous latent phase variables.

    Returns:
        Scalar prior energy loss tensor.
    """

    tempo_t, beat_phase_t, bar_phase_t = z_t[:, 0], z_t[:, 1], z_t[:, 2]
    tempo_t_minus_1, beat_phase_t_minus_1, bar_phase_t_minus_1 = z_t_minus_1[:, 0], z_t_minus_1[:, 1], z_t_minus_1[:, 2]

    expected_beat_phase = beat_phase_t_minus_1 + tempo_t_minus_1
    beat_progression_loss = 1 - torch.cos(2 * math.pi * (beat_phase_t - expected_beat_phase))

    expected_bar_phase = bar_phase_t_minus_1 + (tempo_t_minus_1 / 4.0)
    bar_progression_loss = 1 - torch.cos(2 * math.pi * (bar_phase_t - expected_bar_phase))

    boundary_weight = 1 - torch.cos(2 * math.pi * beat_phase_t_minus_1)
    tempo_change_loss = boundary_weight * torch.abs(tempo_t - tempo_t_minus_1)

    total_prior_energy = torch.mean(beat_progression_loss + bar_progression_loss + tempo_change_loss)
    return total_prior_energy

def compute_svt_loss(
    reconstructed: torch.Tensor,
    targets: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    z_t: torch.Tensor,
    z_t_minus_1: torch.Tensor,
    lambda_prior: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute total SVT loss.

    This placeholder should combine three components:
    - Reconstruction loss (MSE between `reconstructed` and `targets`)
    - KL divergence loss for latent posterior regularization
    - Prior energy loss from `compute_prior_energy_loss`

    Args:
        reconstructed: Decoder output activations.
        targets: Ground-truth activations.
        mu: Latent posterior mean.
        logvar: Latent posterior log-variance.
        z_t: Current latent phase variables.
        z_t_minus_1: Previous latent phase variables.
        lambda_prior: Weighting factor for the prior energy term.

    Returns:
        Tuple containing total loss, reconstruction loss, KL divergence, and prior energy.
    """
    # TODO: Compute and combine MSE, KL divergence, and prior energy terms.
    # 1. Standard VAE Reconstruction (MSE matching the acoustic activations)
    recon_loss = F.mse_loss(reconstructed, targets)
    
    # 2. Standard VAE KL Divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 3. ADD: Our novel Domain-Structured Prior (The Bar Pointer rules)
    prior_energy = compute_prior_energy_loss(z_t, z_t_minus_1)
    
    # The total loss forces the network to balance audio evidence with music theory
    total_loss = recon_loss + kl_loss + (lambda_prior * prior_energy)
    
    return total_loss, recon_loss, kl_loss, prior_energy
