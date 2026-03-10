"""Inference utilities for CHART."""

from __future__ import annotations

import torch

from models.svt_core import SVTModel


def run_inference(model: SVTModel, acoustic_activations: torch.Tensor) -> torch.Tensor:
    """Run autoregressive inference over continuous phase variables.

    This placeholder represents frame-by-frame autoregressive phase sampling
    using the acoustic encoder and causal prior to predict continuous latent
    rhythm states.

    Args:
        model: Trained SVT model.
        acoustic_activations: Input acoustic activation sequence.

    Returns:
        Predicted continuous phase trajectory.
    """
    # TODO: Add autoregressive rollout with causal prior and latent sampling.
    raise NotImplementedError("Implement inference routine.")
