"""Core Sequential Variational Transformer (SVT) model definitions."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding supporting batch-first tensors.

    Args:
        d_model: Embedding dimension.
        max_len: Maximum supported sequence length.
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: Input tensor with shape `(batch, seq_len, d_model)`.

        Returns:
            Tensor with positional encoding added, same shape as input.
        """
        seq_len = x.size(1)
        pe = cast(torch.Tensor, self.pe)
        return x + pe[:, :seq_len, :]


class SVTModel(nn.Module):
    """Sequential Variational Transformer for continuous rhythm phase tracking.

    This class is a boilerplate scaffold for replacing a classical DBN with an
    SVT-style latent-variable model. The architecture contains placeholders for:

    - `acoustic_encoder`: Standard Transformer encoder for acoustic activations.
    - `causal_prior`: Transformer encoder with causal masking for autoregressive
      latent dynamics.
    - `fuser_mu`: Linear projection head for latent mean parameters.
    - `fuser_logvar`: Linear projection head for latent log-variance parameters.
    - `emission_decoder`: Small MLP to reconstruct activations from latent state.

    Notes:
        Internal logic is intentionally omitted for custom implementation.
    """

    def __init__(self, hidden_dim: int = 128, nhead: int = 4, num_layers: int = 2) -> None:
        """Initialize SVT model placeholder modules.

        Args:
            hidden_dim: Transformer/MLP hidden dimensionality.
            nhead: Number of attention heads for transformer layers.
            num_layers: Number of transformer layers for encoder/prior blocks.
        """
        super().__init__()

        input_dim = 2
        latent_dim = 3

        self.pos_encoder_audio = PositionalEncoding(d_model=hidden_dim)
        self.pos_encoder_prior = PositionalEncoding(d_model=hidden_dim)

        self.acoustic_input_proj = nn.Linear(input_dim, hidden_dim)

        self.acoustic_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True),
            num_layers=num_layers,
        )

        self.prior_input_proj = nn.Linear(latent_dim, hidden_dim)

        self.causal_prior = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True),
            num_layers=num_layers,
        )

        self.fuser_bottleneck = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fuser_activation = nn.ReLU()

        self.fuser_mu = nn.Linear(hidden_dim, latent_dim)

        self.fuser_logvar = nn.Linear(hidden_dim, latent_dim)

        self.emission_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Apply the standard VAE reparameterization trick.

        Args:
            mu: Mean tensor of the latent Gaussian.
            logvar: Log-variance tensor of the latent Gaussian.

        Returns:
            Reparameterized latent sample.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, activations: torch.Tensor, z_history: torch.Tensor):
        """Run one forward pass of the SVT model.

        Args:
            activations: Acoustic feature tensor.
            z_history: Previously sampled continuous phase latent variables.

        Returns:
            A tuple `(reconstructed, mu, logvar, z_t)` where:
            - reconstructed: Reconstructed acoustic activations.
            - mu: Latent posterior mean parameters.
            - logvar: Latent posterior log-variance parameters.
            - z_t: Newly sampled latent variables for current step(s).
        """
        # Module A: acoustic encoding path.
        x_audio = self.acoustic_input_proj(activations)
        x_audio = self.pos_encoder_audio(x_audio)
        h_audio = self.acoustic_encoder(x_audio)

        # Module B: causal prior path over latent history.
        x_prior = self.prior_input_proj(z_history)
        x_prior = self.pos_encoder_prior(x_prior)
        seq_len = x_prior.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x_prior.device)
        h_prior = self.causal_prior(x_prior, mask=causal_mask)

        # Module C: bottleneck fusion and posterior parameter heads.
        h_fused = torch.cat([h_audio, h_prior], dim=-1)
        h_bottleneck = self.fuser_activation(self.fuser_bottleneck(h_fused))
        mu = self.fuser_mu(h_bottleneck)
        logvar = self.fuser_logvar(h_bottleneck)

        # Reparameterization for latent sampling.
        z_t = self.reparameterize(mu, logvar)

        # Module D: emission decoding to reconstruct acoustic features.
        reconstructed = self.emission_decoder(z_t)

        return reconstructed, mu, logvar, z_t
