"""Core Sequential Variational Transformer (SVT) model for the bar pointer VAE.

Implements the variational bar pointer model with three structured latent
variables per timestep:

- **Meter** m_t: Categorical(K) with Gumbel-Softmax relaxation.
- **Phase** phi_t: von Mises on [0, 2*pi) with implicit reparameterization.
- **Tempo** phidot_t: Log-Normal (Gaussian in log-space).

The architecture uses Transformers for both the acoustic encoder (bidirectional)
and the causal prior (autoregressive via causal mask).
"""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn

from models.distributions import (
    gumbel_softmax_sample,
    lognormal_sample_logspace,
    von_mises_sample,
)

TWO_PI = 2.0 * math.pi


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding supporting batch-first tensors.

    Args:
        d_model: Embedding dimension.
        max_len: Maximum supported sequence length.
    """

    def __init__(self, d_model: int, max_len: int = 20000) -> None:
        super().__init__()

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        pe = cast(Tensor, self.pe)
        return x + pe[:, :seq_len, :]


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class SVTModel(nn.Module):
    """Sequential Variational Transformer for the bar pointer VAE.

    Encodes acoustic features with a bidirectional Transformer, models latent
    dynamics with a causal Transformer, and produces posterior / prior
    parameters for each of the three structured latent variables.

    Args:
        hidden_dim: Transformer / MLP hidden dimensionality.
        nhead: Number of attention heads.
        num_layers: Number of Transformer layers per encoder.
        num_meter_classes: Number of discrete meter categories (K).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        num_meter_classes: int = 8,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_meter_classes = num_meter_classes
        input_dim = 2  # beat activation + downbeat activation

        # Prior input: [phase (1), log_tempo (1), meter_onehot (K)]
        prior_input_dim = 2 + num_meter_classes

        # --- Acoustic encoder (bidirectional) ---
        self.acoustic_input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder_audio = PositionalEncoding(d_model=hidden_dim)
        self.acoustic_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=nhead, batch_first=True,
            ),
            num_layers=num_layers,
        )

        # --- Causal prior encoder (autoregressive) ---
        self.prior_input_proj = nn.Linear(prior_input_dim, hidden_dim)
        self.pos_encoder_prior = PositionalEncoding(d_model=hidden_dim)
        self.causal_prior = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=nhead, batch_first=True,
            ),
            num_layers=num_layers,
        )

        # --- Posterior heads (from fused h_audio + h_prior) ---
        self.posterior_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        # Meter posterior
        self.posterior_meter = nn.Linear(hidden_dim, num_meter_classes)
        # Phase posterior (mu and log_kappa)
        self.posterior_phase_mu = nn.Linear(hidden_dim, 1)
        self.posterior_phase_log_kappa = nn.Linear(hidden_dim, 1)
        # Tempo posterior (mu and log_sigma in log-space)
        self.posterior_tempo_mu = nn.Linear(hidden_dim, 1)
        self.posterior_tempo_log_sigma = nn.Linear(hidden_dim, 1)

        # --- Prior heads (from h_prior alone) ---
        self.prior_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Meter prior: outputs K*K transition matrix logits
        self.prior_meter_transition = nn.Linear(hidden_dim, num_meter_classes * num_meter_classes)
        # Phase prior: context-dependent concentration
        self.prior_phase_log_kappa = nn.Linear(hidden_dim, 1)
        # Tempo prior: context-dependent std in log-space
        self.prior_tempo_log_sigma = nn.Linear(hidden_dim, 1)

        # --- Emission decoder ---
        # Input: [phase (1), log_tempo (1), meter_soft (K), h_audio (hidden_dim)]
        decoder_input_dim = 2 + num_meter_classes + hidden_dim
        self.emission_decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def encode_acoustics(self, activations: Tensor) -> Tensor:
        """Encode acoustic features (parallel, bidirectional).

        Args:
            activations: ``[B, T, 2]`` acoustic activations.

        Returns:
            ``[B, T, hidden_dim]`` acoustic hidden states.
        """
        x = self.acoustic_input_proj(activations)
        x = self.pos_encoder_audio(x)
        return self.acoustic_encoder(x)

    def encode_prior(self, z_prev_cat: Tensor) -> Tensor:
        """Encode latent history with causal masking.

        Args:
            z_prev_cat: ``[B, T, 2+K]`` concatenated prior input
                (phase, log_tempo, meter_onehot).

        Returns:
            ``[B, T, hidden_dim]`` causal prior hidden states.
        """
        x = self.prior_input_proj(z_prev_cat)
        x = self.pos_encoder_prior(x)
        seq_len = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(
            x.device
        )
        return self.causal_prior(x, mask=causal_mask)

    # ------------------------------------------------------------------
    # Prior parameter computation
    # ------------------------------------------------------------------

    def compute_prior_params(
        self,
        h_prior: Tensor,
        phase_prev: Tensor,
        log_tempo_prev: Tensor,
        meter_onehot_prev: Tensor,
    ) -> dict[str, Tensor]:
        """Compute prior distribution parameters.

        Args:
            h_prior: ``[B, T, hidden_dim]`` or ``[B, 1, hidden_dim]``.
            phase_prev: ``[B, T, 1]`` or ``[B, 1, 1]`` previous phase (radians).
            log_tempo_prev: ``[B, T, 1]`` or ``[B, 1, 1]`` previous log-tempo.
            meter_onehot_prev: ``[B, T, K]`` or ``[B, 1, K]`` previous meter one-hot.

        Returns:
            Dict with keys: ``meter_logits``, ``phase_mu``, ``phase_kappa``,
            ``tempo_mu``, ``tempo_sigma``.
        """
        h = self.prior_proj(h_prior)

        K = self.num_meter_classes
        B = h.size(0)
        T = h.size(1)

        # --- Meter prior ---
        # Transition matrix: [B, T, K, K]
        trans_logits = self.prior_meter_transition(h).view(B, T, K, K)
        trans_log_probs = torch.log_softmax(trans_logits, dim=-1)  # normalize rows

        # Bar boundary detection: phi_{t-1} + phidot_{t-1} >= 2*pi
        tempo_prev = torch.exp(log_tempo_prev)  # [B, T, 1]
        boundary = ((phase_prev + tempo_prev) >= TWO_PI).squeeze(-1)  # [B, T]
        boundary_mask = boundary.unsqueeze(-1).expand_as(meter_onehot_prev)  # [B, T, K]

        # Previous meter as probabilities (one-hot -> row vector)
        # Apply same smoothing for the transition path
        # meter_onehot_prev: [B, T, K] -> [B, T, 1, K]
        meter_prev_smooth = meter_onehot_prev + 1e-3
        meter_prev_smooth = meter_prev_smooth / meter_prev_smooth.sum(dim=-1, keepdim=True)
        meter_row = meter_prev_smooth.unsqueeze(-2)

        # Transition: meter_row @ transition_matrix -> [B, T, 1, K] -> [B, T, K]
        transitioned_log_probs = torch.logsumexp(
            meter_row.log() + trans_log_probs, dim=-2
        ).squeeze(-2)

        # If no boundary: strong preference for previous meter (soft, not hard delta)
        # Add a small uniform floor to avoid infinite KL against a diffuse posterior
        meter_prev_soft = meter_onehot_prev + 1e-3
        meter_prev_soft = meter_prev_soft / meter_prev_soft.sum(dim=-1, keepdim=True)
        no_boundary_log_probs = meter_prev_soft.log()
        meter_prior_logits = torch.where(
            boundary_mask,
            transitioned_log_probs,
            no_boundary_log_probs,
        )

        # --- Phase prior ---
        # mu_p = (phi_{t-1} + phidot_{t-1}) mod 2*pi
        phase_mu = torch.remainder(phase_prev + tempo_prev, TWO_PI).squeeze(-1)  # [B, T]
        phase_kappa = self.prior_phase_log_kappa(h).squeeze(-1).exp()  # [B, T]

        # --- Tempo prior ---
        # mu_p = log(phidot_{t-1}) = log_tempo_prev
        tempo_mu = log_tempo_prev.squeeze(-1)  # [B, T]
        tempo_sigma = self.prior_tempo_log_sigma(h).squeeze(-1).exp()  # [B, T]

        return {
            "meter_logits": meter_prior_logits,  # [B, T, K]
            "phase_mu": phase_mu,                # [B, T]
            "phase_kappa": phase_kappa,          # [B, T]
            "tempo_mu": tempo_mu,                # [B, T]
            "tempo_sigma": tempo_sigma,          # [B, T]
        }

    # ------------------------------------------------------------------
    # Posterior parameter computation
    # ------------------------------------------------------------------

    def compute_posterior_params(self, h_fused: Tensor) -> dict[str, Tensor]:
        """Compute posterior distribution parameters.

        Args:
            h_fused: ``[B, T, hidden_dim]`` fused hidden state.

        Returns:
            Dict with keys: ``meter_logits``, ``phase_mu``, ``phase_log_kappa``,
            ``tempo_mu``, ``tempo_log_sigma``.
        """
        h = self.posterior_proj(h_fused)

        return {
            "meter_logits": self.posterior_meter(h),                      # [B, T, K]
            "phase_mu": self.posterior_phase_mu(h).squeeze(-1),           # [B, T]
            "phase_log_kappa": self.posterior_phase_log_kappa(h).squeeze(-1),  # [B, T]
            "tempo_mu": self.posterior_tempo_mu(h).squeeze(-1),           # [B, T]
            "tempo_log_sigma": self.posterior_tempo_log_sigma(h).squeeze(-1),  # [B, T]
        }

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_latent(
        self,
        posterior: dict[str, Tensor],
        temperature: float = 1.0,
    ) -> dict[str, Tensor]:
        """Draw reparameterized samples from the posterior.

        Args:
            posterior: Dict of posterior parameters (from ``compute_posterior_params``).
            temperature: Gumbel-Softmax temperature for meter.

        Returns:
            Dict with keys: ``meter_soft`` (K-dim), ``meter_onehot`` (K-dim, hard),
            ``phase`` (scalar, radians), ``log_tempo`` (scalar, log-space).
        """
        # Meter: Gumbel-Softmax
        meter_soft = gumbel_softmax_sample(
            posterior["meter_logits"], temperature=temperature, hard=False,
        )
        meter_hard = gumbel_softmax_sample(
            posterior["meter_logits"], temperature=temperature, hard=True,
        )

        # Phase: von Mises
        kappa_q = posterior["phase_log_kappa"].exp()
        phase = von_mises_sample(posterior["phase_mu"], kappa_q)
        phase = torch.remainder(phase, TWO_PI)  # wrap to [0, 2*pi)

        # Tempo: Log-Normal (sample in log-space)
        sigma_q = posterior["tempo_log_sigma"].exp()
        log_tempo = lognormal_sample_logspace(posterior["tempo_mu"], sigma_q)

        return {
            "meter_soft": meter_soft,    # [B, T, K]
            "meter_onehot": meter_hard,  # [B, T, K] (straight-through)
            "phase": phase,              # [B, T]
            "log_tempo": log_tempo,      # [B, T]
        }

    # ------------------------------------------------------------------
    # Emission decoder
    # ------------------------------------------------------------------

    def decode(
        self,
        samples: dict[str, Tensor],
        h_audio: Tensor,
    ) -> Tensor:
        """Decode beat logits from latent samples and acoustic features.

        Args:
            samples: Dict from ``sample_latent``.
            h_audio: ``[B, T, hidden_dim]`` acoustic encoder output.

        Returns:
            ``[B, T, 1]`` beat logits (pre-sigmoid).
        """
        decoder_input = torch.cat(
            [
                samples["phase"].unsqueeze(-1),      # [B, T, 1]
                samples["log_tempo"].unsqueeze(-1),   # [B, T, 1]
                samples["meter_soft"],                # [B, T, K]
                h_audio,                              # [B, T, hidden_dim]
            ],
            dim=-1,
        )
        return self.emission_decoder(decoder_input)  # [B, T, 1]

    # ------------------------------------------------------------------
    # Full forward pass (teacher-forced / parallel)
    # ------------------------------------------------------------------

    def forward(
        self,
        activations: Tensor,
        z_prev: dict[str, Tensor],
        temperature: float = 1.0,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        """Run one forward pass (parallel, teacher-forced).

        This processes all timesteps simultaneously using causal masking in the
        prior.  For autoregressive sampling (Algorithm 1), use
        ``encode_acoustics`` + ``step`` in a loop instead.

        Args:
            activations: ``[B, T, 2]`` acoustic activations.
            z_prev: Dict with keys:
                - ``phase``: ``[B, T, 1]`` previous phase (radians).
                - ``log_tempo``: ``[B, T, 1]`` previous log-tempo.
                - ``meter_onehot``: ``[B, T, K]`` previous meter one-hot.
            temperature: Gumbel-Softmax temperature.

        Returns:
            Dict with keys: ``beat_logits``, ``posterior``, ``prior``, ``samples``.
        """
        # Acoustic encoding (parallel, bidirectional)
        h_audio = self.encode_acoustics(activations)  # [B, T, D]

        # Prior encoding (causal)
        z_prev_cat = torch.cat(
            [z_prev["phase"], z_prev["log_tempo"], z_prev["meter_onehot"]],
            dim=-1,
        )  # [B, T, 2+K]
        h_prior = self.encode_prior(z_prev_cat)  # [B, T, D]

        # Posterior parameters
        h_fused = torch.cat([h_audio, h_prior], dim=-1)  # [B, T, 2D]
        posterior = self.compute_posterior_params(h_fused)

        # Prior parameters
        prior = self.compute_prior_params(
            h_prior,
            phase_prev=z_prev["phase"],
            log_tempo_prev=z_prev["log_tempo"],
            meter_onehot_prev=z_prev["meter_onehot"],
        )

        # Sample
        samples = self.sample_latent(posterior, temperature=temperature)

        # Decode
        beat_logits = self.decode(samples, h_audio)

        return {
            "beat_logits": beat_logits,  # [B, T, 1]
            "posterior": posterior,
            "prior": prior,
            "samples": samples,
        }

    # ------------------------------------------------------------------
    # Step-wise forward (for autoregressive training / inference)
    # ------------------------------------------------------------------

    def step(
        self,
        h_audio_t: Tensor,
        h_prior_t: Tensor,
        phase_prev: Tensor,
        log_tempo_prev: Tensor,
        meter_onehot_prev: Tensor,
        temperature: float = 1.0,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        """Single-step forward for autoregressive rollout.

        Args:
            h_audio_t: ``[B, 1, hidden_dim]`` acoustic features at time t.
            h_prior_t: ``[B, 1, hidden_dim]`` prior hidden state at time t.
            phase_prev: ``[B, 1, 1]`` previous phase.
            log_tempo_prev: ``[B, 1, 1]`` previous log-tempo.
            meter_onehot_prev: ``[B, 1, K]`` previous meter one-hot.
            temperature: Gumbel-Softmax temperature.

        Returns:
            Same structure as ``forward`` but for a single timestep.
        """
        h_fused = torch.cat([h_audio_t, h_prior_t], dim=-1)  # [B, 1, 2D]
        posterior = self.compute_posterior_params(h_fused)
        prior = self.compute_prior_params(
            h_prior_t, phase_prev, log_tempo_prev, meter_onehot_prev,
        )
        samples = self.sample_latent(posterior, temperature=temperature)
        beat_logits = self.decode(samples, h_audio_t)

        return {
            "beat_logits": beat_logits,
            "posterior": posterior,
            "prior": prior,
            "samples": samples,
        }
