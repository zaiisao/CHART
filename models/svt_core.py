"""Core Sequential Variational Transformer (SVT) model for the bar pointer VAE.

Architecture per the pseudocode documents:

Prior (transition_model_pseudocode.pdf):
- Bidirectional Encoder Transformer on x_{1:T} → h_prior_{1:T}
- FFN heads with Softplus for kappa/sigma, delta*Sigmoid for meter epsilon
- All prior params computed in parallel from h_prior
- Means are fixed (bar pointer dynamics), uncertainties are learned

Posterior (posterior_model_final.pdf):
- Encoder-Decoder Transformer: encoder on x_{1:T}, decoder on b_{1:T}
- Cross-attention: Q=b (beat targets), K=V=x (audio features)
- FFN heads with Softplus for kappa/sigma, pi*tanh for phase mu
- All posterior params computed in parallel (no z_{t-1} dependency)
- Does NOT condition on z_{t-1} — purely from (x, b)

Training (Algorithm 1):
- Prior means depend on z_{t-1} (sequential), but prior uncertainties
  and all posterior params are pre-computed in parallel.
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.distributions import (
    gumbel_softmax_sample,
    lognormal_sample_logspace,
    von_mises_sample,
)

TWO_PI = 2.0 * math.pi


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding supporting batch-first tensors."""

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


class SVTModel(nn.Module):
    """Sequential Variational Transformer for the bar pointer VAE.

    Args:
        hidden_dim: Transformer / MLP hidden dimensionality.
        nhead: Number of attention heads.
        num_layers: Number of Transformer layers per encoder.
        num_meter_classes: Number of discrete meter categories (K).
        meter_delta: Upper bound for meter transition epsilon_ij.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        num_meter_classes: int = 8,
        meter_delta: float = 0.001,
        h_prior_bottleneck: int = 0,
        **kwargs,  # absorb extra args like z_context for backward compat
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_meter_classes = num_meter_classes
        self.meter_delta = meter_delta
        K = num_meter_classes
        input_dim = 2  # beat activation + downbeat activation

        # ================================================================
        # PRIOR: Encoder Transformer on x_{1:T} → h_prior_{1:T}
        # (transition_model_pseudocode.pdf, Step 0)
        # ================================================================
        self.prior_input_proj = nn.Linear(input_dim, hidden_dim)
        self.prior_pos_enc = PositionalEncoding(d_model=hidden_dim)
        self.prior_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=nhead, batch_first=True,
            ),
            num_layers=num_layers,
        )

        # Prior FFN heads (applied position-wise to h_prior)
        # Meter: epsilon_ij matrix — delta * sigmoid(FFN) → (0, delta)
        self.prior_meter_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, K * K),
        )
        # Phase: concentration kappa — Softplus(FFN) > 0
        self.prior_phase_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Tempo: std sigma — Softplus(FFN) > 0
        self.prior_tempo_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Learnable initial tempo mean (for t=1)
        self.prior_tempo_mu_init = nn.Parameter(torch.tensor(0.0))

        # ================================================================
        # POSTERIOR: Encoder-Decoder Transformer
        # (posterior_model_final.pdf)
        # Encoder: x_{1:T} → h_x (K, V for cross-attention)
        # Decoder: b_{1:T} → h_post (Q drives cross-attention into h_x)
        # ================================================================
        # Posterior encoder (processes audio features x_{1:T})
        self.post_enc_proj = nn.Linear(input_dim, hidden_dim)
        self.post_enc_pos = PositionalEncoding(d_model=hidden_dim)
        self.post_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=nhead, batch_first=True,
            ),
            num_layers=num_layers,
        )

        # Posterior decoder (processes beat targets b_{1:T}, cross-attends to x)
        # b_{1:T} is 1-dim (binary beat indicator per frame)
        self.post_dec_proj = nn.Linear(1, hidden_dim)
        self.post_dec_pos = PositionalEncoding(d_model=hidden_dim)
        self.post_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim, nhead=nhead, batch_first=True,
            ),
            num_layers=num_layers,
        )

        # Posterior FFN heads (applied position-wise to h_post)
        # Meter: softmax(FFN)
        self.post_meter_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, K),
        )
        # Phase mu: pi * tanh(FFN) → [-pi, pi]
        self.post_phase_mu_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Phase kappa: Softplus(FFN) > 0
        self.post_phase_kappa_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Tempo mu: FFN (unconstrained, log-space)
        self.post_tempo_mu_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Tempo sigma: Softplus(FFN) > 0
        self.post_tempo_sigma_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # ================================================================
        # DECODER (emission model)
        # p_θ(b_t | z_t, h_{1:T}) = Bern(σ(NN_θ(z_t, h_{1:T})))
        # ================================================================
        self.h_prior_bottleneck_dim = h_prior_bottleneck
        if h_prior_bottleneck > 0:
            self.h_prior_bottleneck_proj = nn.Linear(hidden_dim, h_prior_bottleneck)
            h_dim_for_decoder = h_prior_bottleneck
        else:
            h_dim_for_decoder = hidden_dim

        decoder_input_dim = 2 + K + h_dim_for_decoder
        self.emission_decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    # Prior: encode and compute uncertainty params (parallel)
    # ------------------------------------------------------------------

    def encode_prior(self, activations: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Encode audio and compute all prior uncertainty params in parallel.

        Per transition_model_pseudocode.pdf Step 0.

        Args:
            activations: [B, T, 2] acoustic activations.

        Returns:
            h_prior: [B, T, D] prior hidden states.
            prior_params: Dict with pre-computed uncertainty params:
                - epsilon_ij: [B, T, K, K] meter transition probs (0, delta)
                - phase_kappa: [B, T] concentration
                - tempo_sigma: [B, T] std in log-space
        """
        x = self.prior_input_proj(activations)
        x = self.prior_pos_enc(x)
        h_prior = self.prior_encoder(x)  # [B, T, D]

        K = self.num_meter_classes
        B, T, _ = h_prior.shape

        # Meter: delta * sigmoid(FFN) → epsilon_ij in (0, delta)
        epsilon_ij = self.meter_delta * torch.sigmoid(
            self.prior_meter_ffn(h_prior).view(B, T, K, K)
        )

        # Phase: kappa in [1, 200] — concentrated prior around predicted phase
        phase_kappa = 1.0 + 199.0 * torch.sigmoid(
            self.prior_phase_ffn(h_prior).squeeze(-1)
        )

        # Tempo: sigma in [0.01, 0.5] — bounded random walk variance
        tempo_sigma = 0.01 + 0.49 * torch.sigmoid(
            self.prior_tempo_ffn(h_prior).squeeze(-1)
        )

        return h_prior, {
            "epsilon_ij": epsilon_ij,
            "phase_kappa": phase_kappa,
            "tempo_sigma": tempo_sigma,
        }

    def compute_prior_at_t(
        self,
        prior_params: dict[str, Tensor],
        t: int,
        phase_prev: Tensor,
        log_tempo_prev: Tensor,
        meter_onehot_prev: Tensor,
    ) -> dict[str, Tensor]:
        """Compute prior distribution params at time t using z_{t-1}.

        Per transition_model_pseudocode.pdf Steps 1-2.

        Args:
            prior_params: Pre-computed uncertainty params from encode_prior.
            t: Current timestep (0-indexed).
            phase_prev: [B, 1] previous phase.
            log_tempo_prev: [B, 1] previous log-tempo.
            meter_onehot_prev: [B, K] previous meter one-hot.
        """
        K = self.num_meter_classes

        if t == 0:
            # Step 1: Initial state
            B = phase_prev.size(0)
            device = phase_prev.device
            meter_logits = torch.zeros(B, K, device=device)  # uniform
            phase_mu = torch.zeros(B, device=device)
            phase_kappa = prior_params["phase_kappa"][:, 0]
            tempo_mu = self.prior_tempo_mu_init.expand(B)
            tempo_sigma = prior_params["tempo_sigma"][:, 0]
        else:
            # Step 2: Transition
            # Phase: mu = (phi_{t-1} + phidot_{t-1}) mod 2pi
            tempo_prev = torch.exp(log_tempo_prev.clamp(max=10.0))
            phase_mu = torch.remainder(
                phase_prev.squeeze(-1) + tempo_prev.squeeze(-1), TWO_PI
            )

            # Phase kappa from pre-computed
            phase_kappa = prior_params["phase_kappa"][:, t]

            # Tempo: mu = log(phidot_{t-1}) = log_tempo_prev
            tempo_mu = log_tempo_prev.squeeze(-1)
            tempo_sigma = prior_params["tempo_sigma"][:, t]

            # Meter: boundary-gated transition
            wrap_t = (phase_prev.squeeze(-1) + tempo_prev.squeeze(-1)) >= TWO_PI  # [B]
            eps_t = prior_params["epsilon_ij"][:, t]  # [B, K, K]

            # Build transition: pi[i,j] = eps[i,j] * wrap, pi[i,i] = 1 - sum(eps[i,:]) * wrap
            # For soft meter_onehot_prev, compute expected transition
            diag_mask = torch.eye(K, device=eps_t.device).unsqueeze(0)  # [1, K, K]
            off_diag_eps = eps_t * (1.0 - diag_mask)  # zero diagonal
            row_sum = off_diag_eps.sum(dim=-1, keepdim=True)  # [B, K, 1]
            stay_prob = 1.0 - row_sum  # [B, K, 1]
            trans_matrix = off_diag_eps + stay_prob * diag_mask  # [B, K, K]

            # Gate by wrap: if no wrap, stay in same meter
            wrap_mask = wrap_t.unsqueeze(-1).unsqueeze(-1).float()  # [B, 1, 1]
            trans_matrix = wrap_mask * trans_matrix + (1.0 - wrap_mask) * diag_mask.expand_as(trans_matrix)

            # Apply: pi_t = meter_prev @ trans_matrix
            meter_prev_prob = meter_onehot_prev + 1e-6
            meter_prev_prob = meter_prev_prob / meter_prev_prob.sum(dim=-1, keepdim=True)
            meter_logits = torch.log(
                torch.bmm(meter_prev_prob.unsqueeze(1), trans_matrix).squeeze(1) + 1e-10
            )

        return {
            "meter_logits": meter_logits,       # [B, K]
            "phase_mu": phase_mu,               # [B]
            "phase_kappa": phase_kappa,          # [B]
            "tempo_mu": tempo_mu,               # [B]
            "tempo_sigma": tempo_sigma,          # [B]
        }

    # ------------------------------------------------------------------
    # Posterior: encode-decode and compute all params (parallel)
    # ------------------------------------------------------------------

    def encode_posterior(
        self,
        activations: Tensor,
        beat_targets: Tensor,
    ) -> dict[str, Tensor]:
        """Compute all posterior params in parallel via encoder-decoder.

        Per posterior_model_final.pdf.

        Args:
            activations: [B, T, 2] acoustic activations (x_{1:T}).
            beat_targets: [B, T] binary beat indicators (b_{1:T}).

        Returns:
            Dict with posterior params for all T timesteps:
                - meter_logits: [B, T, K]
                - phase_mu: [B, T]
                - phase_kappa: [B, T]  (not log_kappa — direct Softplus output)
                - tempo_mu: [B, T]
                - tempo_sigma: [B, T]  (not log_sigma — direct Softplus output)
        """
        # Encoder: x_{1:T} → h_x
        x_emb = self.post_enc_proj(activations)
        x_emb = self.post_enc_pos(x_emb)
        h_x = self.post_encoder(x_emb)  # [B, T, D]

        # Decoder: b_{1:T} drives Q, h_x provides K,V
        b_emb = self.post_dec_proj(beat_targets.unsqueeze(-1))  # [B, T, 1] → [B, T, D]
        b_emb = self.post_dec_pos(b_emb)
        h_post = self.post_decoder(b_emb, h_x)  # [B, T, D]

        # FFN heads (position-wise, shared weights)
        K = self.num_meter_classes
        return {
            "meter_logits": self.post_meter_ffn(h_post),                              # [B, T, K]
            "phase_mu": math.pi * torch.tanh(self.post_phase_mu_ffn(h_post).squeeze(-1)),  # [B, T]
            "phase_kappa": 0.1 + 299.9 * torch.sigmoid(                                 # [B, T] in [0.1, 300]
                self.post_phase_kappa_ffn(h_post).squeeze(-1)),
            "tempo_mu": self.post_tempo_mu_ffn(h_post).squeeze(-1),                   # [B, T]
            "tempo_sigma": 0.01 + 1.99 * torch.sigmoid(                               # [B, T] in [0.01, 2.0]
                self.post_tempo_sigma_ffn(h_post).squeeze(-1)),
        }

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_latent_at_t(
        self,
        posterior: dict[str, Tensor],
        t: int,
        temperature: float = 1.0,
    ) -> dict[str, Tensor]:
        """Sample z_t from posterior at timestep t.

        Args:
            posterior: Full posterior params [B, T, ...] from encode_posterior.
            t: Timestep index.
            temperature: Gumbel-Softmax temperature.

        Returns:
            Dict with samples at time t (each [B] or [B, K]).
        """
        meter_logits_t = posterior["meter_logits"][:, t, :]  # [B, K]
        meter_soft = gumbel_softmax_sample(meter_logits_t, temperature=temperature, hard=False)
        meter_hard = gumbel_softmax_sample(meter_logits_t, temperature=temperature, hard=True)

        phase = von_mises_sample(posterior["phase_mu"][:, t], posterior["phase_kappa"][:, t])
        phase = torch.remainder(phase, TWO_PI)

        log_tempo = lognormal_sample_logspace(
            posterior["tempo_mu"][:, t], posterior["tempo_sigma"][:, t],
        )

        return {
            "meter_soft": meter_soft,      # [B, K]
            "meter_onehot": meter_hard,    # [B, K]
            "phase": phase,                # [B]
            "log_tempo": log_tempo,        # [B]
        }

    # ------------------------------------------------------------------
    # Emission decoder
    # ------------------------------------------------------------------

    def decode_at_t(
        self,
        samples: dict[str, Tensor],
        h_prior_t: Tensor,
    ) -> Tensor:
        """Decode beat logit at time t."""
        h = h_prior_t
        if self.h_prior_bottleneck_dim > 0:
            h = self.h_prior_bottleneck_proj(h)
        decoder_input = torch.cat([
            samples["phase"].unsqueeze(-1),       # [B, 1]
            samples["log_tempo"].unsqueeze(-1),    # [B, 1]
            samples["meter_soft"],                 # [B, K]
            h,                                     # [B, D'] or [B, D]
        ], dim=-1)
        return self.emission_decoder(decoder_input)  # [B, 1]

    # ------------------------------------------------------------------
    # Forward: Algorithm 1 sequential loop
    # ------------------------------------------------------------------

    def forward(
        self,
        activations: Tensor,
        z_prev_init: dict[str, Tensor],
        temperature: float = 1.0,
        beat_targets: Tensor | None = None,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        """Algorithm 1: sequential autoregressive forward.

        Prior uncertainties and all posterior params are pre-computed in
        parallel. The sequential loop only evaluates prior means (which
        depend on z_{t-1}) and samples from the posterior.

        Args:
            activations: [B, T, 2] acoustic activations.
            z_prev_init: Dict with initial z_prev (t=0 seed):
                - phase: [B, 1, 1]
                - log_tempo: [B, 1, 1]
                - meter_onehot: [B, 1, K]
            temperature: Gumbel-Softmax temperature.
            beat_targets: [B, T] binary beat indicators (required for
                posterior encoder-decoder). If None, uses zeros.

        Returns:
            Dict with beat_logits, posterior, prior, samples.
        """
        B, T, _ = activations.shape
        device = activations.device

        # Pre-compute beat targets if not provided
        if beat_targets is None:
            beat_targets = torch.zeros(B, T, device=device)

        # Step 0: Prior encoder (parallel, one pass)
        h_prior, prior_params = self.encode_prior(activations)

        # Step 0: Posterior encoder-decoder (parallel, one pass)
        posterior = self.encode_posterior(activations, beat_targets)

        # Reformat posterior for KL compatibility:
        # The loss expects "phase_log_kappa" and "tempo_log_sigma",
        # but we use Softplus (direct kappa/sigma). Store both forms.
        posterior_for_loss = {
            "meter_logits": posterior["meter_logits"],
            "phase_mu": posterior["phase_mu"],
            "phase_log_kappa": torch.log(posterior["phase_kappa"] + 1e-8),
            "tempo_mu": posterior["tempo_mu"],
            "tempo_log_sigma": torch.log(posterior["tempo_sigma"] + 1e-8),
        }

        # Current z_prev state
        phase_prev = z_prev_init["phase"].squeeze(1)          # [B, 1]
        log_tempo_prev = z_prev_init["log_tempo"].squeeze(1)  # [B, 1]
        meter_prev = z_prev_init["meter_onehot"].squeeze(1)   # [B, K]

        # Output accumulators
        all_beat_logits = []
        all_prior_meter = []
        all_prior_phase_mu = []
        all_prior_phase_kappa = []
        all_prior_tempo_mu = []
        all_prior_tempo_sigma = []
        all_samp = {k: [] for k in ("meter_soft", "meter_onehot", "phase", "log_tempo")}

        for t in range(T):
            # Prior at time t (means depend on z_{t-1}, uncertainties pre-computed)
            prior_t = self.compute_prior_at_t(
                prior_params, t, phase_prev, log_tempo_prev, meter_prev,
            )

            # Sample from posterior at time t
            samp_t = self.sample_latent_at_t(posterior, t, temperature=temperature)

            # Decode
            beat_logit_t = self.decode_at_t(samp_t, h_prior[:, t, :])

            # Accumulate
            all_beat_logits.append(beat_logit_t)
            all_prior_meter.append(prior_t["meter_logits"])
            all_prior_phase_mu.append(prior_t["phase_mu"])
            all_prior_phase_kappa.append(prior_t["phase_kappa"])
            all_prior_tempo_mu.append(prior_t["tempo_mu"])
            all_prior_tempo_sigma.append(prior_t["tempo_sigma"])
            for k in all_samp:
                all_samp[k].append(samp_t[k])

            # Update z_prev for next step (Algorithm 1)
            phase_prev = samp_t["phase"].unsqueeze(-1)       # [B, 1]
            log_tempo_prev = samp_t["log_tempo"].unsqueeze(-1)  # [B, 1]
            meter_prev = samp_t["meter_onehot"]              # [B, K]

        # Stack across time
        prior_out = {
            "meter_logits": torch.stack(all_prior_meter, dim=1),       # [B, T, K]
            "phase_mu": torch.stack(all_prior_phase_mu, dim=1),        # [B, T]
            "phase_kappa": torch.stack(all_prior_phase_kappa, dim=1),  # [B, T]
            "tempo_mu": torch.stack(all_prior_tempo_mu, dim=1),        # [B, T]
            "tempo_sigma": torch.stack(all_prior_tempo_sigma, dim=1),  # [B, T]
        }
        samples_out = {k: torch.stack(v, dim=1) for k, v in all_samp.items()}

        return {
            "beat_logits": torch.stack(all_beat_logits, dim=1),  # [B, T, 1]
            "posterior": posterior_for_loss,
            "prior": prior_out,
            "samples": samples_out,
        }
