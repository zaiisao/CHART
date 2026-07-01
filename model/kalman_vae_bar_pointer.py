"""KVAE bar-pointer prototype: an EXACT-filter posterior in place of the amortized GRU encoder.

This is a from-scratch reproduction (against the CURRENT VBPM codebase's data/config conventions) of a
prior experiment ("M1") that broke the amortized-encoder wall described in bar_pointer_vae.py's module
docstring: a plain feed-forward GRU posterior, trained with per-frame KL gradients, drags the bar-phase
latent toward the prior and it never learns to rotate (downbeat F stuck at 0.000, see the historical
baselines in this project's notes). Kalman-VAE (Fraccaro et al. 2017, "A Disentangled Recognition and
Nonlinear Dynamics Model for Unsupervised Learning") replaces the amortized q(z|h) with the EXACT
posterior of a (locally) linear-Gaussian state-space model, computed by a Kalman filter/RTS-smoother --
gradient descent cannot shortcut an exact conditional the way it can an amortized approximation.

Architecture (mirrors third_party/kalman-vae's KalmanVariationalAutoencoder, swapping its conv image
VAE for an MLP over our [512]-dim frozen Beat-This frontend features):

    h_t [feature_dim]  --MLP encoder-->  a_t ~ N(mu_a, sigma_a)     (pseudo-observation, the "what")
    a_{1:T}             --StateSpaceModel (reused verbatim)-->  z    (locally-linear-Gaussian latent,
                                                                       filtered/smoothed EXACTLY, the "where")
    a_t                 --MLP decoder-->  reconstruct h_t            (KVAE reconstruction likelihood)
    z_t                 --MLP head-->     Bernoulli(beat, downbeat)  (read out of the FILTERED latent)

Deployment (evaluate_kvae.py): encoder(h) -> a (mean, no sampling) -> Kalman FILTER (causal, "sample_control
mean" so it's deterministic) -> head(filtered z) -> peak-pick beat/downbeat times. This is a LEARNED-HEAD
read-out (not the phase-geometric wrap read-out in model/readout.py's phase_to_*_times): the historical
run found the geometric wrap read-out on a freely-filtered rotational z structurally fails (the filter's
observation update corrects z back toward a static estimate rather than letting it accumulate a monotonic
phase -- z oscillates, it does not rotate). The learned-head read-out is what reached the historical
0.878/0.833 numbers, so this prototype preserves that design rather than the geometric wrap.

This IS a legitimate inference method, not a supervised bypass: the Kalman filter computes the exact
posterior for a linear-Gaussian SSM given the (pseudo-)observations -- the ELBO below is the standard
KVAE ELBO (reconstruction + regularization + Kalman observation/transition/entropy terms), no ground-truth
phase supervision or computed (non-inferred) tempo is used anywhere.
"""
from __future__ import annotations

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

_THIRD_PARTY_KVAE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "third_party", "kalman-vae")
if _THIRD_PARTY_KVAE not in sys.path:
    sys.path.insert(0, _THIRD_PARTY_KVAE)

from kvae.state_space_model import StateSpaceModel     # noqa: E402  (reused verbatim, see module docstring)
from kvae.sample_control import SampleControl           # noqa: E402


class ObservationEncoder(nn.Module):
    """h_t [feature_dim] -> a_t ~ N(mu, sigma): the KVAE pseudo-observation posterior q(a|h)."""

    def __init__(self, feature_dim: int, a_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.mean_head = nn.Linear(hidden, a_dim)
        self.std_head = nn.Linear(hidden, a_dim)

    def forward(self, features: torch.Tensor) -> D.Normal:
        hidden = self.net(features)
        return D.Normal(self.mean_head(hidden), F.softplus(self.std_head(hidden)) + 1e-4)


class ObservationDecoder(nn.Module):
    """a_t [a_dim] -> reconstruct h_t [feature_dim]: Gaussian likelihood, fixed unit scale."""

    def __init__(self, a_dim: int, feature_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(a_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, feature_dim))

    def forward(self, a: torch.Tensor) -> D.Normal:
        mean = self.net(a)
        return D.Normal(mean, torch.ones_like(mean))


class KalmanVAEBarPointer(nn.Module):
    """Encoder -> pseudo-observation -> exact Kalman filter/smoother -> beat/downbeat head."""

    def __init__(self, feature_dim: int = 512, a_dim: int = 8, z_dim: int = 8, K: int = 5, hidden: int = 256,
                Q_reg: float = 1e-3):
        super().__init__()
        self.encoder = ObservationEncoder(feature_dim, a_dim, hidden)
        self.decoder = ObservationDecoder(a_dim, feature_dim, hidden)
        self.ssm = StateSpaceModel(
            a_dim=a_dim, z_dim=z_dim, K=K,
            dynamics_parameter_network="lstm", hidden_dim=64, num_layers=1,
            learn_noise_covariance=True, init_noise_scale=1.0,
            # Q_reg is an ADDITIVE floor on the learned process-noise covariance mat_Q = L L^T + Q_reg*I
            # (see StateSpaceModel.mat_Q) -- raising it above the vendored default (1e-3) loosens how
            # tightly the filter's dynamics prior can pin z_t to A@z_{t-1}, i.e. more tempo/phase rubato
            # tolerance. Exposed here (not hardcoded) for the Q_reg generalization sweep.
            Q_reg=Q_reg,
        )
        self.head = nn.Sequential(nn.Linear(z_dim, 64), nn.ReLU(), nn.Linear(64, 2))  # (beat, downbeat) logits
        self.feature_dim, self.a_dim, self.z_dim = feature_dim, a_dim, z_dim


def kvae_elbo(model: KalmanVAEBarPointer, features_time_major: torch.Tensor, sample_control: SampleControl,
             reconstruction_weight: float = 0.3, regularization_weight: float = 1.0, kalman_weight: float = 1.0):
    """The KVAE ELBO (Fraccaro et al. 2017 eq. as ported in third_party/kalman-vae's KalmanVariationalAutoencoder.elbo).

    features_time_major: [num_frames, batch, feature_dim] (the SSM's filter/smoother are time-major).
    Returns (elbo_to_maximize, smoothed_z_sample, info_dict).

    Weight defaults (0.3 / 1.0 / 1.0) are inherited UNCHANGED from the historical M1 run's kvae_run.py
    (its kvae_elbo(..., recon_w=0.3, reg_w=1.0, kal_w=1.0)) that first reached the 0.878/0.833 target this
    module reproduces -- kept as-is for a faithful reproduction rather than re-tuned. reconstruction_weight
    is down-weighted below 1.0 because the reconstruction term (ln p(h|a) over all feature_dim=512
    dimensions, summed) is numerically much larger in magnitude than the Kalman terms (which sum over only
    a_dim/z_dim=8 dimensions), so 0.3 keeps it from dominating the loss purely by dimensionality.
    """
    num_frames, batch_size, feature_dim = features_time_major.shape
    a_distribution = model.encoder(features_time_major.reshape(-1, feature_dim))
    a = a_distribution.rsample().view(num_frames, batch_size, model.a_dim)

    # ln p(h|a)  (reconstruction) and  -ln q(a|h)  (regularization, keeps a's posterior from collapsing)
    reconstruction = model.decoder(a.view(-1, model.a_dim)).log_prob(features_time_major.reshape(-1, feature_dim))
    reconstruction = reconstruction.view(num_frames, batch_size, feature_dim).sum(-1).mean()
    regularization = a_distribution.log_prob(a.view(-1, model.a_dim)).view(num_frames, batch_size, model.a_dim).sum(-1).mean()

    filter_means, filter_covs, next_means, next_covs, mat_As, mat_Cs, _, _ = model.ssm.kalman_filter(a, sample_control=sample_control)
    smooth_means, smooth_covs, z_smoothed, _ = model.ssm.kalman_smooth(
        a, filter_means, filter_covs, next_means, next_covs, mat_As, mat_Cs, sample_control=sample_control)

    z_distribution = D.MultivariateNormal(smooth_means.view(num_frames, batch_size, model.z_dim),
                                          smooth_covs.view(num_frames, batch_size, model.z_dim, model.z_dim))
    z = z_distribution.rsample()

    # ln p(a|z)  (Kalman observation likelihood under the filtered/smoothed transition matrices)
    kalman_observation = D.MultivariateNormal(
        (mat_Cs[:-1] @ z.unsqueeze(-1)).view(-1, model.a_dim), model.ssm.mat_R
    ).log_prob(a.view(-1, model.a_dim)).view(num_frames, batch_size).mean()

    # ln p(z) = ln p(z_0) + sum_t ln p(z_t | z_{t-1})  (the locally-linear-Gaussian dynamics prior)
    prior_means = torch.cat([
        model.ssm.initial_state_mean.repeat(1, batch_size, 1),
        (mat_As[1:-1] @ z[:-1].unsqueeze(-1)).squeeze(-1),
    ])
    prior_covariances = torch.cat([
        model.ssm.initial_state_covariance.repeat(1, batch_size, 1, 1),
        model.ssm.mat_Q.repeat(num_frames - 1, batch_size, 1, 1),
    ])
    kalman_transition = D.MultivariateNormal(
        prior_means.view(num_frames, batch_size, model.z_dim),
        prior_covariances.view(num_frames, batch_size, model.z_dim, model.z_dim),
    ).log_prob(z).mean()

    # ln q(z|a)  (posterior entropy term, from the smoother's own Gaussian)
    kalman_posterior_entropy = z_distribution.log_prob(z).mean()

    elbo = (reconstruction_weight * reconstruction - regularization_weight * regularization
           + kalman_weight * (kalman_observation + kalman_transition - kalman_posterior_entropy))
    info = {
        "recon": float(reconstruction), "reg": float(regularization),
        "kal_obs": float(kalman_observation), "kal_trans": float(kalman_transition),
        "kal_post": float(kalman_posterior_entropy),
    }
    return elbo, z, info
