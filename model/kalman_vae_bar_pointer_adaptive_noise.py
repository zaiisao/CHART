"""KalmanVAEBarPointer variant with PER-FRAME, PER-SONG ADAPTIVE process/observation noise (mat_Q_t/mat_R_t)
instead of the base KalmanVAEBarPointer's fixed, global mat_Q/mat_R.

See model/adaptive_noise_filter.py's module docstring for the full motivation and the forked-recursion
design (adaptive_kalman_filter/adaptive_kalman_smooth). This module wires that fork into a
KalmanVAEBarPointer-shaped model (encoder/decoder/head IDENTICAL to model/kalman_vae_bar_pointer.py's
KalmanVAEBarPointer, only the filter/smoother + ELBO differ) plus an adaptive_kvae_elbo that keeps the
ELBO's transition/observation log-prob terms consistent with the per-frame mat_Q_t/mat_R_t the filter
itself used (not the base ssm.mat_Q/mat_R) -- otherwise the ELBO would be optimizing against a covariance
the filter doesn't actually use, an inconsistency that would corrupt training.

Correctness of the fork itself (adaptive_kalman_filter/adaptive_kalman_smooth reproducing the vendored
kalman_filter/kalman_smooth exactly when the noise head is forced to constant 1.0) is verified SEPARATELY
in experiments/kvae_prototype/verify_adaptive_noise_correctness.py (bit-exact, 0.000e+00 max-abs-diff) --
that gate must pass before this module is used for training, per the coordinator's mandatory check.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_THIRD_PARTY_KVAE = os.path.join(os.path.dirname(_THIS_DIR), "third_party", "kalman-vae")
if _THIRD_PARTY_KVAE not in sys.path:
    sys.path.insert(0, _THIRD_PARTY_KVAE)

from kvae.state_space_model import StateSpaceModel      # noqa: E402  (reused verbatim; only mat_Q/mat_R usage is forked)
from kvae.sample_control import SampleControl            # noqa: E402

from .kalman_vae_bar_pointer import ObservationEncoder, ObservationDecoder   # noqa: E402  (identical, reused)
from .adaptive_noise_filter import NoiseScaleHead, adaptive_kalman_filter, adaptive_kalman_smooth  # noqa: E402


class KalmanVAEBarPointerAdaptiveNoise(nn.Module):
    """Encoder -> pseudo-observation -> ADAPTIVE-noise Kalman filter/smoother -> beat/downbeat head.

    IDENTICAL to KalmanVAEBarPointer except: adds a NoiseScaleHead and routes the filter/smoother through
    model/adaptive_noise_filter.py's forked recursion instead of StateSpaceModel.kalman_filter/kalman_smooth
    directly. Encoder/decoder/head/StateSpaceModel construction (mat_A_K, mat_C_K, weight_model, base
    mat_Q/mat_R, Q_reg) are all unchanged from the base model.
    """

    def __init__(self, feature_dim: int = 512, a_dim: int = 8, z_dim: int = 8, K: int = 5, hidden: int = 256,
                Q_reg: float = 1e-3):
        super().__init__()
        self.encoder = ObservationEncoder(feature_dim, a_dim, hidden)
        self.decoder = ObservationDecoder(a_dim, feature_dim, hidden)
        self.ssm = StateSpaceModel(
            a_dim=a_dim, z_dim=z_dim, K=K,
            dynamics_parameter_network="lstm", hidden_dim=64, num_layers=1,
            learn_noise_covariance=True, init_noise_scale=1.0, Q_reg=Q_reg,
        )
        self.noise_head = NoiseScaleHead(a_dim=a_dim)
        self.head = nn.Sequential(nn.Linear(z_dim, 64), nn.ReLU(), nn.Linear(64, 2))
        self.feature_dim, self.a_dim, self.z_dim = feature_dim, a_dim, z_dim


def adaptive_kvae_elbo(model: KalmanVAEBarPointerAdaptiveNoise, features_time_major: torch.Tensor,
                       sample_control: SampleControl, reconstruction_weight: float = 0.3,
                       regularization_weight: float = 1.0, kalman_weight: float = 1.0):
    """Line-for-line the same ELBO structure as model.kalman_vae_bar_pointer.kvae_elbo, with the filter/
    smoother calls replaced by the adaptive fork, and the transition/observation log-prob terms using the
    PER-FRAME mat_Qs/mat_Rs the filter actually used (not a fixed model.ssm.mat_Q/mat_R) -- see module
    docstring for why this consistency matters.

    Returns (elbo_to_maximize, smoothed_z_sample, info_dict) -- info_dict additionally reports
    scale_Q_mean/scale_R_mean/scale_Q_std/scale_R_std (the noise head's output statistics this step, for
    monitoring whether it's genuinely adaptive during training).
    """
    num_frames, batch_size, feature_dim = features_time_major.shape
    a_distribution = model.encoder(features_time_major.reshape(-1, feature_dim))
    a = a_distribution.rsample().view(num_frames, batch_size, model.a_dim)

    reconstruction = model.decoder(a.view(-1, model.a_dim)).log_prob(features_time_major.reshape(-1, feature_dim))
    reconstruction = reconstruction.view(num_frames, batch_size, feature_dim).sum(-1).mean()
    regularization = a_distribution.log_prob(a.view(-1, model.a_dim)).view(num_frames, batch_size, model.a_dim).sum(-1).mean()

    (filter_means, filter_covs, next_means, next_covs, mat_As, mat_Cs,
     mat_Qs, mat_Rs, scale_Q_trace, scale_R_trace) = adaptive_kalman_filter(
        model.ssm, a, model.noise_head, sample_control)
    smooth_means, smooth_covs, z_smoothed, _ = adaptive_kalman_smooth(
        model.ssm, filter_means, filter_covs, next_means, next_covs, mat_As, mat_Cs, mat_Rs, sample_control)

    z_distribution = D.MultivariateNormal(smooth_means.view(num_frames, batch_size, model.z_dim),
                                          smooth_covs.view(num_frames, batch_size, model.z_dim, model.z_dim))
    z = z_distribution.rsample()

    # ln p(a|z) -- uses mat_Rs[:-1] (the PER-FRAME adaptive observation-noise covariance actually used by
    # the filter at each frame), matching the base ELBO's use of mat_Cs[:-1] (both index frames 0..T-2,
    # since mat_Cs/mat_Rs have T+1 and T entries respectively -- mat_Rs has exactly T entries, one per
    # frame, so no [:-1] slice is needed there, unlike mat_Cs which has an extra T-th "next" entry appended).
    kalman_observation = D.MultivariateNormal(
        (mat_Cs[:-1] @ z.unsqueeze(-1)).view(-1, model.a_dim),
        mat_Rs.view(-1, model.a_dim, model.a_dim),
    ).log_prob(a.view(-1, model.a_dim)).view(num_frames, batch_size).mean()

    # ln p(z) = ln p(z_0) + sum_t ln p(z_t | z_{t-1}) -- transition covariance is mat_Qs[1:] (the adaptive
    # process-noise covariance at frames 1..T-1, matching the base ELBO's mat_As[1:-1] indexing convention
    # for the SAME frame range: frame 0 uses initial_state_covariance, frames 1..T-1 use the process noise).
    prior_means = torch.cat([
        model.ssm.initial_state_mean.repeat(1, batch_size, 1),
        (mat_As[1:-1] @ z[:-1].unsqueeze(-1)).squeeze(-1),
    ])
    prior_covariances = torch.cat([
        model.ssm.initial_state_covariance.repeat(1, batch_size, 1, 1),
        mat_Qs[1:],
    ])
    kalman_transition = D.MultivariateNormal(
        prior_means.view(num_frames, batch_size, model.z_dim),
        prior_covariances.view(num_frames, batch_size, model.z_dim, model.z_dim),
    ).log_prob(z).mean()

    kalman_posterior_entropy = z_distribution.log_prob(z).mean()

    elbo = (reconstruction_weight * reconstruction - regularization_weight * regularization
           + kalman_weight * (kalman_observation + kalman_transition - kalman_posterior_entropy))
    info = {
        "recon": float(reconstruction), "reg": float(regularization),
        "kal_obs": float(kalman_observation), "kal_trans": float(kalman_transition),
        "kal_post": float(kalman_posterior_entropy),
        "scale_Q_mean": float(scale_Q_trace.mean()), "scale_Q_std": float(scale_Q_trace.std()),
        "scale_R_mean": float(scale_R_trace.mean()), "scale_R_std": float(scale_R_trace.std()),
    }
    return elbo, z, info
