"""Forked Kalman filter/smoother recursion with PER-FRAME, PER-SONG adaptive process/observation noise.

Motivation (coordinator's finding, verified before implementing): third_party/kalman-vae/kvae/
state_space_model.py's StateSpaceModel.mat_Q / .mat_R are each a SINGLE learned Cholesky factor
(self._mat_Q_L / self._mat_R_L, plain nn.Parameters) plus an additive Q_reg/R_reg floor -- fixed,
identical at every timestep/song/regime (confirmed: grep shows self.mat_Q/self.mat_R referenced as bare
(z_dim,z_dim)/(a_dim,a_dim) matrices with no batch or time dimension at every call site in kalman_filter/
kalman_smooth). This is unlike mat_A/mat_C, which ARE genuinely per-frame-adaptive via the K=5 mixture
(einsum("tbk,kij->bij", weight, mat_A_K), weight predicted per-frame by weight_model's LSTM). Since the
Kalman gain (trust-dynamics-prediction vs trust-current-observation) is derived directly from the Q/R
ratio, the vendored filter has exactly ONE global, track-independent trust-balance setting baked in at
training time -- the same class of limitation the "SMC Blind Spot" paper found for a DBN's fixed global
transition_lambda (no single value serves both clean and noisy/rubato tracks; per-track-adaptive lambda
was worth +0.050 F for them). This also retroactively explains why the earlier Q_reg sweep (a single
GLOBAL floor on an already-global parameter) failed to close the SMC gap.

This experiment: make mat_Q (and mat_R) a function of context -- a small head predicts a per-frame,
per-song positive SCALE multiplier on the base learned covariance, the same spirit as weight_model already
does for mat_A/mat_C's mixture weights, so the filter can genuinely lower its trust in the dynamics
prediction on frames/songs where the audio suggests instability (rubato) and raise it when things are
steady.

Per this project's established practice (see feedback_reuse_official_repos; fast_deploy.py's precedent):
does NOT modify third_party/kalman-vae in place. This module FORKS the filter/smoother recursion (the only
way to make mat_Q/mat_R genuinely batched-per-timestep tensors instead of fixed matrices -- not a
monkey-patchable change, per the coordinator's note), reusing everything else from the passed-in
StateSpaceModel instance UNCHANGED: weight_model (mixture weights -> mat_A/mat_C, identical to the
original), mat_A_K/mat_C_K, initial_state_mean/covariance, and the BASE mat_Q/mat_R (used as the floor/
prior the adaptive scale multiplies). Math is otherwise line-for-line identical to kalman_filter/
kalman_smooth -- see the MANDATORY correctness gate in the driver script (train_kvae_adaptive_noise.py)
that verifies this reproduces the unforked recursion exactly when the scale head is forced to output 1.0.

NOTE ON SHAPE: mat_Q/mat_R become (batch, z_dim, z_dim) / (batch, a_dim, a_dim) tensors instead of
(z_dim, z_dim) / (a_dim, a_dim) -- every `+ self.mat_Q` / `+ self.mat_R` in the original becomes an
elementwise `+ mat_Q_t` / `+ mat_R_t` against the per-timestep tensor (broadcasting was already fine for
the fixed-matrix case since torch broadcasts (z,z) against (batch,z,z); the adaptive case just makes that
batch dimension explicit and genuinely time-varying instead of implicit-and-constant).
"""
from __future__ import annotations

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F


class NoiseScaleHead(nn.Module):
    """a_t [batch, a_dim] -> (scale_Q, scale_R) [batch] each a positive multiplier on the BASE learned
    mat_Q/mat_R (base acts as a floor/prior; the scale is the adaptive per-frame/per-song part).
    softplus(raw)+eps keeps scale > 0 and smooth (no hard clamp discontinuity); output is centered near 1.0
    at initialization (small last-layer weights + softplus(0) != 1, corrected via a learned bias init to
    softplus^{-1}(1) so the model starts near the base covariance and has to learn to deviate)."""

    def __init__(self, a_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(a_dim, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        # softplus(x) = 1.0  =>  x = log(e^1 - 1) ~= 0.5413 -- init the output bias there so scale_Q/scale_R
        # both start at ~1.0 (adaptive filter == base filter at init, before any training pulls it away).
        with torch.no_grad():
            self.net[-1].bias.fill_(0.5413)
            self.net[-1].weight.mul_(0.01)  # small initial sensitivity to a_t, avoids a noisy start

    def forward(self, a_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(a_t)                                    # [batch, 2]
        scale = F.softplus(raw) + 1e-3                          # [batch, 2], > 0
        return scale[:, 0], scale[:, 1]                         # scale_Q [batch], scale_R [batch]


def adaptive_kalman_filter(ssm, as_: torch.Tensor, noise_head: NoiseScaleHead, sample_control,
                           symmetrize_covariance: bool = True):
    """Line-for-line fork of StateSpaceModel.kalman_filter (fully-observed path only -- observation_mask
    is not used anywhere in this project's KVAE training/eval, so it is dropped here for clarity), with
    mat_Q/mat_R replaced by PER-FRAME PER-SONG adaptive tensors: mat_Q_t = ssm.mat_Q * scale_Q_t (broadcast
    over the (z_dim,z_dim) matrix by a per-batch-item scalar), mat_R_t = ssm.mat_R * scale_R_t, where
    (scale_Q_t, scale_R_t) = noise_head(a_t).

    Returns (filtered_means, filtered_covariances, next_means, next_covariances, mat_As, mat_Cs,
             mat_Qs, mat_Rs, scale_Q_trace, scale_R_trace)
    mat_Qs/mat_Rs are [T, batch, z_dim, z_dim] / [T, batch, a_dim, a_dim] -- the actual per-frame
    covariances used, needed by adaptive_kvae_elbo so the ELBO's transition/observation terms use the SAME
    covariance the filter itself used (keeping the ELBO and the filter mutually consistent, unlike naively
    reusing the fixed model.ssm.mat_Q/mat_R in the ELBO while the filter uses adaptive ones).
    """
    sequence_length, batch_size = as_.size()[:2]
    device, dtype = as_.device, as_.dtype

    ssm.weight_model.clear_hidden_state()
    weight_next = torch.ones(1, batch_size, ssm.K, device=device, dtype=dtype) / ssm.K

    mean_t_plus = ssm.initial_state_mean.repeat(batch_size, 1)
    cov_t_plus = ssm.initial_state_covariance.unsqueeze(0).repeat(batch_size, 1, 1)

    base_mat_Q, base_mat_R = ssm.mat_Q, ssm.mat_R                # [z_dim,z_dim], [a_dim,a_dim] (base/floor)

    means, covariances, next_means, next_covariances = [], [], [], []
    mat_As_list, mat_Cs_list, mat_Qs_list, mat_Rs_list = [], [], [], []
    scale_Q_trace, scale_R_trace = [], []

    for t in range(sequence_length):
        weight = weight_next
        if t == 0:
            mat_A = torch.einsum("tbk,kij->bij", weight, ssm.mat_A_K)
        else:
            mat_A = mat_A_next
        mat_C = torch.einsum("tbk,kij->bij", weight, ssm.mat_C_K)

        a_observed = as_[t]                                     # [batch, a_dim] -- fully observed path

        # ---- ADAPTIVE NOISE (the fork's one substantive change): per-frame scale on the base mat_Q/mat_R
        scale_Q, scale_R = noise_head(a_observed)                                  # [batch], [batch]
        mat_Q_t = base_mat_Q.unsqueeze(0) * scale_Q.view(batch_size, 1, 1)         # [batch, z_dim, z_dim]
        mat_R_t = base_mat_R.unsqueeze(0) * scale_R.view(batch_size, 1, 1)         # [batch, a_dim, a_dim]

        if sample_control.state_transition == "sample":
            z_sample = D.MultivariateNormal(
                torch.bmm(mat_A, mean_t_plus.unsqueeze(-1)).squeeze(-1), mat_Q_t
            ).rsample()
        elif sample_control.state_transition == "mean":
            z_sample = D.MultivariateNormal(
                torch.bmm(mat_A, mean_t_plus.unsqueeze(-1)).squeeze(-1), mat_Q_t
            ).mean
        else:
            raise ValueError(f"invalid sample_control.state_transition: {sample_control.state_transition}")

        a_for_weight = a_observed  # fully-observed path: a_for_weight is always the real observation

        weight_next = ssm.weight_model(a_for_weight.unsqueeze(0))
        mat_A_next = torch.einsum("tbk,kij->bij", weight_next, ssm.mat_A_K)

        mat_As_list.append(mat_A)
        mat_Cs_list.append(mat_C)
        mat_Qs_list.append(mat_Q_t)
        mat_Rs_list.append(mat_R_t)
        scale_Q_trace.append(scale_Q)
        scale_R_trace.append(scale_R)

        # Kalman gain -- uses mat_R_t (the ADAPTIVE observation-noise covariance) instead of the fixed self.mat_R
        K_t = (
            cov_t_plus @ mat_C.transpose(1, 2)
            @ torch.inverse(mat_C @ cov_t_plus @ mat_C.transpose(1, 2) + mat_R_t)
        )
        mean_t = mean_t_plus + torch.bmm(
            K_t, (as_[t].unsqueeze(-1) - torch.bmm(mat_C, mean_t_plus.unsqueeze(-1)))
        ).squeeze(-1)
        cov_t = cov_t_plus - K_t @ mat_C @ cov_t_plus
        if symmetrize_covariance:
            cov_t = (cov_t + cov_t.transpose(1, 2)) / 2.0

        mean_t_plus = torch.bmm(mat_A_next, mean_t.unsqueeze(-1)).squeeze(-1)
        # predicted covariance -- uses mat_Q_t (the ADAPTIVE process-noise covariance) instead of fixed self.mat_Q
        cov_t_plus = mat_A_next @ cov_t @ mat_A_next.transpose(1, 2) + mat_Q_t
        if symmetrize_covariance:
            cov_t_plus = (cov_t_plus + cov_t_plus.transpose(1, 2)) / 2.0

        means.append(mean_t)
        covariances.append(cov_t)
        next_means.append(mean_t_plus)
        next_covariances.append(cov_t_plus)

    mat_C_next = torch.einsum("tbk,kij->bij", weight_next, ssm.mat_C_K)
    mat_As_list.append(mat_A_next)
    mat_Cs_list.append(mat_C_next)

    return (
        torch.stack(means), torch.stack(covariances), torch.stack(next_means), torch.stack(next_covariances),
        torch.stack(mat_As_list), torch.stack(mat_Cs_list),
        torch.stack(mat_Qs_list), torch.stack(mat_Rs_list),
        torch.stack(scale_Q_trace), torch.stack(scale_R_trace),
    )


def adaptive_kalman_smooth(ssm, filter_means, filter_covariances, filter_next_means, filter_next_covariances,
                           mat_As, mat_Cs, mat_Rs, sample_control, symmetrize_covariance: bool = True):
    """Line-for-line fork of StateSpaceModel.kalman_smooth, using the PER-FRAME mat_Rs (from
    adaptive_kalman_filter) instead of the fixed self.mat_R in the observation-distribution terms. The
    smoother's own recursion (J_t, mean_t, cov_t) does not reference mat_Q/mat_R at all in the original
    (it only uses mat_As and the filter's own means/covariances) -- verified by reading kalman_smooth --
    so mat_Q does not need to be threaded here; only the observation-distribution `a_distrib` calls
    (which sample a pseudo-observation FROM z, not used by kvae_elbo's a/z terms but returned for parity)
    need mat_Rs[t] instead of a fixed self.mat_R.
    """
    sequence_length = filter_means.shape[0]
    batch_size = filter_means.shape[1]
    z_dim = ssm.z_dim

    means = [filter_means[-1]]
    covariances = [filter_covariances[-1]]

    z_distrib = D.MultivariateNormal(filter_means[-1].view(-1, z_dim), filter_covariances[-1])
    z = z_distrib.rsample() if sample_control.state_transition == "sample" else z_distrib.mean

    a_distrib = D.MultivariateNormal(torch.bmm(mat_Cs[-1], z.unsqueeze(-1)).squeeze(-1), mat_Rs[-1])
    a = a_distrib.rsample() if sample_control.observation == "sample" else a_distrib.mean

    zs_list, as_list = [z], [a]

    for t in reversed(range(sequence_length - 1)):
        J_t = filter_covariances[t] @ mat_As[t + 1].transpose(1, 2) @ torch.inverse(filter_next_covariances[t])
        mean_t = filter_means[t] + torch.bmm(J_t, (means[0] - filter_next_means[t]).unsqueeze(-1)).squeeze(-1)
        cov_t = filter_covariances[t] + J_t @ (covariances[0] - filter_next_covariances[t]) @ J_t.transpose(1, 2)
        if symmetrize_covariance:
            cov_t = (cov_t + cov_t.transpose(1, 2)) / 2.0

        z_distrib = D.MultivariateNormal(mean_t.view(batch_size, z_dim), cov_t)
        z = z_distrib.rsample() if sample_control.state_transition == "sample" else z_distrib.mean

        a_distrib = D.MultivariateNormal(torch.bmm(mat_Cs[t], z.unsqueeze(-1)).squeeze(-1), mat_Rs[t])
        a = a_distrib.rsample() if sample_control.observation == "sample" else a_distrib.mean

        zs_list.insert(0, z)
        as_list.insert(0, a)
        means.insert(0, mean_t)
        covariances.insert(0, cov_t)

    return torch.stack(means), torch.stack(covariances), torch.stack(zs_list), torch.stack(as_list)
