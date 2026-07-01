"""Fast batched Kalman-filter deploy path for KalmanVAEBarPointer -- eval-only speedup, math-equivalent
to StateSpaceModel.kalman_filter under the deploy configuration (fully observed, deterministic).

WHY: train_kvae.py's evaluate_leak_condition and the OOD eval scripts (eval_smc_ood_fresh.py,
eval_gtzan_ood.py) called kalman_filter with batch_size=1, once per song -- 993 separate sequential
Python/CUDA loops for GTZAN (~185 min with zero results after the SMC run already validated the leak-test
pattern). Root cause (verified by reading third_party/kalman-vae/kvae/state_space_model.py's
kalman_filter): it loops over `sequence_length` in plain Python; each step does 2 MultivariateNormal
constructions (Cholesky decomps of tiny 8x8 matrices) plus a single-frame LSTM call for the dynamics-
mixture weights. This is latency-bound (kernel-launch overhead), not compute-bound, so batching songs
together (instead of one-at-a-time) turns ~993 sequential T-length loops into ONE T_max-length loop -- the
per-step work barely changes (matrices just gain a batch dimension), so wall-clock should drop by roughly
the song count.

TWO speedups, both applied only to the DEPLOY read (never used for training/ELBO, which still goes through
the vendored StateSpaceModel.kalman_filter unmodified via kvae_elbo in model/kalman_vae_bar_pointer.py):

  1. Batch across songs: pad every song's [T_i, a_dim] pseudo-observation sequence up to a common T_max
     with zeros, run ONE kalman_filter-equivalent pass over [T_max, N_songs, a_dim], then slice each
     song's filtered mean back to its true T_i before peak-picking. A per-song, per-frame boolean mask
     freezes the state (mean/cov held constant, exactly mirroring kalman_filter's own
     `observation_mask=0` unobserved-step branch) once a song's real audio has ended, so padding frames
     cannot corrupt that song's estimate or leak into another song's Kalman gain (state/gain updates stay
     strictly per-batch-column via the existing batched bmm/einsum -- no cross-song mixing is introduced).
  2. Precompute the dynamics-mixture LSTM weights in ONE call over the whole padded sequence instead of
     stepping the LSTM one frame at a time. This is EXACTLY equivalent to the original step-by-step calls:
     nn.LSTM (batch_first=False) is inherently sequential over its time axis and carries hidden state
     between calls; feeding it [T,B,a_dim] in one call vs. T separate calls of [1,B,a_dim] with the hidden
     state threaded through by hand produces IDENTICAL per-timestep outputs (verified below in
     verify_matches_reference, comparing element-wise against the unmodified vendored kalman_filter on a
     real batch before this is trusted for evaluation).

This file does NOT modify third_party/kalman-vae/ (kept verbatim per this project's reuse-official-repos
rule) -- it is a separate, eval-only reimplementation of the same recursion, gated by the deploy
SampleControl (encoder="mean", state_transition="mean", observation="mean", fully observed) under which
the state-transition/observation "sample" branches in the original code are provably unreachable anyway.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def batched_kalman_filter_deploy(ssm, a_padded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Deploy-only fast path: filtered posterior MEAN, batched across songs.

    a_padded: [T_max, N, a_dim] pseudo-observations (encoder means), zero-padded past each song's length.
    lengths:  [N] true (unpadded) length of each song, in frames.
    Returns:  [T_max, N, z_dim] filtered means (mean_t, i.e. \\hat{z}_{t|t} -- the SAME tensor
              kalman_filter's first return value is; only valid for t < lengths[n] per song n).

    Math: identical recursion to StateSpaceModel.kalman_filter with sample_control.{state_transition,
    observation}="mean" (so those branches reduce to `z_sample = mat_A @ mean_t_plus`, unused beyond that --
    the ORIGINAL code computes `a_unobserved` in that branch too, but only to feed `a_for_weight` when
    observation_mask flags an unobserved step; our fully-observed deploy path never takes that branch, so
    that computation is dead code here and is correctly omitted) and observation_mask = (t < lengths[n]),
    i.e. a step is "observed" for every real frame and "unobserved" (state held constant, exactly
    kalman_filter's own unobserved-step formula) for every pad frame.
    """
    T_max, N, a_dim = a_padded.shape
    device, dtype = a_padded.device, a_padded.dtype
    z_dim, K = ssm.z_dim, ssm.K
    observed = (torch.arange(T_max, device=device).unsqueeze(1) < lengths.unsqueeze(0)).to(dtype)  # [T_max, N]

    # ---- speedup 2: one LSTM call over the whole (zero-padded) sequence, not T_max separate calls.
    # Mirrors kalman_filter's per-step `weight_next = weight_model(a_for_weight.unsqueeze(0))` with
    # a_for_weight = a_observed (fully-observed deploy path) -- feeding the whole [T,N,a_dim] sequence to
    # the same nn.LSTM in one shot reproduces those per-timestep outputs exactly (LSTM is causal/sequential
    # internally regardless of call granularity). weight_next at t is USED to build mat_A/mat_C at t+1 in
    # the original loop (see mat_A_next / "if t==0 ... else mat_A = mat_A_next"); we reproduce that exact
    # shift below rather than re-deriving it, so any off-by-one in the source is inherited faithfully.
    ssm.weight_model.clear_hidden_state()
    weights_from_a = ssm.weight_model(a_padded)                                  # [T_max, N, K], weight_model(a[t]) at index t
    uniform_weight = torch.full((1, N, K), 1.0 / K, device=device, dtype=dtype)
    # weight used to build mat_A/mat_C AT frame t: uniform at t=0, else weights_from_a[t-1] (the weight
    # computed from a[t-1], exactly matching kalman_filter's `weight = weight_next` carried from the
    # PREVIOUS iteration's weight_model call).
    weight_at_t = torch.cat([uniform_weight, weights_from_a[:-1]], dim=0)        # [T_max, N, K]
    mat_A_t = torch.einsum("tnk,kij->tnij", weight_at_t, ssm.mat_A_K)            # [T_max, N, z_dim, z_dim]
    mat_C_t = torch.einsum("tnk,kij->tnij", weight_at_t, ssm.mat_C_K)            # [T_max, N, a_dim, z_dim]
    # mat_A_next at t (used to predict z_{t+1|t} from z_{t|t}) is mat_A built from weights_from_a[t].
    mat_A_next_t = torch.einsum("tnk,kij->tnij", weights_from_a, ssm.mat_A_K)    # [T_max, N, z_dim, z_dim]

    mat_Q, mat_R = ssm.mat_Q, ssm.mat_R                                          # [z_dim,z_dim], [a_dim,a_dim]
    mean_t_plus = ssm.initial_state_mean.unsqueeze(0).expand(N, -1).clone()      # [N, z_dim]
    cov_t_plus = ssm.initial_state_covariance.unsqueeze(0).expand(N, -1, -1).clone()  # [N, z_dim, z_dim]

    filtered_means = torch.zeros(T_max, N, z_dim, device=device, dtype=dtype)

    # ---- speedup 1: this loop is over T_max ONCE for the whole batch of N songs (not N separate T_i loops).
    for t in range(T_max):
        mat_A, mat_C = mat_A_t[t], mat_C_t[t]                                    # [N, z_dim, z_dim], [N, a_dim, z_dim]
        mask = observed[t].view(N, 1, 1)                                         # [N,1,1] broadcastable

        kalman_gain = (
            cov_t_plus @ mat_C.transpose(1, 2)
            @ torch.inverse(mat_C @ cov_t_plus @ mat_C.transpose(1, 2) + mat_R)
        )
        mean_t_observed = mean_t_plus + torch.bmm(
            kalman_gain, (a_padded[t].unsqueeze(-1) - torch.bmm(mat_C, mean_t_plus.unsqueeze(-1)))
        ).squeeze(-1)
        cov_t_observed = cov_t_plus - kalman_gain @ mat_C @ cov_t_plus

        mean_t = mask.squeeze(-1) * mean_t_observed + (1.0 - mask.squeeze(-1)) * mean_t_plus
        cov_t = mask * cov_t_observed + (1.0 - mask) * cov_t_plus
        cov_t = (cov_t + cov_t.transpose(1, 2)) / 2.0

        filtered_means[t] = mean_t

        mat_A_next = mat_A_next_t[t]
        mean_t_plus = torch.bmm(mat_A_next, mean_t.unsqueeze(-1)).squeeze(-1)
        cov_t_plus = mat_A_next @ cov_t @ mat_A_next.transpose(1, 2) + mat_Q
        cov_t_plus = (cov_t_plus + cov_t_plus.transpose(1, 2)) / 2.0

    return filtered_means


@torch.no_grad()
def batched_kalman_filter_deploy_with_weights(ssm, a_padded: torch.Tensor, lengths: torch.Tensor):
    """Identical to batched_kalman_filter_deploy, but ALSO returns the per-frame K=5 dynamics-mixture
    softmax weight used to build mat_A/mat_C at each frame (weight_at_t in the sibling function) --
    for the K=5 mixture-collapse diagnostic (TASK B). Not used by any already-validated eval path; kept
    separate so the regression-tested batched_kalman_filter_deploy above is untouched.

    Returns (filtered_means [T_max, N, z_dim], mixture_weights [T_max, N, K]).
    """
    T_max, N, a_dim = a_padded.shape
    device, dtype = a_padded.device, a_padded.dtype
    z_dim, K = ssm.z_dim, ssm.K
    observed = (torch.arange(T_max, device=device).unsqueeze(1) < lengths.unsqueeze(0)).to(dtype)

    ssm.weight_model.clear_hidden_state()
    weights_from_a = ssm.weight_model(a_padded)
    uniform_weight = torch.full((1, N, K), 1.0 / K, device=device, dtype=dtype)
    weight_at_t = torch.cat([uniform_weight, weights_from_a[:-1]], dim=0)        # [T_max, N, K]
    mat_A_t = torch.einsum("tnk,kij->tnij", weight_at_t, ssm.mat_A_K)
    mat_C_t = torch.einsum("tnk,kij->tnij", weight_at_t, ssm.mat_C_K)
    mat_A_next_t = torch.einsum("tnk,kij->tnij", weights_from_a, ssm.mat_A_K)

    mat_Q, mat_R = ssm.mat_Q, ssm.mat_R
    mean_t_plus = ssm.initial_state_mean.unsqueeze(0).expand(N, -1).clone()
    cov_t_plus = ssm.initial_state_covariance.unsqueeze(0).expand(N, -1, -1).clone()

    filtered_means = torch.zeros(T_max, N, z_dim, device=device, dtype=dtype)

    for t in range(T_max):
        mat_A, mat_C = mat_A_t[t], mat_C_t[t]
        mask = observed[t].view(N, 1, 1)

        kalman_gain = (
            cov_t_plus @ mat_C.transpose(1, 2)
            @ torch.inverse(mat_C @ cov_t_plus @ mat_C.transpose(1, 2) + mat_R)
        )
        mean_t_observed = mean_t_plus + torch.bmm(
            kalman_gain, (a_padded[t].unsqueeze(-1) - torch.bmm(mat_C, mean_t_plus.unsqueeze(-1)))
        ).squeeze(-1)
        cov_t_observed = cov_t_plus - kalman_gain @ mat_C @ cov_t_plus

        mean_t = mask.squeeze(-1) * mean_t_observed + (1.0 - mask.squeeze(-1)) * mean_t_plus
        cov_t = mask * cov_t_observed + (1.0 - mask) * cov_t_plus
        cov_t = (cov_t + cov_t.transpose(1, 2)) / 2.0

        filtered_means[t] = mean_t

        mat_A_next = mat_A_next_t[t]
        mean_t_plus = torch.bmm(mat_A_next, mean_t.unsqueeze(-1)).squeeze(-1)
        cov_t_plus = mat_A_next @ cov_t @ mat_A_next.transpose(1, 2) + mat_Q
        cov_t_plus = (cov_t_plus + cov_t_plus.transpose(1, 2)) / 2.0

    return filtered_means, weight_at_t


@torch.no_grad()
def batched_adaptive_kalman_filter_deploy(ssm, noise_head, a_padded: torch.Tensor, lengths: torch.Tensor):
    """Deploy-time batched fast path for the ADAPTIVE-noise filter (model/adaptive_noise_filter.py's
    adaptive_kalman_filter), analogous to batched_kalman_filter_deploy above but with mat_Q_t/mat_R_t =
    base_mat_Q/R * noise_head(a_t) instead of the fixed base mat_Q/mat_R. Same padding/masking scheme as
    batched_kalman_filter_deploy (observed = t < lengths[n], unobserved steps hold state constant).

    Returns (filtered_means [T_max, N, z_dim], scale_Q [T_max, N], scale_R [T_max, N]) -- the scale traces
    are returned too so eval scripts can report the Task-B-style entropy/variance-of-adaptation check
    without a second forward pass.
    """
    T_max, N, a_dim = a_padded.shape
    device, dtype = a_padded.device, a_padded.dtype
    z_dim, K = ssm.z_dim, ssm.K
    observed = (torch.arange(T_max, device=device).unsqueeze(1) < lengths.unsqueeze(0)).to(dtype)

    ssm.weight_model.clear_hidden_state()
    weights_from_a = ssm.weight_model(a_padded)
    uniform_weight = torch.full((1, N, K), 1.0 / K, device=device, dtype=dtype)
    weight_at_t = torch.cat([uniform_weight, weights_from_a[:-1]], dim=0)
    mat_A_t = torch.einsum("tnk,kij->tnij", weight_at_t, ssm.mat_A_K)
    mat_C_t = torch.einsum("tnk,kij->tnij", weight_at_t, ssm.mat_C_K)
    mat_A_next_t = torch.einsum("tnk,kij->tnij", weights_from_a, ssm.mat_A_K)

    base_mat_Q, base_mat_R = ssm.mat_Q, ssm.mat_R                                # [z_dim,z_dim], [a_dim,a_dim]
    scale_Q_all, scale_R_all = noise_head(a_padded.reshape(-1, a_dim))
    scale_Q_all = scale_Q_all.view(T_max, N)                                     # [T_max, N]
    scale_R_all = scale_R_all.view(T_max, N)

    mean_t_plus = ssm.initial_state_mean.unsqueeze(0).expand(N, -1).clone()
    cov_t_plus = ssm.initial_state_covariance.unsqueeze(0).expand(N, -1, -1).clone()

    filtered_means = torch.zeros(T_max, N, z_dim, device=device, dtype=dtype)

    for t in range(T_max):
        mat_A, mat_C = mat_A_t[t], mat_C_t[t]
        mask = observed[t].view(N, 1, 1)

        mat_Q_t = base_mat_Q.unsqueeze(0) * scale_Q_all[t].view(N, 1, 1)         # [N, z_dim, z_dim]
        mat_R_t = base_mat_R.unsqueeze(0) * scale_R_all[t].view(N, 1, 1)         # [N, a_dim, a_dim]

        kalman_gain = (
            cov_t_plus @ mat_C.transpose(1, 2)
            @ torch.inverse(mat_C @ cov_t_plus @ mat_C.transpose(1, 2) + mat_R_t)
        )
        mean_t_observed = mean_t_plus + torch.bmm(
            kalman_gain, (a_padded[t].unsqueeze(-1) - torch.bmm(mat_C, mean_t_plus.unsqueeze(-1)))
        ).squeeze(-1)
        cov_t_observed = cov_t_plus - kalman_gain @ mat_C @ cov_t_plus

        mean_t = mask.squeeze(-1) * mean_t_observed + (1.0 - mask.squeeze(-1)) * mean_t_plus
        cov_t = mask * cov_t_observed + (1.0 - mask) * cov_t_plus
        cov_t = (cov_t + cov_t.transpose(1, 2)) / 2.0

        filtered_means[t] = mean_t

        mat_A_next = mat_A_next_t[t]
        mean_t_plus = torch.bmm(mat_A_next, mean_t.unsqueeze(-1)).squeeze(-1)
        cov_t_plus = mat_A_next @ cov_t @ mat_A_next.transpose(1, 2) + mat_Q_t
        cov_t_plus = (cov_t_plus + cov_t_plus.transpose(1, 2)) / 2.0

    return filtered_means, scale_Q_all, scale_R_all


@torch.no_grad()
def verify_adaptive_matches_reference(model, songs_features: list[torch.Tensor], device: str) -> float:
    """Sanity check: batched_adaptive_kalman_filter_deploy vs. model/adaptive_noise_filter.py's
    adaptive_kalman_filter, called ONE SONG AT A TIME (the slow, already-correctness-gated way), on a
    handful of real songs. Returns the max absolute difference in filtered means -- should be at/near
    float32 numerical noise, the same bar batched_kalman_filter_deploy's own verify_matches_reference used.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                    "third_party", "kalman-vae"))
    from kvae.sample_control import SampleControl
    from model.adaptive_noise_filter import adaptive_kalman_filter

    sample_control = SampleControl(encoder="mean", decoder="mean", state_transition="mean", observation="mean")
    lengths = torch.tensor([f.shape[0] for f in songs_features], device=device)
    T_max = int(lengths.max())
    N = len(songs_features)

    a_list = []
    for features in songs_features:
        a_mean = model.encoder(features.to(device)).mean
        padded = torch.zeros(T_max, model.a_dim, device=device, dtype=a_mean.dtype)
        padded[:a_mean.shape[0]] = a_mean
        a_list.append(padded)
    a_padded = torch.stack(a_list, dim=1)

    batched_means, _, _ = batched_adaptive_kalman_filter_deploy(model.ssm, model.noise_head, a_padded, lengths)

    max_diff = 0.0
    for i, features in enumerate(songs_features):
        T_i = features.shape[0]
        a_mean = model.encoder(features.to(device)).mean.view(T_i, 1, model.a_dim)
        reference_out = adaptive_kalman_filter(model.ssm, a_mean, model.noise_head, sample_control)
        reference_means = reference_out[0].view(T_i, model.z_dim)
        diff = (batched_means[:T_i, i] - reference_means).abs().max().item()
        max_diff = max(max_diff, diff)
    return max_diff


@torch.no_grad()
def verify_matches_reference(model, songs_features: list[torch.Tensor], device: str, atol: float = 1e-3) -> float:
    """Sanity check: batched_kalman_filter_deploy vs. the vendored kalman_filter, called ONE SONG AT A TIME
    (the original, slow way), on a handful of real songs. Returns the max absolute difference in the
    filtered means over all valid (unpadded) frames -- should be at/near float32 numerical noise.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                    "third_party", "kalman-vae"))
    from kvae.sample_control import SampleControl

    sample_control = SampleControl(encoder="mean", decoder="mean", state_transition="mean", observation="mean")
    lengths = torch.tensor([f.shape[0] for f in songs_features], device=device)
    T_max = int(lengths.max())
    N = len(songs_features)

    a_list = []
    for features in songs_features:
        a_mean = model.encoder(features.to(device)).mean          # [T_i, a_dim]
        padded = torch.zeros(T_max, model.a_dim, device=device, dtype=a_mean.dtype)
        padded[:a_mean.shape[0]] = a_mean
        a_list.append(padded)
    a_padded = torch.stack(a_list, dim=1)                          # [T_max, N, a_dim]

    batched_means = batched_kalman_filter_deploy(model.ssm, a_padded, lengths)  # [T_max, N, z_dim]

    max_diff = 0.0
    for i, features in enumerate(songs_features):
        T_i = features.shape[0]
        a_mean = model.encoder(features.to(device)).mean.view(T_i, 1, model.a_dim)
        reference_means, *_ = model.ssm.kalman_filter(a_mean, sample_control=sample_control)  # [T_i, 1, z_dim]
        reference_means = reference_means.view(T_i, model.z_dim)
        diff = (batched_means[:T_i, i] - reference_means).abs().max().item()
        max_diff = max(max_diff, diff)
    return max_diff
