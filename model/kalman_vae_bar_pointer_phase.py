"""KalmanVAEBarPointer variant with an INTERPRETABLE, geometrically-read-out-able bar phase, restoring the
"manually repiece together beats/downbeats from phi" property that a plain learned head on an opaque z
discards -- while KEEPING the exact-Kalman-filter backbone (already proven to generalize: in-domain
0.895/0.845, SMC zero-shot 0.654, GTZAN zero-shot 0.879/0.789).

============================================================================================================
PROVENANCE -- this is NOT a new idea, it is a port of a result already achieved (and diagnosed in detail)
on a DIFFERENT, pre-refactor architecture. See memory/project_sawtooth_phase_breakthrough.md and the
archived CHART_archive_2026-06-30/experiments/diagram_arch/{sawtooth_aux*.py, autocorr_tempo.py,
faithful_autocorr_filter.py, tempo_grad_sawtooth.py} + their .log files (read before touching this file).
Three things were established there, in this order, and this module's design follows directly from them:
  1. A sawtooth phase-supervision loss (built from GT beat/downbeat labels; dense; slope=tempo) grounds
     phase WELL when applied to a FREELY-READ-EVERY-FRAME phase (in-domain beat ~0.84).
  2. A separate FREE-LEARNED tempo latent never gets grounded by that loss (gradient test,
     tempo_grad_sawtooth.py: aux loss doesn't reach the tempo latent's own parameters -- it only grounds
     phase directly; a pointwise/free-learned "rate" variable is fundamentally the wrong kind of quantity
     for gradient descent to learn well). A pure phi=integral(phidot) INTEGRATOR + sawtooth fails outright
     (beat 0.186 -- no per-frame re-anchoring, drifts over long sequences, non-convex).
  3. The fix that actually worked (faithful_autocorr_filter.py, "A+B"): COMPUTE tempo via a differentiable
     AUTOCORRELATION head on the features (a windowed/periodicity operator -- the mathematically correct
     way to recover a RATE, unlike a pointwise derivative or a free latent), use that to drive a PREDICT
     step, then CORRECT toward the audio every frame (a filter). beat 0.772 db 0.817, tempo-from-phidot
     AND tempo-from-phase agreed with each other and GT (r 0.70-0.74 raw, 0.94-0.999 octave-tolerant).
  4. WARNING carried forward: the simpler sawtooth-only version was tested on SMC zero-shot ONCE and
     collapsed (F=0.192 OOD vs 0.84 in-domain) -- diagnosed as overfitting the training mix's steady tempo,
     unable to handle SMC's rubato. The fuller autocorr+filter+sawtooth combo (point 3) was NEVER tested
     against SMC -- this module's overnight run is what finally answers that question.

WHAT'S REUSED, NOT RE-DERIVED (per this project's reuse-official/reuse-existing-code convention):
  * model/divergences.py's AutocorrelationTempoHead -- ALREADY the same differentiable-tempogram design as
    the archived autocorr_tempo.py's TempoNet (onset-strength projection -> windowed autocorrelation over
    AUTOCORR_LAG_FRAMES candidate periods -> softmax-able lag scores), already ported and unit-tested in
    THIS codebase (used by BarPointerVAE's --divergence_tempo_source autocorr ablation). Used verbatim here.
  * data/targets.py's build_sawtooth_phase_target_batch -- ALREADY the same unified bar-phase sawtooth
    target (0 at downbeats, +2*pi/beats_per_bar per beat) as the archived sawtooth_aux3.py's gt_batch_uni
    (the specific variant faithful_autocorr_filter.py -- the A+B winner -- actually used, not the weaker
    bar-only target from sawtooth_aux.py which lost downbeat quality). Used verbatim here.
  * model/readout.py's phase_to_beat_times / phase_to_downbeat_times -- the existing VBPM geometric
    read-out (phase-wrap detection), used verbatim on the phase RECOVERED from this module's z[...,0:2].

============================================================================================================
DESIGN: unit-circle embedding + rotation-block prediction (NOT the archived code's von-Mises-space blend())

The archived code represented phase as a raw angle handled by a von Mises posterior with its own closed-
form KL and an explicit circular blend() for predict/correct. The Kalman filter's z is flat Euclidean/
Gaussian by construction -- a raw angle with wraparound is NOT linear-Gaussian (exactly why the original
Fraccaro/VBPM math needs von Mises for phase and exactly why an earlier, DIFFERENT attempt on this
project's KVAE -- a rotational z read out via atan2 with rotation baked into the LEARNED K=5 mixture
matrices, see kvae_geom.py in the archived kvae_barpointer campaign -- failed to rotate: the observation
correction kept pulling the freely-mixture-driven z back toward a near-static estimate).

The fix here: dedicate z[..., 0:2] to hold (cos(phi), sin(phi)) directly -- THIS pair of coordinates IS
linear-Gaussian (a rotation by a FIXED angle is a linear map on (cos,sin) -- unlike phi itself, which wraps
non-linearly). Per frame:
  1. COMPUTE the bar-phase advance (phidot, radians/frame) via AutocorrelationTempoHead on the raw features
     h (NOT a free latent, NOT z-derived -- matches the archived A+B design exactly: tempo is COMPUTED,
     grounded by its own cross-entropy loss against the GT beat period, detached from the phase rollout's
     gradient, same division of labor as model/bar_pointer_vae.py's existing --divergence_tempo_source
     autocorr ablation).
  2. PREDICT: override the phase sub-block of the per-frame effective mat_A with an exact 2x2 ROTATION
     matrix by phidot (so z[...,0:2]_predicted = R(phidot) @ z[...,0:2]_filtered -- an exact, linear
     "predict via tempo" step, not an approximation). The REMAINING 6 dims of z keep the standard learned
     K=5-mixture dynamics from StateSpaceModel (mat_A_K), UNCHANGED -- only phase's own prediction is
     overridden; the other dims may still read phase (and vice versa isn't blocked either, since mat_C's
     correction step, below, is completely untouched).
  3. CORRECT: the exact Kalman update (gain, mat_C, mat_R) is left COMPLETELY UNMODIFIED from the vendored
     StateSpaceModel.kalman_filter's own math -- this is already the "correct toward what the audio
     suggests" step the archived code had to hand-build via blend(); the Kalman filter already does this
     exactly, for free, once phase is a genuine z sub-block instead of a von-Mises-parameterized scalar.
  4. RECOVER: phi = atan2(z[...,1], z[...,0]) whenever an angle is needed (sawtooth loss, geometric
     read-out) -- never store or propagate phi directly, only its (cos,sin) embedding.

Per this project's "fork, don't modify" convention (feedback_reuse_official_repos; adaptive_noise_filter.py
is the direct precedent for HOW to fork just the recursion, reused here): does NOT modify
third_party/kalman-vae or model/kalman_vae_bar_pointer.py. Everything not explicitly mentioned above
(encoder, decoder, mat_C_K/mixture weights, mat_Q/mat_R, Kalman gain, smoother) is IDENTICAL to the base
KalmanVAEBarPointer -- this is a minimal, targeted, auditable change to the prediction step only.
"""
from __future__ import annotations

import math
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

from kvae.state_space_model import StateSpaceModel      # noqa: E402  (reused verbatim; only the phase
                                                          # sub-block of the per-frame mat_A is overridden)
from kvae.sample_control import SampleControl             # noqa: E402

from .kalman_vae_bar_pointer import ObservationEncoder, ObservationDecoder   # noqa: E402  (identical, reused)
from .divergences import AutocorrelationTempoHead                            # noqa: E402  (reused verbatim)

TWO_PI = 2.0 * math.pi
PHASE_DIMS = 2   # z[..., 0:2] = (cos phi, sin phi); the remaining z_dim - 2 dims are free, as in the base model


def phase_angle_from_z(z_phase_pair: torch.Tensor) -> torch.Tensor:
    """z[..., 0:2] = (cos phi, sin phi) -> phi in [0, 2*pi). Never store/propagate phi itself, only recover
    it on demand (this is what keeps the representation in flat Euclidean/Gaussian space for the filter)."""
    return torch.atan2(z_phase_pair[..., 1], z_phase_pair[..., 0]) % TWO_PI


def rotation_matrix_2x2(angle: torch.Tensor) -> torch.Tensor:
    """angle [...] (radians) -> [..., 2, 2] rotation matrix R(angle) such that R(angle) @ (cos a, sin a)^T
    = (cos(a+angle), sin(a+angle))^T -- i.e. applying R(phidot) to (cos phi, sin phi) advances the phase by
    phidot exactly, with no small-angle approximation (this is an EXACT linear map, unlike phi + phidot
    which only works because phi is never itself propagated -- see module docstring)."""
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    row0 = torch.stack([cos_a, -sin_a], dim=-1)
    row1 = torch.stack([sin_a, cos_a], dim=-1)
    return torch.stack([row0, row1], dim=-2)   # [..., 2, 2]


class KalmanVAEBarPointerPhase(nn.Module):
    """Encoder -> pseudo-observation -> Kalman filter/smoother with a rotation-block phase prediction ->
    beat/downbeat via EITHER a geometric phase read-out OR a learned head on the full z (both reported)."""

    def __init__(self, feature_dim: int = 512, a_dim: int = 8, z_dim: int = 8, K: int = 5, hidden: int = 256,
                Q_reg: float = 1e-3, beats_per_bar: int = 4):
        super().__init__()
        assert z_dim >= PHASE_DIMS, "z_dim must be >= 2 to hold the (cos phi, sin phi) phase sub-block"
        self.encoder = ObservationEncoder(feature_dim, a_dim, hidden)
        self.decoder = ObservationDecoder(a_dim, feature_dim, hidden)
        self.ssm = StateSpaceModel(
            a_dim=a_dim, z_dim=z_dim, K=K,
            dynamics_parameter_network="lstm", hidden_dim=64, num_layers=1,
            learn_noise_covariance=True, init_noise_scale=1.0, Q_reg=Q_reg,
        )
        self.tempo_head = AutocorrelationTempoHead(feature_dim)
        self.head = nn.Sequential(nn.Linear(z_dim, 64), nn.ReLU(), nn.Linear(64, 2))  # learned-head alternative
        self.feature_dim, self.a_dim, self.z_dim, self.beats_per_bar = feature_dim, a_dim, z_dim, beats_per_bar

        # z[0:2]'s initial state prior mean is set to (cos 0, sin 0) = (1, 0) rather than the default zero
        # vector -- a mean of (0,0) has no well-defined ANGLE (atan2(0,0) is degenerate), so a nonzero unit-
        # length prior mean avoids seeding the filter with an ill-posed phase at frame 0.
        with torch.no_grad():
            self.ssm.initial_state_mean[0] = 1.0
            self.ssm.initial_state_mean[1] = 0.0


def _effective_mat_A(mixture_mat_A: torch.Tensor, rotation_2x2: torch.Tensor) -> torch.Tensor:
    """mixture_mat_A [batch, z_dim, z_dim] (the standard K=5-mixture dynamics), rotation_2x2 [batch, 2, 2]
    (this frame's phidot-driven rotation) -> [batch, z_dim, z_dim] with ROWS 0:2 replaced by
    [rotation_2x2 | zeros], rows 2: left exactly as the mixture produced them. This means:
      - z[...,0:2]_predicted = rotation_2x2 @ z[...,0:2]_filtered  (phase evolves ONLY by rotation, exactly
        the archived design's "predict via computed tempo, decoupled from everything else in the predict
        step" -- see faithful_autocorr_filter.py's phi_pred = phiprev + phidot, which this generalizes to
        the (cos,sin) pair).
      - z[...,2:]_predicted = mixture_mat_A[2:, :] @ z_filtered  (the other z_dim-2 dims keep the FULL
        learned mixture dynamics, including any coupling they have TO the phase dims -- only phase's OWN
        prediction is overridden, nothing else is restricted).
    """
    batch_size, z_dim, _ = mixture_mat_A.shape
    device, dtype = mixture_mat_A.device, mixture_mat_A.dtype
    phase_rows = torch.cat([
        rotation_2x2, torch.zeros(batch_size, PHASE_DIMS, z_dim - PHASE_DIMS, device=device, dtype=dtype),
    ], dim=-1)                                                     # [batch, 2, z_dim]
    remaining_rows = mixture_mat_A[:, PHASE_DIMS:, :]               # [batch, z_dim-2, z_dim], UNCHANGED
    return torch.cat([phase_rows, remaining_rows], dim=1)           # [batch, z_dim, z_dim]


def phase_kalman_filter(model: KalmanVAEBarPointerPhase, as_: torch.Tensor, phidot_per_clip: torch.Tensor,
                        sample_control: SampleControl, symmetrize_covariance: bool = True):
    """Fork of StateSpaceModel.kalman_filter (fully-observed path only, as in adaptive_noise_filter.py) with
    ONE change: the phase sub-block (rows 0:2) of the per-frame effective mat_A is replaced by an exact
    rotation-by-phidot matrix instead of the K=5-mixture's own rows for those dims (see _effective_mat_A).
    Everything else -- mat_C from the mixture, the Kalman gain, mat_Q/mat_R, the correction step -- is
    IDENTICAL to the vendored recursion; only the PREDICT step for z[...,0:2] is touched.

    phidot_per_clip: [batch] -- ONE bar-phase advance (radians/frame) per clip (not per-frame; matches the
    archived A+B design's clip_tempo(), which reads phidot from a per-clip autocorrelation over the whole
    crop -- windowed autocorrelation needs enough frames to resolve a period, so it is not meaningfully
    "per-frame" the way the K=5 mixture weights are; a per-clip constant tempo within a ~3s training crop
    is also a much closer match to how AutocorrelationTempoHead was trained (its own CE loss is per-clip)).

    Returns (filtered_means, filtered_covariances, next_means, next_covariances, mat_As, mat_Cs) -- same
    return shape/semantics as the vendored kalman_filter's first six outputs (weight/as_for_weight dropped,
    unused by kvae_elbo-style callers, matching adaptive_noise_filter.py's precedent of dropping unused
    vendored outputs in a fork).
    """
    ssm = model.ssm
    sequence_length, batch_size = as_.size()[:2]
    device, dtype = as_.device, as_.dtype

    ssm.weight_model.clear_hidden_state()
    weight_next = torch.ones(1, batch_size, ssm.K, device=device, dtype=dtype) / ssm.K

    mean_t_plus = ssm.initial_state_mean.repeat(batch_size, 1)
    cov_t_plus = ssm.initial_state_covariance.unsqueeze(0).repeat(batch_size, 1, 1)

    rotation_2x2 = rotation_matrix_2x2(phidot_per_clip)   # [batch, 2, 2] -- same rotation used every frame
                                                           # (phidot is one computed value per clip, see above)

    means, covariances, next_means, next_covariances = [], [], [], []
    mat_As_list, mat_Cs_list = [], []

    for t in range(sequence_length):
        weight = weight_next
        if t == 0:
            mixture_mat_A = torch.einsum("tbk,kij->bij", weight, ssm.mat_A_K)
            mat_A = _effective_mat_A(mixture_mat_A, rotation_2x2)
        else:
            mat_A = mat_A_next
        mat_C = torch.einsum("tbk,kij->bij", weight, ssm.mat_C_K)

        a_observed = as_[t]

        weight_next = ssm.weight_model(a_observed.unsqueeze(0))
        mixture_mat_A_next = torch.einsum("tbk,kij->bij", weight_next, ssm.mat_A_K)
        mat_A_next = _effective_mat_A(mixture_mat_A_next, rotation_2x2)

        mat_As_list.append(mat_A)
        mat_Cs_list.append(mat_C)

        # ---- Kalman gain + correction: IDENTICAL to the vendored kalman_filter, no changes ----
        K_t = (
            cov_t_plus @ mat_C.transpose(1, 2)
            @ torch.inverse(mat_C @ cov_t_plus @ mat_C.transpose(1, 2) + ssm.mat_R)
        )
        mean_t = mean_t_plus + torch.bmm(
            K_t, (as_[t].unsqueeze(-1) - torch.bmm(mat_C, mean_t_plus.unsqueeze(-1)))
        ).squeeze(-1)
        cov_t = cov_t_plus - K_t @ mat_C @ cov_t_plus
        if symmetrize_covariance:
            cov_t = (cov_t + cov_t.transpose(1, 2)) / 2.0

        # ---- predict step: rotation-block mat_A_next for phase, mixture mat_A_next for the rest ----
        mean_t_plus = torch.bmm(mat_A_next, mean_t.unsqueeze(-1)).squeeze(-1)
        cov_t_plus = mat_A_next @ cov_t @ mat_A_next.transpose(1, 2) + ssm.mat_Q
        if symmetrize_covariance:
            cov_t_plus = (cov_t_plus + cov_t_plus.transpose(1, 2)) / 2.0

        means.append(mean_t)
        covariances.append(cov_t)
        next_means.append(mean_t_plus)
        next_covariances.append(cov_t_plus)

    mixture_mat_C_next = torch.einsum("tbk,kij->bij", weight_next, ssm.mat_C_K)
    mat_As_list.append(mat_A_next)
    mat_Cs_list.append(mixture_mat_C_next)

    return (
        torch.stack(means), torch.stack(covariances), torch.stack(next_means), torch.stack(next_covariances),
        torch.stack(mat_As_list), torch.stack(mat_Cs_list),
    )


def phase_kalman_smooth(model: KalmanVAEBarPointerPhase, filter_means, filter_covariances, filter_next_means,
                        filter_next_covariances, mat_As, mat_Cs, sample_control: SampleControl,
                        symmetrize_covariance: bool = True):
    """Fork of StateSpaceModel.kalman_smooth -- IDENTICAL math to the vendored version (the smoother's own
    recursion only uses mat_As and the filter's means/covariances, never mat_Q/mat_R/mat_C directly except
    in the observation-distribution `a_distrib` calls, which are not used by kvae_elbo's a/z terms but are
    kept here for parity/diagnostics). This fork exists only because the vendored kalman_smooth's signature
    doesn't accept mat_As/mat_Cs produced OUTSIDE its own kalman_filter -- the math itself is unchanged."""
    ssm = model.ssm
    sequence_length = filter_means.shape[0]
    batch_size = filter_means.shape[1]
    z_dim = ssm.z_dim

    means = [filter_means[-1]]
    covariances = [filter_covariances[-1]]

    z_distrib = D.MultivariateNormal(filter_means[-1].view(-1, z_dim), filter_covariances[-1])
    z = z_distrib.rsample() if sample_control.state_transition == "sample" else z_distrib.mean

    a_distrib = D.MultivariateNormal(torch.bmm(mat_Cs[-1], z.unsqueeze(-1)).squeeze(-1), ssm.mat_R)
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

        a_distrib = D.MultivariateNormal(torch.bmm(mat_Cs[t], z.unsqueeze(-1)).squeeze(-1), ssm.mat_R)
        a = a_distrib.rsample() if sample_control.observation == "sample" else a_distrib.mean

        zs_list.insert(0, z)
        as_list.insert(0, a)
        means.insert(0, mean_t)
        covariances.insert(0, cov_t)

    return torch.stack(means), torch.stack(covariances), torch.stack(zs_list), torch.stack(as_list)


def phase_kvae_elbo(model: KalmanVAEBarPointerPhase, features_time_major: torch.Tensor,
                    beat_targets_time_major: torch.Tensor, downbeat_targets_time_major: torch.Tensor,
                    sample_control: SampleControl, sawtooth_weight: float = 0.5,
                    reconstruction_weight: float = 0.3, regularization_weight: float = 1.0,
                    kalman_weight: float = 1.0):
    """The KVAE ELBO (identical structure/weights to model.kalman_vae_bar_pointer.kvae_elbo -- see that
    module's docstring for the 0.3/1.0/1.0 provenance) PLUS the sawtooth phase-supervision term, added the
    same way losses.py's compute_loss adds it to the base VBPM model: lambda * sum_t (1 - cos(phi_t -
    phi_gt_t)) * mask_t, using data/targets.py's build_sawtooth_phase_target_batch verbatim.

    features_time_major/beat_targets_time_major/downbeat_targets_time_major: [num_frames, batch, ...].
    Returns (elbo_to_maximize_INCLUDING_sawtooth, smoothed_z_sample, info_dict).

    NOTE: unlike the base kvae_elbo, the returned scalar is elbo MINUS the sawtooth loss already folded in
    (i.e. it is directly `-loss` in the training script's sign convention) -- see the info dict's separate
    "sawtooth" entry if the pre-sawtooth ELBO value specifically is needed for comparison against the base
    model's logged elbo.
    """
    from data.targets import build_sawtooth_phase_target_batch   # local import: avoids a hard dependency
                                                                   # for callers that only need the filter fork

    num_frames, batch_size, feature_dim = features_time_major.shape
    a_distribution = model.encoder(features_time_major.reshape(-1, feature_dim))
    a = a_distribution.rsample().view(num_frames, batch_size, model.a_dim)

    # tempo is COMPUTED (autocorrelation over the whole clip's raw features), not a free latent -- see
    # module docstring point 2/3. Detached: trained by its own CE loss below, not by ELBO/sawtooth backprop
    # (matches AutocorrelationTempoHead.bar_phase_advance's own contract and the archived A+B design).
    phidot_per_clip, tempo_lag_scores = model.tempo_head.bar_phase_advance(
        features_time_major.transpose(0, 1), model.beats_per_bar)   # tempo_head expects [batch, T, feature_dim]

    reconstruction = model.decoder(a.view(-1, model.a_dim)).log_prob(features_time_major.reshape(-1, feature_dim))
    reconstruction = reconstruction.view(num_frames, batch_size, feature_dim).sum(-1).mean()
    regularization = a_distribution.log_prob(a.view(-1, model.a_dim)).view(num_frames, batch_size, model.a_dim).sum(-1).mean()

    filter_means, filter_covs, next_means, next_covs, mat_As, mat_Cs = phase_kalman_filter(
        model, a, phidot_per_clip, sample_control)
    smooth_means, smooth_covs, z_smoothed, _ = phase_kalman_smooth(
        model, filter_means, filter_covs, next_means, next_covs, mat_As, mat_Cs, sample_control)

    z_distribution = D.MultivariateNormal(smooth_means.view(num_frames, batch_size, model.z_dim),
                                          smooth_covs.view(num_frames, batch_size, model.z_dim, model.z_dim))
    z = z_distribution.rsample()

    kalman_observation = D.MultivariateNormal(
        (mat_Cs[:-1] @ z.unsqueeze(-1)).view(-1, model.a_dim), model.ssm.mat_R
    ).log_prob(a.view(-1, model.a_dim)).view(num_frames, batch_size).mean()

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

    kalman_posterior_entropy = z_distribution.log_prob(z).mean()

    elbo = (reconstruction_weight * reconstruction - regularization_weight * regularization
           + kalman_weight * (kalman_observation + kalman_transition - kalman_posterior_entropy))

    # ---- sawtooth phase supervision on the RECOVERED angle (atan2 of z's phase sub-block) ----
    beat_targets_batch_major = beat_targets_time_major.transpose(0, 1)          # build_sawtooth_* wants [batch, T]
    downbeat_targets_batch_major = downbeat_targets_time_major.transpose(0, 1)
    phase_target, valid_mask = build_sawtooth_phase_target_batch(
        beat_targets_batch_major, downbeat_targets_batch_major, model.beats_per_bar)   # [batch, T]
    phase_target = phase_target.transpose(0, 1)          # -> [T, batch], matches z's time-major layout
    valid_mask = valid_mask.transpose(0, 1)

    recovered_phase = phase_angle_from_z(z[..., :PHASE_DIMS])                    # [T, batch]
    per_frame_sawtooth = (1.0 - torch.cos(recovered_phase - phase_target)) * valid_mask
    sawtooth_loss = per_frame_sawtooth.sum(dim=0) / valid_mask.sum(dim=0).clamp(min=1.0)   # [batch]

    # ---- tempo head's own cross-entropy against GT period (trained separately, not via ELBO backprop) ----
    tempo_loss = _autocorr_cross_entropy(tempo_lag_scores, beat_targets_batch_major)

    total_loss = -elbo + sawtooth_weight * num_frames * sawtooth_loss.mean() + tempo_loss

    info = {
        "recon": float(reconstruction), "reg": float(regularization),
        "kal_obs": float(kalman_observation), "kal_trans": float(kalman_transition),
        "kal_post": float(kalman_posterior_entropy), "elbo": float(elbo),
        "sawtooth": float(sawtooth_loss.mean()), "tempo_ce": float(tempo_loss),
        "phidot_mean": float(phidot_per_clip.mean()), "phidot_std": float(phidot_per_clip.std()),
    }
    return -total_loss, z, info    # return NEGATIVE total_loss so callers can keep using `loss = -elbo_like_value`


def _autocorr_cross_entropy(tempo_lag_scores: torch.Tensor, beat_targets_batch_major: torch.Tensor) -> torch.Tensor:
    """Same construction as losses.py's _autocorr_target_lag_indices + its tempo_ce loss (duplicated here,
    not imported, because losses.py's version is coupled to the base BarPointerVAE's RolloutResult type) --
    for each clip, the index (into AUTOCORR_LAG_FRAMES) of the lag nearest the GT beat period, trained by
    cross-entropy against the tempo head's own lag-score logits."""
    import numpy as np
    from model.divergences import AUTOCORR_LAG_FRAMES

    beat_numpy = beat_targets_batch_major.detach().cpu().numpy()
    target_indices, valid = [], []
    for example in beat_numpy:
        beat_frames = np.where(example > 0.5)[0]
        if len(beat_frames) < 2:
            target_indices.append(0)
            valid.append(0.0)
        else:
            period = float(np.median(np.diff(beat_frames)))
            target_indices.append(int(np.argmin(np.abs(AUTOCORR_LAG_FRAMES - period))))
            valid.append(1.0)
    device = beat_targets_batch_major.device
    target_indices_t = torch.tensor(target_indices, device=device)
    valid_t = torch.tensor(valid, device=device)
    cross_entropy = F.cross_entropy(tempo_lag_scores, target_indices_t, reduction="none") * valid_t
    return cross_entropy.sum() / valid_t.sum().clamp(min=1.0)
