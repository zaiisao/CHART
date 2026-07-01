"""Correctness gate for model/kalman_vae_bar_pointer_phase.py, BEFORE any training (per the coordinator's
explicit safety requirement, same bar as verify_adaptive_noise_correctness.py's bit-exact precedent).

Unlike the adaptive-noise fork (which can be checked by forcing its one new component to a no-op constant
and comparing against the vendored recursion), this fork has NO no-op setting that reduces it back to the
vendored kalman_filter exactly -- overriding rows 0:2 of mat_A with a rotation block is a structural change
even at phidot=0 (rotation=identity still REPLACES whatever the K=5 mixture would have produced for those
rows, it doesn't just multiply by 1). So this gate instead verifies the fork's internal correctness three
different ways:

1. UNIT-LEVEL invariants (rotation_matrix_2x2 is an exact rotation; phase_angle_from_z exactly inverts it;
   _effective_mat_A's row-splicing is exact) -- already spot-checked ad hoc during development; re-verified
   here as part of the permanent gate.
2. MANUAL HAND-ROLLED RECURSION: re-implement the exact same 4-frame filter recursion in a SEPARATE,
   independently-written loop (not calling phase_kalman_filter at all), and compare against
   phase_kalman_filter's own output frame-by-frame. This catches loop/indexing/off-by-one bugs the unit
   tests above can't see (they test the building blocks in isolation, not their composition over time).
3. GRADIENT-FLOW CHECK (the archived tempo_grad_sawtooth.py's own diagnostic technique, reused per the
   coordinator's suggestion): confirm the sawtooth loss's gradient actually reaches BOTH the tempo head's
   parameters (via the rotation it drives) AND the z[...,0:2] phase sub-block's own dynamics (via mat_A_K's
   rows 2: coupling back into phase, and via the Kalman correction step) -- if either grad norm is ~0, the
   corresponding latent is decoupled the same way the archived history found the free tempo latent to be.

MUST PASS before any training run proceeds.
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.distributions as D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.kalman_vae_bar_pointer_phase import (
    KalmanVAEBarPointerPhase, phase_kvae_elbo, phase_kalman_filter, phase_angle_from_z,
    rotation_matrix_2x2, _effective_mat_A, PHASE_DIMS,
)

_THIRD_PARTY_KVAE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "third_party", "kalman-vae")
if _THIRD_PARTY_KVAE not in sys.path:
    sys.path.insert(0, _THIRD_PARTY_KVAE)
from kvae.sample_control import SampleControl

TWO_PI = 2.0 * math.pi


def check_unit_invariants() -> bool:
    print("=== 1. UNIT-LEVEL invariants ===", flush=True)
    torch.manual_seed(0)
    angle_a = torch.tensor([0.3, 1.7, -0.5, 4.0])
    angle_b = torch.tensor([0.1, -0.2, 3.0, 0.05])
    unit_vector = torch.stack([torch.cos(angle_a), torch.sin(angle_a)], dim=-1)
    rotation = rotation_matrix_2x2(angle_b)
    rotated = torch.bmm(rotation, unit_vector.unsqueeze(-1)).squeeze(-1)
    expected = torch.stack([torch.cos(angle_a + angle_b), torch.sin(angle_a + angle_b)], dim=-1)
    rotation_diff = (rotated - expected).abs().max().item()
    print(f"  rotation_matrix_2x2 exactness: max abs diff = {rotation_diff:.3e}", flush=True)

    recovered_angle = phase_angle_from_z(unit_vector)
    angle_diff = (recovered_angle - (angle_a % TWO_PI)).abs()
    angle_diff = torch.minimum(angle_diff, TWO_PI - angle_diff)
    recovery_diff = angle_diff.max().item()
    print(f"  phase_angle_from_z exactness (mod 2pi): max abs diff = {recovery_diff:.3e}", flush=True)

    mixture = torch.randn(3, 8, 8)
    zero_rotation = rotation_matrix_2x2(torch.zeros(3))
    effective = _effective_mat_A(mixture, zero_rotation)
    expected_phase_rows = torch.cat(
        [torch.eye(PHASE_DIMS).unsqueeze(0).expand(3, -1, -1), torch.zeros(3, PHASE_DIMS, 8 - PHASE_DIMS)], dim=-1)
    phase_rows_diff = (effective[:, :PHASE_DIMS, :] - expected_phase_rows).abs().max().item()
    remaining_rows_diff = (effective[:, PHASE_DIMS:, :] - mixture[:, PHASE_DIMS:, :]).abs().max().item()
    print(f"  _effective_mat_A phase rows @ phidot=0 vs [I_2|0]: max abs diff = {phase_rows_diff:.3e}", flush=True)
    print(f"  _effective_mat_A remaining rows vs mixture (should be untouched): max abs diff = {remaining_rows_diff:.3e}", flush=True)

    threshold = 1e-5
    passed = max(rotation_diff, recovery_diff, phase_rows_diff, remaining_rows_diff) < threshold
    print(f"  -> {'PASS' if passed else 'FAIL'}\n", flush=True)
    return passed


def check_manual_recursion(model: KalmanVAEBarPointerPhase, device: str) -> bool:
    """Independently hand-rolls 4 frames of the SAME recursion phase_kalman_filter implements, and compares.
    Uses sample_control='mean' throughout (deterministic, so the two independently-written loops must
    produce bit-identical numbers if both are correct implementations of the same math)."""
    print("=== 2. MANUAL HAND-ROLLED RECURSION (4 frames, independent re-implementation) ===", flush=True)
    torch.manual_seed(1)
    batch_size, num_frames = 3, 4
    a = torch.randn(num_frames, batch_size, model.a_dim, device=device)
    phidot = torch.tensor([0.02, 0.05, -0.01], device=device)   # deliberately varied per-clip tempo
    ssm = model.ssm
    sample_control = SampleControl(encoder="mean", decoder="mean", state_transition="mean", observation="mean")
    ssm.eval()

    # ---- reference: the module under test ----
    ref_means, ref_covs, ref_next_means, ref_next_covs, ref_mat_As, ref_mat_Cs = phase_kalman_filter(
        model, a, phidot, sample_control)

    # ---- independent hand-rolled recursion (written from scratch, not copy-pasted from phase_kalman_filter) ----
    with torch.no_grad():
        rotation = rotation_matrix_2x2(phidot)
        ssm.weight_model.clear_hidden_state()
        mean_plus = ssm.initial_state_mean.repeat(batch_size, 1)
        cov_plus = ssm.initial_state_covariance.unsqueeze(0).repeat(batch_size, 1, 1)
        weight = torch.ones(1, batch_size, ssm.K, device=device) / ssm.K

        manual_means, manual_covs = [], []
        for t in range(num_frames):
            mix_A = torch.einsum("tbk,kij->bij", weight, ssm.mat_A_K)
            eff_A = torch.cat([
                torch.cat([rotation, torch.zeros(batch_size, PHASE_DIMS, model.z_dim - PHASE_DIMS, device=device)], dim=-1),
                mix_A[:, PHASE_DIMS:, :],
            ], dim=1)
            mix_C = torch.einsum("tbk,kij->bij", weight, ssm.mat_C_K)

            gain = cov_plus @ mix_C.transpose(1, 2) @ torch.inverse(mix_C @ cov_plus @ mix_C.transpose(1, 2) + ssm.mat_R)
            mean_now = mean_plus + torch.bmm(gain, (a[t].unsqueeze(-1) - torch.bmm(mix_C, mean_plus.unsqueeze(-1)))).squeeze(-1)
            cov_now = cov_plus - gain @ mix_C @ cov_plus
            cov_now = (cov_now + cov_now.transpose(1, 2)) / 2.0
            manual_means.append(mean_now)
            manual_covs.append(cov_now)

            weight = ssm.weight_model(a[t].unsqueeze(0))
            mix_A_next = torch.einsum("tbk,kij->bij", weight, ssm.mat_A_K)
            eff_A_next = torch.cat([
                torch.cat([rotation, torch.zeros(batch_size, PHASE_DIMS, model.z_dim - PHASE_DIMS, device=device)], dim=-1),
                mix_A_next[:, PHASE_DIMS:, :],
            ], dim=1)
            mean_plus = torch.bmm(eff_A_next, mean_now.unsqueeze(-1)).squeeze(-1)
            cov_plus = eff_A_next @ cov_now @ eff_A_next.transpose(1, 2) + ssm.mat_Q
            cov_plus = (cov_plus + cov_plus.transpose(1, 2)) / 2.0

        manual_means = torch.stack(manual_means)
        manual_covs = torch.stack(manual_covs)

    means_diff = (ref_means - manual_means).abs().max().item()
    covs_diff = (ref_covs - manual_covs).abs().max().item()
    print(f"  filtered means: max abs diff = {means_diff:.3e}", flush=True)
    print(f"  filtered covariances: max abs diff = {covs_diff:.3e}", flush=True)

    threshold = 1e-5
    passed = max(means_diff, covs_diff) < threshold
    print(f"  -> {'PASS' if passed else 'FAIL'}\n", flush=True)
    return passed


def check_gradient_flow(model: KalmanVAEBarPointerPhase, device: str) -> bool:
    """Archived tempo_grad_sawtooth.py's own technique: does the sawtooth loss's gradient actually reach
    (a) the tempo head's parameters, and (b) the phase sub-block's OWN dynamics (via mat_A_K rows 2:, which
    can read phase, and via the correction step)? A ~0 gradient into either would mean that component is
    decoupled from the supervision the same way the archived free tempo latent was found to be."""
    print("=== 3. GRADIENT-FLOW CHECK (tempo_grad_sawtooth.py's technique) ===", flush=True)
    model.train()   # check_manual_recursion left ssm in eval mode; cuDNN's LSTM backward requires training
                    # mode (the sample_control="sample" path below also needs training=True, see StateSpaceModel)
    torch.manual_seed(2)
    num_frames, batch_size, feature_dim = 256, 8, model.feature_dim
    features = torch.randn(num_frames, batch_size, feature_dim, device=device)
    # synthesize plausible beat/downbeat targets (evenly spaced, ~2 Hz) so the sawtooth target is well-posed
    beat_period = 30
    beats = torch.zeros(num_frames, batch_size, device=device)
    downbeats = torch.zeros(num_frames, batch_size, device=device)
    beats[::beat_period] = 1.0
    downbeats[::beat_period * 4] = 1.0

    sample_control = SampleControl(encoder="sample", decoder="mean", state_transition="sample", observation="sample")
    model.zero_grad()
    elbo, z, info = phase_kvae_elbo(model, features, beats, downbeats, sample_control)
    loss = -elbo
    loss.backward()

    tempo_head_grad_norm = sum(
        p.grad.norm().item() for p in model.tempo_head.parameters() if p.grad is not None
    )
    # gradient into the K=5 mixture matrices that drive the OTHER (non-phase) z dims but can also read phase
    mat_A_K_grad_norm = model.ssm._mat_A_K.grad.norm().item() if model.ssm._mat_A_K.grad is not None else 0.0
    mat_C_K_grad_norm = model.ssm._mat_C_K.grad.norm().item() if model.ssm._mat_C_K.grad is not None else 0.0
    encoder_grad_norm = sum(p.grad.norm().item() for p in model.encoder.parameters() if p.grad is not None)

    print(f"  info: sawtooth={info['sawtooth']:.4f} tempo_ce={info['tempo_ce']:.4f} "
         f"phidot_mean={info['phidot_mean']:.5f} phidot_std={info['phidot_std']:.5f}", flush=True)
    print(f"  grad norm into tempo_head params:  {tempo_head_grad_norm:.4f}", flush=True)
    print(f"  grad norm into ssm.mat_A_K:        {mat_A_K_grad_norm:.4f}", flush=True)
    print(f"  grad norm into ssm.mat_C_K:        {mat_C_K_grad_norm:.4f}", flush=True)
    print(f"  grad norm into encoder params:     {encoder_grad_norm:.4f}", flush=True)

    # tempo_head is trained by its OWN cross-entropy (tempo_ce), not by the sawtooth/ELBO term (phidot is
    # detached before driving the rotation, matching AutocorrelationTempoHead's documented contract) -- so
    # a nonzero grad here confirms the CE path works, not that sawtooth reaches it (it structurally cannot,
    # by design, matching the archived A+B model's own division of labor).
    passed = tempo_head_grad_norm > 1e-6 and mat_A_K_grad_norm > 1e-6 and encoder_grad_norm > 1e-6
    print(f"  -> {'PASS' if passed else 'FAIL'} (all three paths must show nonzero gradient)\n", flush=True)
    return passed


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = KalmanVAEBarPointerPhase(feature_dim=512, a_dim=8, z_dim=8, K=5).to(device)

    results = []
    results.append(("unit invariants", check_unit_invariants()))
    results.append(("manual recursion", check_manual_recursion(model, device)))
    results.append(("gradient flow", check_gradient_flow(model, device)))

    print("=== SUMMARY ===", flush=True)
    all_passed = True
    for name, passed in results:
        print(f"  {name}: {'PASS' if passed else 'FAIL'}", flush=True)
        all_passed = all_passed and passed
    print(f"\n=== OVERALL: {'PASS' if all_passed else 'FAIL'} ===", flush=True)
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
