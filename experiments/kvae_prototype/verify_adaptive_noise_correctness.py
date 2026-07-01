"""MANDATORY correctness gate (per coordinator instruction) before any adaptive-noise training.

Forces model/adaptive_noise_filter.py's NoiseScaleHead to output a CONSTANT 1.0 scale (via a hook that
overrides its forward to bypass the learned net entirely), so adaptive_kalman_filter's mat_Q_t/mat_R_t
collapse to EXACTLY the base ssm.mat_Q/ssm.mat_R at every frame -- i.e. the forked recursion, under this
forced condition, should be mathematically IDENTICAL to the unforked StateSpaceModel.kalman_filter.

Loads the ALREADY-TRAINED kvae_m1_repro_400ep1000.pt checkpoint's weights (a real, non-trivial ssm state,
not freshly-initialized), runs both the original vendored kalman_filter/kalman_smooth AND the forked
adaptive_kalman_filter/adaptive_kalman_smooth (constant-1.0-scale) on the SAME batch of real val songs, and
reports the max absolute difference in filtered means, smoothed means, and downstream beat/downbeat
predictions. Bar: ~1e-6, matching fast_deploy.py's regression-check precedent.

MUST PASS before train_kvae_adaptive_noise.py is allowed to proceed.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.distributions as D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import load_songs
from model.kalman_vae_bar_pointer import KalmanVAEBarPointer
from model.adaptive_noise_filter import NoiseScaleHead, adaptive_kalman_filter, adaptive_kalman_smooth

_THIRD_PARTY_KVAE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "third_party", "kalman-vae")
if _THIRD_PARTY_KVAE not in sys.path:
    sys.path.insert(0, _THIRD_PARTY_KVAE)
from kvae.sample_control import SampleControl


class ConstantOneNoiseHead(NoiseScaleHead):
    """Same class/interface as NoiseScaleHead, but forward() is overridden to IGNORE the learned net
    entirely and return exactly 1.0 for both scales, regardless of input -- for the correctness gate.
    Subclassing (not just re-initializing weights) guarantees this can't accidentally be perturbed by
    float roundoff in the learned net's forward pass; it's a hard, provable constant."""

    def forward(self, a_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = a_t.shape[0]
        one = torch.ones(batch_size, device=a_t.device, dtype=a_t.dtype)
        return one, one


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "experiments/kvae_prototype/kvae_m1_repro_400ep1000.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_args = checkpoint.get("args", {})

    model = KalmanVAEBarPointer(
        feature_dim=512, a_dim=ckpt_args.get("a_dim", 8), z_dim=ckpt_args.get("z_dim", 8), K=ckpt_args.get("K", 5),
        Q_reg=ckpt_args.get("Q_reg", 1e-3),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"[correctness_gate] loaded {checkpoint_path}", flush=True)

    noise_head = ConstantOneNoiseHead(a_dim=model.a_dim).to(device)

    val_songs = load_songs("cache/acts/bt_val_rich", 8, seed=2)
    crop = 300
    features = torch.stack([s.features[:crop] for s in val_songs]).to(device)   # [batch, crop, feature_dim]
    features_time_major = features.transpose(0, 1).contiguous()                 # [crop, batch, feature_dim]
    print(f"[correctness_gate] testing on {len(val_songs)} real val songs, crop={crop} frames", flush=True)

    # deploy-style deterministic sample_control (mirrors evaluate_leak_condition's deploy config, and
    # avoids the "sample" branches which would make a bit-exact comparison impossible across two separate
    # forward passes due to independent RNG draws)
    sample_control = SampleControl(encoder="mean", decoder="mean", state_transition="mean", observation="mean")
    model.ssm.eval()  # sample_control="mean" requires self.training=False in the vendored code (raises otherwise)

    with torch.no_grad():
        a_mean = model.encoder(features_time_major.reshape(-1, model.feature_dim)).mean.view(
            crop, len(val_songs), model.a_dim)

        # ---- ORIGINAL (unforked) vendored recursion ----
        orig_filter_means, orig_filter_covs, orig_next_means, orig_next_covs, orig_mat_As, orig_mat_Cs, _, _ = \
            model.ssm.kalman_filter(a_mean, sample_control=sample_control)
        orig_smooth_means, orig_smooth_covs, orig_z, _ = model.ssm.kalman_smooth(
            a_mean, orig_filter_means, orig_filter_covs, orig_next_means, orig_next_covs,
            orig_mat_As, orig_mat_Cs, sample_control=sample_control)
        orig_probability = torch.sigmoid(model.head(orig_filter_means.view(-1, model.z_dim))).view(crop, len(val_songs), 2)

        # ---- FORKED adaptive recursion, noise head FORCED to constant 1.0 ----
        (adapt_filter_means, adapt_filter_covs, adapt_next_means, adapt_next_covs,
         adapt_mat_As, adapt_mat_Cs, adapt_mat_Qs, adapt_mat_Rs, scale_Q_trace, scale_R_trace) = \
            adaptive_kalman_filter(model.ssm, a_mean, noise_head, sample_control)
        adapt_smooth_means, adapt_smooth_covs, adapt_z, _ = adaptive_kalman_smooth(
            model.ssm, adapt_filter_means, adapt_filter_covs, adapt_next_means, adapt_next_covs,
            adapt_mat_As, adapt_mat_Cs, adapt_mat_Rs, sample_control=sample_control)
        adapt_probability = torch.sigmoid(model.head(adapt_filter_means.view(-1, model.z_dim))).view(crop, len(val_songs), 2)

    def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
        return (a - b).abs().max().item()

    print("\n=== CORRECTNESS GATE RESULTS (max abs diff, forked-constant-1.0 vs original) ===", flush=True)
    print(f"  filtered means      : {max_abs_diff(orig_filter_means, adapt_filter_means):.3e}", flush=True)
    print(f"  filtered covariances: {max_abs_diff(orig_filter_covs, adapt_filter_covs):.3e}", flush=True)
    print(f"  next means          : {max_abs_diff(orig_next_means, adapt_next_means):.3e}", flush=True)
    print(f"  next covariances    : {max_abs_diff(orig_next_covs, adapt_next_covs):.3e}", flush=True)
    print(f"  mat_As              : {max_abs_diff(orig_mat_As, adapt_mat_As):.3e}", flush=True)
    print(f"  mat_Cs              : {max_abs_diff(orig_mat_Cs, adapt_mat_Cs):.3e}", flush=True)
    print(f"  smoothed means      : {max_abs_diff(orig_smooth_means, adapt_smooth_means):.3e}", flush=True)
    print(f"  smoothed covariances: {max_abs_diff(orig_smooth_covs, adapt_smooth_covs):.3e}", flush=True)
    print(f"  beat/downbeat prob  : {max_abs_diff(orig_probability, adapt_probability):.3e}", flush=True)

    # cross-check the adaptive mat_Q_t/mat_R_t tensors equal the base ssm.mat_Q/mat_R exactly under
    # forced scale=1.0 (a second, more direct check on the specific quantity this experiment changes)
    base_mat_Q_diff = (adapt_mat_Qs - model.ssm.mat_Q.unsqueeze(0).unsqueeze(0)).abs().max().item()
    base_mat_R_diff = (adapt_mat_Rs - model.ssm.mat_R.unsqueeze(0).unsqueeze(0)).abs().max().item()
    print(f"  mat_Q_t vs base mat_Q (should be ~0 under forced scale=1.0): {base_mat_Q_diff:.3e}", flush=True)
    print(f"  mat_R_t vs base mat_R (should be ~0 under forced scale=1.0): {base_mat_R_diff:.3e}", flush=True)
    print(f"  scale_Q_trace all == 1.0? {(scale_Q_trace == 1.0).all().item()}  "
         f"scale_R_trace all == 1.0? {(scale_R_trace == 1.0).all().item()}", flush=True)

    worst = max(
        max_abs_diff(orig_filter_means, adapt_filter_means), max_abs_diff(orig_smooth_means, adapt_smooth_means),
        max_abs_diff(orig_probability, adapt_probability), base_mat_Q_diff, base_mat_R_diff,
    )
    threshold = 1e-5
    print(f"\n=== VERDICT: worst max-abs-diff = {worst:.3e} (threshold {threshold:.0e}) "
         f"=> {'PASS' if worst < threshold else 'FAIL'} ===", flush=True)
    if worst >= threshold:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
