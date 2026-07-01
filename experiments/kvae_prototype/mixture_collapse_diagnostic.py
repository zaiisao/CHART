"""TASK B: K=5 dynamics-mixture collapse diagnostic (SMC vs in-domain).

Adapted from arXiv:2605.12287 Section 5.4.6: no single GLOBAL DBN smoothing parameter serves both easy
and hard tracks -- clean-activation tracks want minimal smoothing, degraded-activation tracks want heavy
smoothing, so a fixed value costs real F either way. This reframes the just-completed Q_reg sweep (a
single GLOBAL process-noise floor, which only made things WORSE as it increased -- see qreg_sweep.py's
results, gap widened 0.241->0.257): raising a single global knob is the wrong lever. The mixture-of-K=5
locally-linear dynamics (StateSpaceModel's mat_A_K/mat_C_K, blended per-frame by weight_model's LSTM
softmax -- third_party/kalman-vae/kvae/dynamics_parameter_network.py) is SUPPOSED to already provide a
per-frame-adaptive alternative to one global smoothing constant. This script checks whether that mixture
is actually doing its job on SMC, or has collapsed to one dominant regime that doesn't fit SMC's tempo
variability.

For SMC (217 songs) and bt_val_rich (40 songs), instruments the deploy pass (fast_deploy's
batched_kalman_filter_deploy_with_weights -- verified bit-identical to the already-validated
batched_kalman_filter_deploy on filtered means) to record the per-frame softmax weight over K=5:
  (a) mean entropy of the K=5 weight distribution per frame, SMC vs in-domain (low entropy = collapsed
      to ~1 dominant component; max possible entropy for K=5 is ln(5)=1.609 nats).
  (b) how many of the K=5 components ever exceed 10% weight on a given song (song-level "components used"
      count), SMC vs in-domain.
  (c) whether SMC's lowest-F songs show LOWER entropy (more collapse) than SMC's highest-F songs --
      correlation between per-song mean entropy and per-song beat F.

Usage:
    python experiments/kvae_prototype/mixture_collapse_diagnostic.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FRAMES_PER_SECOND
from data.dataset import load_songs, Song
from data.targets import ground_truth_beat_times
from model import readout
from model.kalman_vae_bar_pointer import KalmanVAEBarPointer
from extract_ood_features import FRAMES_PER_SECOND as SMC_FPS
from eval_smc_ood_fresh import build_smc_index, load_smc_songs, FEATURE_CACHE_DIR as SMC_FEATURE_CACHE_DIR
from fast_deploy import batched_kalman_filter_deploy_with_weights


def entropy_nats(weights: np.ndarray) -> np.ndarray:
    """weights [..., K] (each row sums to 1) -> entropy [...] in nats."""
    return -np.sum(weights * np.log(weights + 1e-12), axis=-1)


@torch.no_grad()
def instrument_songs(model: KalmanVAEBarPointer, song_specs: list, device: str, frames_per_second: float,
                     tolerance_seconds: float, batch_chunk_size: int = 32) -> list[dict]:
    """song_specs: list of (tid, features [T,feature_dim], reference_beats np.ndarray).
    Returns per-song dicts with f_measure, mean_entropy, components_used (>10% weight at any frame)."""
    model.eval()
    rows = []
    num_songs = len(song_specs)
    K = model.ssm.K

    for chunk_start in range(0, num_songs, batch_chunk_size):
        chunk = song_specs[chunk_start:chunk_start + batch_chunk_size]
        lengths = [features.shape[0] for _, features, _ in chunk]
        T_max = max(lengths)

        a_means = []
        for _, features, _ in chunk:
            a_mean = model.encoder(features.to(device)).mean
            padded = torch.zeros(T_max, model.a_dim, device=device, dtype=a_mean.dtype)
            padded[:a_mean.shape[0]] = a_mean
            a_means.append(padded)
        a_padded = torch.stack(a_means, dim=1)
        lengths_tensor = torch.tensor(lengths, device=device)

        filtered_means, mixture_weights = batched_kalman_filter_deploy_with_weights(model.ssm, a_padded, lengths_tensor)
        probabilities = torch.sigmoid(model.head(filtered_means)).cpu().numpy()
        mixture_weights_np = mixture_weights.cpu().numpy()   # [T_max, chunk_N, K]

        for local_position, (tid, features, reference_beats) in enumerate(chunk):
            num_frames = lengths[local_position]
            probability = probabilities[:num_frames, local_position]
            song_weights = mixture_weights_np[:num_frames, local_position]        # [T_i, K]

            estimated_beats = readout.peak_pick_times(probability[:, 0], frames_per_second)
            f_measure = readout.f_measure(reference_beats, estimated_beats, tolerance_seconds) if len(reference_beats) >= 2 else float("nan")

            frame_entropy = entropy_nats(song_weights)                            # [T_i]
            components_used = int(np.sum(song_weights.max(axis=0) > 0.10))        # how many of K ever exceed 10%
            dominant_component_mean_weight = float(song_weights.mean(axis=0).max())

            rows.append({
                "tid": tid, "f_measure": f_measure, "mean_entropy": float(frame_entropy.mean()),
                "components_used": components_used, "dominant_component_mean_weight": dominant_component_mean_weight,
            })
    return rows


def summarize(label: str, rows: list[dict]) -> None:
    entropies = np.array([r["mean_entropy"] for r in rows])
    components = np.array([r["components_used"] for r in rows])
    max_entropy = np.log(5)
    print(f"{label}: n={len(rows)}", flush=True)
    print(f"  mean entropy per song: mean={entropies.mean():.3f} median={np.median(entropies):.3f} "
         f"(max possible = ln(5) = {max_entropy:.3f} nats, {100*entropies.mean()/max_entropy:.0f}% of max)", flush=True)
    print(f"  components used (>10% weight, out of K=5): mean={components.mean():.2f} median={np.median(components):.1f} "
         f"| distribution: {np.bincount(components, minlength=6)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default="experiments/kvae_prototype/kvae_m1_repro_400ep1000.pt")
    parser.add_argument("--val_feature_dir", type=str, default="cache/acts/bt_val_rich")
    parser.add_argument("--num_val_songs", type=int, default=40)
    parser.add_argument("--eval_beat_tolerance_seconds", type=float, default=0.07)
    parser.add_argument("--low_f_threshold", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    model = KalmanVAEBarPointer(
        feature_dim=512, a_dim=ckpt_args.get("a_dim", 8), z_dim=ckpt_args.get("z_dim", 8), K=ckpt_args.get("K", 5),
        Q_reg=ckpt_args.get("Q_reg", 1e-3),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"[mixture_collapse] loaded {args.checkpoint}", flush=True)

    # ---- in-domain ----
    val_songs = load_songs(args.val_feature_dir, args.num_val_songs, seed=2)
    val_specs = [(f"val_{i}", song.features[:1600], ground_truth_beat_times(song.beat_targets.numpy()[:1600], FRAMES_PER_SECOND))
                for i, song in enumerate(val_songs)]
    in_domain_rows = instrument_songs(model, val_specs, device, FRAMES_PER_SECOND, args.eval_beat_tolerance_seconds)

    # ---- SMC ----
    smc_index = build_smc_index()
    smc_songs = load_smc_songs(smc_index, SMC_FEATURE_CACHE_DIR)
    smc_rows = instrument_songs(model, smc_songs, device, SMC_FPS, args.eval_beat_tolerance_seconds)

    print(f"\n[mixture_collapse] in-domain mean F = {np.nanmean([r['f_measure'] for r in in_domain_rows]):.3f} "
         f"(sanity check vs 0.895)", flush=True)
    print(f"[mixture_collapse] SMC mean F = {np.nanmean([r['f_measure'] for r in smc_rows]):.3f} "
         f"(sanity check vs 0.654)", flush=True)

    print("\n=== (a)+(b): mixture entropy / components-used, SMC vs in-domain ===", flush=True)
    summarize("in-domain (bt_val_rich)", in_domain_rows)
    summarize("SMC (OOD)", smc_rows)

    print("\n=== (c): does SMC's low-F subset show MORE collapse (lower entropy) than its high-F subset? ===", flush=True)
    smc_valid = [r for r in smc_rows if not np.isnan(r["f_measure"])]
    low_f = [r for r in smc_valid if r["f_measure"] <= args.low_f_threshold]
    high_f = [r for r in smc_valid if r["f_measure"] > args.low_f_threshold]
    summarize(f"  SMC LOW-F (<={args.low_f_threshold})", low_f)
    summarize(f"  SMC HIGH-F (>{args.low_f_threshold})", high_f)

    f_values = np.array([r["f_measure"] for r in smc_valid])
    entropy_values = np.array([r["mean_entropy"] for r in smc_valid])
    correlation = float(np.corrcoef(f_values, entropy_values)[0, 1])
    print(f"\n  Pearson correlation(SMC per-song F, SMC per-song mean entropy) = {correlation:.3f}", flush=True)
    print("  (positive => higher-entropy/less-collapsed songs tend to score higher F; "
         "near-zero/negative => entropy/collapse doesn't explain the SMC F variance)", flush=True)


if __name__ == "__main__":
    main()
