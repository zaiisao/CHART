"""Train KalmanVAEBarPointerAdaptiveNoise (model/kalman_vae_bar_pointer_adaptive_noise.py) from scratch on
the ORIGINAL bt_train_rich pool (400 songs -- NOT the rubato-augmented pool, per the coordinator's
instruction to keep this a clean single-variable test of the adaptive-noise idea alone, independent of the
already-negative rubato-augmentation result).

Every other setting is IDENTICAL to kvae_m1_repro_400ep1000 (1000 steps, batch 16, crop 256, a_dim=z_dim=8,
K=5, same pos_weights/lr/Q_reg/seed) -- see train_kvae.py's main() for that baseline's exact defaults.

PREREQUISITE (already run and passed, not re-run here): experiments/kvae_prototype/
verify_adaptive_noise_correctness.py -- bit-exact (0.0e+00) match between the forked adaptive recursion
(noise head forced to constant scale=1.0) and the vendored StateSpaceModel.kalman_filter/kalman_smooth.

Usage:
    python experiments/kvae_prototype/train_kvae_adaptive_noise.py --save_path experiments/kvae_prototype/kvae_adaptive_noise.pt
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FRAMES_PER_SECOND
from data.dataset import load_songs, sample_training_batch, Song
from data.targets import ground_truth_beat_times
from model import readout
from model.kalman_vae_bar_pointer_adaptive_noise import KalmanVAEBarPointerAdaptiveNoise, adaptive_kvae_elbo
from fast_deploy import batched_adaptive_kalman_filter_deploy

_THIRD_PARTY_KVAE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                 "third_party", "kalman-vae")
if _THIRD_PARTY_KVAE not in sys.path:
    sys.path.insert(0, _THIRD_PARTY_KVAE)
from kvae.sample_control import SampleControl


def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def batch_time_major(songs: list[Song], crop_length_frames: int, batch_size: int, device: str):
    features, beats, downbeats = sample_training_batch(songs, crop_length_frames, batch_size, device)
    return features.transpose(0, 1).contiguous(), beats.transpose(0, 1).contiguous(), downbeats.transpose(0, 1).contiguous()


@torch.no_grad()
def evaluate_leak_condition_adaptive(model: KalmanVAEBarPointerAdaptiveNoise, songs: list[Song], device: str,
                                     audio_condition: str, eval_max_frames: int, tolerance_seconds: float,
                                     batch_chunk_size: int = 64) -> dict:
    """Same deploy semantics as train_kvae.py's evaluate_leak_condition_fast, using the adaptive-noise
    batched deploy path (fast_deploy.batched_adaptive_kalman_filter_deploy, verified 4.5e-8 vs the
    correctness-gated reference recursion)."""
    model.eval()
    beat_scores, downbeat_scores = [], []
    num_songs = len(songs)

    for chunk_start in range(0, num_songs, batch_chunk_size):
        chunk_songs = songs[chunk_start:chunk_start + batch_chunk_size]
        chunk_indices = list(range(chunk_start, chunk_start + len(chunk_songs)))

        cropped_features, lengths = [], []
        for local_index, song in zip(chunk_indices, chunk_songs):
            source_features = songs[(local_index + 1) % num_songs].features if audio_condition == "shuffle" else song.features
            num_frames = min(source_features.shape[0], song.beat_targets.shape[0], eval_max_frames)
            if audio_condition == "zero":
                cropped_features.append(torch.zeros(num_frames, model.feature_dim))
            else:
                cropped_features.append(source_features[:num_frames])
            lengths.append(num_frames)

        T_max = max(lengths)
        a_means = []
        for features, length in zip(cropped_features, lengths):
            a_mean = model.encoder(features.to(device)).mean
            padded = torch.zeros(T_max, model.a_dim, device=device, dtype=a_mean.dtype)
            padded[:length] = a_mean
            a_means.append(padded)
        a_padded = torch.stack(a_means, dim=1)
        lengths_tensor = torch.tensor(lengths, device=device)

        filtered_means, _, _ = batched_adaptive_kalman_filter_deploy(model.ssm, model.noise_head, a_padded, lengths_tensor)
        probabilities = torch.sigmoid(model.head(filtered_means)).cpu().numpy()

        for local_position, song in enumerate(chunk_songs):
            num_frames = lengths[local_position]
            probability = probabilities[:num_frames, local_position]
            reference_beats = ground_truth_beat_times(song.beat_targets.numpy()[:num_frames], FRAMES_PER_SECOND)
            reference_downbeats = ground_truth_beat_times(song.downbeat_targets.numpy()[:num_frames], FRAMES_PER_SECOND)
            if len(reference_beats) >= 2:
                estimated_beats = readout.peak_pick_times(probability[:, 0], FRAMES_PER_SECOND)
                beat_scores.append(readout.f_measure(reference_beats, estimated_beats, tolerance_seconds))
            if len(reference_downbeats) >= 2:
                estimated_downbeats = readout.peak_pick_times(probability[:, 1], FRAMES_PER_SECOND)
                downbeat_scores.append(readout.f_measure(reference_downbeats, estimated_downbeats, tolerance_seconds))

    model.train()
    mean = lambda values: float(np.nanmean(values)) if values else float("nan")
    return {"beat_f": mean(beat_scores), "downbeat_f": mean(downbeat_scores)}


def evaluate_with_leak_test_adaptive(model, songs, device, eval_max_frames, tolerance_seconds) -> dict:
    return {
        condition: evaluate_leak_condition_adaptive(model, songs, device, condition, eval_max_frames, tolerance_seconds)
        for condition in ("real", "shuffle", "zero")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train_feature_dir", type=str, default="cache/acts/bt_train_rich")
    parser.add_argument("--val_feature_dir", type=str, default="cache/acts/bt_val_rich")
    parser.add_argument("--num_train_songs", type=int, default=400)
    parser.add_argument("--num_val_songs", type=int, default=40)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--crop_length_frames", type=int, default=256)
    parser.add_argument("--a_dim", type=int, default=8)
    parser.add_argument("--z_dim", type=int, default=8)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--Q_reg", type=float, default=1e-3)
    # See train_kvae.py's identical defaults for provenance: inherited unchanged from the historical M1 run.
    parser.add_argument("--beat_loss_weight", type=float, default=5.0)
    parser.add_argument("--beat_pos_weight", type=float, default=8.0)
    parser.add_argument("--downbeat_pos_weight", type=float, default=20.0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--grad_clip_norm", type=float, default=5.0)
    parser.add_argument("--eval_max_frames", type=int, default=1600)
    parser.add_argument("--eval_beat_tolerance_seconds", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_path", type=str, default="experiments/kvae_prototype/kvae_adaptive_noise.pt")
    parser.add_argument("--eval_every", type=int, default=200)
    args = parser.parse_args()

    set_all_seeds(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    train_songs = load_songs(args.train_feature_dir, args.num_train_songs, seed=1)
    val_songs = load_songs(args.val_feature_dir, args.num_val_songs, seed=2)
    print(f"[train_kvae_adaptive_noise] train={len(train_songs)} val={len(val_songs)} device={device}", flush=True)
    print(f"[train_kvae_adaptive_noise] a_dim={args.a_dim} z_dim={args.z_dim} K={args.K} Q_reg={args.Q_reg} "
         f"steps={args.num_steps} batch={args.batch_size} (ORIGINAL bt_train_rich pool only, "
         f"IDENTICAL budget to kvae_m1_repro_400ep1000 except the adaptive-noise head)", flush=True)

    model = KalmanVAEBarPointerAdaptiveNoise(feature_dim=512, a_dim=args.a_dim, z_dim=args.z_dim, K=args.K,
                                             Q_reg=args.Q_reg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_sample_control = SampleControl(encoder="sample", decoder="mean", state_transition="sample", observation="sample")
    positive_weight = torch.tensor([args.beat_pos_weight, args.downbeat_pos_weight], device=device)

    for step in range(1, args.num_steps + 1):
        features, beats, downbeats = batch_time_major(train_songs, args.crop_length_frames, args.batch_size, device)
        elbo, z, info = adaptive_kvae_elbo(model, features, train_sample_control)
        beat_logits = model.head(z.reshape(-1, model.z_dim)).view(*z.shape[:2], 2)
        beat_bce = F.binary_cross_entropy_with_logits(
            beat_logits, torch.stack([beats, downbeats], dim=-1), pos_weight=positive_weight)
        loss = -elbo + args.beat_loss_weight * beat_bce

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()

        if step % args.eval_every == 0 or step == args.num_steps:
            real = evaluate_leak_condition_adaptive(model, val_songs, device, "real", args.eval_max_frames,
                                                     args.eval_beat_tolerance_seconds)
            print(f"  step {step:5d} | elbo {float(elbo):.1f} | bce {float(beat_bce):.3f} "
                 f"| scaleQ {info['scale_Q_mean']:.3f}+-{info['scale_Q_std']:.3f} "
                 f"| scaleR {info['scale_R_mean']:.3f}+-{info['scale_R_std']:.3f} "
                 f"| GEOM beat {real['beat_f']:.3f} db {real['downbeat_f']:.3f}", flush=True)

    leak = evaluate_with_leak_test_adaptive(model, val_songs, device, args.eval_max_frames, args.eval_beat_tolerance_seconds)
    print("\n[final] Adaptive-noise Kalman-filter deploy:", flush=True)
    for condition in ("real", "shuffle", "zero"):
        print(f"{condition:8s}: beat {leak[condition]['beat_f']:.3f}  downbeat {leak[condition]['downbeat_f']:.3f}", flush=True)

    if args.save_path:
        torch.save({"model": model.state_dict(), "args": vars(args)}, args.save_path)
        print(f"[train_kvae_adaptive_noise] saved -> {args.save_path}", flush=True)


if __name__ == "__main__":
    main()
