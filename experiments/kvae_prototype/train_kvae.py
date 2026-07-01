"""KVAE bar-pointer prototype: train + leak-test evaluate the exact-filter posterior.

Reproduces (against this repo's CURRENT data/config conventions) the historical "M1" result: replace the
amortized GRU posterior in model/bar_pointer_vae.py with the EXACT Kalman-filter posterior of
model/kalman_vae_bar_pointer.py's KalmanVAEBarPointer, and read beats/downbeats off a learned head on the
FILTERED latent (not the geometric phase-wrap read-out -- see kalman_vae_bar_pointer.py's docstring for why).

Data: reuses data/dataset.py's load_songs/sample_training_batch VERBATIM against the same cached
cache/acts/bt_train_rich (2000 songs) / cache/acts/bt_val_rich (118 songs) directories the rest of this
repo trains against.

Evaluation: same leak-test protocol as evaluate.py's evaluate_with_leak_test (real / shuffle / zero audio
conditions) and model/readout.py's peak_pick_times + f_measure (mir_eval beat F-measure, 0.07s tolerance),
applied to the sigmoid of the learned head's logits instead of the geometric phase read-out.

Usage (defaults reproduce the original M1 run's budget: 200 train / 40 val songs, 800 steps, batch 16,
crop 256 frames -- see NOTE in main() about why we default to that budget rather than this repo's 400/40):
    python experiments/kvae_prototype/train_kvae.py
    python experiments/kvae_prototype/train_kvae.py --num_train_songs 400 --num_steps 1000
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

from config import FRAMES_PER_SECOND
from data.dataset import load_songs, sample_training_batch, Song
from data.targets import ground_truth_beat_times
from model import readout
from model.kalman_vae_bar_pointer import KalmanVAEBarPointer, kvae_elbo
from fast_deploy import batched_kalman_filter_deploy

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
    """sample_training_batch gives [batch, frames, ...]; the SSM wants [frames, batch, ...]."""
    features, beats, downbeats = sample_training_batch(songs, crop_length_frames, batch_size, device)
    return features.transpose(0, 1).contiguous(), beats.transpose(0, 1).contiguous(), downbeats.transpose(0, 1).contiguous()


@torch.no_grad()
def evaluate_leak_condition(model: KalmanVAEBarPointer, songs: list[Song], device: str,
                            audio_condition: str, eval_max_frames: int, tolerance_seconds: float) -> dict:
    """Deploy = encoder(h) mean -> Kalman FILTER (causal, deterministic) -> head(z) -> peak-pick."""
    model.eval()
    sample_control = SampleControl(encoder="mean", decoder="mean", state_transition="mean", observation="mean")
    beat_scores, downbeat_scores = [], []
    num_songs = len(songs)

    for song_index, song in enumerate(songs):
        source_features = songs[(song_index + 1) % num_songs].features if audio_condition == "shuffle" else song.features
        num_frames = min(source_features.shape[0], song.beat_targets.shape[0], eval_max_frames)

        if audio_condition == "zero":
            features = torch.zeros(num_frames, 1, model.feature_dim, device=device)
        else:
            features = source_features[:num_frames].unsqueeze(1).to(device)

        a_mean = model.encoder(features.reshape(-1, model.feature_dim)).mean.view(num_frames, 1, model.a_dim)
        filtered_means, *_ = model.ssm.kalman_filter(a_mean, sample_control=sample_control)  # causal filter
        probability = torch.sigmoid(model.head(filtered_means.view(num_frames, model.z_dim))).cpu().numpy()

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


def evaluate_with_leak_test(model: KalmanVAEBarPointer, songs: list[Song], device: str,
                            eval_max_frames: int, tolerance_seconds: float) -> dict:
    return {
        condition: evaluate_leak_condition(model, songs, device, condition, eval_max_frames, tolerance_seconds)
        for condition in ("real", "shuffle", "zero")
    }


# =====================================================================================================
# FAST batched leak-test evaluation (eval-only; verified equivalent to the slow per-song path above --
# see fast_deploy.py's module docstring + verify_matches_reference). Use this for large eval sets (GTZAN's
# 993 songs, sweeps that re-run eval many times) where the per-song loop above is prohibitively slow
# (~993 sequential T-length Python/CUDA loops instead of ~1). Both paths are kept: evaluate_leak_condition
# above remains the trusted slow reference (used to validate this one); this one is what large runs use.
# =====================================================================================================

@torch.no_grad()
def evaluate_leak_condition_fast(model: KalmanVAEBarPointer, songs: list[Song], device: str,
                                 audio_condition: str, eval_max_frames: int, tolerance_seconds: float,
                                 batch_chunk_size: int = 64) -> dict:
    """Same deploy path/semantics as evaluate_leak_condition, but batches songs through ONE Kalman-filter
    pass (per chunk of batch_chunk_size songs, padded to the chunk's max length) instead of one song at a
    time. batch_chunk_size caps peak memory for very large eval sets; chunk boundaries do not affect the
    result (each song's filtered mean only depends on its own column, never on other songs in its chunk --
    see fast_deploy.py's masking)."""
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
            a_mean = model.encoder(features.to(device)).mean            # [length, a_dim]
            padded = torch.zeros(T_max, model.a_dim, device=device, dtype=a_mean.dtype)
            padded[:length] = a_mean
            a_means.append(padded)
        a_padded = torch.stack(a_means, dim=1)                          # [T_max, chunk_N, a_dim]
        lengths_tensor = torch.tensor(lengths, device=device)

        filtered_means = batched_kalman_filter_deploy(model.ssm, a_padded, lengths_tensor)  # [T_max, chunk_N, z_dim]
        probabilities = torch.sigmoid(model.head(filtered_means)).cpu().numpy()             # [T_max, chunk_N, 2]

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


def evaluate_with_leak_test_fast(model: KalmanVAEBarPointer, songs: list[Song], device: str,
                                 eval_max_frames: int, tolerance_seconds: float, batch_chunk_size: int = 64) -> dict:
    return {
        condition: evaluate_leak_condition_fast(model, songs, device, condition, eval_max_frames,
                                                tolerance_seconds, batch_chunk_size)
        for condition in ("real", "shuffle", "zero")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train_feature_dir", type=str, default="cache/acts/bt_train_rich")
    parser.add_argument("--val_feature_dir", type=str, default="cache/acts/bt_val_rich")
    # NOTE: default 200/40/800/16 reproduces the ORIGINAL M1 run's budget exactly (kvae_run.py's own
    # defaults; see m1.log), not this repo's train.py default (400 train songs, 1000 steps). We default to
    # the original budget for a faithful reproduction; pass --num_train_songs 400 --num_steps 1000 to match
    # this repo's other baselines' budget instead (both are run in this prototype's evaluation -- see report).
    parser.add_argument("--num_train_songs", type=int, default=200)
    parser.add_argument("--num_val_songs", type=int, default=40)
    parser.add_argument("--num_steps", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--crop_length_frames", type=int, default=256)
    parser.add_argument("--a_dim", type=int, default=8)
    parser.add_argument("--z_dim", type=int, default=8)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--Q_reg", type=float, default=1e-3, help="StateSpaceModel process-noise floor "
                        "(vendored default 1e-3); raise to loosen the dynamics prior's grip on z_t for "
                        "the Q_reg generalization sweep (rubato/tempo-drift robustness on OOD data).")
    parser.add_argument("--beat_loss_weight", type=float, default=5.0)
    parser.add_argument("--beat_pos_weight", type=float, default=8.0)
    parser.add_argument("--downbeat_pos_weight", type=float, default=20.0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--grad_clip_norm", type=float, default=5.0)
    parser.add_argument("--eval_max_frames", type=int, default=1600)
    parser.add_argument("--eval_beat_tolerance_seconds", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--eval_every", type=int, default=200)
    args = parser.parse_args()

    set_all_seeds(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    train_songs = load_songs(args.train_feature_dir, args.num_train_songs, seed=1)
    val_songs = load_songs(args.val_feature_dir, args.num_val_songs, seed=2)
    print(f"[kvae_prototype] EXACT differentiable Kalman filter (reused third_party/kalman-vae) "
         f"+ MLP VAE on [{512}] Beat-This activations", flush=True)
    print(f"[kvae_prototype] train={len(train_songs)} val={len(val_songs)} device={device} "
         f"a_dim={args.a_dim} z_dim={args.z_dim} K={args.K} Q_reg={args.Q_reg} "
         f"steps={args.num_steps} batch={args.batch_size}", flush=True)

    model = KalmanVAEBarPointer(feature_dim=512, a_dim=args.a_dim, z_dim=args.z_dim, K=args.K, Q_reg=args.Q_reg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_sample_control = SampleControl(encoder="sample", decoder="mean", state_transition="sample", observation="sample")
    positive_weight = torch.tensor([args.beat_pos_weight, args.downbeat_pos_weight], device=device)

    for step in range(1, args.num_steps + 1):
        features, beats, downbeats = batch_time_major(train_songs, args.crop_length_frames, args.batch_size, device)
        elbo, z, info = kvae_elbo(model, features, train_sample_control)
        beat_logits = model.head(z.reshape(-1, model.z_dim)).view(*z.shape[:2], 2)
        beat_bce = F.binary_cross_entropy_with_logits(
            beat_logits, torch.stack([beats, downbeats], dim=-1), pos_weight=positive_weight)
        loss = -elbo + args.beat_loss_weight * beat_bce

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()

        if step % args.eval_every == 0 or step == args.num_steps:
            real = evaluate_leak_condition(model, val_songs, device, "real", args.eval_max_frames, args.eval_beat_tolerance_seconds)
            print(f"  step {step:5d} | elbo {float(elbo):.1f} (recon {info['recon']:.1f} reg {info['reg']:.1f} "
                 f"obs {info['kal_obs']:.1f} trans {info['kal_trans']:.1f} post {info['kal_post']:.1f}) "
                 f"bce {float(beat_bce):.3f} | FILTER deploy: beat {real['beat_f']:.3f} downbeat {real['downbeat_f']:.3f}",
                 flush=True)

    leak = evaluate_with_leak_test(model, val_songs, device, args.eval_max_frames, args.eval_beat_tolerance_seconds)
    print("\n[final] Kalman-filter deploy (learned head on filtered z; decoder/reconstruction discarded):", flush=True)
    for condition in ("real", "shuffle", "zero"):
        print(f"{condition:8s}: beat {leak[condition]['beat_f']:.3f}  downbeat {leak[condition]['downbeat_f']:.3f}", flush=True)
    print("(real high + shuffle/zero collapsed => the read-out genuinely tracks the audio)", flush=True)

    if args.save_path:
        torch.save({"model": model.state_dict(), "args": vars(args)}, args.save_path)
        print(f"[kvae_prototype] saved -> {args.save_path}", flush=True)


if __name__ == "__main__":
    main()
