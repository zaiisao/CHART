"""SMC OOD eval using the FAST batched Kalman-filter deploy path (fast_deploy.py) -- regression-validated
against eval_smc_ood_fresh.py's slow per-song path before being trusted (see train_kvae.py's
evaluate_leak_condition_fast, cross-checked bit-for-bit against evaluate_leak_condition on bt_val_rich).

Reuses the same fresh final0-checkpoint SMC features (cache/acts/smc_fresh_final0, already extracted by
eval_smc_ood_fresh.py) and the same annotation loading; only the Kalman-filter deploy loop is swapped for
the batched version. Target: reproduce real=0.654 / shuffle=0.139 / zero=0.000 (the already-landed slow
result) within float-precision noise.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import readout
from model.kalman_vae_bar_pointer import KalmanVAEBarPointer
from extract_ood_features import FRAMES_PER_SECOND
from eval_smc_ood_fresh import build_smc_index, load_smc_songs, FEATURE_CACHE_DIR
from fast_deploy import batched_kalman_filter_deploy


@torch.no_grad()
def evaluate_smc_condition_fast(model: KalmanVAEBarPointer, songs: list[tuple[str, torch.Tensor, np.ndarray]],
                                device: str, audio_condition: str, tolerance_seconds: float,
                                batch_chunk_size: int = 32) -> dict:
    model.eval()
    beat_scores = []
    num_songs = len(songs)

    for chunk_start in range(0, num_songs, batch_chunk_size):
        chunk = songs[chunk_start:chunk_start + batch_chunk_size]
        chunk_positions = list(range(chunk_start, chunk_start + len(chunk)))

        cropped_features, lengths = [], []
        for position, (tid, features_full, reference_beats) in zip(chunk_positions, chunk):
            source_features = songs[(position + 1) % num_songs][1] if audio_condition == "shuffle" else features_full
            num_frames = min(source_features.shape[0], features_full.shape[0])
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

        filtered_means = batched_kalman_filter_deploy(model.ssm, a_padded, lengths_tensor)
        probabilities = torch.sigmoid(model.head(filtered_means)).cpu().numpy()

        for local_position, (tid, _, reference_beats) in enumerate(chunk):
            num_frames = lengths[local_position]
            probability = probabilities[:num_frames, local_position]
            if len(reference_beats) >= 2:
                estimated_beats = readout.peak_pick_times(probability[:, 0], FRAMES_PER_SECOND)
                beat_scores.append(readout.f_measure(reference_beats, estimated_beats, tolerance_seconds))

    mean = lambda values: float(np.nanmean(values)) if values else float("nan")
    return {"beat_f": mean(beat_scores), "num_songs_scored": len(beat_scores)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="experiments/kvae_prototype/kvae_m1_repro_400ep1000.pt")
    parser.add_argument("--eval_beat_tolerance_seconds", type=float, default=0.07)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--feature_cache_dir", type=str, default=FEATURE_CACHE_DIR)
    parser.add_argument("--batch_chunk_size", type=int, default=32)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    index = build_smc_index()
    print(f"[eval_smc_ood_fast] matched {len(index)} SMC (audio, annotation) pairs", flush=True)
    songs = load_smc_songs(index, args.feature_cache_dir)  # reuses the already-extracted fresh features
    print(f"[eval_smc_ood_fast] loaded {len(songs)} songs (fresh final0 features, already cached)", flush=True)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    model = KalmanVAEBarPointer(
        feature_dim=512, a_dim=ckpt_args.get("a_dim", 8), z_dim=ckpt_args.get("z_dim", 8), K=ckpt_args.get("K", 5),
        Q_reg=ckpt_args.get("Q_reg", 1e-3),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"[eval_smc_ood_fast] loaded {args.checkpoint} (a_dim={model.a_dim} z_dim={model.z_dim} K={model.ssm.K})", flush=True)

    print("\n--- SMC (Holzapfel) OOD leak test: FAST batched Kalman-filter deploy, beat F only ---", flush=True)
    print("--- regression target: real=0.654 shuffle=0.139 zero=0.000 (the already-validated slow result) ---", flush=True)
    for condition in ("real", "shuffle", "zero"):
        result = evaluate_smc_condition_fast(model, songs, device, condition, args.eval_beat_tolerance_seconds,
                                             args.batch_chunk_size)
        print(f"{condition:8s}: beat {result['beat_f']:.4f}   (scored {result['num_songs_scored']}/{len(songs)} songs)", flush=True)


if __name__ == "__main__":
    main()
