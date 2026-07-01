"""Full evaluation of the adaptive-noise KVAE (kvae_adaptive_noise.pt): in-domain leak test, SMC zero-shot
leak test, continuity metrics (CMLc/CMLt/AMLc/AMLt) on both, and a Task-B-style scale-head variation check
(does scale_Q/scale_R actually vary meaningfully across frames/songs, or collapse to a constant?).

Uses fast_deploy.batched_adaptive_kalman_filter_deploy throughout (verified 4.5e-8 vs the
correctness-gated reference recursion in verify_adaptive_noise_correctness.py / fast_deploy's
verify_adaptive_matches_reference).

Usage:
    python experiments/kvae_prototype/eval_adaptive_noise_full.py --checkpoint experiments/kvae_prototype/kvae_adaptive_noise.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import mir_eval
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FRAMES_PER_SECOND
from data.dataset import load_songs
from data.targets import ground_truth_beat_times
from model import readout
from model.kalman_vae_bar_pointer_adaptive_noise import KalmanVAEBarPointerAdaptiveNoise
from extract_ood_features import FRAMES_PER_SECOND as SMC_FPS
from eval_smc_ood_fresh import build_smc_index, load_smc_songs, FEATURE_CACHE_DIR as SMC_FEATURE_CACHE_DIR
from fast_deploy import batched_adaptive_kalman_filter_deploy


def continuity_or_nan(reference_beats: np.ndarray, estimated_beats: np.ndarray) -> tuple:
    if len(reference_beats) < 2:
        return (float("nan"),) * 4
    reference_beats = mir_eval.beat.trim_beats(np.asarray(reference_beats, dtype=float))
    estimated_beats = mir_eval.beat.trim_beats(np.asarray(estimated_beats, dtype=float))
    if len(reference_beats) < 2 or len(estimated_beats) == 0:
        return (0.0, 0.0, 0.0, 0.0) if len(reference_beats) >= 2 else (float("nan"),) * 4
    return mir_eval.beat.continuity(reference_beats, estimated_beats)


@torch.no_grad()
def run_adaptive_deploy(model, song_specs, device, frames_per_second, tolerance_seconds, batch_chunk_size=32):
    """song_specs: [(tid, features [T,feature_dim], reference_beats, reference_downbeats_or_None), ...]
    (downbeats optional -- SMC has none). Returns per-song rows: f_measure, cmlc/cmlt/amlc/amlt,
    downbeat_f (nan if no downbeat annotation given), mean_scale_Q, mean_scale_R, scale_Q_entropy_proxy
    (std/mean, since scale is a scalar not a K-way distribution -- entropy doesn't directly apply the way
    it did for the K=5 mixture; per-song coefficient of variation is the analogous "how much does it move"
    summary)."""
    model.eval()
    num_songs = len(song_specs)
    rows = []

    for chunk_start in range(0, num_songs, batch_chunk_size):
        chunk = song_specs[chunk_start:chunk_start + batch_chunk_size]
        lengths = [features.shape[0] for _, features, _, _ in chunk]
        T_max = max(lengths)

        a_means = []
        for _, features, _, _ in chunk:
            a_mean = model.encoder(features.to(device)).mean
            padded = torch.zeros(T_max, model.a_dim, device=device, dtype=a_mean.dtype)
            padded[:a_mean.shape[0]] = a_mean
            a_means.append(padded)
        a_padded = torch.stack(a_means, dim=1)
        lengths_tensor = torch.tensor(lengths, device=device)

        filtered_means, scale_Q_all, scale_R_all = batched_adaptive_kalman_filter_deploy(
            model.ssm, model.noise_head, a_padded, lengths_tensor)
        probabilities = torch.sigmoid(model.head(filtered_means)).cpu().numpy()
        scale_Q_np = scale_Q_all.cpu().numpy()
        scale_R_np = scale_R_all.cpu().numpy()

        for local_position, (tid, features, reference_beats, reference_downbeats) in enumerate(chunk):
            num_frames = lengths[local_position]
            probability = probabilities[:num_frames, local_position]
            estimated_beats = readout.peak_pick_times(probability[:, 0], frames_per_second)

            f_measure = readout.f_measure(reference_beats, estimated_beats, tolerance_seconds) if len(reference_beats) >= 2 else float("nan")
            cmlc, cmlt, amlc, amlt = continuity_or_nan(reference_beats, estimated_beats)

            downbeat_f = float("nan")
            if reference_downbeats is not None and len(reference_downbeats) >= 2:
                estimated_downbeats = readout.peak_pick_times(probability[:, 1], frames_per_second)
                downbeat_f = readout.f_measure(reference_downbeats, estimated_downbeats, tolerance_seconds)

            song_scale_Q = scale_Q_np[:num_frames, local_position]
            song_scale_R = scale_R_np[:num_frames, local_position]

            rows.append({
                "tid": tid, "f_measure": f_measure, "downbeat_f": downbeat_f,
                "CMLc": cmlc, "CMLt": cmlt, "AMLc": amlc, "AMLt": amlt,
                "mean_scale_Q": float(song_scale_Q.mean()), "std_scale_Q": float(song_scale_Q.std()),
                "mean_scale_R": float(song_scale_R.mean()), "std_scale_R": float(song_scale_R.std()),
            })
    return rows


def aggregate(rows: list[dict], key: str) -> float:
    values = [r[key] for r in rows if not np.isnan(r[key])]
    return float(np.mean(values)) if values else float("nan")


def print_scale_variation_report(label: str, rows: list[dict]) -> None:
    """Task-B-style check: does scale_Q/scale_R vary meaningfully across frames (within-song) and across
    songs (between-song), or collapse to a near-constant value everywhere?"""
    within_song_cv_Q = np.mean([r["std_scale_Q"] / max(r["mean_scale_Q"], 1e-6) for r in rows])
    within_song_cv_R = np.mean([r["std_scale_R"] / max(r["mean_scale_R"], 1e-6) for r in rows])
    between_song_mean_Q = np.array([r["mean_scale_Q"] for r in rows])
    between_song_mean_R = np.array([r["mean_scale_R"] for r in rows])
    print(f"{label}: n={len(rows)}", flush=True)
    print(f"  scale_Q: within-song mean CV (std/mean) = {within_song_cv_Q:.3f} | "
         f"between-song range of per-song mean = [{between_song_mean_Q.min():.3f}, {between_song_mean_Q.max():.3f}] "
         f"(std of per-song means = {between_song_mean_Q.std():.3f})", flush=True)
    print(f"  scale_R: within-song mean CV (std/mean) = {within_song_cv_R:.3f} | "
         f"between-song range of per-song mean = [{between_song_mean_R.min():.3f}, {between_song_mean_R.max():.3f}] "
         f"(std of per-song means = {between_song_mean_R.std():.3f})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default="experiments/kvae_prototype/kvae_adaptive_noise.pt")
    parser.add_argument("--val_feature_dir", type=str, default="cache/acts/bt_val_rich")
    parser.add_argument("--num_val_songs", type=int, default=40)
    parser.add_argument("--eval_max_frames", type=int, default=1600)
    parser.add_argument("--eval_beat_tolerance_seconds", type=float, default=0.07)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    model = KalmanVAEBarPointerAdaptiveNoise(
        feature_dim=512, a_dim=ckpt_args.get("a_dim", 8), z_dim=ckpt_args.get("z_dim", 8), K=ckpt_args.get("K", 5),
        Q_reg=ckpt_args.get("Q_reg", 1e-3),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"[eval_adaptive_noise_full] loaded {args.checkpoint}", flush=True)

    # ---- in-domain val, all 3 leak conditions ----
    val_songs = load_songs(args.val_feature_dir, args.num_val_songs, seed=2)
    print(f"\n=== IN-DOMAIN (bt_val_rich, n={len(val_songs)}) ===", flush=True)
    in_domain_by_condition = {}
    for condition in ("real", "shuffle", "zero"):
        num_songs = len(val_songs)
        specs = []
        for i, song in enumerate(val_songs):
            source = val_songs[(i + 1) % num_songs] if condition == "shuffle" else song
            num_frames = min(source.features.shape[0], song.beat_targets.shape[0], args.eval_max_frames)
            features = torch.zeros(num_frames, model.feature_dim) if condition == "zero" else source.features[:num_frames]
            ref_beats = ground_truth_beat_times(song.beat_targets.numpy()[:num_frames], FRAMES_PER_SECOND)
            ref_downbeats = ground_truth_beat_times(song.downbeat_targets.numpy()[:num_frames], FRAMES_PER_SECOND)
            specs.append((f"val_{i}", features, ref_beats, ref_downbeats))
        rows = run_adaptive_deploy(model, specs, device, FRAMES_PER_SECOND, args.eval_beat_tolerance_seconds)
        in_domain_by_condition[condition] = rows
        print(f"  {condition:8s}: beat {aggregate(rows,'f_measure'):.3f}  downbeat {aggregate(rows,'downbeat_f'):.3f}  "
             f"CMLc {aggregate(rows,'CMLc'):.3f}  CMLt {aggregate(rows,'CMLt'):.3f}  "
             f"AMLc {aggregate(rows,'AMLc'):.3f}  AMLt {aggregate(rows,'AMLt'):.3f}", flush=True)

    # ---- SMC, all 3 leak conditions (beat only, no downbeat annotations) ----
    smc_index = build_smc_index()
    smc_songs = load_smc_songs(smc_index, SMC_FEATURE_CACHE_DIR)
    print(f"\n=== SMC (OOD, n={len(smc_songs)}) ===", flush=True)
    smc_by_condition = {}
    for condition in ("real", "shuffle", "zero"):
        num_songs = len(smc_songs)
        specs = []
        for i, (tid, features_full, reference_beats) in enumerate(smc_songs):
            source_features = smc_songs[(i + 1) % num_songs][1] if condition == "shuffle" else features_full
            num_frames = min(source_features.shape[0], features_full.shape[0])
            features = torch.zeros(num_frames, model.feature_dim) if condition == "zero" else source_features[:num_frames]
            specs.append((tid, features, reference_beats, None))
        rows = run_adaptive_deploy(model, specs, device, SMC_FPS, args.eval_beat_tolerance_seconds)
        smc_by_condition[condition] = rows
        print(f"  {condition:8s}: beat {aggregate(rows,'f_measure'):.3f}  "
             f"CMLc {aggregate(rows,'CMLc'):.3f}  CMLt {aggregate(rows,'CMLt'):.3f}  "
             f"AMLc {aggregate(rows,'AMLc'):.3f}  AMLt {aggregate(rows,'AMLt'):.3f}", flush=True)

    print("\n=== failure-signature read (CMLt-CMLc gap -- the target metric) ===", flush=True)
    for label, rows in (("in-domain", in_domain_by_condition["real"]), ("SMC", smc_by_condition["real"])):
        cmlt_minus_cmlc = aggregate(rows, "CMLt") - aggregate(rows, "CMLc")
        amlt_minus_cmlt = aggregate(rows, "AMLt") - aggregate(rows, "CMLt")
        print(f"  {label}: CMLt-CMLc = {cmlt_minus_cmlc:+.3f}  |  AMLt-CMLt = {amlt_minus_cmlt:+.3f}", flush=True)

    print("\n=== scale-head variation check (Task-B-style) -- does adaptation actually happen? ===", flush=True)
    print_scale_variation_report("in-domain (real audio)", in_domain_by_condition["real"])
    print_scale_variation_report("SMC (real audio)", smc_by_condition["real"])


if __name__ == "__main__":
    main()
