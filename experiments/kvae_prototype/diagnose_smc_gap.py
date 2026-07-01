"""Diagnose the in-domain-to-SMC accuracy gap (0.895 in-domain vs 0.654 SMC real, kvae_m1_repro_400ep1000).

Open investigation (NOT premised on any external citation -- run purely against data already in this repo):
does the gap concentrate in an octave-error pattern (predicted tempo ~= 2x or ~0.5x ground truth) and/or
in slow-tempo songs, which would suggest the model's learned dynamics (mat_A's locally-linear transitions,
fit only to the training mix's tempo distribution) don't cover SMC's actual tempo range? Or is the gap
just generic noisier tracking, uncorrelated with tempo?

For each of the 217 SMC songs (real-audio condition, exact Kalman-filter deploy, same checkpoint/features
already used for the reported 0.654 result):
  1. Per-song beat F-measure (0.07s tolerance, same as the aggregate result).
  2. Ground-truth median tempo: 60 / median(diff(reference_beat_times)).
  3. Predicted median tempo: 60 / median(diff(estimated_beat_times)) from the model's peak-picked beats.
  4. tempo_ratio = predicted_median_tempo / gt_median_tempo -- octave errors show up as this clustering
     near 2.0 (predicting beats twice as often as GT, i.e. we're at double-tempo) or 0.5 (half-tempo).

Also reports the GT tempo distribution for SMC vs. a sample of bt_train_rich (the training mix), so we know
whether SMC actually skews slower -- this project's own cache, no external numbers assumed.

Usage:
    python experiments/kvae_prototype/diagnose_smc_gap.py --checkpoint experiments/kvae_prototype/kvae_m1_repro_400ep1000.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import load_songs
from data.targets import ground_truth_beat_times
from config import FRAMES_PER_SECOND
from model import readout
from model.kalman_vae_bar_pointer import KalmanVAEBarPointer
from extract_ood_features import FRAMES_PER_SECOND as SMC_FPS
from eval_smc_ood_fresh import build_smc_index, load_smc_songs, FEATURE_CACHE_DIR as SMC_FEATURE_CACHE_DIR
from fast_deploy import batched_kalman_filter_deploy

_THIRD_PARTY_KVAE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                 "third_party", "kalman-vae")
if _THIRD_PARTY_KVAE not in sys.path:
    sys.path.insert(0, _THIRD_PARTY_KVAE)
from kvae.sample_control import SampleControl


def median_tempo_bpm(beat_times: np.ndarray) -> float:
    if len(beat_times) < 2:
        return float("nan")
    median_interval = float(np.median(np.diff(beat_times)))
    if median_interval <= 0:
        return float("nan")
    return 60.0 / median_interval


@torch.no_grad()
def per_song_smc_diagnostics(model: KalmanVAEBarPointer, songs, device: str, tolerance_seconds: float,
                             batch_chunk_size: int = 32) -> list[dict]:
    """FAST batched version (fast_deploy.batched_kalman_filter_deploy, regression-validated bit-exact
    against the per-song kalman_filter path -- see fast_deploy.py + train_kvae.py's evaluate_leak_condition
    vs evaluate_leak_condition_fast, and eval_smc_ood_fast.py's 0.6540/0.1392/0.0000 match against the
    slow eval_smc_ood_fresh.py's 0.654/0.139/0.000)."""
    model.eval()
    rows = []
    num_songs = len(songs)

    for chunk_start in range(0, num_songs, batch_chunk_size):
        chunk = songs[chunk_start:chunk_start + batch_chunk_size]
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

        filtered_means = batched_kalman_filter_deploy(model.ssm, a_padded, lengths_tensor)
        probabilities = torch.sigmoid(model.head(filtered_means)).cpu().numpy()

        for local_position, (tid, features, reference_beats) in enumerate(chunk):
            num_frames = lengths[local_position]
            probability = probabilities[:num_frames, local_position]
            estimated_beats = readout.peak_pick_times(probability[:, 0], SMC_FPS)
            f_measure = readout.f_measure(reference_beats, estimated_beats, tolerance_seconds) if len(reference_beats) >= 2 else float("nan")

            gt_tempo = median_tempo_bpm(reference_beats)
            pred_tempo = median_tempo_bpm(estimated_beats)
            tempo_ratio = pred_tempo / gt_tempo if (gt_tempo and gt_tempo > 0 and not np.isnan(pred_tempo)) else float("nan")

            rows.append({
                "tid": tid, "f_measure": f_measure, "gt_tempo_bpm": gt_tempo, "pred_tempo_bpm": pred_tempo,
                "tempo_ratio": tempo_ratio, "n_gt_beats": len(reference_beats), "n_pred_beats": len(estimated_beats),
            })
    return rows


def gt_tempo_distribution_bt_train_rich(feature_dir: str, num_songs: int, seed: int) -> list[float]:
    songs = load_songs(feature_dir, num_songs, seed=seed)
    tempos = []
    for song in songs:
        beat_frames = np.where(song.beat_targets.numpy() > 0.5)[0]
        beat_times = beat_frames / FRAMES_PER_SECOND
        tempo = median_tempo_bpm(beat_times)
        if not np.isnan(tempo):
            tempos.append(tempo)
    return tempos


def summarize_tempo_distribution(label: str, tempos: list[float]) -> None:
    arr = np.array([t for t in tempos if not np.isnan(t)])
    if len(arr) == 0:
        print(f"{label}: no valid tempos", flush=True)
        return
    quartiles = np.percentile(arr, [0, 25, 50, 75, 100])
    print(f"{label}: n={len(arr)} min={quartiles[0]:.1f} Q1={quartiles[1]:.1f} "
         f"median={quartiles[2]:.1f} Q3={quartiles[3]:.1f} max={quartiles[4]:.1f} BPM", flush=True)
    # coarse histogram
    bins = [0, 40, 55, 70, 85, 100, 120, 140, 170, 1000]
    labels = ["<40", "40-55", "55-70", "70-85", "85-100", "100-120", "120-140", "140-170", ">170"]
    counts, _ = np.histogram(arr, bins=bins)
    for lo_label, count in zip(labels, counts):
        bar = "#" * int(count / max(counts.max(), 1) * 40)
        print(f"    {lo_label:>8} BPM: {count:4d} {bar}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default="experiments/kvae_prototype/kvae_m1_repro_400ep1000.pt")
    parser.add_argument("--eval_beat_tolerance_seconds", type=float, default=0.07)
    parser.add_argument("--low_f_threshold", type=float, default=0.3, help="songs at/below this F are "
                        "treated as 'failing' for the octave-error / low-tempo clustering checks")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--train_feature_dir", type=str, default="cache/acts/bt_train_rich")
    parser.add_argument("--train_tempo_sample_songs", type=int, default=200)
    parser.add_argument("--csv_out", type=str, default="experiments/kvae_prototype/logs/smc_per_song_diagnostics.csv")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    index = build_smc_index()
    songs = load_smc_songs(index, SMC_FEATURE_CACHE_DIR)
    print(f"[diagnose_smc_gap] loaded {len(songs)} SMC songs (fresh final0 features)", flush=True)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    model = KalmanVAEBarPointer(
        feature_dim=512, a_dim=ckpt_args.get("a_dim", 8), z_dim=ckpt_args.get("z_dim", 8), K=ckpt_args.get("K", 5),
        Q_reg=ckpt_args.get("Q_reg", 1e-3),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"[diagnose_smc_gap] loaded {args.checkpoint}", flush=True)

    rows = per_song_smc_diagnostics(model, songs, device, args.eval_beat_tolerance_seconds)

    # ---- write per-song CSV ----
    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    with open(args.csv_out, "w") as f:
        f.write("tid,f_measure,gt_tempo_bpm,pred_tempo_bpm,tempo_ratio,n_gt_beats,n_pred_beats\n")
        for row in rows:
            f.write(f"{row['tid']},{row['f_measure']:.4f},{row['gt_tempo_bpm']:.2f},{row['pred_tempo_bpm']:.2f},"
                   f"{row['tempo_ratio']:.4f},{row['n_gt_beats']},{row['n_pred_beats']}\n")
    print(f"[diagnose_smc_gap] wrote per-song CSV -> {args.csv_out}", flush=True)

    f_measures = np.array([r["f_measure"] for r in rows])
    print(f"\n[diagnose_smc_gap] aggregate mean F = {np.nanmean(f_measures):.3f} "
         f"(regression check vs. the reported 0.654)", flush=True)

    # ---- octave-error clustering check ----
    low_f_rows = [r for r in rows if not np.isnan(r["f_measure"]) and r["f_measure"] <= args.low_f_threshold]
    high_f_rows = [r for r in rows if not np.isnan(r["f_measure"]) and r["f_measure"] > args.low_f_threshold]
    print(f"\n[octave-error check] {len(low_f_rows)}/{len(rows)} songs at/below F<={args.low_f_threshold} "
         f"('failing'); {len(high_f_rows)} above", flush=True)

    def ratio_bucket_counts(rows_subset, label):
        ratios = np.array([r["tempo_ratio"] for r in rows_subset if not np.isnan(r["tempo_ratio"])])
        if len(ratios) == 0:
            print(f"  {label}: no valid tempo ratios", flush=True)
            return
        near_double = np.sum((ratios > 1.7) & (ratios < 2.3))
        near_half = np.sum((ratios > 0.35) & (ratios < 0.65))
        near_one = np.sum((ratios > 0.85) & (ratios < 1.15))
        other = len(ratios) - near_double - near_half - near_one
        print(f"  {label}: n={len(ratios)} | near-1x(0.85-1.15): {near_one} ({100*near_one/len(ratios):.0f}%) "
             f"| near-2x(1.7-2.3): {near_double} ({100*near_double/len(ratios):.0f}%) "
             f"| near-0.5x(0.35-0.65): {near_half} ({100*near_half/len(ratios):.0f}%) "
             f"| other: {other} ({100*other/len(ratios):.0f}%)", flush=True)
        print(f"    ratio distribution: median={np.median(ratios):.2f} "
             f"Q1={np.percentile(ratios,25):.2f} Q3={np.percentile(ratios,75):.2f}", flush=True)

    ratio_bucket_counts(low_f_rows, "LOW-F (failing) songs' predicted/GT tempo ratio")
    ratio_bucket_counts(high_f_rows, "HIGH-F (succeeding) songs' predicted/GT tempo ratio")
    ratio_bucket_counts(rows, "ALL songs' predicted/GT tempo ratio")

    # ---- low-tempo concentration check ----
    print(f"\n[low-tempo concentration check] GT tempo of low-F vs high-F songs:", flush=True)
    low_f_gt_tempos = np.array([r["gt_tempo_bpm"] for r in low_f_rows if not np.isnan(r["gt_tempo_bpm"])])
    high_f_gt_tempos = np.array([r["gt_tempo_bpm"] for r in high_f_rows if not np.isnan(r["gt_tempo_bpm"])])
    if len(low_f_gt_tempos):
        print(f"  LOW-F  songs GT tempo: median={np.median(low_f_gt_tempos):.1f} "
             f"Q1={np.percentile(low_f_gt_tempos,25):.1f} Q3={np.percentile(low_f_gt_tempos,75):.1f} BPM", flush=True)
    if len(high_f_gt_tempos):
        print(f"  HIGH-F songs GT tempo: median={np.median(high_f_gt_tempos):.1f} "
             f"Q1={np.percentile(high_f_gt_tempos,25):.1f} Q3={np.percentile(high_f_gt_tempos,75):.1f} BPM", flush=True)

    for threshold in (55.0, 60.0, 70.0, 80.0):
        below = [r for r in rows if not np.isnan(r["gt_tempo_bpm"]) and r["gt_tempo_bpm"] < threshold]
        above = [r for r in rows if not np.isnan(r["gt_tempo_bpm"]) and r["gt_tempo_bpm"] >= threshold]
        below_mean_f = np.nanmean([r["f_measure"] for r in below]) if below else float("nan")
        above_mean_f = np.nanmean([r["f_measure"] for r in above]) if above else float("nan")
        print(f"  threshold {threshold:5.1f} BPM: below-threshold n={len(below):3d} mean F={below_mean_f:.3f} "
             f"| above-threshold n={len(above):3d} mean F={above_mean_f:.3f}", flush=True)

    # ---- tempo distribution: SMC vs training mix ----
    print(f"\n[tempo distribution] SMC (this OOD set) vs bt_train_rich (training mix, sampled)", flush=True)
    smc_tempos = [r["gt_tempo_bpm"] for r in rows]
    summarize_tempo_distribution("SMC (217 songs)", smc_tempos)
    train_tempos = gt_tempo_distribution_bt_train_rich(args.train_feature_dir, args.train_tempo_sample_songs, seed=1)
    summarize_tempo_distribution(f"bt_train_rich (sample of {args.train_tempo_sample_songs})", train_tempos)


if __name__ == "__main__":
    main()
