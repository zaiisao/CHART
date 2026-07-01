"""TASK C: CMLc/CMLt/AMLc/AMLt continuity metrics, SMC vs in-domain.

arXiv:2605.12287 distinguishes beat PLACEMENT (F-measure) from METRICAL COHERENCE (CMLt/AMLt) throughout
-- Section 5.4.5 notes these are empirically independent (perfect tempo improves CMLt universally but
doesn't move F on total-failure/continuity-error tracks). We currently only report F-measure everywhere in
this project's KVAE work. This adds mir_eval.beat.continuity's four metrics on top of the existing F-measure
eval, for BOTH the SMC real-condition and bt_val_rich in-domain, so the failure-signature shape can be read
off directly:
  - high AMLt, low CMLt  => tracking at an allowed metrical level (e.g. double/half tempo) but not the
    exact annotated one -- echoes the octave-error finding from diagnose_smc_gap.py.
  - high CMLt, low CMLc  => locally-correct-at-the-right-tempo tracking that's frequently INTERRUPTED
    (continuity errors) rather than wrong-tempo throughout.
  - all four low          => the "complete failure" (F<0.3) bucket -- consistent with diagnose_smc_gap.py's
    23/217 low-F SMC songs.

Usage:
    python experiments/kvae_prototype/continuity_metrics.py --checkpoint experiments/kvae_prototype/kvae_m1_repro_400ep1000.pt
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
from model.kalman_vae_bar_pointer import KalmanVAEBarPointer
from extract_ood_features import FRAMES_PER_SECOND as SMC_FPS
from eval_smc_ood_fresh import build_smc_index, load_smc_songs, FEATURE_CACHE_DIR as SMC_FEATURE_CACHE_DIR
from fast_deploy import batched_kalman_filter_deploy


def continuity_or_nan(reference_beats: np.ndarray, estimated_beats: np.ndarray) -> tuple:
    if len(reference_beats) < 2:
        return (float("nan"),) * 4
    reference_beats = mir_eval.beat.trim_beats(np.asarray(reference_beats, dtype=float))
    estimated_beats = mir_eval.beat.trim_beats(np.asarray(estimated_beats, dtype=float))
    if len(reference_beats) < 2 or len(estimated_beats) == 0:
        return (0.0, 0.0, 0.0, 0.0) if len(reference_beats) >= 2 else (float("nan"),) * 4
    return mir_eval.beat.continuity(reference_beats, estimated_beats)


@torch.no_grad()
def evaluate_with_continuity(model: KalmanVAEBarPointer, song_specs: list, device: str, frames_per_second: float,
                             tolerance_seconds: float, batch_chunk_size: int = 32) -> dict:
    """song_specs: [(tid, features [T,feature_dim], reference_beats), ...]. Returns aggregate F + continuity."""
    model.eval()
    f_scores, cmlc_scores, cmlt_scores, amlc_scores, amlt_scores = [], [], [], [], []
    num_songs = len(song_specs)

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

        filtered_means = batched_kalman_filter_deploy(model.ssm, a_padded, lengths_tensor)
        probabilities = torch.sigmoid(model.head(filtered_means)).cpu().numpy()

        for local_position, (tid, features, reference_beats) in enumerate(chunk):
            num_frames = lengths[local_position]
            probability = probabilities[:num_frames, local_position]
            estimated_beats = readout.peak_pick_times(probability[:, 0], frames_per_second)

            if len(reference_beats) >= 2:
                f_scores.append(readout.f_measure(reference_beats, estimated_beats, tolerance_seconds))
                cmlc, cmlt, amlc, amlt = continuity_or_nan(reference_beats, estimated_beats)
                cmlc_scores.append(cmlc); cmlt_scores.append(cmlt); amlc_scores.append(amlc); amlt_scores.append(amlt)

    mean = lambda values: float(np.nanmean(values)) if values else float("nan")
    return {
        "F": mean(f_scores), "CMLc": mean(cmlc_scores), "CMLt": mean(cmlt_scores),
        "AMLc": mean(amlc_scores), "AMLt": mean(amlt_scores), "n": len(f_scores),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default="experiments/kvae_prototype/kvae_m1_repro_400ep1000.pt")
    parser.add_argument("--val_feature_dir", type=str, default="cache/acts/bt_val_rich")
    parser.add_argument("--num_val_songs", type=int, default=40)
    parser.add_argument("--eval_max_frames", type=int, default=1600)
    parser.add_argument("--eval_beat_tolerance_seconds", type=float, default=0.07)
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
    print(f"[continuity_metrics] loaded {args.checkpoint}", flush=True)

    val_songs = load_songs(args.val_feature_dir, args.num_val_songs, seed=2)
    val_specs = [(f"val_{i}", song.features[:args.eval_max_frames],
                 ground_truth_beat_times(song.beat_targets.numpy()[:args.eval_max_frames], FRAMES_PER_SECOND))
                for i, song in enumerate(val_songs)]
    in_domain = evaluate_with_continuity(model, val_specs, device, FRAMES_PER_SECOND, args.eval_beat_tolerance_seconds)

    smc_index = build_smc_index()
    smc_songs = load_smc_songs(smc_index, SMC_FEATURE_CACHE_DIR)
    smc = evaluate_with_continuity(model, smc_songs, device, SMC_FPS, args.eval_beat_tolerance_seconds)

    print(f"\n=== continuity metrics (F, CMLc, CMLt, AMLc, AMLt), real-audio condition ===", flush=True)
    header = f"{'set':>20} {'n':>4} {'F':>7} {'CMLc':>7} {'CMLt':>7} {'AMLc':>7} {'AMLt':>7}"
    print(header, flush=True)
    for label, result in (("in-domain (bt_val_rich)", in_domain), ("SMC (OOD)", smc)):
        print(f"{label:>20} {result['n']:>4} {result['F']:>7.3f} {result['CMLc']:>7.3f} "
             f"{result['CMLt']:>7.3f} {result['AMLc']:>7.3f} {result['AMLt']:>7.3f}", flush=True)

    print("\n=== failure-signature read ===", flush=True)
    for label, result in (("in-domain", in_domain), ("SMC", smc)):
        amlt_minus_cmlt = result["AMLt"] - result["CMLt"]
        cmlt_minus_cmlc = result["CMLt"] - result["CMLc"]
        print(f"  {label}: AMLt-CMLt = {amlt_minus_cmlt:+.3f} (large positive => tracking at wrong-but-allowed "
             f"metrical level, echoes octave errors) | CMLt-CMLc = {cmlt_minus_cmlc:+.3f} "
             f"(large positive => correct-tempo-but-interrupted tracking, continuity errors)", flush=True)


if __name__ == "__main__":
    main()
