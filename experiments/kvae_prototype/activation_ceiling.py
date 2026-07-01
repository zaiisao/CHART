"""TASK A: activation-quality vs. dynamics-quality decomposition (SMC gap diagnosis).

Adapted from arXiv:2605.12287 ("The SMC Blind Spot") Section 5.3.2's method: they fed synthetic Gaussian
peaks at ground-truth beat positions through the (unchanged) DBN to isolate "what's the DBN's ceiling given
PERFECT input" (F=0.924) vs. real activations (F=0.585) -- a 0.339 gap, 4x larger on SMC than other
datasets, showing SMC's problem is mostly ACTIVATION quality, not DBN/dynamics quality.

We can't literally replicate that (our pseudo-observation `a` is an 8-dim learned latent, not an
interpretable per-frame scalar activation function -- there's no well-defined "perfect a"). Instead we run
the analogous decomposition the other direction: train a SIMPLE SUPERVISED CLASSIFIER (no temporal
filtering at all, just a per-frame MLP + peak-picking) on the SAME frozen [T,512] Beat-This features and
SAME bt_train_rich training songs our KVAE uses, then compare ITS ceiling on SMC vs. our full KVAE's 0.654:
  - classifier-only SMC F ~= or WORSE than KVAE's 0.654  => bottleneck is the shared FROZEN FRONTEND
    FEATURES themselves ("activation ceiling" -- can't fix by touching the Kalman filter/dynamics).
  - classifier-only SMC F NOTABLY HIGHER than KVAE's 0.654 => the filter/dynamics stage is actively
    degrading things on SMC relative to what's recoverable from the features alone ("dynamics ceiling" --
    the Kalman filter/head is the thing to fix).

ActivationEmissionHead/pretrain_emission are copied (not cross-worktree-imported, for isolation) from
experiments/smc_prototype/emission.py in the sibling worktree agent-a365bc6c9f51b0ad9 (verified to exist
and match this description before reuse) -- a small MLP mapping [feature_dim] -> per-frame (beat,
downbeat) probability via plain BCE, exactly the "small learned MLP on the [512] features" baseline this
decomposition needs.

Usage:
    python experiments/kvae_prototype/activation_ceiling.py --steps 1000
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FRAMES_PER_SECOND
from data.dataset import load_songs, sample_training_batch, Song
from data.targets import ground_truth_beat_times
from model import readout
from extract_ood_features import FRAMES_PER_SECOND as SMC_FPS
from eval_smc_ood_fresh import build_smc_index, load_smc_songs, FEATURE_CACHE_DIR as SMC_FEATURE_CACHE_DIR


class ActivationEmissionHead(nn.Module):
    """[*, feature_dim] -> per-frame (p_beat, p_downbeat) in (0, 1). Copied from
    experiments/smc_prototype/emission.py (sibling worktree agent-a365bc6c9f51b0ad9), verbatim."""

    def __init__(self, feature_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

    def probabilities(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(features))


def pretrain_emission(model: ActivationEmissionHead, songs: list[Song], crop_length_frames: int, batch_size: int,
                      steps: int, device: str, pos_weight_beat: float = 5.0, pos_weight_downbeat: float = 5.0,
                      lr: float = 1e-3, log_every: int = 200) -> ActivationEmissionHead:
    """Copied from experiments/smc_prototype/emission.py's pretrain_emission, verbatim (only the config
    object's two fields it reads -- crop_length_frames, batch_size -- are passed directly instead)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight = torch.tensor([pos_weight_beat, pos_weight_downbeat], device=device)
    model.train()
    for step in range(1, steps + 1):
        features, beat_targets, downbeat_targets = sample_training_batch(songs, crop_length_frames, batch_size, device)
        logits = model.logits(features)
        targets = torch.stack([beat_targets, downbeat_targets], dim=-1)
        loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % log_every == 0 or step == steps:
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                beat_acc = ((probs[..., 0] > 0.5).float() == beat_targets).float().mean()
            print(f"  [emission pretrain] step {step:5d} | bce {loss.item():.4f} | beat_frame_acc {beat_acc.item():.3f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def evaluate_classifier_leak_test(model: ActivationEmissionHead, songs: list[Song], device: str,
                                  frames_per_second: float, tolerance_seconds: float, eval_max_frames: int) -> dict:
    """Same leak-test protocol as everywhere else in this project: real/shuffle/zero, peak-pick, F-measure."""
    model.eval()
    results = {}
    num_songs = len(songs)
    for condition in ("real", "shuffle", "zero"):
        beat_scores, downbeat_scores = [], []
        for song_index, song in enumerate(songs):
            source_features = songs[(song_index + 1) % num_songs].features if condition == "shuffle" else song.features
            num_frames = min(source_features.shape[0], song.beat_targets.shape[0], eval_max_frames)
            features = (torch.zeros(num_frames, source_features.shape[1], device=device) if condition == "zero"
                       else source_features[:num_frames].to(device))
            probability = model.probabilities(features).cpu().numpy()

            reference_beats = ground_truth_beat_times(song.beat_targets.numpy()[:num_frames], frames_per_second)
            reference_downbeats = ground_truth_beat_times(song.downbeat_targets.numpy()[:num_frames], frames_per_second)
            if len(reference_beats) >= 2:
                estimated_beats = readout.peak_pick_times(probability[:, 0], frames_per_second)
                beat_scores.append(readout.f_measure(reference_beats, estimated_beats, tolerance_seconds))
            if len(reference_downbeats) >= 2:
                estimated_downbeats = readout.peak_pick_times(probability[:, 1], frames_per_second)
                downbeat_scores.append(readout.f_measure(reference_downbeats, estimated_downbeats, tolerance_seconds))
        mean = lambda values: float(np.nanmean(values)) if values else float("nan")
        results[condition] = {"beat_f": mean(beat_scores), "downbeat_f": mean(downbeat_scores)}
    return results


@torch.no_grad()
def evaluate_classifier_on_smc(model: ActivationEmissionHead, smc_songs, device: str, tolerance_seconds: float) -> dict:
    model.eval()
    beat_scores = []
    for tid, features, reference_beats in smc_songs:
        probability = model.probabilities(features.to(device)).cpu().numpy()
        if len(reference_beats) >= 2:
            estimated_beats = readout.peak_pick_times(probability[:, 0], SMC_FPS)
            beat_scores.append(readout.f_measure(reference_beats, estimated_beats, tolerance_seconds))
    mean = lambda values: float(np.nanmean(values)) if values else float("nan")
    return {"beat_f": mean(beat_scores), "num_songs_scored": len(beat_scores)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train_feature_dir", type=str, default="cache/acts/bt_train_rich")
    parser.add_argument("--val_feature_dir", type=str, default="cache/acts/bt_val_rich")
    parser.add_argument("--num_train_songs", type=int, default=400)
    parser.add_argument("--num_val_songs", type=int, default=40)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--crop_length_frames", type=int, default=256)
    parser.add_argument("--eval_max_frames", type=int, default=1600)
    parser.add_argument("--eval_beat_tolerance_seconds", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_path", type=str, default="experiments/kvae_prototype/activation_emission.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    train_songs = load_songs(args.train_feature_dir, args.num_train_songs, seed=1)
    val_songs = load_songs(args.val_feature_dir, args.num_val_songs, seed=2)
    print(f"[activation_ceiling] train={len(train_songs)} val={len(val_songs)} device={device}", flush=True)

    model = ActivationEmissionHead(feature_dim=512).to(device)
    model = pretrain_emission(model, train_songs, args.crop_length_frames, args.batch_size, args.steps, device)

    torch.save({"model": model.state_dict()}, args.save_path)
    print(f"[activation_ceiling] saved -> {args.save_path}", flush=True)

    in_domain = evaluate_classifier_leak_test(model, val_songs, device, FRAMES_PER_SECOND,
                                              args.eval_beat_tolerance_seconds, args.eval_max_frames)
    print("\n--- classifier-only (no filter/dynamics) in-domain leak test ---", flush=True)
    for condition in ("real", "shuffle", "zero"):
        print(f"{condition:8s}: beat {in_domain[condition]['beat_f']:.3f}  downbeat {in_domain[condition]['downbeat_f']:.3f}", flush=True)

    smc_index = build_smc_index()
    smc_songs = load_smc_songs(smc_index, SMC_FEATURE_CACHE_DIR)
    smc_result = evaluate_classifier_on_smc(model, smc_songs, device, args.eval_beat_tolerance_seconds)
    print(f"\n--- classifier-only (no filter/dynamics) SMC real-condition ---", flush=True)
    print(f"real    : beat {smc_result['beat_f']:.3f}   (scored {smc_result['num_songs_scored']}/{len(smc_songs)} songs)", flush=True)

    print("\n=== TASK A DECOMPOSITION ===", flush=True)
    print(f"  Full KVAE (filter+dynamics+head) SMC real beat F  = 0.654   (already reported)", flush=True)
    print(f"  Classifier-only (no filter) SMC real beat F       = {smc_result['beat_f']:.3f}", flush=True)
    gap = smc_result["beat_f"] - 0.654
    if gap <= 0.02:
        verdict = "ACTIVATION ceiling: classifier-only is NOT notably better -> bottleneck is the shared frozen frontend features, not the Kalman filter/dynamics."
    else:
        verdict = "DYNAMICS ceiling: classifier-only is notably BETTER -> the Kalman filter/dynamics stage is actively degrading SMC results relative to what the features alone support."
    print(f"  delta = {gap:+.3f}  =>  {verdict}", flush=True)
    print(f"  (for reference, in-domain: full KVAE real beat F = 0.895 vs classifier-only = {in_domain['real']['beat_f']:.3f})", flush=True)


if __name__ == "__main__":
    main()
