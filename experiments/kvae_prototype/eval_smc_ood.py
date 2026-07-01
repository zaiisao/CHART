"""Genuine out-of-domain check: run the trained KVAE checkpoint on SMC (Holzapfel), NOT bt_val_rich.

Why this is needed: bt_train_rich / bt_val_rich (what train_kvae.py trains and validates on) are both
built from the SAME four training datasets (ballroom, beatles, hains, rwc_popular) -- bt_val_rich is a
held-out SPLIT of that pool, not a different distribution. This project's own history (diagram_arch,
sawtooth_phase-adjacent runs) has repeatedly seen models that looked strong on this in-domain val cache
collapse hard on SMC (Holzapfel) -- e.g. in-domain beat 0.84 -> SMC 0.192 for one prior model, and even a
strong discriminative peak-pick baseline dropped 0.93 (clean) -> ~0.63/0.586 (SMC AMLt). SMC is a
genuinely different, harder distribution (expressive/non-mechanical timing) -- the standard way this
project (and the field) checks whether an in-domain win is real generalization or in-domain overfitting.

Data wrinkles specific to this cache (verified empirically before writing this script, not assumed):
  * cache/acts/smc_rich_heldout/*.pt has keys {"feat" [T,512] float16, "act2" [T,2], "tid", "fold"} --
    NOT the {"activations","beat_targets","downbeat_targets"} schema of bt_val_rich. No GT here at all.
  * Ground truth beat times live externally, one file per track:
      /home/sogang/jaehoon/Analyze-SMC/beat_this_annotations/smc/annotations/beats/{tid}.beats
    (plain text, one beat time in seconds per line, standard mir_eval .beats format). Verified all 217
    smc_rich_heldout tids have a matching .beats file (set difference is empty).
  * SMC has NO downbeat annotations (a known property of the dataset) -- beat F-measure only here.
  * FRAME RATE: verified empirically (NOT assumed from any docstring) by cross-referencing feat.shape[0]
    against the actual SMC_MIREX .wav duration for several tracks: every SMC track is exactly 40.0s long,
    and every track has feat.shape[0] == 2001, giving (T-1)/duration = 2000/40.0 = 50.0 fps EXACTLY. This
    is NOT the 86.1328125 fps (22050/256) used for bt_train_rich/bt_val_rich -- despite both caches coming
    from the same Beat-This frontend/frontend-hop, this SMC extraction (tests/extract_smc_rich.py in the
    archived tree) taps the transformer at its native 50 fps rather the 86.13 fps hop VBPM's other caches
    use. Using 86.13 fps here would misconvert every frame index by a factor of ~1.72x and silently produce
    a nonsense (or luckily-noisy) F-measure -- this is exactly the bug this docstring exists to flag.

Deploy path: IDENTICAL to train_kvae.py's evaluate_leak_condition (encoder(h) mean -> causal deterministic
Kalman filter -> head(filtered z) -> sigmoid -> peak-pick), just fed SMC features/GT instead of val_songs.
Same leak-test protocol (real / shuffle / zero) for the same reason as everywhere else in this project:
distinguishes "genuinely tracking OOD audio, just less accurately" from "collapsed to an audio-blind
template that happens to not totally fail" -- an in-domain-trained model could plausibly do the latter on
truly unfamiliar audio.

Usage:
    python experiments/kvae_prototype/eval_smc_ood.py --checkpoint experiments/kvae_prototype/kvae_m1_repro_400ep1000.pt
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model import readout
from model.kalman_vae_bar_pointer import KalmanVAEBarPointer

_THIRD_PARTY_KVAE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                 "third_party", "kalman-vae")
if _THIRD_PARTY_KVAE not in sys.path:
    sys.path.insert(0, _THIRD_PARTY_KVAE)
from kvae.sample_control import SampleControl

# Verified empirically (see module docstring): (T-1)/duration = 2000/40.0 = 50.0 fps EXACTLY for every
# SMC track checked (smc_001, smc_002, smc_003, ..., smc_008 all exactly 40.0s / 2001 frames). This is
# NOT config.FRAMES_PER_SECOND (86.1328125), which is bt_train_rich/bt_val_rich's rate, not this cache's.
SMC_FRAMES_PER_SECOND = 50.0

SMC_FEATURE_DIR = "cache/acts/smc_rich_heldout"
SMC_ANNOTATION_DIR = "/home/sogang/jaehoon/Analyze-SMC/beat_this_annotations/smc/annotations/beats"


def load_smc_songs(feature_dir: str, annotation_dir: str) -> list[tuple[str, torch.Tensor, np.ndarray]]:
    """Returns [(tid, features [T,512] float32, reference_beat_times_seconds), ...] for every track that
    has both a feature file and an annotation file (verified 1:1 coverage, 217/217, before writing this)."""
    songs = []
    for file_path in sorted(glob.glob(f"{feature_dir}/*.pt")):
        record = torch.load(file_path, map_location="cpu")
        tid = record["tid"]
        annotation_path = f"{annotation_dir}/{tid}.beats"
        if not os.path.exists(annotation_path):
            print(f"[eval_smc_ood] WARNING: no annotation for {tid}, skipping", flush=True)
            continue
        reference_beats = np.loadtxt(annotation_path, dtype=float).reshape(-1)
        songs.append((tid, record["feat"].float(), reference_beats))
    return songs


@torch.no_grad()
def evaluate_smc_condition(model: KalmanVAEBarPointer, songs: list[tuple[str, torch.Tensor, np.ndarray]],
                           device: str, audio_condition: str, tolerance_seconds: float) -> dict:
    """Same deploy path as train_kvae.py's evaluate_leak_condition: encoder mean -> causal Kalman filter ->
    head -> peak-pick. Beat F-measure only (SMC has no downbeat annotations)."""
    model.eval()
    sample_control = SampleControl(encoder="mean", decoder="mean", state_transition="mean", observation="mean")
    beat_scores = []
    num_songs = len(songs)

    for song_index, (tid, features_full, reference_beats) in enumerate(songs):
        source_features = songs[(song_index + 1) % num_songs][1] if audio_condition == "shuffle" else features_full
        num_frames = min(source_features.shape[0], features_full.shape[0])

        if audio_condition == "zero":
            features = torch.zeros(num_frames, 1, model.feature_dim, device=device)
        else:
            features = source_features[:num_frames].unsqueeze(1).to(device)

        a_mean = model.encoder(features.reshape(-1, model.feature_dim)).mean.view(num_frames, 1, model.a_dim)
        filtered_means, *_ = model.ssm.kalman_filter(a_mean, sample_control=sample_control)
        probability = torch.sigmoid(model.head(filtered_means.view(num_frames, model.z_dim))).cpu().numpy()

        if len(reference_beats) >= 2:
            estimated_beats = readout.peak_pick_times(probability[:, 0], SMC_FRAMES_PER_SECOND)
            beat_scores.append(readout.f_measure(reference_beats, estimated_beats, tolerance_seconds))

    mean = lambda values: float(np.nanmean(values)) if values else float("nan")
    return {"beat_f": mean(beat_scores), "num_songs_scored": len(beat_scores)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default="experiments/kvae_prototype/kvae_m1_repro_400ep1000.pt")
    parser.add_argument("--feature_dir", type=str, default=SMC_FEATURE_DIR)
    parser.add_argument("--annotation_dir", type=str, default=SMC_ANNOTATION_DIR)
    parser.add_argument("--eval_beat_tolerance_seconds", type=float, default=0.07)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    model = KalmanVAEBarPointer(
        feature_dim=512, a_dim=ckpt_args.get("a_dim", 8), z_dim=ckpt_args.get("z_dim", 8), K=ckpt_args.get("K", 5),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"[eval_smc_ood] loaded {args.checkpoint} (a_dim={model.a_dim} z_dim={model.z_dim} K={model.ssm.K})", flush=True)
    print(f"[eval_smc_ood] SMC frame rate = {SMC_FRAMES_PER_SECOND} fps (verified empirically, "
         f"NOT config.FRAMES_PER_SECOND=86.13 -- see module docstring)", flush=True)

    songs = load_smc_songs(args.feature_dir, args.annotation_dir)
    print(f"[eval_smc_ood] loaded {len(songs)} SMC songs with matched GT annotations "
         f"(beats-only, no downbeat annotations exist for SMC)", flush=True)

    print("\n--- SMC (Holzapfel) OOD leak test: exact Kalman-filter deploy, beat F-measure only ---", flush=True)
    for condition in ("real", "shuffle", "zero"):
        result = evaluate_smc_condition(model, songs, device, condition, args.eval_beat_tolerance_seconds)
        print(f"{condition:8s}: beat {result['beat_f']:.3f}   (scored {result['num_songs_scored']}/{len(songs)} songs)", flush=True)
    print("(real high + shuffle/zero collapsed => genuinely OOD-audio-driven, not an in-domain-only template)", flush=True)


if __name__ == "__main__":
    main()
