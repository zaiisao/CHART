"""Genuine OOD check on SMC (Holzapfel) with FRESHLY-EXTRACTED features -- checkpoint-matched this time.

SUPERSEDES eval_smc_ood.py: that script used cache/acts/smc_rich_heldout's PRE-CACHED features, which
turned out to be extracted with 8 DIFFERENT per-fold Beat-This checkpoints (fold0..fold7, see the archived
tests/extract_smc_rich.py's `load_model(f"fold{fo}", dev)`), not "final0" -- the checkpoint
bt_train_rich/bt_val_rich (and hence the trained KVAE model) actually used. That mismatch confounds
"genuine OOD audio difficulty" with "frontend weight mismatch", so those numbers are not trustworthy in
isolation. This script re-extracts from RAW AUDIO with extract_ood_features.BeatThisResampledExtractor
(checkpoint "final0", bt_train_rich's exact 86.1328125fps recipe -- see that module's docstring), removing
the confound and making this directly comparable to the GTZAN OOD check (eval_gtzan_ood.py), which uses
the identical extractor.

Data:
  * Raw audio: /home/sogang/jaehoon/Analyze-SMC/SMC_MIREX/SMC_MIREX_Audio/SMC_{NNN}.wav (217 files).
  * Annotations: .../beat_this_annotations/smc/annotations/beats/smc_{nnn}.beats -- SINGLE-COLUMN (just
    beat times in seconds, one per line). info.json confirms "has_downbeats": false -- SMC genuinely has
    no downbeat annotations (a well-known property of this benchmark, not a bug) -- beat F-measure only.
  * tid casing: audio is SMC_NNN.wav (upper), annotations are smc_nnn.beats (lower) -- matched via the
    zero-padded numeric stem, case-insensitively.

Deploy path: identical to train_kvae.py's evaluate_leak_condition (encoder(h) mean -> causal deterministic
Kalman filter -> head(filtered z) -> sigmoid -> peak-pick), fed the freshly-extracted SMC features. Same
leak-test protocol (real / shuffle / zero) for the same reason as everywhere else in this project.

Usage:
    python experiments/kvae_prototype/eval_smc_ood_fresh.py --checkpoint experiments/kvae_prototype/kvae_m1_repro_400ep1000.pt
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import readout
from model.kalman_vae_bar_pointer import KalmanVAEBarPointer
from extract_ood_features import BeatThisResampledExtractor, FRAMES_PER_SECOND

_THIRD_PARTY_KVAE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                 "third_party", "kalman-vae")
if _THIRD_PARTY_KVAE not in sys.path:
    sys.path.insert(0, _THIRD_PARTY_KVAE)
from kvae.sample_control import SampleControl

SMC_AUDIO_DIR = "/home/sogang/jaehoon/Analyze-SMC/SMC_MIREX/SMC_MIREX_Audio"
SMC_ANNOTATION_DIR = "/home/sogang/jaehoon/Analyze-SMC/beat_this_annotations/smc/annotations/beats"
FEATURE_CACHE_DIR = "cache/acts/smc_fresh_final0"   # cached alongside GTZAN's cache for reproducibility


def _numeric_stem(filename: str) -> str:
    match = re.search(r"(\d+)", os.path.basename(filename))
    return match.group(1) if match else ""


def build_smc_index() -> list[tuple[str, str, str]]:
    """Returns [(tid, wav_path, beats_path), ...] matched case-insensitively on the numeric stem."""
    wav_by_number = {_numeric_stem(f): f for f in glob.glob(f"{SMC_AUDIO_DIR}/SMC_*.wav")}
    beats_by_number = {_numeric_stem(f): f for f in glob.glob(f"{SMC_ANNOTATION_DIR}/smc_*.beats")}
    common_numbers = sorted(set(wav_by_number) & set(beats_by_number))
    return [(f"smc_{number}", wav_by_number[number], beats_by_number[number]) for number in common_numbers]


def extract_and_cache_all(index: list[tuple[str, str, str]], device: str, cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    extractor = BeatThisResampledExtractor(device=device)
    for i, (tid, wav_path, _) in enumerate(index):
        cache_path = f"{cache_dir}/{tid}.pt"
        if os.path.exists(cache_path):
            continue
        features = extractor.extract_from_wav(wav_path)
        torch.save({"features": features, "tid": tid}, cache_path)
        if (i + 1) % 20 == 0:
            print(f"  extracted {i + 1}/{len(index)}", flush=True)


def load_smc_songs(index: list[tuple[str, str, str]], cache_dir: str) -> list[tuple[str, torch.Tensor, np.ndarray]]:
    songs = []
    for tid, _, beats_path in index:
        features = torch.load(f"{cache_dir}/{tid}.pt", map_location="cpu")["features"]
        reference_beats = np.loadtxt(beats_path, dtype=float).reshape(-1)
        songs.append((tid, features, reference_beats))
    return songs


@torch.no_grad()
def evaluate_smc_condition(model: KalmanVAEBarPointer, songs: list[tuple[str, torch.Tensor, np.ndarray]],
                           device: str, audio_condition: str, tolerance_seconds: float) -> dict:
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
            estimated_beats = readout.peak_pick_times(probability[:, 0], FRAMES_PER_SECOND)
            beat_scores.append(readout.f_measure(reference_beats, estimated_beats, tolerance_seconds))

    mean = lambda values: float(np.nanmean(values)) if values else float("nan")
    return {"beat_f": mean(beat_scores), "num_songs_scored": len(beat_scores)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default="experiments/kvae_prototype/kvae_m1_repro_400ep1000.pt")
    parser.add_argument("--eval_beat_tolerance_seconds", type=float, default=0.07)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--feature_cache_dir", type=str, default=FEATURE_CACHE_DIR)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    index = build_smc_index()
    print(f"[eval_smc_ood_fresh] matched {len(index)} SMC (audio, annotation) pairs", flush=True)

    print("[eval_smc_ood_fresh] extracting fresh final0 features at 86.1328125 fps (cached under "
         f"{args.feature_cache_dir}, skips already-cached files)...", flush=True)
    extract_and_cache_all(index, device, args.feature_cache_dir)

    songs = load_smc_songs(index, args.feature_cache_dir)
    print(f"[eval_smc_ood_fresh] loaded {len(songs)} songs with fresh final0 features", flush=True)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    model = KalmanVAEBarPointer(
        feature_dim=512, a_dim=ckpt_args.get("a_dim", 8), z_dim=ckpt_args.get("z_dim", 8), K=ckpt_args.get("K", 5),
        Q_reg=ckpt_args.get("Q_reg", 1e-3),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"[eval_smc_ood_fresh] loaded {args.checkpoint} (a_dim={model.a_dim} z_dim={model.z_dim} K={model.ssm.K})", flush=True)

    print("\n--- SMC (Holzapfel) OOD leak test: FRESH final0 features, exact Kalman-filter deploy, beat F only ---", flush=True)
    for condition in ("real", "shuffle", "zero"):
        result = evaluate_smc_condition(model, songs, device, condition, args.eval_beat_tolerance_seconds)
        print(f"{condition:8s}: beat {result['beat_f']:.3f}   (scored {result['num_songs_scored']}/{len(songs)} songs)", flush=True)
    print("(real high + shuffle/zero collapsed => genuinely OOD-audio-driven, not an in-domain-only template)", flush=True)


if __name__ == "__main__":
    main()
