"""Generate the local-rubato-augmented training cache: for each of a sample of raw training songs (from
ballroom/beatles/hains/rwc_popular -- the SAME 4 datasets bt_train_rich itself draws from, per
run_rich.sh's --dataset_include; SMC and GTZAN audio are never touched here), synthesize 1-2 rubato variants
(rubato_augment.py: independent per-segment stretch, cut at beat boundaries) and extract fresh [T,512]
final0-checkpoint features (extract_ood_features.BeatThisResampledExtractor, the SAME extractor/frame-rate
convention already validated for SMC/GTZAN) plus correspondingly-warped beat/downbeat targets.

Output schema matches data/dataset.py's Song / load_songs expectations exactly (activations/beat_targets/
downbeat_targets/fps), written to a SEPARATE cache dir (cache/acts/bt_train_rubato) -- ADDED to, not
replacing, the original bt_train_rich pool (train_kvae_rubato.py samples from the union).

Usage:
    python experiments/kvae_prototype/extract_rubato_cache.py --num_source_songs 400 --variants_per_song 2
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_ood_features import BeatThisResampledExtractor, FRAMES_PER_SECOND, TARGET_SAMPLE_RATE
from raw_dataset_loader import load_all_raw_songs, RawSong
from rubato_augment import rubato_augment_song


def binarize(times: np.ndarray, num_frames: int, frames_per_second: float) -> torch.Tensor:
    target = torch.zeros(num_frames)
    for t in times:
        frame_index = round(t * frames_per_second)
        if 0 <= frame_index < num_frames:
            target[frame_index] = 1.0
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", type=str, default="cache/acts/bt_train_rubato")
    parser.add_argument("--num_source_songs", type=int, default=400)
    parser.add_argument("--variants_per_song", type=int, default=2)
    parser.add_argument("--beats_per_segment", type=int, default=8)
    parser.add_argument("--stretch_min_percent", type=float, default=-25.0)
    parser.add_argument("--stretch_max_percent", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=1)  # matches bt_train_rich's own seed=1 selection convention
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    all_songs = load_all_raw_songs()
    print(f"[extract_rubato_cache] {len(all_songs)} raw songs available across ballroom/beatles/hains/rwc_popular", flush=True)

    rng_select = np.random.default_rng(args.seed)
    indices = rng_select.permutation(len(all_songs))[: args.num_source_songs]
    source_songs: list[RawSong] = [all_songs[i] for i in indices]
    print(f"[extract_rubato_cache] sampled {len(source_songs)} source songs (seed={args.seed})", flush=True)

    extractor = BeatThisResampledExtractor(device=device)

    written, skipped = 0, 0
    for song_index, song in enumerate(source_songs):
        try:
            waveform, sample_rate = torchaudio.load(song.audio_path)
        except Exception as error:
            print(f"  SKIP {song.audio_path}: load failed ({error})", flush=True)
            skipped += 1
            continue
        waveform = waveform.mean(dim=0)
        if sample_rate != TARGET_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, TARGET_SAMPLE_RATE)

        for variant_index in range(args.variants_per_song):
            out_path = f"{args.out_dir}/{song.dataset}_{song_index:04d}_v{variant_index}.pt"
            if os.path.exists(out_path):
                written += 1
                continue
            rng = np.random.default_rng(args.seed * 100000 + song_index * 10 + variant_index)
            try:
                augmented_waveform, warped_beats, warped_downbeats = rubato_augment_song(
                    waveform, TARGET_SAMPLE_RATE, song.beat_times, song.downbeat_times,
                    beats_per_segment=args.beats_per_segment,
                    stretch_range_percent=(args.stretch_min_percent, args.stretch_max_percent), rng=rng)
            except Exception as error:
                print(f"  SKIP {song.audio_path} variant {variant_index}: augment failed ({error})", flush=True)
                skipped += 1
                continue

            features = extractor.extract_from_waveform(augmented_waveform, TARGET_SAMPLE_RATE)
            num_frames = features.shape[0]
            beat_targets = binarize(warped_beats, num_frames, FRAMES_PER_SECOND)
            downbeat_targets = binarize(warped_downbeats, num_frames, FRAMES_PER_SECOND)
            if beat_targets.sum() < 8:
                skipped += 1
                continue

            torch.save({
                "activations": features, "beat_targets": beat_targets, "downbeat_targets": downbeat_targets,
                "fps": FRAMES_PER_SECOND, "source_dataset": song.dataset, "source_audio": song.audio_path,
            }, out_path)
            written += 1

        if (song_index + 1) % 25 == 0:
            print(f"  processed {song_index + 1}/{len(source_songs)} source songs (written={written} skipped={skipped})", flush=True)

    print(f"[extract_rubato_cache] DONE: wrote {written} augmented songs to {args.out_dir} (skipped {skipped})", flush=True)


if __name__ == "__main__":
    main()
