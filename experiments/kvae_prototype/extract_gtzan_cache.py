"""Extract a bt_train_rich-COMPATIBLE cache for GTZAN (genuine OOD: never in ballroom/beatles/hains/rwc_popular).

GTZAN was NOT in bt_train_rich/bt_val_rich's training mix (that cache's run_rich.sh used
--dataset_include ballroom,beatles,hains,rwc_popular -- gtzan is absent), so this is a real held-out
generalization check, same spirit as the SMC check but with real downbeat labels (SMC has none).

Unlike SMC, GTZAN's raw audio+annotations let us build a cache in the EXACT SAME .pt schema
data/dataset.py's load_songs expects (activations/beat_targets/downbeat_targets), so evaluation can reuse
load_songs directly with zero custom loader code -- see eval_gtzan_ood.py.

Data:
  * Audio: /home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data/gtzan/data/NNNN_genre.NNNNN.wav (993).
  * Annotations: .../gtzan/label/gtzan_genre_NNNNN.beats -- TWO-COLUMN (time_seconds \t beat_position_in_bar),
    position 1 == downbeat (verified by sampling; matches the archived training/phase_generation/gtzan.py
    convention). 999 label files exist but only 993 have a matching audio file (6 GTZAN jazz tracks are
    famously corrupted/missing in this distribution -- a known quirk, not a bug here); matched by
    (genre, 5-digit-index) parsed from both filename patterns.

Features: extract_ood_features.BeatThisResampledExtractor ("final0" checkpoint, bt_train_rich's exact
86.1328125fps recipe: native-hop LogMelSpect -> Beat-This transformer -> linear-interpolate up to the
22050/256 frame grid) -- the SAME extractor eval_smc_ood_fresh.py uses, so both OOD checks share one
frontend/frame-rate convention and are apples-to-apples with each other and with training.

Target binarization: nearest-frame, single-sample spikes (0/1), matching bt_train_rich's own convention --
verified empirically against a real bt_train_rich song (no dilation/smoothing: beat_targets has isolated
1.0s at frame = round(beat_time_seconds * fps), zero elsewhere). downbeat_targets is beat_targets
restricted to rows where the annotation's second column == 1.

Output schema (one .pt per song, saved to --out_dir, default cache/acts/gtzan_rich):
    {"activations": FloatTensor[T,512], "beat_targets": FloatTensor[T], "downbeat_targets": FloatTensor[T],
     "fps": 86.1328125}
This is byte-compatible with data/dataset.py's load_songs (which reads exactly these three tensor keys).

Usage:
    python experiments/kvae_prototype/extract_gtzan_cache.py --out_dir cache/acts/gtzan_rich
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_ood_features import BeatThisResampledExtractor, FRAMES_PER_SECOND

GTZAN_AUDIO_DIR = "/home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data/gtzan/data"
GTZAN_LABEL_DIR = "/home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data/gtzan/label"

_AUDIO_PATTERN = re.compile(r"\d+_([a-z]+)\.(\d+)\.wav$")
_LABEL_PATTERN = re.compile(r"gtzan_([a-z]+)_(\d+)\.beats$")


def build_gtzan_index(audio_dir: str = GTZAN_AUDIO_DIR, label_dir: str = GTZAN_LABEL_DIR) -> list[tuple[str, str, str]]:
    """Returns [(tid, wav_path, beats_path), ...] matched on (genre, 5-digit index)."""
    audio_by_key = {}
    for path in glob.glob(f"{audio_dir}/*.wav"):
        match = _AUDIO_PATTERN.search(os.path.basename(path))
        if match:
            audio_by_key[(match.group(1), match.group(2))] = path
    label_by_key = {}
    for path in glob.glob(f"{label_dir}/*.beats"):
        match = _LABEL_PATTERN.search(os.path.basename(path))
        if match:
            label_by_key[(match.group(1), match.group(2))] = path
    common_keys = sorted(set(audio_by_key) & set(label_by_key))
    return [(f"gtzan_{genre}_{index}", audio_by_key[(genre, index)], label_by_key[(genre, index)])
           for genre, index in common_keys]


def binarize_targets(annotation_path: str, num_frames: int, frames_per_second: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-column (time_seconds, beat_position_in_bar) annotation -> (beat_targets, downbeat_targets) [T]."""
    rows = np.loadtxt(annotation_path, dtype=float).reshape(-1, 2)
    beat_targets = torch.zeros(num_frames)
    downbeat_targets = torch.zeros(num_frames)
    for time_seconds, position in rows:
        frame_index = round(time_seconds * frames_per_second)
        if 0 <= frame_index < num_frames:
            beat_targets[frame_index] = 1.0
            if int(round(position)) == 1:
                downbeat_targets[frame_index] = 1.0
    return beat_targets, downbeat_targets


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", type=str, default="cache/acts/gtzan_rich")
    parser.add_argument("--max_songs", type=int, default=0, help="0 = all matched songs")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    index = build_gtzan_index()
    if args.max_songs > 0:
        index = index[:args.max_songs]
    print(f"[extract_gtzan_cache] matched {len(index)} GTZAN (audio, annotation) pairs", flush=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    extractor = BeatThisResampledExtractor(device=device)

    written, skipped_short = 0, 0
    beats_per_song = []
    for i, (tid, wav_path, beats_path) in enumerate(index):
        out_path = f"{args.out_dir}/{tid}.pt"
        if os.path.exists(out_path):
            written += 1
            continue
        try:
            features = extractor.extract_from_wav(wav_path)
        except Exception as error:
            print(f"  SKIP {tid}: extraction failed ({error})", flush=True)
            continue
        num_frames = features.shape[0]
        beat_targets, downbeat_targets = binarize_targets(beats_path, num_frames, FRAMES_PER_SECOND)
        if beat_targets.sum() < 2:
            skipped_short += 1
            continue
        torch.save({
            "activations": features, "beat_targets": beat_targets, "downbeat_targets": downbeat_targets,
            "fps": FRAMES_PER_SECOND,
        }, out_path)
        beats_per_song.append(float(beat_targets.sum()))
        written += 1
        if (i + 1) % 50 == 0:
            print(f"  extracted {i + 1}/{len(index)} (written={written})", flush=True)

    print(f"[extract_gtzan_cache] wrote {written}/{len(index)} songs to {args.out_dir} "
         f"(skipped {skipped_short} with <2 beats)", flush=True)
    if beats_per_song:
        print(f"[extract_gtzan_cache] mean beats/song = {np.mean(beats_per_song):.1f} "
             f"(sanity: should be well above the 8-beat load_songs floor)", flush=True)


if __name__ == "__main__":
    main()
