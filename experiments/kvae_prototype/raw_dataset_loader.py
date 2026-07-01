"""Minimal loaders for the 4 raw training datasets (ballroom, beatles, hains, rwc_popular) -- audio path +
beat/downbeat annotation times, for the rubato-augmentation pipeline (which needs to touch raw audio, not
the already-extracted bt_train_rich cache -- that cache has no traceable link back to source filenames).

Per-dataset annotation format and audio/annotation matching logic is COPIED from (not re-derived, to avoid
re-guessing the four formats) the vendored WaveBeat DownbeatDataset.load_annot
(archived at CHART_archive_2026-06-30/extractors/wavebeat/wavebeat/data.py, the SAME class
training/extractors/beat_this_backend.py reuses to build bt_train_rich itself -- see that file's
`from wavebeat.data import DownbeatDataset`), adapted only for this local filesystem's actual layout
(verified empirically before writing this: no `_L+R.wav` suffix on beatles/rwc_popular here, unlike what
that class's directory-discovery assumed for a different filesystem; beatles/label is flat, not nested by
album; rwc_popular audio is nested in numeric-range subdirs but labels are flat, matched by stem).

    ballroom:    ANNOT_DIR/{stem}.beats            -- "time_sec beat_index" (space or tab separated)
    beatles:     ANNOT_DIR/{stem}.txt               -- "time_sec  beat_index" (double-space separated)
    hains:       ANNOT_DIR/{stem}.txt               -- "time_sec beat_index" (single-space separated)
    rwc_popular: ANNOT_DIR/{stem}.BEAT.TXT           -- "time_100ms\\ttime_100ms\\tbeat_code" (beat_code==384 -> downbeat)

beat_index == 1 (ballroom/beatles/hains) or beat_code == 384 (rwc_popular) marks a DOWNBEAT.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

DATASET_ROOT = "/home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data"


@dataclass
class RawSong:
    dataset: str
    audio_path: str
    beat_times: np.ndarray       # seconds, includes downbeats
    downbeat_times: np.ndarray   # seconds, subset of beat_times


def _parse_ballroom(annot_path: str) -> tuple[np.ndarray, np.ndarray]:
    beat_times, downbeat_times = [], []
    for line in open(annot_path):
        line = line.strip("\n").replace("\t", " ")
        parts = [p for p in line.split(" ") if p]
        if len(parts) < 2:
            continue
        time_sec, beat = float(parts[0]), int(float(parts[1]))
        beat_times.append(time_sec)
        if beat == 1:
            downbeat_times.append(time_sec)
    return np.array(beat_times), np.array(downbeat_times)


def _parse_beatles_or_hains(annot_path: str) -> tuple[np.ndarray, np.ndarray]:
    beat_times, downbeat_times = [], []
    for line in open(annot_path):
        line = line.strip("\n").replace("\t", " ")
        parts = [p for p in line.split(" ") if p]
        if len(parts) < 2:
            continue
        time_sec, beat = float(parts[0]), int(float(parts[1]))
        beat_times.append(time_sec)
        if beat == 1:
            downbeat_times.append(time_sec)
    return np.array(beat_times), np.array(downbeat_times)


def _parse_rwc_popular(annot_path: str) -> tuple[np.ndarray, np.ndarray]:
    beat_times, downbeat_times = [], []
    for line in open(annot_path):
        line = line.strip("\n")
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        time_sec = int(parts[0]) / 100.0
        beat = 1 if int(parts[2]) == 384 else 2
        beat_times.append(time_sec)
        if beat == 1:
            downbeat_times.append(time_sec)
    return np.array(beat_times), np.array(downbeat_times)


_PARSERS = {
    "ballroom": (_parse_ballroom, ".beats"),
    "beatles": (_parse_beatles_or_hains, ".txt"),
    "hains": (_parse_beatles_or_hains, ".txt"),
    "rwc_popular": (_parse_rwc_popular, ".BEAT.TXT"),
}


def load_raw_songs(dataset: str, root: str = DATASET_ROOT) -> list[RawSong]:
    """All (audio, annotation) pairs for one dataset, matched by filename stem."""
    if dataset not in _PARSERS:
        raise ValueError(f"unknown dataset: {dataset}")
    parser, annot_ext = _PARSERS[dataset]

    data_dir = os.path.join(root, dataset, "data")
    label_dir = os.path.join(root, dataset, "label")

    audio_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True))
    label_by_stem = {}
    for label_path in glob.glob(os.path.join(label_dir, "**", f"*{annot_ext}"), recursive=True):
        stem = os.path.basename(label_path)
        if stem.endswith(annot_ext):
            stem = stem[: -len(annot_ext)]
        label_by_stem[stem] = label_path

    songs = []
    for audio_path in audio_files:
        stem = os.path.splitext(os.path.basename(audio_path))[0]
        label_path = label_by_stem.get(stem)
        if label_path is None:
            continue
        beat_times, downbeat_times = parser(label_path)
        if len(beat_times) < 8:
            continue
        songs.append(RawSong(dataset=dataset, audio_path=audio_path,
                             beat_times=beat_times, downbeat_times=downbeat_times))
    return songs


def load_all_raw_songs(datasets: tuple[str, ...] = ("ballroom", "beatles", "hains", "rwc_popular"),
                       root: str = DATASET_ROOT) -> list[RawSong]:
    songs = []
    for dataset in datasets:
        songs.extend(load_raw_songs(dataset, root))
    return songs
