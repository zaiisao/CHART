"""The song catalog and the CV split, plus the true-phase reference used for scoring."""
from __future__ import annotations

import numpy as np

from .songs import ANNOTATIONS_ROOT, AUDIO_BY_STEM, Song

MIN_DOWNBEATS = 4


def true_phase(crop):
    """(phase [T], valid [T]) -- the true bar phase per frame. SCORING ONLY."""
    n = len(crop["y"])
    times = crop["t0"] + np.arange(n) / crop["fps"]
    downbeats = crop["anchors"]
    turns = 2.0 * np.pi * np.arange(len(downbeats))
    phase = np.interp(times, downbeats, turns, left=np.nan, right=np.nan)
    valid = ~np.isnan(phase)
    phase = np.where(valid, phase, 0.0)
    return np.mod(phase, 2.0 * np.pi).astype(np.float32), valid


def load_catalog(datasets=None) -> dict:
    """The song catalog, grouped by CV fold (fold -> [Song])."""
    songs_by_fold: dict = {}
    for dataset_dir in sorted(ANNOTATIONS_ROOT.iterdir()):
        dataset = dataset_dir.name

        if datasets is not None and dataset not in datasets:
            continue

        beats_dir = dataset_dir / "annotations" / "beats"
        audio_dir = AUDIO_BY_STEM / dataset

        if not beats_dir.is_dir() or not audio_dir.is_dir():
            continue

        audio_by_stem = {p.stem: p for p in audio_dir.iterdir()}
        split_file = dataset_dir / "8-folds.split"

        if split_file.exists():
            entries = [line.split("\t") for line in split_file.read_text().splitlines()]
        else:
            # No CV split = a test-only dataset in the Beat This protocol
            entries = [(p.stem, None) for p in sorted(beats_dir.glob("*.beats"))]

        for song_id, fold in entries:
            audio_path = audio_by_stem.get(song_id)
            if audio_path is not None and (beats_dir / f"{song_id}.beats").exists():
                fold = int(fold) if fold is not None else None
                song = Song(song_id, dataset, fold, audio_path, beats_dir / f"{song_id}.beats")
                songs_by_fold.setdefault(fold, []).append(song)

    return songs_by_fold


def split_songs(val_fold, limit_per_fold=None):
    """(train, val, test) Song lists: train folds / fold VAL_FOLD / fold None (gtzan)."""
    # JA: folds is a dictionary with keys 0, 1, 2, 3, 4, 5, 6, 7, None. Each key
    # corresponds to a fold number and the value is a list of Song objects in that fold.
    # None corresponds to the test-only group that is held out of every checkpoint.
    folds = load_catalog()

    if limit_per_fold is not None:
        folds = {fold: songs[:limit_per_fold] for fold, songs in folds.items()}

    train_folds = [fold for fold in folds if fold not in (None, val_fold)]

    train_songs = [song for fold in train_folds for song in folds[fold]]
    val_songs = folds.get(val_fold, [])
    test_songs = folds.get(None, [])

    return train_songs, val_songs, test_songs
