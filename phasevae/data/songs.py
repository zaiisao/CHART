"""Live song catalog: Beat This annotations + official 8-fold splits + local audio. NO CACHES.

User decision 2026-07-15: activation caches are retired. The June cache builder was a second,
uncertified code path through the frontend (its own audio loading, padding, unchunked forward);
computing activations live through frontends/ makes live == eval by construction, so the whole
"does the cache match the wrapper" question can never arise again. Cost: ~1-2 s of frontend forward
per song per run, paid at run start instead of cached on disk.

This module is the data half of that: it enumerates songs as (stem, dataset, fold, audio_path,
beats_path) from
    dataset_store/beat_this_annotations/<dataset>/           the official annotations + splits
    dataset_store/beat_tracking_db1/.../labeled_data/<dir>/data/   local audio
and parses the .beats files. It deliberately knows nothing about frontends or bar-pointer models.

Fold-honesty (standing directive): `fold` is the Beat This CV fold this song is HELD OUT of, read
from the official 8-folds.split. Any evaluation on song s must use checkpoint fold{s.fold}; final0
saw s in training.
"""
import re
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ANNOTATIONS_ROOT = _REPO_ROOT / "dataset_store" / "beat_this_annotations"
AUDIO_ROOT = (_REPO_ROOT / "dataset_store" / "beat_tracking_db1" / "beat-tracking"
              / "labeled_data")

# annotation dataset name -> audio directory name, where they differ
AUDIO_DIR_NAMES = {"hainsworth": "hains"}
AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3")

# ---------------------------------------------------------------------------------------
# RWC: audio lives OUTSIDE the labeled_data tree and under a different naming scheme.
#
# The annotations are named by disc and track (``rwc_classical_CD4_07``); the audio is named
# by sequential piece number (``RWC_C038.wav``), with multi-movement pieces suffixed A-E. The
# two share no stem, so the ordinary matcher finds nothing and rwc silently reported 0 usable
# songs -- which is how a whole corpus went missing. The audio was there the entire time.
#
# This matters more than one dataset's worth of songs: rwc_classical is 2:17 / 3:18 / 4:19,
# the only near-balanced meter stratum available anywhere, against a corpus that is otherwise
# 86% 4/4. rwc_popular is 4:99 and carries no meter signal at all.
#
# The pairing is positional (both sequences are in disc order) and is VERIFIED, not assumed:
# corr(last beat time, audio duration) = 0.9999 / 0.9986 / 0.9972 for classical / jazz /
# royalty-free, with zero songs whose last beat falls past the end of the audio. Shuffling the
# pairing collapses that to corr ~0.01 with roughly half the songs overrunning.
RWC_AUDIO_ROOT = Path("/disk3/jaehoon/beat-tracking-dataset/_raw")
RWC_SUBSETS = {"classical": ("rwc_classical", "RWC-C"),
               "jazz": ("rwc_jazz", "RWC-J"),
               "royalty-free": ("rwc_royaltyfree", "RWC-R")}


# ---------------------------------------------------------------------------------------
# ASAP: audio also lives outside labeled_data. It was assembled from MAESTRO recordings into
# ``_repos/asap-dataset/<Composer>/<Piece>/<Performer>.wav``, and the annotation stem is that
# relative path with separators replaced by underscores -- so unlike rwc the mapping is
# deterministic, not positional. 469 of 473 annotations match exactly.
#
# The 4 that do not are a ``no_repeat`` token present on one side only (e.g. annotation
# ``Beethoven_Piano_Sonatas_21-1_Sekino05M`` vs audio ``..._21-1_no_repeat_Sekino05M``).
# They are DROPPED, deliberately: a repeat and a no-repeat performance are different lengths,
# so fuzzy-matching them would pair annotations with audio they do not describe. Four missing
# songs is cheap; a silently misaligned pair is the failure that quarantined harmonix.
ASAP_AUDIO_ROOT = Path("/disk3/jaehoon/beat-tracking-dataset/_repos/asap-dataset")


def _asap_audio_index() -> dict:
    """Annotation stem -> audio path, by exact structural name match only."""
    if not ASAP_AUDIO_ROOT.is_dir():
        return {}
    return {"_".join(w.relative_to(ASAP_AUDIO_ROOT).with_suffix("").parts): w
            for w in ASAP_AUDIO_ROOT.rglob("*.wav")}


def _rwc_sort_key(path: Path):
    """Disc-order sort key: C023A sorts after C022 and before C024.

    Plain lexical order would agree too, but only by luck of zero-padding; making the
    intent explicit means a 3-digit overflow cannot reorder it.
    """
    match = re.match(r"RWC_[CJR](\d+)([A-Z]?)", path.stem)
    return (int(match.group(1)), match.group(2)) if match else (10**6, path.stem)


def _rwc_audio_index() -> dict:
    """Annotation stem WITHOUT the ``rwc_`` prefix -> audio path, by position in each subset.

    Keyed unprefixed because ``iter_songs`` strips the dataset prefix before matching.
    """
    index = {}
    for subset, (dirname, inner) in RWC_SUBSETS.items():
        audio_dir = RWC_AUDIO_ROOT / dirname / inner
        if not audio_dir.is_dir():
            continue
        audio = sorted(audio_dir.glob("*.wav"), key=_rwc_sort_key)
        annotations = sorted(
            (ANNOTATIONS_ROOT / "rwc").rglob(f"rwc_{subset}_*.beats"),
            key=lambda p: tuple(int(g) for g in re.search(r"_CD(\d+)_(\d+)", p.stem).groups()))
        if len(audio) != len(annotations):
            # Positional pairing is only meaningful if the two sequences are the same length.
            # Skip rather than pair anything: a silently misaligned corpus is far worse than
            # an absent one, and this project has already quarantined one for that reason.
            continue
        index.update({a.stem[len("rwc_"):]: w for a, w in zip(annotations, audio)})
    return index


class Song:
    """One annotated song: (stem, dataset, fold, audio_path, beats_path)."""

    def __init__(self, stem, dataset, fold, audio_path, beats_path):
        self.stem, self.dataset, self.fold = stem, dataset, fold
        self.audio_path, self.beats_path = audio_path, beats_path

    def beats(self):
        """(beat_times, downbeat_times) in seconds, from the official .beats annotation."""
        annotation = np.loadtxt(self.beats_path, ndmin=2)
        beat_times = annotation[:, 0]
        downbeat_times = beat_times[annotation[:, 1] == 1] if annotation.shape[1] > 1 else \
            np.array([])
        return beat_times, downbeat_times

    def __repr__(self):
        return f"Song({self.stem}, fold={self.fold})"


def _index_audio(dataset: str) -> dict:
    """filename-sans-extension -> path, recursive (rwc nests audio in subdirectories)."""
    if dataset == "rwc":
        return _rwc_audio_index()          # positional pairing, not stem matching -- see above
    if dataset == "asap":
        return _asap_audio_index()         # structural path->stem match -- see above
    audio_dir = AUDIO_ROOT / AUDIO_DIR_NAMES.get(dataset, dataset)
    if not audio_dir.is_dir():
        return {}
    return {p.stem: p for ext in AUDIO_EXTENSIONS for p in audio_dir.rglob(f"*{ext}")}


def _match_audio(unprefixed_stem: str, audio_index: dict) -> Optional[Path]:
    """Match an annotation stem to an audio file.

    Exact match first; else tolerate an index prefix and '.'/'_' differences on the audio
    side (gtzan stores '0001_blues.00000.wav' for the stem 'gtzan_blues_00000').
    """
    if unprefixed_stem in audio_index:
        return audio_index[unprefixed_stem]
    normalized_stem = unprefixed_stem.replace(".", "_")
    for name, path in audio_index.items():
        normalized_name = name.replace(".", "_")
        if (normalized_name == normalized_stem
                or normalized_name.endswith("_" + normalized_stem)
                or normalized_name.endswith("-" + normalized_stem)):
            return path
    return None


def iter_songs(datasets=None, folds=None):
    """All songs with BOTH an annotation and local audio. datasets/folds filter if given.

    Songs whose dataset has annotations but no local audio are silently absent -- use
    coverage_report() to see exactly what is and is not available, so missing audio is a known
    fact rather than a silent hole.
    """
    songs = []
    for dataset_dir in sorted(ANNOTATIONS_ROOT.iterdir()):
        dataset = dataset_dir.name
        if datasets is not None and dataset not in datasets:
            continue
        audio_index = _index_audio(dataset)
        beats_dir = dataset_dir / "annotations" / "beats"
        if not audio_index or not beats_dir.is_dir():
            continue
        split_file = dataset_dir / "8-folds.split"
        if split_file.exists():
            entries = [line.split("\t") for line in split_file.read_text().splitlines()]
        else:
            # No CV split = a test-only dataset in the Beat This protocol (gtzan). fold = None:
            # it was held out of EVERY checkpoint, so any checkpoint is fold-honest on it.
            entries = [(p.stem, None) for p in sorted(beats_dir.glob("*.beats"))]
        for stem, fold in entries:
            fold = int(fold) if fold is not None else None
            if folds is not None and fold not in folds:
                continue
            beats_path = beats_dir / f"{stem}.beats"
            unprefixed = stem[len(dataset) + 1:] if stem.startswith(dataset + "_") else stem
            audio_path = _match_audio(unprefixed, audio_index)
            if audio_path is not None and beats_path.exists():
                songs.append(Song(stem, dataset, fold, audio_path, beats_path))
    return songs


def coverage_report() -> str:
    """Per annotated dataset: how many songs have local audio. Missing audio must be VISIBLE."""
    lines = [f"{'dataset':16s} {'annotated':>9s} {'with audio':>10s}"]
    for dataset_dir in sorted(ANNOTATIONS_ROOT.iterdir()):
        beats_dir = dataset_dir / "annotations" / "beats"
        if not beats_dir.is_dir():
            continue
        split_file = dataset_dir / "8-folds.split"
        annotated = (len(split_file.read_text().splitlines()) if split_file.exists()
                     else len(list(beats_dir.glob("*.beats"))))
        with_audio = len(iter_songs(datasets=[dataset_dir.name]))
        marker = "" if with_audio else "   <- NO LOCAL AUDIO"
        if not split_file.exists() and with_audio:
            marker = "   (test-only: no CV folds)"
        lines.append(f"{dataset_dir.name:16s} {annotated:9d} {with_audio:10d}{marker}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(coverage_report())
    songs = iter_songs()
    print(f"\ntotal usable songs: {len(songs)}")
    per_fold = {f: sum(1 for s in songs if s.fold == f) for f in range(8)}
    print(f"per fold: {per_fold}")
    example = songs[0]
    beat_times, downbeat_times = example.beats()
    print(f"\nexample: {example} -> {len(beat_times)} beats, {len(downbeat_times)} downbeats, "
          f"audio={example.audio_path.name}")
