"""Build the by-stem audio store: every corpus, one layout, exact names.

``dataset_store/audio_by_stem/<dataset>/<annotation_stem>.<ext>`` is a tree of
SYMLINKS, one per song, named exactly by the Beat This annotation stem. The loader
never matches names at runtime -- audio lookup is a dict ``.get`` on the stem -- and
all the naming archaeology lives HERE, once, where its output is verified. Sources
are never moved or copied; re-running rebuilds and RE-VERIFIES the whole mapping
from the raw stores, so the layout stays re-derivable rather than becoming a one-off
artifact nobody can certify.

Three pairing regimes:

  * ordinary corpora (gtzan, ballroom, beatles, hjdb, hainsworth): audio lives under
    labeled_data/<dir>/ with names that ALMOST match the annotation stems -- an index
    prefix and '.'-vs-'_' differences (gtzan stores 0001_blues.00000.wav for the stem
    gtzan_blues_00000). ``match_audio`` tolerates exactly those differences.
  * rwc: annotations are named by disc and track (rwc_classical_CD4_07); the audio
    rips carry RWC catalog numbers (RWC_C038.wav, multi-movement pieces suffixed
    A-E). The two share no stem, so the pairing is POSITIONAL -- both sequences
    sorted into disc order. That is inherently risky, so it is verified, not
    assumed: equal counts per subset (else the subset is refused outright), zero
    songs whose last annotated beat falls past the end of the audio, and
    corr(last beat time, audio duration) > 0.99 per subset. A shuffled pairing
    collapses that correlation to ~0.01.
  * asap: audio was assembled from MAESTRO recordings into
    _repos/asap-dataset/<Composer>/<Piece>/<Performer>.wav, and the annotation stem
    is exactly that relative path with separators replaced by underscores --
    deterministic, not positional. Annotations with no exact structural match (a
    ``no_repeat`` token on one side only) are DROPPED, deliberately: a repeat and a
    no-repeat performance are different lengths, so fuzzy-matching would pair
    annotations with audio they do not describe -- the silent-misalignment failure
    that quarantined harmonix.

Usage:
    /disk4/anaconda3/envs/vbpm/bin/python -m phasevae.data.unify_audio_layout
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile

from .songs import ANNOTATIONS_ROOT, AUDIO_BY_STEM

LABELED_DATA = (ANNOTATIONS_ROOT.parent / "beat_tracking_db1" / "beat-tracking"
                / "labeled_data")
AUDIO_DIR_NAMES = {"hainsworth": "hains"}   # annotation name -> audio dir, where they differ
AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3")

RWC_AUDIO_ROOT = Path("/disk3/jaehoon/beat-tracking-dataset/_raw")
RWC_SUBSETS = {"classical": ("rwc_classical", "RWC-C"),
               "jazz": ("rwc_jazz", "RWC-J"),
               "royalty-free": ("rwc_royaltyfree", "RWC-R")}
ASAP_AUDIO_ROOT = Path("/disk3/jaehoon/beat-tracking-dataset/_repos/asap-dataset")


def _link(target_dir: Path, stem: str, audio_path: Path):
    link_path = target_dir / f"{stem}{audio_path.suffix}"
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(audio_path)


def match_audio(unprefixed_stem: str, audio_index: dict) -> Optional[Path]:
    """Match an annotation stem to an audio file.

    Exact match first; else tolerate an index prefix and '.'/'_' differences on the
    audio side (gtzan stores '0001_blues.00000.wav' for the stem 'gtzan_blues_00000').
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


def build_ordinary(dataset: str) -> int:
    """Link every annotation whose audio the tolerant matcher can resolve."""
    audio_dir = LABELED_DATA / AUDIO_DIR_NAMES.get(dataset, dataset)
    beats_dir = ANNOTATIONS_ROOT / dataset / "annotations" / "beats"
    if not audio_dir.is_dir() or not beats_dir.is_dir():
        return 0
    audio_index = {p.stem: p for ext in AUDIO_EXTENSIONS
                   for p in audio_dir.rglob(f"*{ext}")}
    if not audio_index:
        return 0
    target = AUDIO_BY_STEM / dataset
    target.mkdir(parents=True, exist_ok=True)
    made, unmatched = 0, 0
    for beats_path in sorted(beats_dir.glob("*.beats")):
        stem = beats_path.stem
        unprefixed = stem[len(dataset) + 1:] if stem.startswith(dataset + "_") else stem
        audio_path = match_audio(unprefixed, audio_index)
        if audio_path is None:
            unmatched += 1
            continue
        _link(target, stem, audio_path)
        made += 1
    print(f"  {dataset}: {made} links, {unmatched} annotations without local audio")
    return made


def _rwc_sort_key(path: Path):
    """Disc-order sort key: C023A sorts after C022 and before C024.

    Plain lexical order would agree too, but only by luck of zero-padding; making the
    intent explicit means a 3-digit overflow cannot reorder it.
    """
    match = re.match(r"RWC_[CJR](\d+)([A-Z]?)", path.stem)
    return (int(match.group(1)), match.group(2)) if match else (10**6, path.stem)


def build_rwc() -> int:
    """Verify the positional disc-order pairing per subset, then link by stem."""
    target = AUDIO_BY_STEM / "rwc"
    target.mkdir(parents=True, exist_ok=True)
    made = 0
    for subset, (dirname, inner) in RWC_SUBSETS.items():
        audio = sorted((RWC_AUDIO_ROOT / dirname / inner).glob("*.wav"),
                       key=_rwc_sort_key)
        annotations = sorted(
            (ANNOTATIONS_ROOT / "rwc").rglob(f"rwc_{subset}_*.beats"),
            key=lambda p: tuple(int(g) for g in re.search(r"_CD(\d+)_(\d+)", p.stem).groups()))
        assert audio and len(audio) == len(annotations), (
            f"rwc {subset}: {len(audio)} audio vs {len(annotations)} annotations -- "
            "positional pairing is only meaningful over equal-length sequences; refusing")

        last_beats = np.array([np.loadtxt(a, ndmin=2)[-1, 0] for a in annotations])
        durations = np.array([soundfile.info(str(w)).duration for w in audio])
        overruns = int((last_beats > durations).sum())
        corr = float(np.corrcoef(last_beats, durations)[0, 1])
        assert overruns == 0 and corr > 0.99, (
            f"rwc {subset}: positional pairing FAILED verification "
            f"(corr {corr:.4f}, {overruns} beat-past-audio songs); refusing to link")

        for annotation, wav in zip(annotations, audio):
            _link(target, annotation.stem, wav)
            made += 1
        print(f"  rwc/{subset}: {len(audio)} links, corr {corr:.4f}, 0 overruns")
    return made


def build_asap() -> int:
    """Link by exact structural path->stem match; drop annotations with no exact match."""
    target = AUDIO_BY_STEM / "asap"
    target.mkdir(parents=True, exist_ok=True)
    audio_by_stem = {"_".join(w.relative_to(ASAP_AUDIO_ROOT).with_suffix("").parts): w
                     for w in ASAP_AUDIO_ROOT.rglob("*.wav")}
    made, dropped = 0, []
    for annotation in sorted((ANNOTATIONS_ROOT / "asap").rglob("*.beats")):
        wav = audio_by_stem.get(annotation.stem)
        if wav is None:
            dropped.append(annotation.stem)
            continue
        _link(target, annotation.stem, wav)
        made += 1
    print(f"  asap: {made} links; {len(dropped)} annotations without an exact structural "
          f"match, dropped deliberately (no_repeat variants): {dropped}")
    return made


if __name__ == "__main__":
    print(f"linking into {AUDIO_BY_STEM}")
    total = 0
    for dataset_dir in sorted(ANNOTATIONS_ROOT.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if dataset_dir.name == "rwc":
            total += build_rwc()
        elif dataset_dir.name == "asap":
            total += build_asap()
        else:
            total += build_ordinary(dataset_dir.name)
    print(f"done: {total} symlinks")
