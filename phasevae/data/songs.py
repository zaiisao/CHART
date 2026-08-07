"""Live song catalog: Beat This annotations + official 8-fold splits + local audio. NO CACHES.

User decision 2026-07-15: activation caches are retired. The June cache builder was a second,
uncertified code path through the frontend (its own audio loading, padding, unchunked forward);
computing activations live through frontends/ makes live == eval by construction, so the whole
"does the cache match the wrapper" question can never arise again. Cost: ~1-2 s of frontend forward
per song per run, paid at run start instead of cached on disk.

This module is the data half of that: the Song record and the coverage report, over
    dataset_store/beat_this_annotations/<dataset>/    the official annotations + splits
    dataset_store/audio_by_stem/<dataset>/            audio SYMLINKS named by annotation stem
The audio tree is built and VERIFIED by ``unify_audio_layout.py``; because link names are
annotation stems, audio lookup at load time is exact -- no name matching. The enumeration
itself (walk the annotations, look up audio, group by fold) is written out in
``dataset.load_dataset``. This module deliberately knows nothing about frontends or crops.

Fold-honesty (standing directive): `fold` is the Beat This CV fold this song is HELD OUT of, read
from the official 8-folds.split. Any evaluation on song s must use checkpoint fold{s.fold}; final0
saw s in training.
"""
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ANNOTATIONS_ROOT = _REPO_ROOT / "dataset_store" / "beat_this_annotations"
AUDIO_BY_STEM = _REPO_ROOT / "dataset_store" / "audio_by_stem"


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


def coverage_report() -> str:
    """Per annotated dataset: how many songs have local audio. Missing audio must be VISIBLE."""
    lines = [f"{'dataset':16s} {'annotated':>9s} {'with audio':>10s}"]
    for dataset_dir in sorted(ANNOTATIONS_ROOT.iterdir()):
        dataset = dataset_dir.name
        beats_dir = dataset_dir / "annotations" / "beats"
        if not beats_dir.is_dir():
            continue
        split_file = dataset_dir / "8-folds.split"
        annotated = (len(split_file.read_text().splitlines()) if split_file.exists()
                     else len(list(beats_dir.glob("*.beats"))))
        audio_dir = AUDIO_BY_STEM / dataset
        linked = ({p.stem for p in audio_dir.iterdir() if p.exists()}   # p.exists()
                  if audio_dir.is_dir() else set())                     # skips dead links
        with_audio = sum(1 for p in beats_dir.glob("*.beats") if p.stem in linked)
        marker = "" if with_audio else "   <- NO LOCAL AUDIO"
        if not split_file.exists() and with_audio:
            marker = "   (test-only: no CV folds)"
        lines.append(f"{dataset:16s} {annotated:9d} {with_audio:10d}{marker}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(coverage_report())
