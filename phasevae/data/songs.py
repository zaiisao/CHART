"""Live song catalog: Beat This annotations + official 8-fold splits + local audio. NO CACHES."""
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ANNOTATIONS_ROOT = _REPO_ROOT / "dataset_store" / "beat_this_annotations"
AUDIO_BY_STEM = _REPO_ROOT / "dataset_store" / "audio_by_stem"


class Song:
    """One annotated song: (song_id, dataset, fold, audio_path, beats_path)."""

    def __init__(self, song_id, dataset, fold, audio_path, beats_path):
        self.song_id, self.dataset, self.fold = song_id, dataset, fold
        self.audio_path, self.beats_path = audio_path, beats_path

    def beats(self):
        """(beat_times, downbeat_times) in seconds, from the official .beats annotation."""
        annotation = np.loadtxt(self.beats_path, ndmin=2)
        beat_times = annotation[:, 0]
        downbeat_times = np.array([])

        assert annotation.shape[1] in (1, 2), \
            f"unexpected annotation shape {annotation.shape} in {self.beats_path}"

        if annotation.shape[1] == 2:
            # JA: datasets like smc and simac do not have downbeat annotations
            beat_in_bar = annotation[:, 1]
            downbeat_times = beat_times[beat_in_bar == 1]

        return beat_times, downbeat_times

    def __repr__(self):
        return f"Song({self.song_id}, fold={self.fold})"


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
