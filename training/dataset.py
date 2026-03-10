"""Dataset definitions for CHART training."""

from __future__ import annotations

import bisect
from collections.abc import Sized
from dataclasses import dataclass
from pathlib import Path
import random
from typing import cast

import numpy as np
import torch
from torch.utils.data import Dataset


class ActivationDataset(Dataset):
    """Dataset for paired activation/phase `.npy` files used by CHART.

    The dataset scans two directories, matches files by basename, and yields
    fixed-length training windows for autoregressive sequence-to-sequence
    training.
    """

    def __init__(self, activations_dir: str | Path, phases_dir: str | Path, seq_len: int = 1024) -> None:
        """Initialize dataset by matching `.npy` files across two directories.

        Args:
            activations_dir: Directory containing activation arrays with shape `[T, 2]`.
            phases_dir: Directory containing phase arrays with shape `[T, 3]`.
            seq_len: Fixed sequence length produced by each dataset item.
        """
        self.activations_dir = Path(activations_dir)
        self.phases_dir = Path(phases_dir)
        self.seq_len = seq_len

        if seq_len <= 0:
            raise ValueError("seq_len must be a positive integer.")
        if not self.activations_dir.is_dir():
            raise FileNotFoundError(f"Activations directory not found: {self.activations_dir}")
        if not self.phases_dir.is_dir():
            raise FileNotFoundError(f"Phases directory not found: {self.phases_dir}")

        activation_map = {path.stem: path for path in self.activations_dir.rglob("*.npy")}
        phase_map = {path.stem: path for path in self.phases_dir.rglob("*.npy")}

        common_keys = sorted(set(activation_map).intersection(phase_map))
        self.matched_files: list[tuple[Path, Path]] = [
            (activation_map[key], phase_map[key]) for key in common_keys
        ]

        if len(self.matched_files) == 0:
            raise RuntimeError(
                "No matched .npy files found between activations_dir and phases_dir. "
                "Expected shared basenames (e.g., song_01.npy)."
            )

    def __len__(self) -> int:
        """Return total number of matched activation/phase files."""
        return len(self.matched_files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Load, resize to fixed length, and build autoregressive history.

        Args:
            index: Dataset index.

        Returns:
            Dictionary with:
            - `activations`: `(seq_len, 2)` float32 tensor
            - `z_target`: `(seq_len, 3)` float32 tensor
            - `z_history`: `(seq_len, 3)` float32 tensor shifted right by 1
        """
        activation_path, phase_path = self.matched_files[index]

        activation_np = np.load(activation_path)
        phase_np = np.load(phase_path)

        activations = torch.as_tensor(activation_np, dtype=torch.float32)
        z_target = torch.as_tensor(phase_np, dtype=torch.float32)

        if activations.ndim != 2 or activations.shape[1] != 2:
            raise ValueError(
                f"Expected activations shape [T, 2], got {tuple(activations.shape)} for {activation_path}"
            )
        if z_target.ndim != 2 or z_target.shape[1] != 3:
            raise ValueError(
                f"Expected phases shape [T, 3], got {tuple(z_target.shape)} for {phase_path}"
            )
        if activations.shape[0] != z_target.shape[0]:
            raise ValueError(
                "Activation and phase lengths must match: "
                f"{activations.shape[0]} vs {z_target.shape[0]} for {activation_path.stem}"
            )

        total_len = activations.shape[0]

        if total_len < self.seq_len:
            pad_len = self.seq_len - total_len
            activations = torch.cat(
                [activations, torch.zeros(pad_len, 2, dtype=torch.float32)], dim=0
            )
            z_target = torch.cat(
                [z_target, torch.zeros(pad_len, 3, dtype=torch.float32)], dim=0
            )
        elif total_len > self.seq_len:
            start = random.randint(0, total_len - self.seq_len)
            end = start + self.seq_len
            activations = activations[start:end]
            z_target = z_target[start:end]

        z_history = torch.zeros_like(z_target)
        z_history[1:] = z_target[:-1]

        return {
            "activations": activations,
            "z_target": z_target,
            "z_history": z_history,
        }


class AudioPhaseBridgeDataset(Dataset):
    """Bridge dataset for end-to-end extractor -> CHART training.

    This dataset wraps an audio-based extractor dataset (expected tuple output
    `(audio, target, ...)` and `audio_files` metadata) and pairs each audio file
    with a phase `.npy` file for CHART.
    """

    def __init__(self, source_dataset: Dataset, phases_dir: str | Path) -> None:
        self.source_dataset = source_dataset
        self.phases_dir = Path(phases_dir)

        if not self.phases_dir.is_dir():
            raise FileNotFoundError(f"Phases directory not found: {self.phases_dir}")

        phase_files = list(self.phases_dir.rglob("*.npy"))
        if len(phase_files) == 0:
            raise RuntimeError(f"No .npy phase files found in: {self.phases_dir}")

        self.phase_by_stem: dict[str, Path] = {phase_path.stem: phase_path for phase_path in phase_files}

        audio_files = getattr(self.source_dataset, "audio_files", None)
        if audio_files is None:
            raise AttributeError("source_dataset must expose 'audio_files' for filename matching.")

        self.phase_path_by_audio_stem: dict[str, Path] = {}
        for audio_path in audio_files:
            phase_path = self._resolve_phase_path_from_audio(audio_path)
            audio_stem = Path(audio_path).stem.replace("_L+R", "")
            self.phase_path_by_audio_stem[audio_stem] = phase_path

    def __len__(self) -> int:
        if isinstance(self.source_dataset, Sized):
            return len(cast(Sized, self.source_dataset))
        raise TypeError("source_dataset must implement __len__().")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.source_dataset[index]
        if not isinstance(sample, tuple) or len(sample) < 2:
            raise TypeError("Expected tuple sample from source dataset: (audio, target, ...)")

        audio = sample[0].float()
        extractor_target = sample[1].float()

        audio_path = self._get_audio_path_for_index(index)

        audio_stem = Path(audio_path).stem.replace("_L+R", "")
        phase_path = self.phase_path_by_audio_stem.get(audio_stem)
        if phase_path is None:
            phase_path = self._resolve_phase_path_from_audio(audio_path)
            self.phase_path_by_audio_stem[audio_stem] = phase_path
        z_target = torch.as_tensor(np.load(phase_path), dtype=torch.float32)

        if z_target.ndim != 2 or z_target.shape[1] != 3:
            raise ValueError(
                f"Expected phase shape [T, 3], got {tuple(z_target.shape)} for {phase_path}"
            )

        target_len = extractor_target.shape[-1]
        z_target = self._fit_phase_length(z_target, target_len)

        z_history = torch.zeros_like(z_target)
        z_history[1:] = z_target[:-1]

        return {
            "audio": audio,
            "extractor_target": extractor_target,
            "z_target": z_target,
            "z_history": z_history,
            "audio_path": str(audio_path),
            "phase_path": str(phase_path),
        }

    def _get_audio_path_for_index(self, index: int) -> str:
        get_audio_path_for_index = getattr(self.source_dataset, "get_audio_path_for_index", None)
        if callable(get_audio_path_for_index):
            return str(get_audio_path_for_index(index))

        audio_files = getattr(self.source_dataset, "audio_files", None)
        if audio_files is None or len(audio_files) == 0:
            raise AttributeError(
                "source_dataset must expose 'audio_files' or implement get_audio_path_for_index(index)."
            )

        audio_idx = index % len(audio_files)
        return str(audio_files[audio_idx])

    def _resolve_phase_path_from_audio(self, audio_path: str | Path) -> Path:
        basename = Path(audio_path).name
        candidates = [
            Path(basename).stem,
            Path(basename).stem.replace("_L+R", ""),
            basename.replace(".wav", ""),
            basename.replace("_L+R.wav", ""),
        ]

        for key in candidates:
            if key in self.phase_by_stem:
                return self.phase_by_stem[key]

        raise FileNotFoundError(
            f"No phase file found for audio '{audio_path}'. Tried keys: {candidates} in {self.phases_dir}"
        )

    @staticmethod
    def _fit_phase_length(z_target: torch.Tensor, target_len: int) -> torch.Tensor:
        current_len = z_target.shape[0]

        if current_len == target_len:
            return z_target

        if current_len > target_len:
            start = (current_len - target_len) // 2
            end = start + target_len
            return z_target[start:end]

        pad_len = target_len - current_len
        pad = torch.zeros(pad_len, z_target.shape[1], dtype=z_target.dtype)
        return torch.cat([z_target, pad], dim=0)


class WaveBeatPhaseDataset(AudioPhaseBridgeDataset):
    """Backward-compatible alias for previous class name."""


@dataclass(frozen=True)
class WaveBeatDatasetSpec:
    """Resolved dataset specification for a WaveBeat-compatible source."""

    key: str
    wavebeat_dataset: str
    audio_dir: Path
    annot_dir: Path


class BaseWaveBeatSpecResolver:
    """Base class for root-folder dataset resolvers."""

    key: str

    def resolve(self, root_dir: Path) -> WaveBeatDatasetSpec | None:
        raise NotImplementedError


def _resolve_data_label_dirs(root_dir: Path, dataset_dir_name: str) -> tuple[Path, Path] | None:
    layout_candidates = [
        root_dir / "labeled_data" / dataset_dir_name,
        root_dir / dataset_dir_name,
    ]

    for dataset_root in layout_candidates:
        audio_dir = dataset_root / "data"
        annot_dir = dataset_root / "label"
        if audio_dir.is_dir() and annot_dir.is_dir():
            return audio_dir, annot_dir

    return None


class BallroomSpecResolver(BaseWaveBeatSpecResolver):
    key = "ballroom"

    def resolve(self, root_dir: Path) -> WaveBeatDatasetSpec | None:
        resolved_dirs = _resolve_data_label_dirs(root_dir, "ballroom")
        if resolved_dirs is not None:
            audio_dir, annot_dir = resolved_dirs
            return WaveBeatDatasetSpec(
                key=self.key,
                wavebeat_dataset="ballroom",
                audio_dir=audio_dir,
                annot_dir=annot_dir,
            )
        return None


class BallroomTestSpecResolver(BaseWaveBeatSpecResolver):
    key = "ballroom_test"

    def resolve(self, root_dir: Path) -> WaveBeatDatasetSpec | None:
        resolved_dirs = _resolve_data_label_dirs(root_dir, "ballroom_test")
        if resolved_dirs is not None:
            audio_dir, annot_dir = resolved_dirs
            return WaveBeatDatasetSpec(
                key=self.key,
                wavebeat_dataset="ballroom",
                audio_dir=audio_dir,
                annot_dir=annot_dir,
            )
        return None


class BeatlesSpecResolver(BaseWaveBeatSpecResolver):
    key = "beatles"

    def resolve(self, root_dir: Path) -> WaveBeatDatasetSpec | None:
        resolved_dirs = _resolve_data_label_dirs(root_dir, "beatles")
        if resolved_dirs is not None:
            audio_dir, annot_dir = resolved_dirs
            return WaveBeatDatasetSpec(
                key=self.key,
                wavebeat_dataset="beatles",
                audio_dir=audio_dir,
                annot_dir=annot_dir,
            )
        return None


class BeatlesOldSpecResolver(BaseWaveBeatSpecResolver):
    key = "beatles_old"

    def resolve(self, root_dir: Path) -> WaveBeatDatasetSpec | None:
        resolved_dirs = _resolve_data_label_dirs(root_dir, "beatles_old")
        if resolved_dirs is not None:
            audio_dir, annot_dir = resolved_dirs
            return WaveBeatDatasetSpec(
                key=self.key,
                wavebeat_dataset="beatles",
                audio_dir=audio_dir,
                annot_dir=annot_dir,
            )
        return None


class GTZANSpecResolver(BaseWaveBeatSpecResolver):
    key = "gtzan"

    def resolve(self, root_dir: Path) -> WaveBeatDatasetSpec | None:
        resolved_dirs = _resolve_data_label_dirs(root_dir, "gtzan")
        if resolved_dirs is not None:
            audio_dir, annot_dir = resolved_dirs
            return WaveBeatDatasetSpec(
                key=self.key,
                wavebeat_dataset="gtzan",
                audio_dir=audio_dir,
                annot_dir=annot_dir,
            )
        return None


class HainsSpecResolver(BaseWaveBeatSpecResolver):
    key = "hains"

    def resolve(self, root_dir: Path) -> WaveBeatDatasetSpec | None:
        resolved_dirs = _resolve_data_label_dirs(root_dir, "hains")
        if resolved_dirs is not None:
            audio_dir, annot_dir = resolved_dirs
            return WaveBeatDatasetSpec(
                key=self.key,
                wavebeat_dataset="hainsworth",
                audio_dir=audio_dir,
                annot_dir=annot_dir,
            )
        return None


class RWCPopularSpecResolver(BaseWaveBeatSpecResolver):
    key = "rwc_popular"

    def resolve(self, root_dir: Path) -> WaveBeatDatasetSpec | None:
        resolved_dirs = _resolve_data_label_dirs(root_dir, "rwc_popular")
        if resolved_dirs is not None:
            audio_dir, annot_dir = resolved_dirs
            return WaveBeatDatasetSpec(
                key=self.key,
                wavebeat_dataset="rwc_popular",
                audio_dir=audio_dir,
                annot_dir=annot_dir,
            )
        return None


def discover_wavebeat_dataset_specs(
    root_dir: str | Path,
    include_keys: set[str] | None = None,
) -> list[WaveBeatDatasetSpec]:
    """Discover known dataset folders from a single root directory."""
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset_root not found: {root}")

    if include_keys is not None and len(include_keys) == 0:
        raise ValueError("include_keys must be non-empty when provided.")

    resolvers: list[BaseWaveBeatSpecResolver] = [
        BallroomSpecResolver(),
        BallroomTestSpecResolver(),
        BeatlesSpecResolver(),
        BeatlesOldSpecResolver(),
        GTZANSpecResolver(),
        HainsSpecResolver(),
        RWCPopularSpecResolver(),
    ]

    specs: list[WaveBeatDatasetSpec] = []
    for resolver in resolvers:
        if include_keys is not None and resolver.key not in include_keys:
            continue
        if spec := resolver.resolve(root):
            specs.append(spec)

    return specs


class MultiSourceAudioDataset(Dataset):
    """Dataset adapter that combines multiple audio extractor datasets."""

    def __init__(self, source_datasets: list[Dataset], source_keys: list[str]) -> None:
        if len(source_datasets) == 0:
            raise ValueError("source_datasets must contain at least one dataset.")
        if len(source_datasets) != len(source_keys):
            raise ValueError("source_datasets and source_keys must have the same length.")

        self.source_datasets = source_datasets
        self.source_keys = source_keys

        self._lengths: list[int] = []
        self._offsets: list[int] = [0]
        self.audio_files: list[str] = []

        for dataset in self.source_datasets:
            if not isinstance(dataset, Sized):
                raise TypeError("Each source dataset must implement __len__().")

            dataset_len = len(cast(Sized, dataset))
            self._lengths.append(dataset_len)
            self._offsets.append(self._offsets[-1] + dataset_len)

            dataset_audio_files = getattr(dataset, "audio_files", None)
            if dataset_audio_files is None:
                raise AttributeError("Each source dataset must expose 'audio_files'.")
            self.audio_files.extend([str(path) for path in dataset_audio_files])

    def __len__(self) -> int:
        return self._offsets[-1]

    def _find_source_index(self, index: int) -> int:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index out of range: {index}")
        return bisect.bisect_right(self._offsets, index) - 1

    def get_audio_path_for_index(self, index: int) -> str:
        source_idx = self._find_source_index(index)
        local_index = index - self._offsets[source_idx]
        audio_files = getattr(self.source_datasets[source_idx], "audio_files", None)
        if audio_files is None or len(audio_files) == 0:
            raise RuntimeError("Source dataset has no audio_files for index mapping.")
        return str(audio_files[local_index % len(audio_files)])

    def __getitem__(self, index: int) -> object:
        source_idx = self._find_source_index(index)
        local_index = index - self._offsets[source_idx]
        return self.source_datasets[source_idx][local_index]
