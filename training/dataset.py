"""Dataset definitions for CHART training."""

from __future__ import annotations

from pathlib import Path
import random

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

        activation_map = {path.stem: path for path in self.activations_dir.glob("*.npy")}
        phase_map = {path.stem: path for path in self.phases_dir.glob("*.npy")}

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
