"""Dataset definitions for CHART training."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class ActivationDataset(Dataset):
    """Dataset for pre-computed acoustic activations and phase targets.

    Expects paired file lists where each activation `.npy` path corresponds to a
    ground-truth phase `.npy` path.
    """

    def __init__(self, activation_files: Sequence[str | Path], phase_files: Sequence[str | Path]) -> None:
        """Initialize dataset with activation and phase file paths.

        Args:
            activation_files: Paths to pre-computed activation `.npy` files.
            phase_files: Paths to corresponding ground-truth phase `.npy` files.
        """
        # TODO: Validate file list lengths and store resolved paths.
        self.activation_files = [Path(path) for path in activation_files]
        self.phase_files = [Path(path) for path in phase_files]

    def __len__(self) -> int:
        """Return number of paired training examples."""
        # TODO: Optionally add integrity checks for paired file counts.
        return len(self.activation_files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Load a single activation/phase pair from `.npy` files.

        Args:
            index: Dataset index.

        Returns:
            Dictionary with tensor entries for activations and phases.
        """
        # TODO: Add robust I/O, dtype handling, and shape normalization.
        activation = np.load(self.activation_files[index])
        phase = np.load(self.phase_files[index])

        return {
            "activations": torch.from_numpy(activation).float(),
            "phases": torch.from_numpy(phase).float(),
        }
