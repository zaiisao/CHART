"""Training entrypoints for CHART."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from models.svt_core import SVTModel
from training.dataset import ActivationDataset


def train_epoch(model: SVTModel, dataloader: DataLoader, optimizer: torch.optim.Optimizer) -> None:
    """Run one training epoch placeholder.

    Args:
        model: SVT model instance.
        dataloader: DataLoader yielding training batches.
        optimizer: Optimizer instance.
    """
    # TODO: Add mini-batch training logic, loss computation, and optimization steps.
    raise NotImplementedError("Implement per-epoch training logic.")


def main() -> None:
    """Initialize model, dataset, optimizer, and run a dummy training loop."""
    # TODO: Replace with real file discovery for your activation/phase datasets.
    activation_files: list[str] = []
    phase_files: list[str] = []

    # TODO: Set task-specific dimensions and hyperparameters.
    model = SVTModel(input_dim=128, latent_dim=16, hidden_dim=128, num_layers=2)
    dataset = ActivationDataset(activation_files=activation_files, phase_files=phase_files)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # TODO: Replace with actual training schedule and checkpointing.
    for _ in range(1):
        pass
        # train_epoch(model, dataloader, optimizer)


if __name__ == "__main__":
    main()
