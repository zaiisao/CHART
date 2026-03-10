"""Training entrypoints for CHART."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence

import torch
from torch import optim
from torch.utils.data import DataLoader

from models.svt_core import SVTModel
from models.loss import compute_svt_loss
from training.dataset import ActivationDataset


def _select_device() -> torch.device:
    """Select the best available device in CUDA -> MPS -> CPU order."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    has_mps = bool(getattr(torch, "has_mps", False))
    if has_mps:
        return torch.device("mps")

    return torch.device("cpu")


def _unpack_batch(batch: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack batch data from either dict-style or tuple/list-style loaders.

    Args:
        batch: A dataloader batch containing activations, z_target, z_history.

    Returns:
        A tuple of `(activations, z_target, z_history)` tensors.
    """
    if isinstance(batch, Mapping):
        activations = batch["activations"]
        z_target = batch["z_target"]
        z_history = batch["z_history"]
        return activations, z_target, z_history

    if isinstance(batch, Sequence) and len(batch) >= 3:
        activations, z_target, z_history = batch[0], batch[1], batch[2]
        return activations, z_target, z_history

    raise TypeError(
        "Batch must be a mapping with keys {'activations','z_target','z_history'} "
        "or a sequence/tuple of length >= 3."
    )


def train_epoch(
    model: SVTModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    lambda_prior: float = 1.0,
) -> tuple[float, float, float, float]:
    """Run one training epoch for CHART.

    Args:
        model: SVT model instance.
        dataloader: DataLoader yielding training batches.
        optimizer: Optimizer instance.
        device: Target device for model and tensors.
        lambda_prior: Weight applied to prior-energy loss term.

    Returns:
        Tuple of average losses:
        `(avg_total, avg_reconstruction, avg_kl, avg_prior_energy)`.
    """
    model.train()

    total_sum = 0.0
    recon_sum = 0.0
    kl_sum = 0.0
    prior_sum = 0.0
    num_batches = 0

    for batch in dataloader:
        activations, z_target, z_history = _unpack_batch(batch)

        activations = activations.to(device)
        z_target = z_target.to(device)
        z_history = z_history.to(device)

        optimizer.zero_grad(set_to_none=True)

        reconstructed, mu, logvar, z_t = model(activations, z_history)

        total_loss, recon_loss, kl_loss, prior_loss = compute_svt_loss(
            reconstructed=reconstructed,
            targets=z_target,
            mu=mu,
            logvar=logvar,
            z_t=z_t,
            z_t_minus_1=z_history,
            lambda_prior=lambda_prior,
        )

        total_loss.backward()
        optimizer.step()

        total_sum += float(total_loss.detach().item())
        recon_sum += float(recon_loss.detach().item())
        kl_sum += float(kl_loss.detach().item())
        prior_sum += float(prior_loss.detach().item())
        num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0, 0.0, 0.0

    avg_total = total_sum / num_batches
    avg_recon = recon_sum / num_batches
    avg_kl = kl_sum / num_batches
    avg_prior = prior_sum / num_batches
    return avg_total, avg_recon, avg_kl, avg_prior


def main() -> None:
    """Initialize data/model/optimizer and run a basic training loop."""
    parser = argparse.ArgumentParser(description="Train CHART SVT model")
    parser.add_argument("--activations_dir", type=str, required=True, help="Directory with activation .npy files")
    parser.add_argument("--phases_dir", type=str, required=True, help="Directory with phase .npy files")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length for training windows")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lambda_prior", type=float, default=1.0, help="Prior energy loss weight")
    args = parser.parse_args()

    device = _select_device()
    print(f"Using device: {device}")

    dataset = ActivationDataset(
        activations_dir=args.activations_dir,
        phases_dir=args.phases_dir,
        seq_len=args.seq_len,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = SVTModel(hidden_dim=128, nhead=4, num_layers=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(1, args.num_epochs + 1):
        avg_total, avg_recon, avg_kl, avg_prior = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            lambda_prior=args.lambda_prior,
        )

        print(
            f"[Epoch {epoch:03d}/{args.num_epochs:03d}] "
            f"total={avg_total:.6f} | recon={avg_recon:.6f} | "
            f"kl={avg_kl:.6f} | prior={avg_prior:.6f}"
        )


if __name__ == "__main__":
    main()
