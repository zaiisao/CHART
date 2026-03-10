"""Training entrypoints for CHART."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence

import torch
from torch import optim
from torch.utils.data import DataLoader

from models.svt_core import SVTModel
from models.loss import compute_svt_loss
from training.dataset import ActivationDataset
from training.extractors import get_extractor_backend, list_extractor_backends
from training.extractors.base import ExtractorBackend


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


def _center_crop_seq_dim(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.shape[1] == length:
        return x
    if x.shape[1] < length:
        raise ValueError(f"Cannot crop sequence length {x.shape[1]} to larger length {length}")
    start = (x.shape[1] - length) // 2
    end = start + length
    return x[:, start:end, :]


def train_epoch(
    model: SVTModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    lambda_prior: float = 1.0,
    epoch: int = 1,
    num_epochs: int = 1,
    log_interval: int = 1,
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

    num_total_batches = max(1, len(dataloader))

    for batch_idx, batch in enumerate(dataloader, start=1):
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

        if log_interval > 0 and (batch_idx % log_interval == 0 or batch_idx == num_total_batches):
            avg_so_far = total_sum / num_batches
            sys.stdout.write(
                f"\r[Epoch {epoch:03d}/{num_epochs:03d}] "
                f"step {batch_idx:04d}/{num_total_batches:04d} "
                f"total={avg_so_far:.6f}"
            )
            sys.stdout.flush()

    if num_batches > 0:
        sys.stdout.write("\n")

    if num_batches == 0:
        return 0.0, 0.0, 0.0, 0.0

    avg_total = total_sum / num_batches
    avg_recon = recon_sum / num_batches
    avg_kl = kl_sum / num_batches
    avg_prior = prior_sum / num_batches
    return avg_total, avg_recon, avg_kl, avg_prior


def train_epoch_end_to_end(
    extractor_model: torch.nn.Module,
    extractor_backend: ExtractorBackend,
    svt_model: SVTModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    lambda_prior: float,
    extractor_loss_weight: float,
    svt_loss_weight: float,
    epoch: int = 1,
    num_epochs: int = 1,
    log_interval: int = 1,
) -> tuple[float, float, float, float, float, float]:
    extractor_model.train()
    svt_model.train()

    total_sum = 0.0
    extractor_sum = 0.0
    svt_total_sum = 0.0
    svt_recon_sum = 0.0
    svt_kl_sum = 0.0
    svt_prior_sum = 0.0
    num_batches = 0

    num_total_batches = max(1, len(dataloader))

    for batch_idx, batch in enumerate(dataloader, start=1):
        audio = batch["audio"].to(device)
        extractor_target = batch["extractor_target"].to(device)
        z_target = batch["z_target"].to(device)
        z_history = batch["z_history"].to(device)

        optimizer.zero_grad(set_to_none=True)

        extractor_loss, activations = extractor_backend.compute_loss_and_activations(
            model=extractor_model,
            audio=audio,
            target=extractor_target,
        )
        z_target = _center_crop_seq_dim(z_target, activations.shape[1])
        z_history = _center_crop_seq_dim(z_history, activations.shape[1])

        reconstructed, mu, logvar, z_t = svt_model(activations, z_history)
        svt_total, svt_recon, svt_kl, svt_prior = compute_svt_loss(
            reconstructed=reconstructed,
            targets=z_target,
            mu=mu,
            logvar=logvar,
            z_t=z_t,
            z_t_minus_1=z_history,
            lambda_prior=lambda_prior,
        )

        total_loss = extractor_loss_weight * extractor_loss + svt_loss_weight * svt_total
        total_loss.backward()
        optimizer.step()

        total_sum += float(total_loss.detach().item())
        extractor_sum += float(extractor_loss.detach().item())
        svt_total_sum += float(svt_total.detach().item())
        svt_recon_sum += float(svt_recon.detach().item())
        svt_kl_sum += float(svt_kl.detach().item())
        svt_prior_sum += float(svt_prior.detach().item())
        num_batches += 1

        if log_interval > 0 and (batch_idx % log_interval == 0 or batch_idx == num_total_batches):
            avg_total_so_far = total_sum / num_batches
            avg_extractor_so_far = extractor_sum / num_batches
            avg_svt_so_far = svt_total_sum / num_batches
            sys.stdout.write(
                f"\r[Epoch {epoch:03d}/{num_epochs:03d}] "
                f"step {batch_idx:04d}/{num_total_batches:04d} "
                f"total={avg_total_so_far:.6f} "
                f"extractor={avg_extractor_so_far:.6f} "
                f"svt={avg_svt_so_far:.6f}"
            )
            sys.stdout.flush()

    if num_batches > 0:
        sys.stdout.write("\n")

    if num_batches == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    return (
        total_sum / num_batches,
        extractor_sum / num_batches,
        svt_total_sum / num_batches,
        svt_recon_sum / num_batches,
        svt_kl_sum / num_batches,
        svt_prior_sum / num_batches,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CHART SVT model")
    parser.add_argument("--mode", choices=["activation", "end2end"], default="activation")
    parser.add_argument("--extractor", type=str, default="wavebeat", choices=list_extractor_backends())
    parser.add_argument("--activations_dir", type=str, default=None, help="Directory with activation .npy files")
    parser.add_argument(
        "--phases_dir",
        type=str,
        default=None,
        help="Directory with phase .npy files (optional in end2end when --dataset_root is set)",
    )
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length for training windows")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--log_interval", type=int, default=1, help="Update progress every N batches")
    parser.add_argument("--lambda_prior", type=float, default=1.0, help="Prior energy loss weight")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    parser.add_argument("--extractor_ckpt", type=str, default=None, help="Optional checkpoint path for extractor")
    parser.add_argument("--freeze_extractor", action="store_true", help="Freeze extractor parameters")
    parser.add_argument("--extractor_loss_weight", type=float, default=1.0, help="Weight for extractor loss")
    parser.add_argument("--svt_loss_weight", type=float, default=1.0, help="Weight for SVT loss")

    parser.add_argument("--wavebeat_ckpt", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--freeze_wavebeat", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--wavebeat_loss_weight", type=float, default=None, help=argparse.SUPPRESS)

    parser.add_argument("--save_ckpt_path", type=str, default=None, help="Optional path to save combined model checkpoint")

    known_args, _ = parser.parse_known_args()
    extractor_backend = get_extractor_backend(known_args.extractor)
    extractor_backend.add_cli_args(parser)
    return parser


def _normalize_backward_compat_args(args: argparse.Namespace) -> None:
    if args.wavebeat_ckpt is not None and args.extractor_ckpt is None:
        args.extractor_ckpt = args.wavebeat_ckpt
    if bool(args.freeze_wavebeat):
        args.freeze_extractor = True
    if args.wavebeat_loss_weight is not None:
        args.extractor_loss_weight = args.wavebeat_loss_weight


def main() -> None:
    """Initialize data/model/optimizer and run a basic training loop."""
    parser = _build_parser()
    args = parser.parse_args()
    _normalize_backward_compat_args(args)

    device = _select_device()
    print(f"Using device: {device}")

    if args.mode == "activation":
        if args.activations_dir is None or args.phases_dir is None:
            raise ValueError("--activations_dir and --phases_dir are required for mode=activation")

        dataset = ActivationDataset(
            activations_dir=args.activations_dir,
            phases_dir=args.phases_dir,
            seq_len=args.seq_len,
        )
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        model = SVTModel(hidden_dim=128, nhead=4, num_layers=2).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

        for epoch in range(1, args.num_epochs + 1):
            avg_total, avg_recon, avg_kl, avg_prior = train_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                device=device,
                lambda_prior=args.lambda_prior,
                epoch=epoch,
                num_epochs=args.num_epochs,
                log_interval=args.log_interval,
            )

            print(
                f"[Epoch {epoch:03d}/{args.num_epochs:03d}] "
                f"total={avg_total:.6f} | recon={avg_recon:.6f} | "
                f"kl={avg_kl:.6f} | prior={avg_prior:.6f}"
            )

        if args.save_ckpt_path:
            torch.save({"svt_model": model.state_dict()}, args.save_ckpt_path)
        return

    extractor_backend = get_extractor_backend(args.extractor)
    dataloader = extractor_backend.build_dataloader(args)
    extractor_model = extractor_backend.build_model(args, device)
    extractor_backend.load_checkpoint(extractor_model, args, device)

    if args.freeze_extractor:
        for parameter in extractor_model.parameters():
            parameter.requires_grad = False

    svt_model = SVTModel(hidden_dim=128, nhead=4, num_layers=2).to(device)

    trainable_parameters = [
        parameter
        for parameter in list(extractor_model.parameters()) + list(svt_model.parameters())
        if parameter.requires_grad
    ]
    optimizer = optim.AdamW(trainable_parameters, lr=args.lr)

    for epoch in range(1, args.num_epochs + 1):
        avg_total, avg_extractor, avg_svt, avg_recon, avg_kl, avg_prior = train_epoch_end_to_end(
            extractor_model=extractor_model,
            extractor_backend=extractor_backend,
            svt_model=svt_model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            lambda_prior=args.lambda_prior,
            extractor_loss_weight=args.extractor_loss_weight,
            svt_loss_weight=args.svt_loss_weight,
            epoch=epoch,
            num_epochs=args.num_epochs,
            log_interval=args.log_interval,
        )

        print(
            f"[Epoch {epoch:03d}/{args.num_epochs:03d}] "
            f"total={avg_total:.6f} | extractor={avg_extractor:.6f} | svt={avg_svt:.6f} | "
            f"recon={avg_recon:.6f} | kl={avg_kl:.6f} | prior={avg_prior:.6f}"
        )

    if args.save_ckpt_path:
        os.makedirs(os.path.dirname(args.save_ckpt_path) or ".", exist_ok=True)
        torch.save(
            {
                "extractor": args.extractor,
                "extractor_model": extractor_model.state_dict(),
                "svt_model": svt_model.state_dict(),
                "args": vars(args),
            },
            args.save_ckpt_path,
        )


if __name__ == "__main__":
    main()
