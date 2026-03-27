"""Training entrypoints for CHART."""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Mapping, Sequence

import heapq
import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from models.svt_core import SVTModel
from models.loss import compute_elbo_loss
from evaluation.phase_converter import extract_beat_timestamps, extract_downbeat_timestamps
from evaluation.score import evaluate_beats, evaluate_downbeats, frames_to_beat_times
from training.dataset import ActivationDataset
from training.extractors import get_extractor_backend, list_extractor_backends
from training.extractors.base import ExtractorBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    has_mps = bool(getattr(torch, "has_mps", False))
    if has_mps:
        return torch.device("mps")
    return torch.device("cpu")


def _gumbel_temperature(epoch: int, num_epochs: int, start: float, end: float) -> float:
    """Linear annealing of Gumbel-Softmax temperature."""
    if num_epochs <= 1:
        return end
    t = min(epoch / max(num_epochs - 1, 1), 1.0)
    return start + (end - start) * t


def _kl_beta(epoch: int, anneal_epochs: int) -> float:
    """Linear KL annealing from 0 to 1."""
    if anneal_epochs <= 0:
        return 1.0
    return min(epoch / anneal_epochs, 1.0)


def _build_z_prev(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    """Extract z_prev dict from a batch mapping."""
    return {
        "phase": batch["phase_prev"].to(device),
        "log_tempo": batch["log_tempo_prev"].to(device),
        "meter_onehot": batch["meter_onehot_prev"].to(device),
    }


def _center_crop_seq_dim(x: Tensor, length: int) -> Tensor:
    if x.shape[1] == length:
        return x
    if x.shape[1] < length:
        raise ValueError(f"Cannot crop sequence length {x.shape[1]} to larger length {length}")
    start = (x.shape[1] - length) // 2
    return x[:, start : start + length]


def _center_crop_seq_dim_1d(x: Tensor, length: int) -> Tensor:
    """Center-crop a 1D or 2D tensor along dim 0 (seq dimension)."""
    if x.shape[0] == length:
        return x
    if x.shape[0] < length:
        raise ValueError(f"Cannot crop length {x.shape[0]} to {length}")
    start = (x.shape[0] - length) // 2
    return x[start : start + length]


# ---------------------------------------------------------------------------
# Training epochs
# ---------------------------------------------------------------------------

def train_epoch(
    model: SVTModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    temperature: float = 1.0,
    beta: float = 1.0,
    pos_weight: float = 20.0,
    epoch: int = 1,
    num_epochs: int = 1,
    log_interval: int = 1,
) -> tuple[float, dict[str, float]]:
    """Run one training epoch (teacher-forced parallel mode).

    Returns:
        Tuple of (avg_total_loss, avg_component_dict).
    """
    model.train()

    total_sum = 0.0
    comp_sums: dict[str, float] = {}
    num_batches = 0
    num_total_batches = max(1, len(dataloader))

    for batch_idx, batch in enumerate(dataloader, start=1):
        activations = batch["activations"].to(device)
        beat_targets = batch["beat_targets"].to(device)
        z_prev = _build_z_prev(batch, device)

        optimizer.zero_grad(set_to_none=True)

        # Algorithm 1 pass 1: rollout to get model samples for z_prev
        with torch.no_grad():
            out_rollout = model(activations, z_prev, temperature=temperature)
            samp = out_rollout["samples"]
            z_prev_sampled = {
                "phase": torch.cat(
                    [z_prev["phase"][:, :1, :],
                     samp["phase"][:, :-1].unsqueeze(-1)], dim=1,
                ),
                "log_tempo": torch.cat(
                    [z_prev["log_tempo"][:, :1, :],
                     samp["log_tempo"][:, :-1].unsqueeze(-1)], dim=1,
                ),
                "meter_onehot": torch.cat(
                    [z_prev["meter_onehot"][:, :1, :],
                     samp["meter_onehot"][:, :-1, :]], dim=1,
                ),
            }

        # Algorithm 1 pass 2: forward with sampled z_prev
        out = model(activations, z_prev_sampled, temperature=temperature)

        total_loss, components = compute_elbo_loss(
            beat_logits=out["beat_logits"],
            beat_targets=beat_targets,
            posterior=out["posterior"],
            prior=out["prior"],
            beta=beta,
            pos_weight=pos_weight,
        )

        total_loss.backward()
        optimizer.step()

        total_sum += float(total_loss.detach().item())
        for k, v in components.items():
            comp_sums[k] = comp_sums.get(k, 0.0) + float(v.item())
        num_batches += 1

        if log_interval > 0 and (batch_idx % log_interval == 0 or batch_idx == num_total_batches):
            avg = total_sum / num_batches
            sys.stdout.write(
                f"\r[Epoch {epoch:03d}/{num_epochs:03d}] "
                f"step {batch_idx:04d}/{num_total_batches:04d} "
                f"total={avg:.6f}"
            )
            sys.stdout.flush()

    if num_batches > 0:
        sys.stdout.write("\n")
    if num_batches == 0:
        return 0.0, {}

    avg_total = total_sum / num_batches
    avg_comps = {k: v / num_batches for k, v in comp_sums.items()}
    return avg_total, avg_comps


def train_epoch_end_to_end(
    extractor_model: torch.nn.Module,
    extractor_backend: ExtractorBackend,
    svt_model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    temperature: float = 1.0,
    beta: float = 1.0,
    pos_weight: float = 20.0,
    free_bits: float = 0.0,
    extractor_loss_weight: float = 1.0,
    svt_loss_weight: float = 1.0,
    epoch: int = 1,
    num_epochs: int = 1,
    log_interval: int = 1,
    is_main: bool = True,
) -> tuple[float, float, float, dict[str, float]]:
    """Run one end-to-end training epoch.

    Returns:
        (avg_total, avg_extractor, avg_svt, avg_components).
    """
    extractor_model.train()
    svt_model.train()

    total_sum = 0.0
    extractor_sum = 0.0
    svt_sum = 0.0
    comp_sums: dict[str, float] = {}
    num_batches = 0
    num_total_batches = max(1, len(dataloader))

    for batch_idx, batch in enumerate(dataloader, start=1):
        audio = batch["audio"].to(device)
        extractor_target = batch["extractor_target"].to(device)
        beat_targets = batch["beat_targets"].to(device)

        optimizer.zero_grad(set_to_none=True)

        # Stage 1: Extractor forward
        extractor_loss, activations = extractor_backend.compute_loss_and_activations(
            model=extractor_model, audio=audio, target=extractor_target,
        )

        # Crop structured targets to match extractor output length
        T_act = activations.shape[1]
        beat_targets_cropped = _center_crop_seq_dim(beat_targets.unsqueeze(-1), T_act).squeeze(-1)

        # Ground-truth z_prev (used only for the initial frame seed)
        z_prev_gt = {
            "phase": _center_crop_seq_dim(batch["phase_prev"].to(device), T_act),
            "log_tempo": _center_crop_seq_dim(batch["log_tempo_prev"].to(device), T_act),
            "meter_onehot": _center_crop_seq_dim(batch["meter_onehot_prev"].to(device), T_act),
        }

        # Algorithm 1: pass 1 — rollout under current parameters (no grad)
        # to obtain model samples ẑ that will serve as z_prev for pass 2.
        with torch.no_grad():
            out_rollout = svt_model(activations.detach(), z_prev_gt, temperature=temperature)
            samp = out_rollout["samples"]
            # Build z_prev_sampled: z_prev_sampled[t] = ẑ[t-1].
            # Frame 0 keeps the ground-truth seed (equivalent to f_init_ψ).
            z_prev_sampled = {
                "phase": torch.cat(
                    [z_prev_gt["phase"][:, :1, :],
                     samp["phase"][:, :-1].unsqueeze(-1)], dim=1,
                ),
                "log_tempo": torch.cat(
                    [z_prev_gt["log_tempo"][:, :1, :],
                     samp["log_tempo"][:, :-1].unsqueeze(-1)], dim=1,
                ),
                "meter_onehot": torch.cat(
                    [z_prev_gt["meter_onehot"][:, :1, :],
                     samp["meter_onehot"][:, :-1, :]], dim=1,
                ),
            }

        # Algorithm 1: pass 2 — forward with sampled z_prev, compute ELBO
        out = svt_model(activations, z_prev_sampled, temperature=temperature)
        svt_total, components = compute_elbo_loss(
            beat_logits=out["beat_logits"],
            beat_targets=beat_targets_cropped,
            posterior=out["posterior"],
            prior=out["prior"],
            beta=beta,
            pos_weight=pos_weight,
            free_bits=free_bits,
        )

        total_loss = extractor_loss_weight * extractor_loss + svt_loss_weight * svt_total
        total_loss.backward()
        optimizer.step()

        total_sum += float(total_loss.detach().item())
        extractor_sum += float(extractor_loss.detach().item())
        svt_sum += float(svt_total.detach().item())
        for k, v in components.items():
            comp_sums[k] = comp_sums.get(k, 0.0) + float(v.item())
        num_batches += 1

        if is_main and log_interval > 0 and (batch_idx % log_interval == 0 or batch_idx == num_total_batches):
            sys.stdout.write(
                f"\r[Epoch {epoch:03d}/{num_epochs:03d}] "
                f"step {batch_idx:04d}/{num_total_batches:04d} "
                f"total={total_sum / num_batches:.6f} "
                f"ext={extractor_sum / num_batches:.6f} "
                f"svt={svt_sum / num_batches:.6f}"
            )
            sys.stdout.flush()
            if _WANDB_AVAILABLE and _wandb.run is not None:
                global_step = (epoch - 1) * num_total_batches + batch_idx
                step_log: dict = {
                    "global_step": global_step,
                    "train_step/total_loss": total_sum / num_batches,
                    "train_step/ext_loss": extractor_sum / num_batches,
                    "train_step/svt_loss": svt_sum / num_batches,
                }
                for k, v in comp_sums.items():
                    step_log[f"train_step/{k}"] = v / num_batches
                _wandb.log(step_log)

    if is_main and num_batches > 0:
        sys.stdout.write("\n")
    if num_batches == 0:
        return 0.0, 0.0, 0.0, {}

    avg_comps = {k: v / num_batches for k, v in comp_sums.items()}
    return (
        total_sum / num_batches,
        extractor_sum / num_batches,
        svt_sum / num_batches,
        avg_comps,
    )


@torch.no_grad()
def val_epoch_end_to_end(
    extractor_model: torch.nn.Module,
    extractor_backend: ExtractorBackend,
    svt_model: SVTModel,
    dataloader: DataLoader,
    device: torch.device,
    temperature: float = 1.0,
    beta: float = 1.0,
    pos_weight: float = 20.0,
    free_bits: float = 0.0,
    extractor_loss_weight: float = 1.0,
    svt_loss_weight: float = 1.0,
    fps: float = 172.265625,
) -> tuple[float, float, float, dict[str, float], dict[str, float]]:
    """Run one validation epoch (no gradient updates).

    Returns:
        (avg_total, avg_extractor, avg_svt, avg_loss_components, avg_mir_eval_metrics).
    """
    extractor_model.eval()
    svt_model.eval()

    total_sum = 0.0
    extractor_sum = 0.0
    svt_sum = 0.0
    comp_sums: dict[str, float] = {}
    metric_sums: dict[str, float] = {}
    num_batches = 0
    num_eval_samples = 0

    for batch in dataloader:
        audio = batch["audio"].to(device)
        extractor_target = batch["extractor_target"].to(device)
        beat_targets = batch["beat_targets"].to(device)

        extractor_loss, activations = extractor_backend.compute_loss_and_activations(
            model=extractor_model, audio=audio, target=extractor_target,
        )

        # Cap sequence length so Transformer attention doesn't OOM on long songs.
        # train_length=65536 samples / target_factor=256 = 256 frames normally,
        # but val audio is full-length. Cap at 4096 frames (~24 s at 172 fps).
        _MAX_VAL_FRAMES = 4096
        T_act = min(activations.shape[1], _MAX_VAL_FRAMES)
        activations = activations[:, :T_act, :]

        # Use first T_act frames of beat_targets (consistent with activations[:T_act]).
        # Center-cropping would compare model output from song start to reference
        # beats from song middle, giving spuriously low F-measure.
        if beat_targets.shape[1] >= T_act:
            beat_targets_cropped = beat_targets[:, :T_act]
        else:
            beat_targets_cropped = _center_crop_seq_dim(beat_targets.unsqueeze(-1), T_act).squeeze(-1)

        # Seed z_prev from GT initial frame, then roll out with model samples
        # (same two-pass logic as training — Algorithm 1).
        z_prev_gt = {
            "phase": _center_crop_seq_dim(batch["phase_prev"].to(device), T_act),
            "log_tempo": _center_crop_seq_dim(batch["log_tempo_prev"].to(device), T_act),
            "meter_onehot": _center_crop_seq_dim(batch["meter_onehot_prev"].to(device), T_act),
        }
        out_rollout = svt_model(activations, z_prev_gt, temperature=temperature)
        samp = out_rollout["samples"]
        z_prev_sampled = {
            "phase": torch.cat(
                [z_prev_gt["phase"][:, :1, :],
                 samp["phase"][:, :-1].unsqueeze(-1)], dim=1,
            ),
            "log_tempo": torch.cat(
                [z_prev_gt["log_tempo"][:, :1, :],
                 samp["log_tempo"][:, :-1].unsqueeze(-1)], dim=1,
            ),
            "meter_onehot": torch.cat(
                [z_prev_gt["meter_onehot"][:, :1, :],
                 samp["meter_onehot"][:, :-1, :]], dim=1,
            ),
        }
        out = svt_model(activations, z_prev_sampled, temperature=temperature)
        svt_total, components = compute_elbo_loss(
            beat_logits=out["beat_logits"],
            beat_targets=beat_targets_cropped,
            posterior=out["posterior"],
            prior=out["prior"],
            beta=beta,
            pos_weight=pos_weight,
            free_bits=free_bits,
        )

        total_loss = extractor_loss_weight * extractor_loss + svt_loss_weight * svt_total

        total_sum += float(total_loss.item())
        extractor_sum += float(extractor_loss.item())
        svt_sum += float(svt_total.item())
        for k, v in components.items():
            comp_sums[k] = comp_sums.get(k, 0.0) + float(v.item())
        num_batches += 1

        # --- mir_eval beat/downbeat metrics per sample in batch ---
        beat_probs = torch.sigmoid(out["beat_logits"].squeeze(-1)).cpu().numpy()  # [B, T]
        bt_ref_np = beat_targets_cropped.cpu().numpy()  # [B, T]
        phase_np = out["samples"]["phase"].cpu().numpy()  # [B, T]
        B = beat_probs.shape[0]

        for b in range(B):
            ref_beats = frames_to_beat_times(bt_ref_np[b], fps)
            if len(ref_beats) < 2:
                continue

            est_beats = extract_beat_timestamps(beat_probs[b], fps=fps)
            if len(est_beats) == 0:
                continue

            # Beat metrics
            beat_scores = evaluate_beats(ref_beats, est_beats)
            for k, v in beat_scores.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + v

            # Downbeat metrics
            est_downbeats = extract_downbeat_timestamps(est_beats, phase_np[b], fps=fps)
            # Reference downbeats: phase near 0 at beat positions
            ref_downbeats = extract_downbeat_timestamps(
                ref_beats,
                _center_crop_seq_dim_1d(
                    batch["phase"][b].squeeze(-1), T_act
                ).numpy() / 1.0,  # already in radians
                fps=fps,
            )
            if len(ref_downbeats) >= 2 and len(est_downbeats) >= 2:
                db_scores = evaluate_downbeats(ref_downbeats, est_downbeats)
                for k, v in db_scores.items():
                    metric_sums[k] = metric_sums.get(k, 0.0) + v

            num_eval_samples += 1

    if num_batches == 0:
        return 0.0, 0.0, 0.0, {}, {}

    avg_comps = {k: v / num_batches for k, v in comp_sums.items()}
    avg_metrics = {k: v / max(num_eval_samples, 1) for k, v in metric_sums.items()}
    return (
        total_sum / num_batches,
        extractor_sum / num_batches,
        svt_sum / num_batches,
        avg_comps,
        avg_metrics,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CHART SVT model")
    parser.add_argument("--mode", choices=["activation", "end2end"], default="activation")
    parser.add_argument("--extractor", type=str, default="wavebeat", choices=list_extractor_backends())
    parser.add_argument("--activations_dir", type=str, default=None)
    parser.add_argument("--phases_dir", type=str, default=None)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Structured latent / ELBO
    parser.add_argument("--num_meter_classes", type=int, default=8)
    parser.add_argument("--gumbel_temp_start", type=float, default=1.0)
    parser.add_argument("--gumbel_temp_end", type=float, default=0.1)
    parser.add_argument("--kl_anneal_epochs", type=int, default=10)
    parser.add_argument(
        "--bce_pos_weight",
        type=float,
        default=20.0,
        help="BCE positive-class weight to compensate for beat-frame class imbalance (~1%% positive rate). Default: 20.0",
    )
    parser.add_argument(
        "--free_bits",
        type=float,
        default=0.0,
        help="Free-bits threshold λ (nats) applied per latent per sample. Prevents KL collapse. Default: 0.0 (disabled).",
    )

    # End-to-end
    parser.add_argument("--extractor_ckpt", type=str, default=None)
    parser.add_argument("--freeze_extractor", action="store_true")
    parser.add_argument("--extractor_loss_weight", type=float, default=1.0)
    parser.add_argument("--svt_loss_weight", type=float, default=1.0)

    # Backward compat
    parser.add_argument("--wavebeat_ckpt", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--freeze_wavebeat", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--wavebeat_loss_weight", type=float, default=None, help=argparse.SUPPRESS)

    parser.add_argument("--save_ckpt_path", type=str, default=None)

    # Weights & Biases
    parser.add_argument("--wandb_project", type=str, default="chart")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")

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
    parser = _build_parser()
    args = parser.parse_args()
    _normalize_backward_compat_args(args)

    # --- Distributed setup ---
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank >= 0
    if is_distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        device = _select_device()
        rank = 0
        world_size = 1
    is_main = rank == 0
    args.dist_rank = rank
    args.dist_world_size = world_size

    if is_main:
        print(f"Using device: {device}" + (f" (DDP world_size={world_size})" if is_distributed else ""))

    # --- Weights & Biases init ---
    use_wandb = is_main and _WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        _wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            resume="allow",
        )
        # Define custom x-axes so step-level and epoch-level metrics don't conflict.
        _wandb.define_metric("global_step")
        _wandb.define_metric("train_step/*", step_metric="global_step")
        _wandb.define_metric("epoch")
        _wandb.define_metric("train/*", step_metric="epoch")
        _wandb.define_metric("val/*", step_metric="epoch")
        _wandb.define_metric("ckpt/*", step_metric="epoch")
    else:
        if not _WANDB_AVAILABLE:
            print("[wandb] not installed, skipping.")
        elif args.no_wandb:
            print("[wandb] disabled via --no_wandb.")

    K = args.num_meter_classes

    if args.mode == "activation":
        if args.activations_dir is None or args.phases_dir is None:
            raise ValueError("--activations_dir and --phases_dir are required for mode=activation")

        dataset = ActivationDataset(
            activations_dir=args.activations_dir,
            phases_dir=args.phases_dir,
            seq_len=args.seq_len,
            num_meter_classes=K,
        )
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        model = SVTModel(
            hidden_dim=128, nhead=4, num_layers=2, num_meter_classes=K,
        ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

        for epoch in range(1, args.num_epochs + 1):
            temp = _gumbel_temperature(
                epoch - 1, args.num_epochs, args.gumbel_temp_start, args.gumbel_temp_end,
            )
            beta = _kl_beta(epoch - 1, args.kl_anneal_epochs)

            avg_total, avg_comps = train_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                device=device,
                temperature=temp,
                beta=beta,
                pos_weight=args.bce_pos_weight,
                epoch=epoch,
                num_epochs=args.num_epochs,
                log_interval=args.log_interval,
            )

            comp_str = " | ".join(f"{k}={v:.6f}" for k, v in avg_comps.items())
            print(
                f"[Epoch {epoch:03d}/{args.num_epochs:03d}] "
                f"total={avg_total:.6f} | {comp_str} | tau={temp:.3f} beta={beta:.3f}"
            )
            if use_wandb:
                log = {"train/total_loss": avg_total, "train/gumbel_temp": temp, "train/kl_beta": beta}
                log.update({f"train/{k}": v for k, v in avg_comps.items()})
                _wandb.log(log, step=epoch)

        if args.save_ckpt_path:
            torch.save({"svt_model": model.state_dict(), "args": vars(args)}, args.save_ckpt_path)
        if use_wandb:
            _wandb.finish()
        return

    # --- End-to-end mode ---
    extractor_backend = get_extractor_backend(args.extractor)
    dataloader = extractor_backend.build_dataloader(args)
    val_dataloader = extractor_backend.build_val_dataloader(args)
    extractor_model = extractor_backend.build_model(args, device)
    extractor_backend.load_checkpoint(extractor_model, args, device)

    if args.freeze_extractor:
        for parameter in extractor_model.parameters():
            parameter.requires_grad = False

    svt_model = SVTModel(
        hidden_dim=128, nhead=4, num_layers=2, num_meter_classes=K,
    ).to(device)

    if is_distributed:
        svt_model = DDP(svt_model, device_ids=[local_rank], find_unused_parameters=False)

    trainable_parameters = [
        p for p in list(extractor_model.parameters()) + list(svt_model.parameters())
        if p.requires_grad
    ]
    optimizer = optim.AdamW(trainable_parameters, lr=args.lr)

    # Top-3 checkpoint tracking: heap of (score, epoch, path)
    # We use a min-heap so the worst of the top-3 is always heap[0].
    top_ckpts: list[tuple[float, int, str]] = []
    ckpt_dir = os.path.dirname(args.save_ckpt_path) if args.save_ckpt_path else "checkpoints"
    ckpt_stem = os.path.splitext(os.path.basename(args.save_ckpt_path))[0] if args.save_ckpt_path else "chart"
    os.makedirs(ckpt_dir or ".", exist_ok=True)

    for epoch in range(1, args.num_epochs + 1):
        # Keep DistributedSampler in sync with epoch for proper shuffling
        if is_distributed and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

        temp = _gumbel_temperature(
            epoch - 1, args.num_epochs, args.gumbel_temp_start, args.gumbel_temp_end,
        )
        beta = _kl_beta(epoch - 1, args.kl_anneal_epochs)

        avg_total, avg_ext, avg_svt, avg_comps = train_epoch_end_to_end(
            extractor_model=extractor_model,
            extractor_backend=extractor_backend,
            svt_model=svt_model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            temperature=temp,
            beta=beta,
            pos_weight=args.bce_pos_weight,
            free_bits=args.free_bits,
            extractor_loss_weight=args.extractor_loss_weight,
            svt_loss_weight=args.svt_loss_weight,
            epoch=epoch,
            num_epochs=args.num_epochs,
            log_interval=args.log_interval,
            is_main=is_main,
        )

        if is_main:
            comp_str = " | ".join(f"{k}={v:.6f}" for k, v in avg_comps.items())
            print(
                f"[Epoch {epoch:03d}/{args.num_epochs:03d}] "
                f"total={avg_total:.6f} | ext={avg_ext:.6f} | svt={avg_svt:.6f} | "
                f"{comp_str} | tau={temp:.3f} beta={beta:.3f}"
            )

            # Gradient norm (computed over all trainable params after last backward)
            grad_norm = 0.0
            for p in trainable_parameters:
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5

            lr_current = optimizer.param_groups[0]["lr"]
            if use_wandb:
                train_log = {
                    "epoch": epoch,
                    "train/total_loss": avg_total,
                    "train/ext_loss": avg_ext,
                    "train/svt_loss": avg_svt,
                    "train/grad_norm": grad_norm,
                    "train/gumbel_temp": temp,
                    "train/kl_beta": beta,
                    "train/lr": lr_current,
                }
                train_log.update({f"train/{k}": v for k, v in avg_comps.items()})
                _wandb.log(train_log)

        val_f_measure = 0.0
        if is_main and val_dataloader is not None:
            val_fps = getattr(args, "audio_sample_rate", 44100) / getattr(args, "target_factor", 256)
            v_total, v_ext, v_svt, v_comps, v_metrics = val_epoch_end_to_end(
                extractor_model=extractor_model,
                extractor_backend=extractor_backend,
                svt_model=svt_model,
                dataloader=val_dataloader,
                device=device,
                temperature=temp,
                beta=beta,
                pos_weight=args.bce_pos_weight,
                free_bits=args.free_bits,
                extractor_loss_weight=args.extractor_loss_weight,
                svt_loss_weight=args.svt_loss_weight,
                fps=val_fps,
            )
            v_comp_str = " | ".join(f"{k}={v:.6f}" for k, v in v_comps.items())
            print(
                f"  [Val] total={v_total:.6f} | ext={v_ext:.6f} | svt={v_svt:.6f} | "
                f"{v_comp_str}"
            )
            val_f_measure = v_metrics.get("F-measure", 0.0)
            m_str = " | ".join(f"{k}={v:.4f}" for k, v in v_metrics.items()) if v_metrics else "F-measure=0.0000"
            print(f"  [Val mir_eval] {m_str}")

            if use_wandb:
                val_log = {
                    "epoch": epoch,
                    "val/total_loss": v_total,
                    "val/ext_loss": v_ext,
                    "val/svt_loss": v_svt,
                }
                val_log.update({f"val/{k}": v for k, v in v_comps.items()})
                val_log.update({f"val/{k}": v for k, v in v_metrics.items()})
                _wandb.log(val_log)

        # --- Save top-3 checkpoints by val beat F-measure (rank 0 only) ---
        if is_main and args.save_ckpt_path:
            svt_state = svt_model.module.state_dict() if is_distributed else svt_model.state_dict()
            ckpt_path = os.path.join(ckpt_dir, f"{ckpt_stem}_ep{epoch:03d}_f{val_f_measure:.4f}.pt")
            ckpt_data = {
                "epoch": epoch,
                "val_f_measure": val_f_measure,
                "extractor": args.extractor,
                "extractor_model": extractor_model.state_dict(),
                "svt_model": svt_state,
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }
            if len(top_ckpts) < 3:
                torch.save(ckpt_data, ckpt_path)
                heapq.heappush(top_ckpts, (val_f_measure, epoch, ckpt_path))
                print(f"  [Ckpt] saved {os.path.basename(ckpt_path)}")
                if use_wandb:
                    _wandb.log({"epoch": epoch, "ckpt/saved_epoch": epoch, "ckpt/val_f_measure": val_f_measure})
            elif val_f_measure > top_ckpts[0][0]:
                # Better than the worst of top-3: evict it
                _, _, old_path = heapq.heapreplace(top_ckpts, (val_f_measure, epoch, ckpt_path))
                torch.save(ckpt_data, ckpt_path)
                if os.path.exists(old_path):
                    os.remove(old_path)
                print(f"  [Ckpt] saved {os.path.basename(ckpt_path)} (replaced {os.path.basename(old_path)})")
                if use_wandb:
                    _wandb.log({"epoch": epoch, "ckpt/saved_epoch": epoch, "ckpt/val_f_measure": val_f_measure})

    if use_wandb:
        _wandb.finish()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
