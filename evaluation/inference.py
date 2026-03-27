"""Inference utilities and CLI for CHART."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from models.svt_core import SVTModel

TWO_PI = 2.0 * math.pi


def _select_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    has_mps = bool(getattr(torch, "has_mps", False))
    if has_mps:
        return torch.device("mps")
    return torch.device("cpu")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CHART inference and save predictions")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_npy", type=str, required=True,
                        help="Input acoustic activations (.npy) with shape [T,2] or [B,T,2]")
    parser.add_argument("--output_npy", type=str, required=True,
                        help="Path to save predictions (.npy)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_meter_classes", type=int, default=8)
    parser.add_argument("--non_autoregressive", action="store_true",
                        help="Single forward pass with zero history (faster, less accurate)")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Gumbel-Softmax temperature for meter (low = more discrete)")
    return parser


def _load_svt_model(
    checkpoint_path: str,
    device: torch.device,
    hidden_dim: int,
    nhead: int,
    num_layers: int,
    num_meter_classes: int,
) -> SVTModel:
    ckpt: Any = torch.load(checkpoint_path, map_location=device, weights_only=False)
    svt_state = ckpt.get("svt_model", ckpt) if isinstance(ckpt, dict) else ckpt

    model = SVTModel(
        hidden_dim=hidden_dim, nhead=nhead, num_layers=num_layers,
        num_meter_classes=num_meter_classes,
    ).to(device)
    model.load_state_dict(svt_state, strict=True)
    model.eval()
    return model


def _prepare_activations(input_path: str, device: torch.device) -> Tensor:
    activations_np = np.load(input_path)
    activations = torch.as_tensor(activations_np, dtype=torch.float32)

    if activations.ndim == 2:
        if activations.shape[1] != 2:
            raise ValueError(f"Expected [T,2] input when ndim=2, got {tuple(activations.shape)}")
        activations = activations.unsqueeze(0)
    elif activations.ndim == 3:
        if activations.shape[2] != 2:
            raise ValueError(f"Expected [B,T,2] input when ndim=3, got {tuple(activations.shape)}")
    else:
        raise ValueError(f"Expected input shape [T,2] or [B,T,2], got {tuple(activations.shape)}")

    return activations.to(device)


def _make_zero_z_prev(
    batch_size: int,
    seq_len: int,
    num_meter_classes: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Tensor]:
    """Create zero-initialized z_prev dict."""
    return {
        "phase": torch.zeros(batch_size, seq_len, 1, device=device, dtype=dtype),
        "log_tempo": torch.zeros(batch_size, seq_len, 1, device=device, dtype=dtype),
        "meter_onehot": torch.ones(batch_size, seq_len, num_meter_classes,
                                   device=device, dtype=dtype) / num_meter_classes,
    }


@torch.no_grad()
def run_inference(
    model: SVTModel,
    acoustic_activations: Tensor,
    *,
    autoregressive: bool = True,
    temperature: float = 0.1,
) -> dict[str, Tensor]:
    """Run CHART inference.

    Args:
        model: Trained SVT model.
        acoustic_activations: ``[B, T, 2]`` acoustic features.
        autoregressive: If True, frame-by-frame rollout feeding predictions back.
        temperature: Gumbel-Softmax temperature for meter.

    Returns:
        Dict with ``beat_probs`` [B,T], ``phase`` [B,T], ``tempo`` [B,T],
        ``meter`` [B,T] (argmax class index).
    """
    if acoustic_activations.ndim != 3 or acoustic_activations.shape[-1] != 2:
        raise ValueError(
            f"acoustic_activations must have shape [B,T,2], got {tuple(acoustic_activations.shape)}"
        )

    B, T, _ = acoustic_activations.shape
    device = acoustic_activations.device
    K = model.num_meter_classes

    if not autoregressive:
        z_prev = _make_zero_z_prev(B, T, K, device)
        out = model(acoustic_activations, z_prev, temperature=temperature)
        beat_probs = torch.sigmoid(out["beat_logits"].squeeze(-1))
        return {
            "beat_probs": beat_probs,
            "phase": out["samples"]["phase"],
            "tempo": out["samples"]["log_tempo"].exp(),
            "meter": out["samples"]["meter_soft"].argmax(dim=-1),
        }

    # Autoregressive rollout
    h_audio = model.encode_acoustics(acoustic_activations)  # [B, T, D]

    # Collect per-frame results
    all_beat_logits = []
    all_phase = []
    all_log_tempo = []
    all_meter_soft = []

    # Running z_prev state — always [B, 1, ?] for step()
    phase_prev = torch.zeros(B, 1, 1, device=device)
    log_tempo_prev = torch.zeros(B, 1, 1, device=device)
    meter_oh_prev = torch.ones(B, 1, K, device=device) / K

    # Build prior input incrementally: [B, T, 2+K]
    z_prev_history = torch.zeros(B, T, 2 + K, device=device)

    for t in range(T):
        # Write current z_prev into history at position t
        z_prev_history[:, t, 0:1] = phase_prev[:, 0, :]        # [B, 1]
        z_prev_history[:, t, 1:2] = log_tempo_prev[:, 0, :]    # [B, 1]
        z_prev_history[:, t, 2:] = meter_oh_prev[:, 0, :]      # [B, K]

        # Run causal prior on history up to t+1
        h_prior = model.encode_prior(z_prev_history[:, : t + 1])  # [B, t+1, D]
        h_prior_t = h_prior[:, t : t + 1, :]  # [B, 1, D]
        h_audio_t = h_audio[:, t : t + 1, :]  # [B, 1, D]

        result = model.step(
            h_audio_t, h_prior_t,
            phase_prev, log_tempo_prev, meter_oh_prev,
            temperature=temperature,
        )

        # result samples: phase [B,1], log_tempo [B,1], meter_soft [B,1,K], meter_onehot [B,1,K]
        all_beat_logits.append(result["beat_logits"])           # [B, 1, 1]
        all_phase.append(result["samples"]["phase"])            # [B, 1]
        all_log_tempo.append(result["samples"]["log_tempo"])    # [B, 1]
        all_meter_soft.append(result["samples"]["meter_soft"])  # [B, 1, K]

        # Update running state for next step
        phase_prev = result["samples"]["phase"].unsqueeze(-1)        # [B, 1] -> [B, 1, 1]
        log_tempo_prev = result["samples"]["log_tempo"].unsqueeze(-1)  # [B, 1] -> [B, 1, 1]
        meter_oh_prev = result["samples"]["meter_onehot"]              # [B, 1, K]

    beat_logits = torch.cat(all_beat_logits, dim=1).squeeze(-1)  # [B, T]
    phase = torch.cat(all_phase, dim=1)                          # [B, T]
    log_tempo = torch.cat(all_log_tempo, dim=1)                  # [B, T]
    meter_soft = torch.cat(all_meter_soft, dim=1)                # [B, T, K]

    return {
        "beat_probs": torch.sigmoid(beat_logits),
        "phase": phase,
        "tempo": log_tempo.exp(),
        "meter": meter_soft.argmax(dim=-1),
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    device = _select_device(args.device)
    model = _load_svt_model(
        checkpoint_path=args.checkpoint,
        device=device,
        hidden_dim=args.hidden_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        num_meter_classes=args.num_meter_classes,
    )

    activations = _prepare_activations(args.input_npy, device)
    results = run_inference(
        model=model,
        acoustic_activations=activations,
        autoregressive=not args.non_autoregressive,
        temperature=args.temperature,
    )

    # Save beat probabilities as primary output
    beat_probs_np = results["beat_probs"].detach().cpu().numpy()
    if beat_probs_np.shape[0] == 1:
        beat_probs_np = beat_probs_np[0]

    output_path = Path(args.output_npy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, beat_probs_np)

    print(f"Saved beat probabilities to: {output_path}")
    print(f"Output shape: {beat_probs_np.shape}")


if __name__ == "__main__":
    main()
