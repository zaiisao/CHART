"""Inference utilities and CLI for CHART.

At inference time, beat annotations (b_{1:T}) are unavailable. The model
uses the PRIOR (not posterior) to generate latent trajectories from audio
alone. Beats are extracted from the phase trajectory by detecting
wrap-around points (phase crossing 2*pi -> 0), following the bar pointer
model's physics. The decoder is a training-time loss signal; at inference
we read beats directly from the learned dynamics.

This mirrors Stable Diffusion's approach: the encoder (posterior) provides
structured training targets, but inference uses the generative model
(prior/diffusion) without the encoder.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from models.svt_core import SVTModel
from models.distributions import (
    gumbel_softmax_sample,
    lognormal_sample_logspace,
    von_mises_sample,
)
from evaluation.phase_converter import extract_beats_from_phase_trajectory

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
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Gumbel-Softmax temperature for meter (low = more discrete)")
    parser.add_argument("--fps", type=float, default=86.1328125,
                        help="Frames per second of the input activations")
    return parser


def _load_svt_model(
    checkpoint_path: str,
    device: torch.device,
    hidden_dim: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    num_meter_classes: int = 8,
) -> SVTModel:
    ckpt: Any = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract model args from checkpoint if available
    if isinstance(ckpt, dict) and "args" in ckpt:
        saved_args = ckpt["args"]
        num_meter_classes = saved_args.get("num_meter_classes", num_meter_classes)

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


@torch.no_grad()
def run_inference(
    model: SVTModel,
    acoustic_activations: Tensor,
    *,
    temperature: float = 0.1,
    fps: float = 86.1328125,
) -> dict[str, Tensor | np.ndarray]:
    """Run CHART inference using the PRIOR (no beat annotations needed).

    The prior encoder processes the audio to learn uncertainty parameters.
    The sequential loop samples from the prior's transition model
    (bar pointer dynamics) to generate phase/tempo/meter trajectories.
    Beats are extracted from phase wrap-arounds.

    Args:
        model: Trained SVT model.
        acoustic_activations: ``[B, T, 2]`` acoustic features.
        temperature: Gumbel-Softmax temperature for meter.
        fps: Frames per second for beat timestamp extraction.

    Returns:
        Dict with:
        - ``beat_times``: list of 1-D arrays of beat timestamps (seconds) per batch
        - ``phase``: ``[B, T]`` phase trajectory
        - ``log_tempo``: ``[B, T]`` log-tempo trajectory
        - ``tempo``: ``[B, T]`` tempo in rad/frame
        - ``meter``: ``[B, T]`` meter class index
        - ``beat_logits``: ``[B, T]`` decoder beat logits (for comparison)
        - ``beat_probs``: ``[B, T]`` sigmoid of beat_logits
    """
    B, T, _ = acoustic_activations.shape
    device = acoustic_activations.device
    K = model.num_meter_classes

    # Step 0: Prior encoder — compute h_prior and uncertainty params
    h_prior, prior_params = model.encode_prior(acoustic_activations)

    # Initialize z_prev for t=0
    phase_prev = torch.zeros(B, 1, device=device)
    log_tempo_prev = torch.zeros(B, 1, device=device)
    meter_prev = torch.ones(B, K, device=device) / K

    # Accumulators
    all_beat_logits = []
    all_phase = []
    all_log_tempo = []
    all_meter = []

    for t in range(T):
        # Compute prior at time t
        prior_t = model.compute_prior_at_t(
            prior_params, t, phase_prev, log_tempo_prev, meter_prev,
        )

        # Sample from PRIOR (not posterior — no beat annotations available)
        # Meter
        meter_soft = gumbel_softmax_sample(
            prior_t["meter_logits"], temperature=temperature, hard=False,
        )
        meter_hard = gumbel_softmax_sample(
            prior_t["meter_logits"], temperature=temperature, hard=True,
        )

        # Phase: sample from vM(prior_mu, prior_kappa)
        phase = von_mises_sample(prior_t["phase_mu"], prior_t["phase_kappa"])
        phase = torch.remainder(phase, TWO_PI)

        # Tempo: sample from LogNormal(prior_mu, prior_sigma)
        log_tempo = lognormal_sample_logspace(
            prior_t["tempo_mu"], prior_t["tempo_sigma"],
        )

        # Decode (for comparison — not used for beat extraction)
        samp_t = {
            "phase": phase,
            "log_tempo": log_tempo,
            "meter_soft": meter_soft,
            "meter_onehot": meter_hard,
        }
        beat_logit_t = model.decode_at_t(samp_t, h_prior[:, t, :])

        # Accumulate
        all_beat_logits.append(beat_logit_t)
        all_phase.append(phase)
        all_log_tempo.append(log_tempo)
        all_meter.append(meter_hard.argmax(dim=-1))

        # Update z_prev for next step
        phase_prev = phase.unsqueeze(-1)
        log_tempo_prev = log_tempo.unsqueeze(-1)
        meter_prev = meter_hard

    # Stack
    beat_logits = torch.stack(all_beat_logits, dim=1).squeeze(-1)  # [B, T]
    phase_traj = torch.stack(all_phase, dim=1)                      # [B, T]
    log_tempo_traj = torch.stack(all_log_tempo, dim=1)              # [B, T]
    meter_traj = torch.stack(all_meter, dim=1)                      # [B, T]

    # Extract beats from phase wrap-arounds (primary method)
    phase_np = phase_traj.cpu().numpy()
    beat_times_list = []
    for b in range(B):
        bt = extract_beats_from_phase_trajectory(phase_np[b], fps=fps)
        beat_times_list.append(bt)

    return {
        "beat_times": beat_times_list,
        "phase": phase_traj,
        "log_tempo": log_tempo_traj,
        "tempo": log_tempo_traj.exp(),
        "meter": meter_traj,
        "beat_logits": beat_logits,
        "beat_probs": torch.sigmoid(beat_logits),
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
        temperature=args.temperature,
        fps=args.fps,
    )

    # Save phase-based beat timestamps (primary output)
    beat_times = results["beat_times"][0]  # first batch element
    output_path = Path(args.output_npy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, beat_times)

    print(f"Saved {len(beat_times)} beat timestamps to: {output_path}")
    if len(beat_times) > 0:
        avg_bpm = 60.0 / np.mean(np.diff(beat_times)) if len(beat_times) > 1 else 0
        print(f"Estimated tempo: {avg_bpm:.1f} BPM")

    # Also save full trajectories
    traj_path = output_path.with_suffix(".trajectories.npz")
    np.savez(
        traj_path,
        beat_times=beat_times,
        phase=results["phase"][0].cpu().numpy(),
        tempo=results["tempo"][0].cpu().numpy(),
        meter=results["meter"][0].cpu().numpy(),
        beat_probs=results["beat_probs"][0].cpu().numpy(),
    )
    print(f"Saved full trajectories to: {traj_path}")


if __name__ == "__main__":
    main()
