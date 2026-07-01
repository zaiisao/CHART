"""Particle-filter (SMC) deployment mode for the trained KalmanVAEBarPointer checkpoint.

Deploy-time-only add-on: loads the checkpoint trained by train_kvae.py (no retraining), and replaces the
causal Kalman-filter deploy path (evaluate_leak_condition in train_kvae.py, which uses a single
DETERMINISTIC pseudo-observation a_t = encoder(h_t).mean shared by "all particles") with a bootstrap
particle filter that gives EACH particle its own STOCHASTIC a_t^(i) ~ q(a|h_t) = model.encoder(h_t).

Why this can matter (and why the deterministic case is a pure control, not a real test): the SSM's
mixture-of-K weights alpha_k are produced by an LSTM (kvae/dynamics_parameter_network.py's LSTMModel)
that reads the pseudo-observation SEQUENCE a_{1:t}, not z. With one shared deterministic a_t, the mixture
collapses to ONE specific time-varying linear-Gaussian system, and for that fixed system the Kalman
filter's mean is already the closed-form optimal estimate -- an SMC run on the SAME deterministic a would
just be a noisier Monte-Carlo re-derivation of the same answer, not an improvement. Only letting each
particle sample its OWN a_t^(i) lets different particles walk different LSTM hidden-state trajectories and
hence different alpha_k(t) mixture-weight branches -- i.e. genuinely different locally-linear dynamics --
which is the only way SMC could capture something a single deterministic filter pass structurally cannot
(e.g. early-song tempo/phase multi-modality).

Per-step recursion mirrors kvae/state_space_model.py's kalman_filter exactly (same mixture-weight timing:
the weight used to blend mat_A/mat_C at frame t was produced by the weight_model from a_{t-1}, with a
uniform 1/K weight at t=0 -- see that file's `weight = weight_next` / `weight_next = weight_model(...)`
lines), just replaced by a bootstrap-particle-filter recursion instead of the exact Gaussian update:
  1. sample a_t^(i) ~ Normal(encoder(h_t))                              (per particle, i.i.d.)
  2. weight_t^(i) = weight_model([a_1^(i), ..., a_{t-1}^(i)])            (LSTM hidden state per particle)
  3. A_t^(i) = sum_k weight_t^(i)_k * mat_A_K[k],  C_t^(i) similarly
  4. propose z_t^(i) ~ Normal(A_t^(i) @ z_{t-1}^(i), mat_Q)
  5. importance-weight  w_t^(i) *= Normal(C_t^(i) @ z_t^(i), mat_R).log_prob(a_t^(i))   (bootstrap PF)
  6. resample (systematic) when ESS = 1/sum(normalized_w^2) < 0.5 * N
  7. deploy signal = particle-weighted mean of z_t -> the SAME trained model.head -> sigmoid -> peak-pick

Reuses model/readout.py's peak_pick_times/f_measure and data/dataset.py's load_songs, exactly like
train_kvae.py's evaluate_leak_condition, so leak-test numbers are directly comparable.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.distributions as D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import FRAMES_PER_SECOND
from data.dataset import load_songs, Song
from data.targets import ground_truth_beat_times
from model import readout
from model.kalman_vae_bar_pointer import KalmanVAEBarPointer

_THIRD_PARTY_KVAE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                 "third_party", "kalman-vae")
if _THIRD_PARTY_KVAE not in sys.path:
    sys.path.insert(0, _THIRD_PARTY_KVAE)


def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def systematic_resample(log_weights: torch.Tensor) -> torch.Tensor:
    """log_weights [num_particles] (unnormalized) -> particle indices [num_particles] to resample.

    Standard systematic resampling: one uniform random offset, then evenly-spaced draws through the
    empirical CDF of the normalized weights -- lower variance than multinomial resampling.
    """
    num_particles = log_weights.shape[0]
    weights = torch.softmax(log_weights, dim=0)
    cumulative = torch.cumsum(weights, dim=0)
    cumulative[-1] = 1.0  # guard against floating-point shortfall
    offset = torch.rand(1, device=log_weights.device).item() / num_particles
    positions = offset + torch.arange(num_particles, device=log_weights.device, dtype=weights.dtype) / num_particles
    # searchsorted can return num_particles for a position that lands exactly at/past the last CDF bin
    # edge (floating-point boundary case) -- clamp into [0, num_particles - 1] to keep indices valid.
    return torch.searchsorted(cumulative, positions).clamp(max=num_particles - 1)


@torch.no_grad()
def particle_filter_deploy(model: KalmanVAEBarPointer, features: torch.Tensor, num_particles: int,
                           device: str, stochastic_a: bool, ess_resample_fraction: float = 0.5,
                           seed: int | None = None) -> tuple[np.ndarray, dict]:
    """Run the bootstrap particle filter over one song's features [num_frames, feature_dim].

    stochastic_a=False: all particles share ONE deterministic a_t = encoder(h_t).mean (the CONTROL --
        should closely reproduce the deterministic Kalman-filter deploy number).
    stochastic_a=True: each particle draws its OWN a_t^(i) ~ encoder(h_t) (the REAL test).

    Returns (probability [num_frames, 2] beat/downbeat sigmoid from the particle-mean z, diagnostics dict
    with per-frame ESS and the resample rate).
    """
    if seed is not None:
        generator_state = torch.get_rng_state()
        torch.manual_seed(seed)

    num_frames = features.shape[0]
    z_dim, a_dim, K = model.z_dim, model.a_dim, model.ssm.K
    N = num_particles

    weight_model = model.ssm.weight_model
    weight_model.clear_hidden_state()
    weight_model.reset_hidden_state(N, device=device, dtype=features.dtype)

    mat_A_K, mat_C_K = model.ssm.mat_A_K, model.ssm.mat_C_K   # [K, z_dim, z_dim], [K, a_dim, z_dim]
    mat_Q, mat_R = model.ssm.mat_Q, model.ssm.mat_R           # [z_dim, z_dim], [a_dim, a_dim]

    z = model.ssm.initial_state_mean.unsqueeze(0).repeat(N, 1)                       # [N, z_dim]
    log_weights = torch.zeros(N, device=device)                                       # uniform initially
    mixture_weight_next = torch.full((N, K), 1.0 / K, device=device)                 # weight_next at t=0

    z_means = torch.zeros(num_frames, z_dim, device=device)
    ess_trace, resample_count = [], 0

    for t in range(num_frames):
        h_t = features[t].unsqueeze(0).expand(N, -1)                                  # [N, feature_dim]
        a_distribution = model.encoder(h_t)
        a_t = a_distribution.rsample() if stochastic_a else a_distribution.mean.expand(N, a_dim).clone()

        mixture_weight = mixture_weight_next                                          # weight computed at t-1 (or uniform at t=0)
        mat_A = torch.einsum("nk,kij->nij", mixture_weight, mat_A_K)                  # [N, z_dim, z_dim]
        mat_C = torch.einsum("nk,kij->nij", mixture_weight, mat_C_K)                  # [N, a_dim, z_dim]

        # propose: z_t^(i) ~ N(A_t^(i) @ z_{t-1}^(i), Q)
        predicted_mean = torch.bmm(mat_A, z.unsqueeze(-1)).squeeze(-1)                # [N, z_dim]
        z = D.MultivariateNormal(predicted_mean, mat_Q).rsample()

        # importance-weight by the observation likelihood N(C_t^(i) @ z_t^(i), R).log_prob(a_t^(i))
        predicted_observation = torch.bmm(mat_C, z.unsqueeze(-1)).squeeze(-1)         # [N, a_dim]
        log_likelihood = D.MultivariateNormal(predicted_observation, mat_R).log_prob(a_t)
        log_weights = log_weights + log_likelihood

        # weighted mean state estimate for THIS frame (before resampling, using current log_weights)
        normalized_weights = torch.softmax(log_weights, dim=0)
        z_means[t] = (normalized_weights.unsqueeze(-1) * z).sum(dim=0)

        ess = 1.0 / (normalized_weights.pow(2).sum() + 1e-12)
        ess_trace.append(float(ess))
        if ess < ess_resample_fraction * N:
            indices = systematic_resample(log_weights)
            z = z[indices]
            log_weights = torch.zeros(N, device=device)
            resample_count += 1
            # LSTM hidden state must be resampled along with the particles it belongs to
            hidden, cell = weight_model.hidden
            weight_model.hidden = (hidden[:, indices, :], cell[:, indices, :])

        # advance the mixture-weight LSTM with a_t (this frame's pseudo-observation), producing the
        # weight used for mat_A/mat_C at t+1 -- mirrors state_space_model.py's `weight_next = weight_model(...)`
        mixture_weight_next = weight_model(a_t.unsqueeze(0)).squeeze(0)               # [N, K]

    diagnostics = {
        "mean_ess": float(np.mean(ess_trace)), "min_ess": float(np.min(ess_trace)),
        "resample_rate": resample_count / num_frames, "num_particles": N,
    }
    if seed is not None:
        torch.set_rng_state(generator_state)

    probability = torch.sigmoid(model.head(z_means)).cpu().numpy()
    return probability, diagnostics


@torch.no_grad()
def evaluate_smc_condition(model: KalmanVAEBarPointer, songs: list[Song], device: str, audio_condition: str,
                           num_particles: int, stochastic_a: bool, eval_max_frames: int,
                           tolerance_seconds: float, seed: int) -> dict:
    model.eval()
    beat_scores, downbeat_scores = [], []
    ess_means, resample_rates = [], []
    num_songs = len(songs)

    for song_index, song in enumerate(songs):
        source_features = songs[(song_index + 1) % num_songs].features if audio_condition == "shuffle" else song.features
        num_frames = min(source_features.shape[0], song.beat_targets.shape[0], eval_max_frames)

        if audio_condition == "zero":
            features = torch.zeros(num_frames, model.feature_dim, device=device)
        else:
            features = source_features[:num_frames].to(device)

        probability, diagnostics = particle_filter_deploy(
            model, features, num_particles, device, stochastic_a, seed=seed + song_index)
        ess_means.append(diagnostics["mean_ess"])
        resample_rates.append(diagnostics["resample_rate"])

        reference_beats = ground_truth_beat_times(song.beat_targets.numpy()[:num_frames], FRAMES_PER_SECOND)
        reference_downbeats = ground_truth_beat_times(song.downbeat_targets.numpy()[:num_frames], FRAMES_PER_SECOND)
        if len(reference_beats) >= 2:
            estimated_beats = readout.peak_pick_times(probability[:, 0], FRAMES_PER_SECOND)
            beat_scores.append(readout.f_measure(reference_beats, estimated_beats, tolerance_seconds))
        if len(reference_downbeats) >= 2:
            estimated_downbeats = readout.peak_pick_times(probability[:, 1], FRAMES_PER_SECOND)
            downbeat_scores.append(readout.f_measure(reference_downbeats, estimated_downbeats, tolerance_seconds))

    mean = lambda values: float(np.nanmean(values)) if values else float("nan")
    return {
        "beat_f": mean(beat_scores), "downbeat_f": mean(downbeat_scores),
        "mean_ess": mean(ess_means), "resample_rate": mean(resample_rates),
    }


def run_leak_test(model: KalmanVAEBarPointer, songs: list[Song], device: str, num_particles: int,
                  stochastic_a: bool, eval_max_frames: int, tolerance_seconds: float, seed: int) -> dict:
    return {
        condition: evaluate_smc_condition(model, songs, device, condition, num_particles, stochastic_a,
                                          eval_max_frames, tolerance_seconds, seed)
        for condition in ("real", "shuffle", "zero")
    }


def print_leak_test(label: str, leak: dict) -> None:
    print(f"\n--- {label} ---", flush=True)
    for condition in ("real", "shuffle", "zero"):
        row = leak[condition]
        print(f"{condition:8s}: beat {row['beat_f']:.3f}  downbeat {row['downbeat_f']:.3f}   "
             f"(mean ESS {row['mean_ess']:.1f}/{row.get('num_particles', '?')}, "
             f"resample rate {row['resample_rate']:.2f})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default="experiments/kvae_prototype/kvae_m1_repro.pt")
    parser.add_argument("--val_feature_dir", type=str, default="cache/acts/bt_val_rich")
    parser.add_argument("--num_val_songs", type=int, default=40)
    parser.add_argument("--particle_counts", type=int, nargs="+", default=[200, 50])
    parser.add_argument("--eval_max_frames", type=int, default=1600)
    parser.add_argument("--eval_beat_tolerance_seconds", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    set_all_seeds(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    model = KalmanVAEBarPointer(
        feature_dim=512, a_dim=ckpt_args.get("a_dim", 8), z_dim=ckpt_args.get("z_dim", 8), K=ckpt_args.get("K", 5),
        Q_reg=ckpt_args.get("Q_reg", 1e-3),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"[smc_eval] loaded {args.checkpoint} (a_dim={model.a_dim} z_dim={model.z_dim} K={model.ssm.K})", flush=True)

    val_songs = load_songs(args.val_feature_dir, args.num_val_songs, seed=2)
    print(f"[smc_eval] val={len(val_songs)} device={device}", flush=True)

    for num_particles in args.particle_counts:
        print(f"\n===== N={num_particles} particles =====", flush=True)

        control = run_leak_test(model, val_songs, device, num_particles, stochastic_a=False,
                                eval_max_frames=args.eval_max_frames,
                                tolerance_seconds=args.eval_beat_tolerance_seconds, seed=args.seed)
        print_leak_test(f"CONTROL (shared deterministic a, N={num_particles}) "
                        f"-- should reproduce the Kalman-filter deploy number", control)

        stochastic = run_leak_test(model, val_songs, device, num_particles, stochastic_a=True,
                                   eval_max_frames=args.eval_max_frames,
                                   tolerance_seconds=args.eval_beat_tolerance_seconds, seed=args.seed)
        print_leak_test(f"STOCHASTIC-a (per-particle sampled a, N={num_particles}) -- the real test", stochastic)


if __name__ == "__main__":
    main()
