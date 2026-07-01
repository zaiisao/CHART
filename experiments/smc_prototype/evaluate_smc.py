"""Leak-test evaluation for the FIVO/SMC bar-pointer prototype -- mirrors evaluate.py exactly.

Same three audio conditions (real / shuffle / zero) and the same geometric phase->beat/downbeat
read-out (model/readout.py) as the plain VBPM baseline, so numbers are directly comparable.
"""
from __future__ import annotations

import numpy as np
import torch

from config import FRAMES_PER_SECOND
from data.dataset import Song
from data.targets import ground_truth_beat_times
from model import readout

from .bar_pointer_smc import BarPointerFIVO
from .emission import ActivationEmissionHead


@torch.no_grad()
def evaluate_smc_geometric(fivo: BarPointerFIVO, emission_head: ActivationEmissionHead,
                           songs: list[Song], device: str, beats_per_bar: int,
                           num_particles: int = 200, ess_frac: float = 0.5, readout_mode: str = "map",
                           max_frames: int = 1600, eval_tolerance_seconds: float = 0.07,
                           audio_condition: str = "real") -> dict:
    """Mean geometric beat/downbeat F-measure over songs, deployed by the particle filter (no labels)."""
    fivo.eval()
    emission_head.eval()
    beat_scores, downbeat_scores = [], []
    num_songs = len(songs)

    for song_index, song in enumerate(songs):
        if audio_condition == "shuffle":
            source_features = songs[(song_index + 1) % num_songs].features
        else:
            source_features = song.features
        num_frames = min(source_features.shape[0], song.beat_targets.shape[0], max_frames)

        if audio_condition == "zero":
            features = torch.zeros(num_frames, source_features.shape[1], device=device)
        else:
            features = source_features[:num_frames].to(device)

        observed_activations = emission_head.probabilities(features)     # [T, 2]
        phase = fivo.deploy_smc(observed_activations, num_particles=num_particles,
                                ess_frac=ess_frac, readout=readout_mode)
        phase_numpy = phase.cpu().numpy()

        reference_beats = ground_truth_beat_times(song.beat_targets.numpy()[:num_frames], FRAMES_PER_SECOND)
        reference_downbeats = ground_truth_beat_times(song.downbeat_targets.numpy()[:num_frames], FRAMES_PER_SECOND)
        if len(reference_beats) >= 2:
            estimated_beats = readout.phase_to_beat_times(phase_numpy, beats_per_bar, FRAMES_PER_SECOND)
            beat_scores.append(readout.f_measure(reference_beats, estimated_beats, eval_tolerance_seconds))
        if len(reference_downbeats) >= 2:
            estimated_downbeats = readout.phase_to_downbeat_times(phase_numpy, FRAMES_PER_SECOND)
            downbeat_scores.append(readout.f_measure(reference_downbeats, estimated_downbeats, eval_tolerance_seconds))

    fivo.train()
    mean = lambda values: float(np.nanmean(values)) if values else float("nan")
    return {"beat_f": mean(beat_scores), "downbeat_f": mean(downbeat_scores)}


def evaluate_smc_with_leak_test(fivo: BarPointerFIVO, emission_head: ActivationEmissionHead,
                                songs: list[Song], device: str, beats_per_bar: int,
                                num_particles: int = 200, ess_frac: float = 0.5,
                                readout_mode: str = "map", max_frames: int = 1600,
                                eval_tolerance_seconds: float = 0.07) -> dict:
    kwargs = dict(device=device, beats_per_bar=beats_per_bar, num_particles=num_particles,
                 ess_frac=ess_frac, readout_mode=readout_mode, max_frames=max_frames,
                 eval_tolerance_seconds=eval_tolerance_seconds)
    return {
        "real": evaluate_smc_geometric(fivo, emission_head, songs, audio_condition="real", **kwargs),
        "shuffle": evaluate_smc_geometric(fivo, emission_head, songs, audio_condition="shuffle", **kwargs),
        "zero": evaluate_smc_geometric(fivo, emission_head, songs, audio_condition="zero", **kwargs),
    }


def print_leak_test(leak: dict) -> None:
    for condition in ("real", "shuffle", "zero"):
        print(f"{condition:8s}: beat {leak[condition]['beat_f']:.3f}  downbeat {leak[condition]['downbeat_f']:.3f}", flush=True)
