"""Utilities to convert model outputs into beat/downbeat timestamps."""

from __future__ import annotations

import math

import numpy as np


def extract_beat_timestamps(
    beat_probs: np.ndarray,
    fps: float = 100.0,
    threshold: float = 0.5,
    min_distance_sec: float = 0.15,
) -> np.ndarray:
    """Convert beat probability curve to discrete beat timestamps.

    Uses peak-picking with a minimum distance constraint.

    Args:
        beat_probs: 1-D array of beat probabilities, shape ``[T]``.
        fps: Frames per second.
        threshold: Minimum probability to consider as a beat.
        min_distance_sec: Minimum time between consecutive beats.

    Returns:
        1-D array of beat timestamps in seconds.
    """
    if beat_probs.ndim != 1:
        raise ValueError(f"beat_probs must be 1-D, got shape {beat_probs.shape}")

    min_distance_frames = max(1, int(min_distance_sec * fps))
    T = len(beat_probs)

    # Find local maxima above threshold
    candidates = []
    for i in range(1, T - 1):
        if beat_probs[i] < threshold:
            continue
        if beat_probs[i] >= beat_probs[i - 1] and beat_probs[i] >= beat_probs[i + 1]:
            candidates.append(i)

    # Enforce minimum distance (greedy, highest-probability-first)
    candidates.sort(key=lambda i: -beat_probs[i])
    selected: list[int] = []
    taken = set()
    for idx in candidates:
        if any(abs(idx - s) < min_distance_frames for s in selected):
            continue
        selected.append(idx)

    selected.sort()
    return np.array(selected, dtype=np.float64) / fps


def extract_downbeat_timestamps(
    beat_timestamps: np.ndarray,
    phase: np.ndarray,
    fps: float = 100.0,
) -> np.ndarray:
    """Identify downbeats from beat timestamps and phase trajectory.

    A downbeat is a beat where the phase is near the start of a bar
    (i.e., phase close to 0 or 2*pi).

    Args:
        beat_timestamps: 1-D array of beat times in seconds.
        phase: 1-D array of phase values in radians, shape ``[T]``.
        fps: Frames per second of the phase array.

    Returns:
        1-D array of downbeat timestamps in seconds.
    """
    downbeats = []
    TWO_PI = 2.0 * math.pi

    for bt in beat_timestamps:
        frame_idx = int(round(bt * fps))
        frame_idx = min(max(frame_idx, 0), len(phase) - 1)
        p = phase[frame_idx] % TWO_PI
        # Downbeat: phase near 0 (start of bar)
        if p < (TWO_PI * 0.15) or p > (TWO_PI * 0.85):
            downbeats.append(bt)

    return np.array(downbeats, dtype=np.float64)


def extract_beats_from_phase_trajectory(
    phase: np.ndarray,
    fps: float = 86.1328125,
    min_distance_sec: float = 0.15,
) -> np.ndarray:
    """Extract beat timestamps directly from a phase trajectory.

    In the bar pointer model, a beat occurs when the phase wraps from
    near 2*pi back to near 0. This function detects those wrap-around
    points without using the decoder's beat probability output.

    Args:
        phase: 1-D array of phase values in radians, shape ``[T]``.
        fps: Frames per second.
        min_distance_sec: Minimum time between consecutive beats.

    Returns:
        1-D array of beat timestamps in seconds.
    """
    TWO_PI = 2.0 * math.pi
    T = len(phase)
    min_dist = max(1, int(min_distance_sec * fps))

    # Detect phase wraps: large negative jump in phase (2*pi → 0)
    phase_wrapped = phase % TWO_PI
    diffs = np.diff(phase_wrapped)
    # A wrap occurs when diff < -pi (phase dropped by more than half circle)
    wrap_indices = np.where(diffs < -math.pi)[0] + 1  # +1 because diff shifts by 1

    if len(wrap_indices) == 0:
        return np.array([], dtype=np.float64)

    # Enforce minimum distance
    selected = [wrap_indices[0]]
    for idx in wrap_indices[1:]:
        if idx - selected[-1] >= min_dist:
            selected.append(idx)

    return np.array(selected, dtype=np.float64) / fps


def extract_timestamps_from_phase(
    beat_probs: np.ndarray,
    phase: np.ndarray,
    fps: float = 100.0,
    threshold: float = 0.5,
    min_distance_sec: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Full pipeline: beat probabilities + phase -> beat and downbeat timestamps.

    Args:
        beat_probs: Beat probability curve, shape ``[T]``.
        phase: Phase trajectory in radians, shape ``[T]``.
        fps: Frames per second.
        threshold: Beat detection threshold.
        min_distance_sec: Minimum inter-beat interval.

    Returns:
        Tuple of ``(beat_timestamps, downbeat_timestamps)`` in seconds.
    """
    beats = extract_beat_timestamps(beat_probs, fps, threshold, min_distance_sec)
    downbeats = extract_downbeat_timestamps(beats, phase, fps)
    return beats, downbeats
