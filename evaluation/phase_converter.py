"""Utilities to convert continuous phase trajectories into timestamps."""

from __future__ import annotations

import numpy as np


def extract_timestamps_from_phase(phase_array: np.ndarray, fps: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Convert phase sweeps to beat and downbeat timestamps.

    This placeholder converts a continuous phase trajectory (nominally sweeping
    from 0.0 to 1.0) into discrete beat/downbeat timestamps by detecting wrap
    events where the phase returns to approximately 0.

    Args:
        phase_array: Continuous phase values over time.
        fps: Frames per second of the phase sequence.

    Returns:
        A tuple `(beat_timestamps, downbeat_timestamps)`.
    """
    # TODO: Detect phase wrap-around events and map frame indices to seconds.
    raise NotImplementedError("Implement phase-to-timestamp conversion.")
