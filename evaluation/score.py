"""Evaluation wrappers for beat/downbeat tracking metrics using mir_eval."""

from __future__ import annotations

import numpy as np
import mir_eval.beat


def evaluate_beats(
    reference_beats: np.ndarray,
    estimated_beats: np.ndarray,
) -> dict[str, float]:
    """Evaluate beat tracking with mir_eval.

    Args:
        reference_beats: 1-D array of ground truth beat times in seconds.
        estimated_beats: 1-D array of predicted beat times in seconds.

    Returns:
        Dict with F-measure, CMLc, CMLt, AMLc, AMLt.
    """
    reference_beats = np.asarray(reference_beats, dtype=np.float64).ravel()
    estimated_beats = np.asarray(estimated_beats, dtype=np.float64).ravel()

    # mir_eval requires sorted, unique, non-empty arrays
    if len(reference_beats) == 0 or len(estimated_beats) == 0:
        return {"F-measure": 0.0, "CMLc": 0.0, "CMLt": 0.0, "AMLc": 0.0, "AMLt": 0.0}

    reference_beats = np.sort(np.unique(reference_beats))
    estimated_beats = np.sort(np.unique(estimated_beats))

    scores = mir_eval.beat.evaluate(reference_beats, estimated_beats)

    return {
        "F-measure": scores["F-measure"],
        "CMLc": scores["Correct Metric Level Continuous"],
        "CMLt": scores["Correct Metric Level Total"],
        "AMLc": scores["Any Metric Level Continuous"],
        "AMLt": scores["Any Metric Level Total"],
    }


def evaluate_downbeats(
    reference_downbeats: np.ndarray,
    estimated_downbeats: np.ndarray,
) -> dict[str, float]:
    """Evaluate downbeat tracking with mir_eval.

    Uses the same beat evaluation metrics applied to downbeats.

    Args:
        reference_downbeats: 1-D array of ground truth downbeat times in seconds.
        estimated_downbeats: 1-D array of predicted downbeat times in seconds.

    Returns:
        Dict with downbeat F-measure, CMLc, CMLt, AMLc, AMLt.
    """
    reference_downbeats = np.asarray(reference_downbeats, dtype=np.float64).ravel()
    estimated_downbeats = np.asarray(estimated_downbeats, dtype=np.float64).ravel()

    if len(reference_downbeats) < 2 or len(estimated_downbeats) < 2:
        return {
            "db_F-measure": 0.0, "db_CMLc": 0.0, "db_CMLt": 0.0,
            "db_AMLc": 0.0, "db_AMLt": 0.0,
        }

    reference_downbeats = np.sort(np.unique(reference_downbeats))
    estimated_downbeats = np.sort(np.unique(estimated_downbeats))

    scores = mir_eval.beat.evaluate(reference_downbeats, estimated_downbeats)

    return {
        "db_F-measure": scores["F-measure"],
        "db_CMLc": scores["Correct Metric Level Continuous"],
        "db_CMLt": scores["Correct Metric Level Total"],
        "db_AMLc": scores["Any Metric Level Continuous"],
        "db_AMLt": scores["Any Metric Level Total"],
    }


def frames_to_beat_times(
    beat_targets: np.ndarray,
    fps: float,
) -> np.ndarray:
    """Convert binary frame-level beat targets to beat times in seconds.

    Args:
        beat_targets: 1-D binary array, shape ``[T]``.
        fps: Frames per second.

    Returns:
        1-D array of beat times in seconds.
    """
    indices = np.where(beat_targets > 0.5)[0]
    return indices.astype(np.float64) / fps
