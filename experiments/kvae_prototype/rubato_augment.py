"""Local (piecewise) rubato time-stretch augmentation: cut a song into segments at beat boundaries,
independently sample a stretch percentage PER SEGMENT, stretch each segment, concatenate, and warp that
segment's beat/downbeat times by a running time offset.

This differs from Beat This's own augmentation (external/beat_this/beat_this/dataset/augment.py's
augment_tempo/stretch_annotations, verified by reading that file: ONE percentage per whole track, so
tempo is uniformly faster/slower throughout but internally just as metronomically rigid as the original)
specifically to manufacture genuine WITHIN-TRACK tempo instability -- what continuity_metrics.py's
diagnosis (CMLt-CMLc = 0.166 SMC vs 0.030 in-domain) identified as the actual SMC failure signature:
frequent short interruptions in an otherwise-correct track, consistent with rubato knocking the Kalman
filter off a correct lock intermittently. A single global stretch factor cannot simulate that; a
piecewise one can.

Segmentation: cut at existing beat times (a few seconds each -- we group multiple consecutive beats into
one segment rather than cutting at every single beat, to keep segments long enough for the phase vocoder
to have a stable, unambiguous pitch estimate and to keep the number of splice points, and thus splice
artifacts, bounded). Segment boundaries land exactly ON a beat, so no beat is ever split across a splice
seam -- avoiding the "awkward within-beat cuts" the task brief flagged as undesirable.

Annotation warping: within segment i (original time range [t_start_i, t_end_i), stretched by rate_i),
    warped_time = cumulative_output_duration_before_segment_i + (original_time - t_start_i) / rate_i
This is the direct per-segment generalization of stretch_annotations's single `item["beat_time"] / factor`
(that function's `factor` is our per-segment `rate_i`, applied only within each segment's own local time
window with a running offset carried across segments -- see stretch_annotations's docstring for the
single-factor original this generalizes).
"""
from __future__ import annotations

import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rubato_stretch import time_stretch


def make_segments(beat_times: np.ndarray, total_duration: float, beats_per_segment: int = 8,
                  min_segment_seconds: float = 1.5) -> list[tuple[float, float]]:
    """Cut [0, total_duration) into segments at every `beats_per_segment`-th beat time (so every segment
    boundary lands exactly on a beat -- no beat is ever split by a splice seam). Segments shorter than
    min_segment_seconds are merged into the previous one (keeps the phase vocoder well-conditioned and
    limits splice-artifact density)."""
    beat_times = np.sort(beat_times)
    boundary_candidates = [0.0] + list(beat_times[::beats_per_segment]) + [total_duration]
    boundaries = sorted(set(t for t in boundary_candidates if 0.0 <= t <= total_duration))

    segments = []
    start = boundaries[0]
    for end in boundaries[1:]:
        if end - start < min_segment_seconds and end != total_duration:
            continue  # extend the current segment through this boundary instead of splitting here
        segments.append((start, end))
        start = end
    if segments and segments[-1][1] != total_duration:
        segments[-1] = (segments[-1][0], total_duration)
    elif not segments:
        segments = [(0.0, total_duration)]
    return segments


def rubato_augment_song(waveform: torch.Tensor, sample_rate: int, beat_times: np.ndarray,
                        downbeat_times: np.ndarray, beats_per_segment: int = 8,
                        stretch_range_percent: tuple[float, float] = (-25.0, 25.0),
                        rng: np.random.Generator | None = None) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Returns (augmented_waveform, warped_beat_times, warped_downbeat_times).

    Each segment gets an INDEPENDENTLY sampled stretch percentage in stretch_range_percent (e.g. -25% to
    +25%, matching Beat This's own augment_tempo range convention -- percentage is the change in TEMPO,
    so factor = 1 + percentage/100 and output_duration = input_duration / factor, exactly
    stretch_annotations's convention, applied per-segment here instead of once globally).
    """
    if rng is None:
        rng = np.random.default_rng()

    total_duration = waveform.shape[-1] / sample_rate
    segments = make_segments(beat_times, total_duration, beats_per_segment=beats_per_segment)

    stretched_chunks = []
    warped_beat_times, warped_downbeat_times = [], []
    output_offset = 0.0

    for seg_start, seg_end in segments:
        percentage = rng.uniform(*stretch_range_percent)
        factor = 1.0 + percentage / 100.0

        sample_start = int(round(seg_start * sample_rate))
        sample_end = int(round(seg_end * sample_rate))
        segment_waveform = waveform[sample_start:sample_end]
        if segment_waveform.numel() < 256:  # too short to STFT meaningfully; pass through unstretched
            factor = 1.0

        stretched_segment = time_stretch(segment_waveform, factor) if factor != 1.0 else segment_waveform
        stretched_chunks.append(stretched_segment)
        segment_output_duration = stretched_segment.shape[-1] / sample_rate

        # warp every beat/downbeat time that falls in [seg_start, seg_end) by this segment's factor,
        # relative to the running output-time offset (the direct per-segment generalization of
        # stretch_annotations's `beat_time / factor`, see module docstring).
        in_segment_beats = beat_times[(beat_times >= seg_start) & (beat_times < seg_end)]
        warped_beat_times.extend(output_offset + (in_segment_beats - seg_start) / factor)
        in_segment_downbeats = downbeat_times[(downbeat_times >= seg_start) & (downbeat_times < seg_end)]
        warped_downbeat_times.extend(output_offset + (in_segment_downbeats - seg_start) / factor)

        output_offset += segment_output_duration

    augmented_waveform = torch.cat(stretched_chunks)
    return augmented_waveform, np.array(sorted(warped_beat_times)), np.array(sorted(warped_downbeat_times))
