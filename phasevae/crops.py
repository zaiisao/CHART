"""Crops cut at ARBITRARY beat offsets -- the fix that makes this task non-vacuous.

``vbpm.data`` cuts crops at bar boundaries, so the first beat of a crop is a downbeat in
99.998% of cases (measured below). A phase model trained on those learns "phase 0 is at
frame 0" and scores beautifully having learned nothing. Here a crop starts at a beat
index drawn uniformly over the song's valid starts, PER CROP -- not per song, which would
make the offset a song-level constant and therefore perfectly predictable from identity.

Two invariants, both load-bearing:
  * n = crop_bars * m beats exactly, so n % m == 0 and the number of downbeats in a crop
    is len(range(r, n, m)) = crop_bars for EVERY offset r. With n % m != 0 the count
    leaks r (n=13, m=4 gives counts [4, 3, 3, 3]) and a downbeat COUNTER would score
    above chance without ever using phase.
  * the crop's downbeats must be exactly range(r, n, m) -- an internally meter-varying
    stretch has no single r_true, so it is excluded and COUNTED, never relabelled.

r_true is derived here for SCORING ONLY. Nothing that runs at deployment may read it.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from vbpm.data import FPS, derive_y

CROP_BARS = 8          # n = 8 * m beats: 32 beats at m = 4, ~16 s at 120 bpm
DOWNBEAT_TOL_S = 0.02


def song_bar_length(y):
    """The song's own median bar length in beats, or None if it has < 2 downbeats."""
    downbeats = np.flatnonzero(np.asarray(y))
    if len(downbeats) < 2:
        return None
    return int(np.floor(float(np.median(np.diff(downbeats))) + 0.5))


def crop_starts(y, m: int, crop_bars: int = CROP_BARS):
    """(valid_starts, rejects) over a whole song's per-beat downbeat indicator.

    A start s is valid iff the window [s, s+n) has downbeats at exactly range(r, n, m)
    for some r, i.e. the window is internally consistent with beats-per-bar m.

    Call this only on songs whose own median bar length IS m. On a 3/4 song every
    candidate window fails, and tallying those failures as "the meter varies inside this
    window" would conflate the corpus restriction (this song is not in m) with the
    exclusion this counter exists to measure (this window has no single r_true). The
    first population outnumbers the second ~40:1, so the merged number is unreadable and
    makes a benign filter look catastrophic. ``song_crops`` enforces the split.

    Args:
        y: [num_beats] uint8 downbeat indicator for the whole song.
        m: beats per bar to cut for.
        crop_bars: bars per crop; n = crop_bars * m.

    Returns:
        (list of (start, r_true), Counter of rejection reasons, including the
        ``candidate_windows`` denominator that makes the failure rate legible).
    """
    n = crop_bars * m
    y = np.asarray(y)
    rejects: Counter = Counter()
    starts = []
    for s in range(0, len(y) - n + 1):
        rejects["candidate_windows"] += 1
        idx = np.flatnonzero(y[s:s + n])
        if len(idx) == 0:
            rejects["no_downbeat_in_window"] += 1
            continue
        r = int(idx[0])
        if not np.array_equal(idx, np.arange(r, n, m)):
            rejects["meter_varies_within_window"] += 1
            continue
        starts.append((s, r))
    return starts, rejects


def per_frame_ibi(beat_times, frame_times):
    """Local inter-beat interval (seconds) at each frame time, by containing interval.

    Frames before the first / after the last beat take the first / last interval: the
    grid is given, so extrapolating it is a statement about the grid, not about audio.
    """
    intervals = np.diff(beat_times)
    which = np.clip(np.searchsorted(beat_times, frame_times, side="right") - 1,
                    0, len(intervals) - 1)
    return intervals[which]


def build_crop(features, beat_times, start: int, r_true: int, m: int,
               crop_bars: int = CROP_BARS, fps: float = FPS):
    """One crop dict, or None if the frame window falls outside the feature array.

    Keys: h [T, D] float32, delta [T] the deterministic per-frame bar-phase advance
    2*pi / (m * IBI * fps), beat_frames [n] int, y [n] uint8, r_true int, m int.
    """
    n = crop_bars * m
    beats = beat_times[start:start + n]
    lo = int(np.floor(beats[0] * fps))
    hi = int(np.ceil(beats[-1] * fps)) + 1
    if lo < 0 or hi > len(features) or hi - lo < 2:
        return None
    h = np.asarray(features[lo:hi], dtype=np.float32)
    frame_times = (lo + np.arange(hi - lo)) / fps
    ibi = per_frame_ibi(beat_times, frame_times)
    delta = 2.0 * np.pi / (m * ibi * fps)
    beat_frames = np.clip(np.round(beats * fps).astype(int) - lo, 0, hi - lo - 1)
    y = np.zeros(n, dtype=np.uint8)
    y[r_true::m] = 1
    return {"h": h, "delta": delta.astype(np.float32), "beat_frames": beat_frames,
            "y": y, "r_true": r_true, "m": m}


def song_crops(features, song, m: int, rng, max_crops: int = 3,
               crop_bars: int = CROP_BARS, aligned: bool = False):
    """(crops, rejects) for one song: up to ``max_crops`` starts drawn uniformly.

    ``aligned=True`` reproduces the bar-aligned cropping this build exists to replace;
    it is the control arm, not a training option.
    """
    beat_times, downbeat_times = song.beats()
    rejects: Counter = Counter()
    if len(downbeat_times) == 0 or len(beat_times) < crop_bars * m + 1:
        rejects["song_too_short_or_unannotated"] += 1
        return [], rejects

    y_song, unmatched = derive_y(beat_times, downbeat_times, DOWNBEAT_TOL_S)
    rejects["unmatched_downbeats"] += unmatched
    # the corpus restriction is a SONG-level fact and is counted once per song; only
    # songs already in m reach the per-window meter-variation counter (see crop_starts)
    bar_length = song_bar_length(y_song)
    if bar_length is None:
        rejects["song_fewer_than_two_downbeats"] += 1
        return [], rejects
    if bar_length != m:
        rejects[f"song_meter_is_not_m(={bar_length})"] += 1
        rejects["songs_excluded_by_meter_restriction"] += 1
        return [], rejects
    rejects["songs_in_m"] += 1
    starts, start_rejects = crop_starts(y_song, m, crop_bars)
    rejects.update(start_rejects)
    if not starts:
        rejects["no_valid_start_for_m"] += 1
        return [], rejects

    if aligned:
        starts = [(s, r) for s, r in starts if r == 0] or starts
    chosen = [starts[i] for i in rng.choice(len(starts), size=min(max_crops, len(starts)),
                                            replace=False)]
    crops = []
    for start, r_true in chosen:
        crop = build_crop(features, beat_times, start, r_true, m, crop_bars)
        if crop is None:
            rejects["frame_window_outside_features"] += 1
            continue
        crop.update(dataset=song.dataset, stem=song.stem, start=start)
        crops.append(crop)
    return crops, rejects
