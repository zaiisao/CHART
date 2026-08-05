"""Crops for a pure downbeat tracker: time windows, every meter, no beat grid.

What this replaces, and why each thing went:

  * ``m`` (beats per bar) is gone. It only ever entered as a route to the BAR period via
    ``m * IBI``, and the bar period is directly observable from consecutive annotated
    downbeats. Carrying ``m`` also silently turned a downbeat tracker into a joint
    beat-and-downbeat tracker, and forced a {2,3,4} vocabulary that cannot express 6/8
    or 5/4.
  * ``r`` (the discrete bar offset) is gone. It came from a model whose latent was
    discrete; with a continuous phase you read the phase, and a crop-level integer offset
    is a lossy summary of it. Dropping it also drops the rule that a crop span whole
    bars, which existed ONLY because the downbeat COUNT leaks r when it does not.
  * The "meter varies in this window" exclusion is gone. A mid-song meter change is just
    the bar period changing -- something the model tracks, not a reason to discard the
    window. The previous rule threw away ~96% of candidate windows.

So: every song, every meter, no vocabulary, no exclusion beyond "the audio or the
annotation is missing".

Crop starts are uniform in TIME and drawn per crop, never per song. A per-song phase is
uniform in aggregate while being perfectly predictable from song identity -- a leak that
looks like a healthy distribution.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from vbpm.data import FPS

MAX_CROP_SECONDS = 45.0  # MEASURED, not chosen. Scoring the true phase trajectory against
                         # a rigid constant-rate ramp under this exact ELBO, with the
                         # emission fitted to the oracle:
                         #     16 s  oracle -469.66  metronome -469.44  -> metronome wins
                         #     30 s  oracle -863.54  metronome -930.99  -> oracle +67
                         #     45 s  oracle -1300.7  metronome -1466.3  -> oracle +166
                         #     60 s  oracle -1724.6  metronome -1996.9  -> oracle +272
                         # At 16 s the objective is INDIFFERENT between tracking the truth
                         # and coasting from the right start, so anchor-and-coast is what
                         # it asks for. Real tempo drift accumulates with time and is what
                         # separates them; 45 s clears the tie without the memory cost of
                         # 60 s. Defined in TIME -- there is no beat grid here.
                         #
                         # It is a CAP, not a fixed length, because gtzan and ballroom are
                         # 30-second excerpts (median duration 29.0 s and 29.1 s, 0% over
                         # 45 s). A fixed 45 s silently deleted both -- 1,808 of 2,899
                         # songs, including the entire held-out test corpus -- and left a
                         # training set that was half asap with no transfer split at all.
                         # Each song now gets the longest crop its audio supports, so the
                         # short corpora sit near 29 s (margin ~65 nats, still clear) and
                         # the long ones get the full 45 s.
                         #
                         # Padding the short songs up to 45 s would NOT help: the margin
                         # comes from real accumulated drift, and padded frames carry no
                         # audio and no downbeats. Padding to a fixed length is the same
                         # computation as a shorter masked crop.
MIN_CROP_SECONDS = 24.0  # below this the oracle-vs-metronome margin gets thin (it is
                         # -0.21 nats at 16 s), so a shorter window is not worth training on
MIN_DOWNBEATS = 4        # fewer than this carries too little phase evidence to score
EDGE_MARGIN_S = 0.5      # keep the window clear of the very end of the feature array
MIN_START_RANGE_S = 4.0  # a song must offer at least this much range of start times, or
                         # the crop is shortened until it does -- otherwise every crop
                         # from a short song begins at t0 = 0 and the phase becomes a
                         # per-song constant


def bar_period(downbeat_times, lo_t: float, hi_t: float):
    """One constant bar period (seconds) for the window, or None if undefined.

    ONE CONSTANT PER CROP, and this is load-bearing rather than a convenience. A
    piecewise-constant per-frame period built from consecutive downbeat intervals has its
    discontinuities exactly ON the downbeats, so a model could read the step positions and
    recover every bar without consulting the audio at all -- scoring beautifully having
    learned nothing. A single median over the window carries no such information: it says
    how long a bar is, never where one starts.
    """
    inside = downbeat_times[(downbeat_times >= lo_t) & (downbeat_times <= hi_t)]
    if len(inside) < MIN_DOWNBEATS:
        return None
    return float(np.median(np.diff(inside)))


def build_crop(features, downbeat_times, lo_t: float, fps: float = FPS,
               crop_seconds: float = MAX_CROP_SECONDS, target_tol_frames: int = 1):
    """One crop dict, or None if the window is unusable.

    Keys: h [T, D] float32; delta scalar per-frame advance 2*pi/(bar_period*fps); y [T]
    per-frame downbeat target, widened to +-``target_tol_frames`` because the emission
    ``a + b cos(phi)`` is smooth and a single-frame spike would fight its shape;
    downbeat_times and t0 for SCORING ONLY.
    """
    hi_t = lo_t + crop_seconds
    period = bar_period(downbeat_times, lo_t, hi_t)
    if period is None or period <= 0:
        return None

    lo = int(round(lo_t * fps))
    hi = lo + int(round(crop_seconds * fps))
    if lo < 0 or hi > len(features):
        return None

    h = np.asarray(features[lo:hi], dtype=np.float32)
    inside = downbeat_times[(downbeat_times >= lo_t) & (downbeat_times <= hi_t)]
    # phase anchors BRACKET the window: interpolation needs a downbeat on each side or
    # the frames before the first inside-downbeat have no left anchor and are undefined --
    # which silently made the crop-start phase (the uniformity control) unmeasurable
    first = np.searchsorted(downbeat_times, lo_t, side="left")
    last = np.searchsorted(downbeat_times, hi_t, side="right")
    anchors = downbeat_times[max(0, first - 1):min(len(downbeat_times), last + 1)]
    y = np.zeros(hi - lo, dtype=np.float32)
    for t in inside:
        centre = int(round(t * fps)) - lo
        y[max(0, centre - target_tol_frames):centre + target_tol_frames + 1] = 1.0

    return {"h": h, "delta": float(2.0 * np.pi / (period * fps)), "y": y,
            "downbeat_times": np.asarray(inside, dtype=np.float64),
            "anchors": np.asarray(anchors, dtype=np.float64), "t0": float(lo_t),
            "bar_period": period}


def song_crops(features, song, rng, max_crops: int = 3, fps: float = FPS,
               crop_seconds: float = MAX_CROP_SECONDS):
    """(crops, rejects) for one song: ``max_crops`` windows at uniformly random times.

    The window is the LONGEST the song supports, capped at ``crop_seconds`` -- so a
    30-second excerpt yields a ~29-second crop rather than being discarded. See
    MAX_CROP_SECONDS for why this is a cap.

    ``max_crops`` is a SAMPLING cap, not a filter. The usable-crop total is set by it, and
    quoting that total as a filtering outcome misreads the data.
    """
    rejects: Counter = Counter()
    _beat_times, downbeat_times = song.beats()
    downbeat_times = np.asarray(downbeat_times, dtype=np.float64)
    duration = len(features) / fps - EDGE_MARGIN_S
    span = min(crop_seconds, duration)
    if len(downbeat_times) < MIN_DOWNBEATS or span < MIN_CROP_SECONDS:
        rejects["song_shorter_than_min_crop_or_unannotated"] += 1
        return [], rejects

    # Shorten the window when the song cannot offer a distinct start for every crop. A
    # 30-second excerpt with a 45-second cap gave span == duration, hence
    # uniform(0, 0) == 0.0 EVERY TIME, hence three byte-identical crops -- 41% of the
    # corpus duplicated and 100% of gtzan pinned to t0 = 0, which is exactly the
    # per-song-phase leak this module's docstring warns about. Reserve a start range of
    # at least one bar so the offsets are genuinely random, and never emit two crops with
    # the same start.
    span = min(span, max(MIN_CROP_SECONDS, duration - MIN_START_RANGE_S))
    if span < MIN_CROP_SECONDS:
        rejects["song_shorter_than_min_crop_or_unannotated"] += 1
        return [], rejects

    crops, seen = [], set()
    for _ in range(max_crops * 8):                 # a few tries per wanted crop
        if len(crops) >= max_crops:
            break
        lo_t = float(rng.uniform(0.0, duration - span))
        if round(lo_t, 3) in seen:                 # never the same window twice
            continue
        seen.add(round(lo_t, 3))
        crop = build_crop(features, downbeat_times, lo_t, fps, span)
        if crop is None:
            rejects["window_has_too_few_downbeats"] += 1
            continue
        crop.update(dataset=song.dataset, stem=song.stem)
        crops.append(crop)

    if not crops:
        rejects["no_usable_window"] += 1
    return crops, rejects


def true_phase(crop, fps: float = FPS):
    """(phase [T], valid [T]) -- the true bar phase per frame. SCORING ONLY.

    Linear interpolation between consecutive annotated downbeats, so phase is exactly 0 at
    each one and 2*pi just before the next. Frames outside the annotated span are marked
    invalid rather than extrapolated. Nothing on the deployable path may read this.
    """
    n = len(crop["y"])
    times = crop["t0"] + np.arange(n) / fps
    downbeats = crop["anchors"]
    turns = 2.0 * np.pi * np.arange(len(downbeats))
    phase = np.interp(times, downbeats, turns, left=np.nan, right=np.nan)
    valid = ~np.isnan(phase)
    phase = np.where(valid, phase, 0.0)
    return np.mod(phase, 2.0 * np.pi).astype(np.float32), valid
