"""The Stage-P synthetic bench (SS6.5): crops of KNOWN phase, with slip and meter change.

SS6.5 is blunt about why this exists: *"a phase model that cannot recover known phase from
clean synthetic input is broken, and this must be established before any real-data number
exists."* It is equally blunt that the bench cannot substitute for real-data controls -- it
is balanced by construction and the real corpus is not.

Design, and the leaks it closes
-------------------------------
Crops are cut from a LONG track at arbitrary beat starts, never at bar boundaries. That is
SS6.1's mandate made structural rather than promised: ``r_true`` is then uniform over
``0..m-1`` because the start beat is uniform, and SS3 P2 ("any path assuming ``y[0] == 1``
is wrong") has teeth. Stage 0's own bench cut at bar boundaries and put ``r = 0`` in ~99%
of crops (SS10.1), which is precisely how phase went unstudied for the whole project.

Cutting from a shared track also makes SS8.4's shift-consistency control *exact*: two crops
that differ by one beat of start position are two real cuts of the same audio, so ``h``,
the beat grid and ``y`` all move together. Shifting ``y`` alone would test the emission and
quietly leave the deployable path out of the control.

``h`` is genuinely phase-bearing: channel 0 has bumps at every beat, channel 1 at downbeats
only. This is the mechanism under test, so it must be present -- but note what that means
for interpretation. SS10.6 measured a synthetic-``h`` control at 0.988/0.991 and warns that
the bumps sit at the annotations that define the labels. A high score here proves the
crop/label/prior/read-out chain is sound. It cannot prove a frontend carries phase.

Slip and meter change
---------------------
SS4.1's ``eps_hold`` / ``eps_skip`` are exactly the events that a mid-crop meter change
produces in a fixed-``m`` model: inside an ``m = 4`` pointer, one bar of 3 IS a skip and one
bar of 5 IS a hold. The bench therefore generates both from one mechanism, and a crop
containing a slip event is automatically one on which SS6.2's ``r_true`` is undefined --
the ~2% that must be excluded and counted at P1, and the interesting cases at P2.
"""
from __future__ import annotations

import math

import numpy as np

import reference_p as R

DEFAULT_FPS = R.DEFAULT_FPS
BUMP_SIGMA_S = 0.05


def slip_schedule(n_beats: int, events) -> np.ndarray:
    """Return the ``[n_beats]`` pointer trajectory implied by a list of slip events.

    Args:
        n_beats: Number of beats in the track.
        events: Iterable of ``(beat_index, kind)`` with kind in ``{"hold", "skip"}``. The
            slip is applied on the transition INTO ``beat_index``.

    Returns:
        An int array of pointer states ``s_i``, starting from ``s_0 = 0``.

    Raises:
        ValueError: On an unknown slip kind.
    """
    sched = {int(i): str(k) for i, k in events}
    s = np.zeros(n_beats, dtype=int)
    for i in range(1, n_beats):
        kind = sched.get(i, "advance")
        if kind == "advance":
            step = 1
        elif kind == "hold":
            step = 0
        elif kind == "skip":
            step = 2
        else:
            raise ValueError(f"unknown slip kind {kind!r}")
        s[i] = (s[i - 1] + step) % R.STAGE_P_M
    return s


def make_track(n_beats: int = 96, m: int = R.STAGE_P_M, beat_s: float = 0.55,
               fps: float = DEFAULT_FPS, slip_events=(), noise: float = 0.0,
               seed: int = 0) -> dict:
    """Build one long synthetic track with a known pointer trajectory.

    Args:
        n_beats: Number of beats on the track.
        m: Beats per bar (SS2.1: a parameter, never inlined).
        beat_s: Beat period in seconds, constant.
        fps: Frames per second of the activation.
        slip_events: Slip schedule, see :func:`slip_schedule`.
        noise: Standard deviation of Gaussian noise added to ``h``.
        seed: Seed for the noise draw.

    Returns:
        A dict with ``h`` ``[T, 2]``, ``beat_frames`` ``[n_beats]``, ``y`` ``[n_beats]``,
        ``pointer`` ``[n_beats]``, ``m``, and ``fps``.
    """
    rng = np.random.default_rng(seed)
    pointer = slip_schedule(n_beats, slip_events) if slip_events else \
        np.arange(n_beats, dtype=int) % m
    y = (pointer == 0).astype(np.uint8)

    beats_s = np.arange(n_beats, dtype=np.float64) * beat_s + 1.0
    duration = float(beats_s[-1] + 2.0 * beat_s)
    n_frames = int(math.ceil(duration * fps))
    t = (np.arange(n_frames, dtype=np.float64) + 0.5) / fps

    def bumps(times):
        acc = np.zeros(n_frames, dtype=np.float64)
        for tc in times:
            acc += np.exp(-0.5 * ((t - tc) / BUMP_SIGMA_S) ** 2)
        return acc

    h = np.zeros((n_frames, 2), dtype=np.float64)
    h[:, 0] = bumps(beats_s)
    h[:, 1] = bumps(beats_s[y == 1])
    if noise > 0:
        h = h + rng.normal(0.0, noise, size=h.shape)

    beat_frames = np.clip(np.round(beats_s * fps - 0.5).astype(int), 0, n_frames - 1)
    return {"h": h, "beat_frames": beat_frames, "y": y, "pointer": pointer,
            "m": int(m), "fps": float(fps), "beats_s": beats_s}


def crop_at(track: dict, start_beat: int, n_beats: int) -> dict:
    """Cut one crop starting at an arbitrary BEAT offset -- SS6.1's un-aligned crop.

    ``h`` is sliced to the frame span of the crop and ``beat_frames`` is rebased, so the
    crop is a self-contained observation with no reference to its position in the track.
    That is what makes the shift control honest: nothing downstream can recover the start
    beat except through phase.

    Args:
        track: A track from :func:`make_track`.
        start_beat: Index of the crop's first beat.
        n_beats: Number of beats in the crop.

    Returns:
        A crop dict with ``h``, ``beat_frames``, ``y``, ``pointer``, ``m``, and ``r_true``
        (None when no single offset fits, per SS6.2).

    Raises:
        ValueError: If the requested span runs off the end of the track, or if the crop
            does not span whole bars.
    """
    n_total = len(track["y"])
    if start_beat < 0 or start_beat + n_beats > n_total:
        raise ValueError(f"crop [{start_beat}, {start_beat + n_beats}) outside {n_total} beats")
    # Un-aligned in PHASE, whole-numbered in BARS. A crop with n % m != 0 hands out its
    # offset for free through the downbeat-slot count alone, which would make every
    # chance-level gate in SS8.3 wrong; reference_p.assert_whole_bars states the arithmetic.
    R.assert_whole_bars(int(n_beats), int(track["m"]))
    bf = track["beat_frames"]
    lo = int(bf[start_beat])
    hi = int(bf[start_beat + n_beats - 1]) + 1
    y = np.asarray(track["y"][start_beat:start_beat + n_beats]).copy()
    return {
        "h": np.asarray(track["h"][lo:hi]).copy(),
        "beat_frames": (bf[start_beat:start_beat + n_beats] - lo).astype(int),
        "y": y,
        "pointer": np.asarray(track["pointer"][start_beat:start_beat + n_beats]).copy(),
        "m": int(track["m"]),
        "start_beat": int(start_beat),
        "r_true": R.derive_r_true(y, int(track["m"])),
    }


def make_dataset(n_crops: int = 24, n_beats: int = 16, m: int = R.STAGE_P_M,
                 seed: int = 0, slip_events=(), noise: float = 0.0,
                 n_track_beats: int = 96) -> list:
    """Build a set of un-aligned crops with known phase, balanced over offsets.

    The start beat is swept rather than sampled, so every offset appears equally often even
    at small ``n_crops``. SS3 P5 warns that run-to-run margins are +-0.08; a bench whose
    class balance itself wobbles with the seed would add a second noise source to every
    property that thresholds an accuracy.

    Args:
        n_crops: Number of crops to cut.
        n_beats: Beats per crop.
        m: Beats per bar.
        seed: Seed for the track's noise.
        slip_events: Slip schedule passed to :func:`make_track`.
        noise: Activation noise.
        n_track_beats: Length of the underlying track.

    Returns:
        A list of crop dicts, including any on which ``r_true`` is None.
    """
    track = make_track(n_beats=n_track_beats, m=m, slip_events=slip_events,
                       noise=noise, seed=seed)
    span = n_track_beats - n_beats
    starts = [(i * 1) % span for i in range(n_crops)]
    return [crop_at(track, s, n_beats) for s in starts]


def make_labelled_dataset(n_crops: int = 24, n_beats: int = 16, m: int = R.STAGE_P_M,
                          seed: int = 0, noise: float = 0.0) -> list:
    """Build a slip-free dataset on which every crop has a well-defined ``r_true``.

    Args:
        n_crops: Number of crops.
        n_beats: Beats per crop.
        m: Beats per bar.
        seed: Seed for the track's noise.
        noise: Activation noise.

    Returns:
        A list of crop dicts, each with an integer ``r_true``.
    """
    crops = make_dataset(n_crops=n_crops, n_beats=n_beats, m=m, seed=seed, noise=noise)
    return [c for c in crops if c["r_true"] is not None]


def make_meter_change_dataset(n_crops: int = 24, n_beats: int = 16,
                              m: int = R.STAGE_P_M, seed: int = 0) -> list:
    """Build crops around a mid-track meter change -- SS6.5's P2 case, SS6.2's exclusions.

    One bar of 3 inside an ``m = 4`` pointer is a "skip"; one bar of 5 is a "hold". Crops
    straddling either event have no single fitting offset, so ``r_true`` is None and P1
    must exclude and count them. P2 is supposed to explain them.

    Args:
        n_crops: Number of crops.
        n_beats: Beats per crop.
        m: Beats per bar.
        seed: Seed for the track's noise.

    Returns:
        A list of crop dicts, a mixture of labelled and unlabelled.
    """
    events = ((40, "skip"), (64, "hold"))
    return make_dataset(n_crops=n_crops, n_beats=n_beats, m=m, seed=seed,
                        slip_events=events)
