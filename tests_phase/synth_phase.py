"""Phase-bearing synthetic bench: the same idea as ``tests/synth_bench.py``, plus an offset.

``tests/reference.py``'s bench sets ``y[::m] = 1``, so its bar offset is identically 0 —
exactly the property that makes the REAL crops useless for a phase model (18902 of 18902 at
r = 0; see docs/PHASE_PLAN.md). A bench for a phase model must therefore draw the offset,
and this one does.

Everything else is carried over deliberately: noise-free, exactly periodic, and NOT
BYPASSABLE. The beat period is drawn independently of ``m`` and the beat COUNT is fixed, so
neither tempo nor length carries the meter (docs/SPEC.md section 6.4 derives both leaks).
This is a machinery check, not a benchmark; do not harden it.
"""
from __future__ import annotations

import numpy as np

DEFAULT_VALUES = (2, 3, 4)
FPS = 50.0


def make_song(m: int, r0: int, beat_s: float, n_beats: int, rng, noise: float = 0.0,
              in_dim: int = 2):
    """One song: a beat grid whose downbeats start at bar offset ``r0``.

    Args:
        m: beats per bar, a COUNT.
        r0: the bar offset of beat 0, in ``0..m-1``. Beat ``i`` is a downbeat iff
            ``(i + r0) % m == 0``.
        beat_s: the beat period in seconds, drawn independently of ``m``.
        n_beats: beats in the song, drawn independently of ``m``.
        rng: a numpy Generator.
        noise: standard deviation of Gaussian noise added to the features.
        in_dim: feature width; channel 0 bumps at beats, channel 1 at downbeats.

    Returns:
        A crop dict with ``beat_h``, ``y``, ``m_true``, ``r0`` and ``downbeat_index``.
    """
    beats = (np.arange(n_beats, dtype=np.float64) + 0.5) * beat_s
    first = (-r0) % m
    downbeat_index = np.arange(first, n_beats, m)
    y = np.zeros(n_beats, dtype=np.float64)
    y[downbeat_index] = 1.0

    # beat-synchronous features directly: channel 0 marks every beat, channel 1 downbeats.
    # (beat_sync() is exercised separately; the bench hands the model beat-level features
    #  so a bench failure is never ambiguous between pooling and the chain.)
    beat_h = np.zeros((n_beats, in_dim), dtype=np.float64)
    beat_h[:, 0] = 1.0
    beat_h[:, 1] = y
    if noise > 0:
        beat_h = beat_h + rng.normal(0.0, noise, size=beat_h.shape)

    return {"beat_h": beat_h, "y": y, "m_true": int(m), "r0": int(r0),
            "downbeat_index": downbeat_index, "beats": beats, "beat_s": float(beat_s)}


def make_dataset(n_per_class: int, values=DEFAULT_VALUES, seed: int = 0,
                 noise: float = 0.0, n_beats: int = 32, in_dim: int = 2):
    """A balanced bench with a RANDOM bar offset per song — the phase bench.

    ``n_beats`` is held constant across meters and ``beat_s`` is drawn independently of
    ``m``, so a downbeat-blind classifier can read neither length nor tempo for the meter.
    """
    rng = np.random.default_rng(seed)
    songs = []
    for _ in range(n_per_class):
        for m in values:
            beat_s = float(rng.uniform(0.4, 0.8))
            r0 = int(rng.integers(0, m))
            songs.append(make_song(m, r0, beat_s, n_beats, rng, noise=noise,
                                   in_dim=in_dim))
    return songs


def make_meter_change_song(m_before: int, m_after: int, r0: int, beat_s: float,
                           bars_before: int, bars_after: int, rng,
                           noise: float = 0.0, in_dim: int = 2):
    """One song whose meter changes ONCE, at a bar boundary — Stage 0 cannot represent this.

    The pointer runs ``m_before`` for ``bars_before`` bars from offset ``r0``, then
    ``m_after``. The returned ``change_beat`` is the index of the first downbeat of the new
    meter, which is what a decode is scored against.
    """
    r = r0
    beat_meter, downbeat_index, bars_seen = [], [], 0
    m = m_before
    change_beat = None
    i = 0
    while bars_seen < bars_before + bars_after:
        if r == 0:
            downbeat_index.append(i)
            if bars_seen == bars_before:
                change_beat = i
        beat_meter.append(m)
        r += 1
        if r == m:
            r = 0
            bars_seen += 1
            if bars_seen == bars_before:
                m = m_after
        i += 1

    n_beats = i
    y = np.zeros(n_beats, dtype=np.float64)
    y[np.array(downbeat_index, dtype=int)] = 1.0
    beat_h = np.zeros((n_beats, in_dim), dtype=np.float64)
    beat_h[:, 0] = 1.0
    beat_h[:, 1] = y
    if noise > 0:
        beat_h = beat_h + rng.normal(0.0, noise, size=beat_h.shape)

    return {"beat_h": beat_h, "y": y, "m_true": int(m_before), "r0": int(r0),
            "downbeat_index": np.array(downbeat_index, dtype=int),
            "change_beat": int(change_beat), "m_after": int(m_after),
            "beats": (np.arange(n_beats) + 0.5) * beat_s}


def make_meter_change_dataset(n_songs: int, values=DEFAULT_VALUES, seed: int = 0,
                              noise: float = 0.0, in_dim: int = 2):
    """Songs with exactly one mid-song meter change at a bar boundary."""
    rng = np.random.default_rng(seed)
    songs = []
    for _ in range(n_songs):
        m_before = int(rng.choice(values))
        m_after = int(rng.choice([v for v in values if v != m_before]))
        songs.append(make_meter_change_song(
            m_before, m_after, int(rng.integers(0, m_before)),
            float(rng.uniform(0.4, 0.8)), 5, 5, rng, noise=noise, in_dim=in_dim))
    return songs
