"""Real data -> training CROPS {h, y, m_true} for Stage 0.

y (per-beat downbeat indicator) and m_true (beats per bar) come from the trusted
beat/downbeat annotation files via data/songs.py. h is live Beat This output computed
through frontends/ — never cached (standing decision: a cache is a second, uncertified
code path through the frontend) — and fold-honestly: each song goes through the
checkpoint that held that song out of training (final0 for gtzan, which no checkpoint
ever trained on).

Every excluded song or crop is counted by reason and reported — rejections are
surfaced, never silent, because a silently absent corpus once looked like a fact
about the data when it was a fact about the matcher.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from data.songs import iter_songs

from .stage0 import DEFAULT_VALUES

MIN_BARS = 3
DOWNBEAT_TOL_S = 0.02
VALUES = DEFAULT_VALUES  # the meter vocabulary has ONE owner (vbpm.stage0); alias only
FPS = 50.0               # frames-per-second has ONE owner (this module): everything
                         # frame-related reads THIS, and the frontend pass asserts the
                         # frontend agrees — frame/second confusions are silent otherwise
MIN_BEATS = 12          # a crop must cover >= 3 bars at the largest meter (4): 12 beats
CROP_BARS = 8           # complete bars per crop: 16/24/32 beats at m = 2/3/4, all >= 12.
                        # One m per CROP, and a song is NOT one crop: ~4% of songs change
                        # meter mid-song (concentrated in asap), so a whole-song median
                        # m_true would fabricate a label wrong for the entire song.


def derive_y(beat_times, downbeat_times, tol_s: float = DOWNBEAT_TOL_S):
    """Per-beat downbeat indicator y, plus the count of unmatched downbeats.

    y_i = 1 iff beat i has an annotated downbeat within tol seconds.

    A downbeat that matches no beat is a data error to surface, not a beat to insert —
    downbeats are by definition a subset of beats.
    """
    y = np.zeros(len(beat_times), dtype=np.uint8)
    unmatched = 0
    for d in downbeat_times:
        i = int(np.argmin(np.abs(beat_times - d)))
        if abs(beat_times[i] - d) <= tol_s:
            y[i] = 1
        else:
            unmatched += 1
    return y, unmatched


def derive_m_true(beat_times, downbeat_times, min_bars: int = MIN_BARS,
                  tol_s: float = DOWNBEAT_TOL_S):
    """Beats-per-bar label: the median bar length in beats, over complete bars.

    Counts the beats falling in each bar [d_k, d_{k+1}); ties round half-up. Returns
    None (crop rejected) when there are fewer than min_bars complete bars to take a
    median over.
    """
    n_bars = len(downbeat_times) - 1
    if n_bars < min_bars:
        return None
    counts = [int(np.sum((beat_times >= downbeat_times[k] - tol_s)
                         & (beat_times < downbeat_times[k + 1] - tol_s)))
              for k in range(n_bars)]
    return int(math.floor(float(np.median(counts)) + 0.5))


def make_crops(beat_times, downbeat_times, crop_bars: int = CROP_BARS):
    """Bar-aligned crops: consecutive complete bars in chunks of ``crop_bars``.

    Yields (crop_beat_times, bar_bounds) with bar_bounds = the B+1 downbeat times
    delimiting the crop's complete bars, so the m_true median runs over exactly those
    bars. Beats before the first downbeat and after the last complete bar belong to
    incomplete bars, which carry no label, so they land in no crop. A tail shorter than
    MIN_BARS complete bars cannot be labeled and is dropped.
    """
    tol = DOWNBEAT_TOL_S
    n_bars = len(downbeat_times) - 1
    crops = []
    for start_bar in range(0, n_bars, crop_bars):
        end_bar = min(start_bar + crop_bars, n_bars)
        if end_bar - start_bar < MIN_BARS:
            break
        bounds = downbeat_times[start_bar:end_bar + 1]
        sel = (beat_times >= bounds[0] - tol) & (beat_times < bounds[-1] - tol)
        if not sel.any():
            continue
        crops.append((beat_times[sel], bounds))
    return crops


def extract_crops(beat_times, downbeat_times, values=VALUES):
    """(crops, rejects): every valid labeled crop of one song, and why the rest fell.

    Each crop is {"crop": index, "beats": times, "bounds": bar bounds, "y": indicator,
    "m_true": count}. The single authority for crop validity — experiments must not
    re-implement this chain (a five-way copy is how policies silently diverge).
    """
    rejects: Counter = Counter()
    if len(downbeat_times) == 0:
        rejects["no_downbeat_annotation"] += 1
        return [], rejects
    if len(downbeat_times) - 1 < MIN_BARS:
        rejects[f"fewer_than_{MIN_BARS}_bars"] += 1
        return [], rejects

    crops = []
    for crop_index, (crop_beats, bar_bounds) in enumerate(
            make_crops(beat_times, downbeat_times)):
        m_true = derive_m_true(crop_beats, bar_bounds)
        if m_true is None:
            rejects["crop_fewer_bars_than_min"] += 1
            continue
        if m_true not in values:
            rejects[f"crop_m_out_of_vocabulary({m_true})"] += 1
            continue
        if len(crop_beats) < MIN_BEATS:
            rejects[f"crop_fewer_than_{MIN_BEATS}_beats"] += 1
            continue

        # y is matched against the crop's bar STARTS (bounds[:-1]); the closing bound
        # is the next crop's first downbeat, not a downbeat of this crop
        y, unmatched = derive_y(crop_beats, bar_bounds[:-1])
        rejects["unmatched_downbeats"] += unmatched
        crops.append({"crop": crop_index, "beats": crop_beats, "bounds": bar_bounds,
                      "y": y, "m_true": m_true})
    if not crops:
        rejects["no_usable_crops"] += 1
    return crops, rejects


def iter_frontend_features(datasets=None, device: str = "cuda", limit_per_fold=None,
                           output: str = "activations", verbose: bool = True):
    """Yield (song, features) fold-honestly: each song through the checkpoint that held it out.

    The single authority for the frontend pass (checkpoint selection, audio load, mono
    mix) — this is where fold-honesty lives, so it must exist exactly once.
    """
    import soundfile
    from frontends.beat_this import BeatThisFrontend

    by_fold: dict = {}
    for s in iter_songs(datasets=datasets):
        by_fold.setdefault(s.fold, []).append(s)

    for fold, members in sorted(by_fold.items(), key=lambda kv: (kv[0] is None, kv[0])):
        checkpoint = "final0" if fold is None else f"fold{fold}"
        frontend = BeatThisFrontend(checkpoint=checkpoint, device=device, output=output)
        assert frontend.fps == FPS, \
            f"frontend fps {frontend.fps} != vbpm.data.FPS {FPS}: one module owns fps"
        if limit_per_fold is not None:
            members = members[:limit_per_fold]

        for s in members:
            signal, sample_rate = soundfile.read(str(s.audio_path), dtype="float32")
            if signal.ndim > 1:
                signal = signal.mean(axis=1)
            yield s, frontend.get_features(signal, sample_rate).numpy()

        del frontend
        if verbose:
            print(f"  {checkpoint}: done", flush=True)


def to_prob(h):
    """sigmoid: the frontend emits LOGITS; this is the one owner of that conversion."""
    return 1.0 / (1.0 + np.exp(-np.asarray(h, dtype=np.float64)))


def slice_h(features, crop_beats):
    """The crop's frame window of a whole-song feature array, plus its start time.

    Frame count tracks beat count: the features handed to the model span exactly the
    beats the crop's y describes.
    """
    lo = max(0, int(math.floor(crop_beats[0] * FPS)))
    hi = min(len(features), int(math.ceil(crop_beats[-1] * FPS)) + 1)
    return features[lo:hi], lo / FPS


def default_entry(song, crop, h_crop, t0):
    """The standard training-crop dict; loaders needing more override make_entry."""
    return {"h": h_crop, "y": crop["y"], "m_true": crop["m_true"],
            "dataset": song.dataset, "fold": song.fold,
            "stem": song.stem, "crop": crop["crop"]}


def load_crops(datasets=None, device: str = "cuda", limit_per_fold=None,
               values=VALUES, verbose: bool = True, output: str = "activations",
               make_entry=default_entry):
    """(crops, report). One entry per valid crop, built by ``make_entry``.

    The single assembly path for every consumer (package and experiments): the frontend
    pass, the crop-validity chain and the reject accounting exist here ONCE.
    ``make_entry`` receives (song, crop, h_crop, t0) with h sliced to the crop's frames.
    """
    crops, rejects = [], Counter()
    for s, h in iter_frontend_features(datasets=datasets, device=device,
                                       limit_per_fold=limit_per_fold, verbose=verbose,
                                       output=output):
        beat_times, downbeat_times = s.beats()
        song_crops, song_rejects = extract_crops(beat_times, downbeat_times, values)
        rejects.update(song_rejects)

        for c in song_crops:
            h_crop, t0 = slice_h(h, c["beats"])
            crops.append(make_entry(s, c, h_crop, t0))

    class_counts_by_dataset: dict = {}
    for c in crops:
        class_counts_by_dataset.setdefault(c["dataset"], Counter())[c["m_true"]] += 1

    unmatched = rejects.pop("unmatched_downbeats", 0)
    report = {"usable": len(crops), "rejects": dict(rejects),
              "unmatched_downbeats": unmatched,
              "class_counts_by_dataset": {dataset: dict(counts) for dataset, counts
                                          in class_counts_by_dataset.items()}}
    return crops, report
