"""Real data -> training CROPS {h, y, m_true} for Stage 0.

y (per-beat downbeat indicator) and m_true (beats per bar) come from the trusted
beat/downbeat annotation files via data/songs.py. h is Beat This output computed through
frontends/ — fold-honestly: each song goes through the checkpoint that held that song
out of training (final0 for gtzan, which no checkpoint ever trained on). Features are
memoized on disk (user decision 2026-08-01, revising the earlier no-caches rule: the cache
stores the certified path's own output, and one song per checkpoint group is recomputed
live each run and compared). That sampled check catches GLOBAL drift — a changed
checkpoint, frontend or audio path — and does not catch per-song corruption; see
``iter_frontend_features`` for the exact limits, which are narrower than "verified".

Every excluded song or crop is counted by reason and reported — rejections are
surfaced, never silent, because a silently absent corpus once looked like a fact
about the data when it was a fact about the matcher.
"""
from __future__ import annotations

import math
import pathlib
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


FEATURE_CACHE_DIR = "/disk4/jaehoon/vbpm_feature_cache"   # user decision 2026-08-01:
# memoize the CERTIFIED pass's output (float32, verified against a live recompute on
# every load) — this is not a second pipeline, it is the one pipeline remembered.


def _compute_features(frontend, song):
    """One song through the frontend: audio load, mono mix, forward."""
    import soundfile
    signal, sample_rate = soundfile.read(str(song.audio_path), dtype="float32")
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    return frontend.get_features(signal, sample_rate).numpy()


def iter_frontend_features(datasets=None, device: str = "cuda", limit_per_fold=None,
                           output: str = "activations", verbose: bool = True,
                           folds=None, cache_dir: str = FEATURE_CACHE_DIR,
                           override_checkpoint: str | None = None):
    """Yield (song, features) fold-honestly: each song through the checkpoint that held it out.

    The single authority for the frontend pass (checkpoint selection, audio load, mono
    mix) — this is where fold-honesty lives, so it must exist exactly once.

    Features are memoized under cache_dir (cache_dir=None forces fully-live computation).
    For each checkpoint group that served any cache hit, ONE cached song is recomputed live
    and compared. Be precise about what that buys: one probe per group detects GLOBAL drift
    (a changed checkpoint, a changed frontend, a changed audio path) and cannot detect
    per-song corruption of the other ~250 songs in the group. It is also an ``assert``, so
    it disappears under ``python -O``, and it runs after the group has been yielded -- a
    consumer that breaks out of the generator early never reaches it.

    ``folds`` filters which checkpoint groups run (integers 0-7, or None-in-list for the
    test-only/final0 group) — the knob that lets a multi-GPU warmer shard the pass.

    ``override_checkpoint`` forces EVERY song through one named checkpoint. It exists for
    the checkpoint-swap probe alone: it deliberately breaks fold-honesty (songs go through
    a checkpoint that trained on them), so its output is a diagnostic about the FEATURES
    and must never be reported as a fold-honest score.
    """
    from frontends.beat_this import BeatThisFrontend

    by_fold: dict = {}
    for s in iter_songs(datasets=datasets):
        by_fold.setdefault(s.fold, []).append(s)

    for fold, members in sorted(by_fold.items(), key=lambda kv: (kv[0] is None, kv[0])):
        if folds is not None and fold not in folds:
            continue
        checkpoint = override_checkpoint or ("final0" if fold is None else f"fold{fold}")
        if limit_per_fold is not None:
            members = members[:limit_per_fold]
        stems = [s.stem for s in members]
        assert len(set(stems)) == len(stems), \
            f"{checkpoint}: song stems are the cache key and are not unique in this group"
        group_dir = (pathlib.Path(cache_dir) / checkpoint / output.replace("+", "_")
                     if cache_dir else None)

        frontend = None   # instantiated lazily: a fully-cached group may not need the GPU
        served_from_cache = []
        for s in members:
            cache_path = group_dir / f"{s.stem}.npy" if group_dir else None
            if cache_path is not None and cache_path.exists():
                features = np.load(cache_path)
                served_from_cache.append(s)
            else:
                if frontend is None:
                    frontend = BeatThisFrontend(checkpoint=checkpoint, device=device,
                                                output=output)
                    assert frontend.fps == FPS, \
                        f"frontend fps {frontend.fps} != vbpm.data.FPS {FPS}"
                features = _compute_features(frontend, s)
                if cache_path is not None:
                    group_dir.mkdir(parents=True, exist_ok=True)
                    # write-then-rename: a reader racing a warmer must never see a
                    # half-written array (rename is atomic within a filesystem)
                    partial = cache_path.with_suffix(".npy.partial")
                    # np.save APPENDS ".npy" to any path not ending in it, silently
                    # renaming the temp file; an open handle keeps the name as given
                    with open(partial, "wb") as fh:
                        np.save(fh, features.astype(np.float32))
                    partial.replace(cache_path)
            yield s, features

        if served_from_cache:
            # re-earn trust: one cached song per group is recomputed live and compared
            probe = served_from_cache[0]
            if frontend is None:
                frontend = BeatThisFrontend(checkpoint=checkpoint, device=device,
                                            output=output)
            live = _compute_features(frontend, probe)
            cached = np.load(group_dir / f"{probe.stem}.npy")
            drift = float(np.max(np.abs(live.astype(np.float64)
                                        - cached.astype(np.float64))))
            assert drift < 1e-3, (
                f"feature cache DRIFT on {probe.stem} ({checkpoint}/{output}): "
                f"max|live - cached| = {drift:.3e} — delete the cache and recompute")

        del frontend
        if verbose:
            print(f"  {checkpoint}: done ({len(served_from_cache)}/{len(members)} cached)",
                  flush=True)


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
