"""The dataset layer: crop construction, assembly, caching, folds and batching.

CROP MATH (was crops.py):
Crops for a pure downbeat tracker: time windows, every meter, no beat grid.

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

ASSEMBLY/BATCHING (was loading.py):
Crops in, padded GPU batches out. Fold-honest
features, one-time collation.
"""
from __future__ import annotations

import numpy as np
import pathlib
import pickle
import torch

from .features import FEATURE_CACHE_DIR, FPS, atomic_save_npy, compute_features
from .songs import ANNOTATIONS_ROOT, AUDIO_BY_STEM, Song
from collections import Counter

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
            "bar_period": period, "fps": float(fps)}


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
        crop.update(dataset=song.dataset, stem=song.stem, fold=song.fold)
        crops.append(crop)

    if not crops:
        rejects["no_usable_window"] += 1
    return crops, rejects


def true_phase(crop):
    """(phase [T], valid [T]) -- the true bar phase per frame. SCORING ONLY.

    Linear interpolation between consecutive annotated downbeats, so phase is exactly 0 at
    each one and 2*pi just before the next. Frames outside the annotated span are marked
    invalid rather than extrapolated. Nothing on the deployable path may read this.
    The frame grid comes from the crop itself (``fps`` key) -- there is no global grid.
    """
    n = len(crop["y"])
    times = crop["t0"] + np.arange(n) / crop["fps"]
    downbeats = crop["anchors"]
    turns = 2.0 * np.pi * np.arange(len(downbeats))
    phase = np.interp(times, downbeats, turns, left=np.nan, right=np.nan)
    valid = ~np.isnan(phase)
    phase = np.where(valid, phase, 0.0)
    return np.mod(phase, 2.0 * np.pi).astype(np.float32), valid


MAX_CROPS = 3         # per song; a SAMPLING cap, not a filter
VAL_FOLD = 7


def load_catalog(datasets=None) -> dict:
    """The song catalog, grouped by CV fold (fold -> [Song]).

    Every song with BOTH a .beats annotation and an audio link. Links are named by
    annotation stem (built and VERIFIED by unify_audio_layout.py) so the lookup is
    exact -- no name matching here. A dataset with no audio is silently absent;
    songs.coverage_report() shows it. fold None = the test-only/gtzan group, held out
    of EVERY checkpoint.
    """
    songs_by_fold: dict = {}
    for dataset_dir in sorted(ANNOTATIONS_ROOT.iterdir()):
        dataset = dataset_dir.name
        if datasets is not None and dataset not in datasets:
            continue
        beats_dir = dataset_dir / "annotations" / "beats"
        audio_dir = AUDIO_BY_STEM / dataset
        if not beats_dir.is_dir() or not audio_dir.is_dir():
            continue
        audio_by_stem = {p.stem: p for p in audio_dir.iterdir()}
        split_file = dataset_dir / "8-folds.split"
        if split_file.exists():
            entries = [line.split("\t") for line in split_file.read_text().splitlines()]
        else:
            # No CV split = a test-only dataset in the Beat This protocol (gtzan).
            entries = [(p.stem, None) for p in sorted(beats_dir.glob("*.beats"))]
        for stem, fold in entries:
            audio_path = audio_by_stem.get(stem)
            if audio_path is not None and (beats_dir / f"{stem}.beats").exists():
                fold = int(fold) if fold is not None else None
                songs_by_fold.setdefault(fold, []).append(
                    Song(stem, dataset, fold, audio_path, beats_dir / f"{stem}.beats"))
    return songs_by_fold


def load_dataset(seed: int = 0, limit_per_fold=None, datasets=None,
                 override_checkpoint: str | None = None):
    """(crops, rejects) for every song, each through the checkpoint that held it out.

    THE frontend pass — checkpoint routing is where fold-honesty lives, so it exists
    exactly once, here. Features are memoized under FEATURE_CACHE_DIR; the memo-cache
    is re-certified every run by the per-group drift probe below.

    ``override_checkpoint`` forces EVERY song through one named checkpoint. For gtzan
    (no checkpoint trained on it) that is still fold-honest — see load_gtzan_through;
    for any song whose checkpoint trained on it, the output is a diagnostic about the
    FEATURES and must never be reported as a fold-honest score.
    """
    from .frontends.beat_this import BeatThisFrontend

    rng = np.random.default_rng(seed)
    crops, rejects = [], Counter()
    songs_by_fold = load_catalog(datasets)

    fold_order = sorted(fold for fold in songs_by_fold if fold is not None)
    if None in songs_by_fold:
        fold_order.append(None)           # the fold-None/test group (gtzan) runs last

    for fold in fold_order:
        songs = songs_by_fold[fold]
        if limit_per_fold is not None:
            songs = songs[:limit_per_fold]

        # Fold-honesty itself: this fold's songs go through the Beat This checkpoint
        # that HELD THEM OUT of its training ("final0" for the fold-None test group,
        # which no fold checkpoint trained on) -- unless the caller forces one.
        checkpoint = override_checkpoint or ("final0" if fold is None else f"fold{fold}")

        # song.stem names each song's cache file, so a repeated stem would silently
        # serve one song's features as another's.
        stems = [song.stem for song in songs]
        assert len(set(stems)) == len(stems), \
            f"{checkpoint}: duplicate song stems in this group"
        group_dir = pathlib.Path(FEATURE_CACHE_DIR) / checkpoint / "features_activations"

        frontend = BeatThisFrontend(checkpoint=checkpoint, device="cuda",
                                    output="features+activations")
        assert frontend.FPS == FPS, \
            f"frontend fps {frontend.FPS} != phasevae.features.FPS {FPS}"

        served_from_cache = []
        for song in songs:
            cache_path = group_dir / f"{song.stem}.npy"
            if cache_path.exists():
                features = np.load(cache_path)
                served_from_cache.append(song)
            else:
                features = compute_features(frontend, song)
                group_dir.mkdir(parents=True, exist_ok=True)
                atomic_save_npy(cache_path, features)

            got, rej = song_crops(features, song, rng, MAX_CROPS)
            crops += got
            rejects.update(rej)

        if served_from_cache:
            # re-earn trust: one cached song per group is recomputed live and compared
            probe = served_from_cache[0]
            live = compute_features(frontend, probe)
            cached = np.load(group_dir / f"{probe.stem}.npy")
            drift = float(np.max(np.abs(live.astype(np.float64)
                                        - cached.astype(np.float64))))
            assert drift < 1e-3, (
                f"feature cache DRIFT on {probe.stem} ({checkpoint}): "
                f"max|live - cached| = {drift:.3e} — delete the cache and recompute")

        del frontend
        print(f"  {checkpoint}: done ({len(served_from_cache)}/{len(songs)} cached)",
              flush=True)

    return crops, rejects


def load_or_build(cache, limit_per_fold):
    """Crops from ``cache`` if it exists, else built and written there."""
    if cache and pathlib.Path(cache).exists():
        with open(cache, "rb") as fh:
            crops, rejects = pickle.load(fh)
        print(f"crops loaded from {cache}", flush=True)
        return crops, rejects

    crops, rejects = load_dataset(limit_per_fold=limit_per_fold)
    if cache:
        with open(cache, "wb") as fh:
            pickle.dump((crops, rejects), fh)
    return crops, rejects


def split_songs(limit_per_fold=None):
    """(train, val, test) Song lists: train folds / fold VAL_FOLD / fold None (gtzan).

    The song-level counterpart of split_folds -- same policy, one owner. limit_per_fold
    trims each fold for smoke runs.
    """
    songs_by_fold = load_catalog()
    if limit_per_fold is not None:
        songs_by_fold = {fold: songs[:limit_per_fold]
                         for fold, songs in songs_by_fold.items()}
    train = [song for fold, group in songs_by_fold.items()
             if fold not in (None, VAL_FOLD) for song in group]
    return train, songs_by_fold.get(VAL_FOLD, []), songs_by_fold.get(None, [])


def split_folds(crops):
    """(train, val, test): train folds / fold VAL_FOLD / fold None (gtzan).

    THE fold split; it was written out inline three times before.
    """
    train = [c for c in crops if c["fold"] not in (None, VAL_FOLD)]
    val = [c for c in crops if c["fold"] == VAL_FOLD]
    test = [c for c in crops if c["fold"] is None]
    return train, val, test


def load_gtzan_through(checkpoint: str):
    """Gtzan test crops through a FOLD checkpoint instead of the cache's final0.

    Any fold checkpoint is fold-honest for gtzan (none trained on it). final0 is an
    activation space the encoder never sees in training, which alone cost the supervised
    probe 0.75 -> 0.06.
    """
    return load_dataset(seed=0, datasets=["gtzan"], override_checkpoint=checkpoint)


def collate(batch):
    """Pad crops into PINNED CPU tensors; ``h`` stays fp16 on the wire.

    The fp32 cast is free on the GPU, so shipping fp16 halves the transfer; pinned
    memory lets the copy overlap compute. ``delta`` is one CONSTANT per crop.
    """
    length = max(len(c["y"]) for c in batch)
    count, width = len(batch), batch[0]["h"].shape[1]
    h = torch.zeros(count, length, width, dtype=torch.float16)
    delta = torch.zeros(count, length)
    mask = torch.zeros(count, length)
    y = torch.zeros(count, length)

    for i, crop in enumerate(batch):
        t = len(crop["y"])
        h[i, :t] = torch.from_numpy(np.asarray(crop["h"], dtype=np.float16))
        delta[i, :t] = float(crop["delta"])
        mask[i, :t] = 1.0
        y[i, :t] = torch.from_numpy(crop["y"].astype(np.float32))

    return {k: v.pin_memory() for k, v in
            {"h": h, "delta": delta, "mask": mask, "y": y}.items()}


def to_device(batch, device):
    """Ship a collated batch, casting h back to fp32 on the GPU where it costs nothing."""
    out = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    out["h"] = out["h"].float()
    return out


class Batches:
    """Crops padded ONCE into fixed length-sorted buckets, reused every epoch.

    Rebuilding padded tensors per epoch was 66% of wall time. The crops never change
    and neither do the buckets -- only the ORDER buckets are visited in -- so padding
    happens once here and shuffling permutes bucket order, never membership.
    """

    def __init__(self, crops, batch_size: int, device):
        order = np.argsort([len(c["y"]) for c in crops])
        self.device = device
        self.chunks = [[crops[i] for i in order[j:j + batch_size]]
                       for j in range(0, len(order), batch_size)]
        self.padded = [collate(chunk) for chunk in self.chunks]

    def __len__(self):
        return len(self.chunks)

    def __call__(self, shuffle: bool = False, rng=None):
        """Yield (raw crops, batch on device)."""
        index = np.arange(len(self.chunks))
        if shuffle:
            rng.shuffle(index)
        for i in index:
            yield self.chunks[i], to_device(self.padded[i], self.device)
