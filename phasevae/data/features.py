"""Fold-honest frontend features: the data layer the bar-phase VAE trains on.

FPS lives here (one owner), and ``iter_frontend_features`` is the single authority for
the frontend pass: checkpoint selection (each song through the checkpoint that held it
out; final0 for gtzan), audio load, mono mix, and the certified feature memo-cache
(one song per checkpoint group recomputed live and compared each run — that sampled
check catches GLOBAL drift and not per-song corruption; see the docstring for exact
limits). The song catalog itself is ``phasevae.songs``; the frontend wrappers are
``phasevae.frontends``.

The beat-grid crop machinery that once shared this module (derive_y, extract_crops,
load_crops — the Stage-0 era surface) had no remaining consumers on this branch and
was removed 2026-08-06; it is recoverable from git history.
"""
from __future__ import annotations

import pathlib

import numpy as np

from .songs import iter_songs

FPS = 50.0               # frames-per-second has ONE owner (this module): everything
                         # frame-related reads THIS, and the frontend pass asserts the
                         # frontend agrees — frame/second confusions are silent otherwise


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
    from .frontends.beat_this import BeatThisFrontend

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
                        f"frontend fps {frontend.fps} != phasevae.features.FPS {FPS}"
                features = _compute_features(frontend, s)
                if cache_path is not None:
                    group_dir.mkdir(parents=True, exist_ok=True)
                    atomic_save_npy(cache_path, features)
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


def atomic_save_npy(cache_path, array):
    """Write-then-rename: a reader racing a writer never sees a half-written array.

    np.save APPENDS ".npy" to any path not ending in it, silently renaming the temp
    file; an open file handle keeps the name as given. Rename is atomic within a
    filesystem. THE cache-write block -- the test suite exercises this function.
    """
    partial = cache_path.with_suffix(".npy.partial")
    with open(partial, "wb") as fh:
        np.save(fh, array.astype(np.float32))
    partial.replace(cache_path)
