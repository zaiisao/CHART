"""Per-song frontend primitives: FPS, the feature cache location, one-song compute.

The corpus pass that drives these (fold grouping, checkpoint routing, the cache drift
probe) is ``dataset.load_dataset`` — the loop lives there, in the open, next to the
crop building it feeds; this module keeps only the pieces it calls per song. The song
catalog is ``phasevae.songs``; the frontend wrappers are ``phasevae.frontends``.

The beat-grid crop machinery that once shared this module (derive_y, extract_crops,
load_crops — the Stage-0 era surface) had no remaining consumers on this branch and
was removed 2026-08-06; it is recoverable from git history.
"""
from __future__ import annotations

import numpy as np

FPS = 50.0               # the LEGACY crop pipeline's grid (checks/ scripts): its crop
                         # builder and frontend pass assert against THIS. The excerpt
                         # pipeline has no global grid — everything reads frontend.FPS
                         # and the per-crop "fps" key instead.


FEATURE_CACHE_DIR = "/disk4/jaehoon/vbpm_feature_cache"   # user decision 2026-08-01:
# memoize the CERTIFIED pass's output (float32, verified against a live recompute on
# every load) — this is not a second pipeline, it is the one pipeline remembered.


def compute_features(frontend, song):
    """One song through the frontend: audio load, mono mix, forward."""
    import soundfile
    signal, sample_rate = soundfile.read(str(song.audio_path), dtype="float32")
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    return frontend.get_features(signal, sample_rate).numpy()


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
