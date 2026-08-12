"""Per-song frontend primitives: FPS, the feature cache location, one-song compute."""
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
    """Write-then-rename: a reader racing a writer never sees a half-written array."""
    partial = cache_path.with_suffix(".npy.partial")
    with open(partial, "wb") as fh:
        np.save(fh, array.astype(np.float32))
    partial.replace(cache_path)
