"""Frontend-agnostic training excerpts: the shared shape of both official pipelines.

What Beat This and Beat Transformer's datasets BOTH do, and this module does once over
any Frontend: train on a cached, model-free input representation (never raw audio in the
Dataset); slice a fixed-length window from each song; rasterize the beat annotations to
framewise targets at the representation's frame rate; pad short songs under a mask.

What they disagree on, and the choice made here:
  * window policy — Beat This draws a FRESH random window per __getitem__ (new windows
    every epoch); Beat Transformer pre-chops fixed non-overlapping chunks at init. We
    take Beat This's: strictly more data diversity, and it subsumes the other (their
    val/test "middle excerpt" is the deterministic=True case).
  * target widening — Beat This widens in the LOSS, Beat Transformer in the DATA
    (half-height shoulders). We keep OUR OWN target convention (build_crop's ±1-frame
    widening) because the consumer is the bar-phase VAE's emission, not their heads.
  * everything frontend-specific (mel recipe, demixing, frame rate, input layout,
    chunking) lives in the Frontend class — this module only ever assumes "axis 0 of
    the cached array is time, at frontend.FPS".

The cache layout mirrors Beat This's spectrogram store one-to-one
(<cache>/<frontend>/<dataset>/<stem>/track.npy) so their precomputed pitch/tempo
augmentation scheme (variant files next to track.npy) can drop in later without a
relayout.

The VAE-side contract per item matches build_crop's: the window carries ONE constant
bar period (median over its downbeats — a per-frame period would leak bar positions
through its discontinuities; see bar_period), a ±1-frame-widened downbeat target y,
and SCORING-ONLY fields (downbeat_times, anchors, t0) that nothing on the training
path may read.
"""
from __future__ import annotations

import pathlib

import numpy as np
import torch

from .dataset import MIN_DOWNBEATS, bar_period
from .features import atomic_save_npy

INPUT_CACHE_DIR = "/disk4/jaehoon/vbpm_input_cache"


def input_cache_path(frontend_name: str, song, cache_root: str = INPUT_CACHE_DIR):
    """<cache>/<frontend>/<dataset>/<stem>/track.npy — Beat This's store layout."""
    return pathlib.Path(cache_root) / frontend_name / song.dataset / song.stem / "track.npy"


def warm_input_cache(songs, frontend, cache_root: str = INPUT_CACHE_DIR, verbose=True):
    """Compute and store every missing per-song model input; returns (done, failed).

    One-time per frontend (prepare_input is checkpoint-independent by contract). Audio
    loading matches features.compute_features: soundfile, mono mix. Failures are
    collected, not fatal — a corrupt file should cost one song, not the warm pass.
    """
    import soundfile

    if verbose:
        print(f"input cache: checking {len(songs)} songs for {frontend.name} "
              f"(first run computes, later runs skip)...", flush=True)
    done, failed = 0, []
    for song in songs:
        path = input_cache_path(frontend.name, song, cache_root)
        if path.exists():
            done += 1
            continue
        try:
            signal, sample_rate = soundfile.read(str(song.audio_path), dtype="float32")
            if signal.ndim > 1:
                signal = signal.mean(axis=1)
            array = frontend.prepare_input(signal, sample_rate)
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_save_npy(path, array)
            done += 1
        except Exception as error:                     # noqa: BLE001 — collected, reported
            failed.append((song.stem, repr(error)))
        if verbose and done % 200 == 0:
            print(f"  input cache: {done}/{len(songs)}", flush=True)
    if verbose and failed:
        print(f"  input cache FAILURES ({len(failed)}): {failed[:5]}...", flush=True)
    return done, failed


class ExcerptDataset(torch.utils.data.Dataset):
    """Per-epoch random windows of cached frontend input + framewise VAE targets.

    Holds NO reference to the frontend object (DataLoader workers pickle the dataset;
    a resident CUDA model must not ride along) — only its name and native frame rate,
    read off the instance at construction. Window draws use np.random (Beat This's
    convention), so DataLoader worker seeding governs reproducibility.

    Songs whose annotations can never yield a valid window (fewer than MIN_DOWNBEATS
    downbeats overall) are rejected at construction and counted in ``rejects``.
    """

    RETRIES = 8            # window draws per __getitem__ before conceding the middle

    def __init__(self, songs, frontend, excerpt_seconds: float = 45.0,
                 deterministic: bool = False, cache_root: str = INPUT_CACHE_DIR):
        self.frontend_name = frontend.name
        self.fps = float(frontend.FPS)
        self.excerpt_frames = int(round(excerpt_seconds * self.fps))
        self.deterministic = deterministic
        self.cache_root = cache_root

        self.items, self.rejects = [], []
        for song in songs:
            _beat_times, downbeat_times = song.beats()
            downbeat_times = np.asarray(downbeat_times, dtype=np.float64)
            path = input_cache_path(self.frontend_name, song, cache_root)
            if len(downbeat_times) < MIN_DOWNBEATS or not path.exists():
                self.rejects.append(song.stem)
                continue
            self.items.append((song, downbeat_times, path))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        song, downbeat_times, path = self.items[index]
        array = np.load(path, mmap_mode="r")
        total = len(array)
        frames = min(self.excerpt_frames, total)
        longer = total - frames

        # Fresh window per call; a draw whose bar period is undefined (too few downbeats
        # inside) is redrawn, and the middle window is the deterministic/backstop choice.
        start, targets = longer // 2, None
        if not self.deterministic:
            for _ in range(self.RETRIES):
                candidate = int(np.random.randint(0, longer + 1))
                targets = self._targets(downbeat_times, candidate, frames)
                if targets is not None:
                    start = candidate
                    break
        if targets is None:
            targets = self._targets(downbeat_times, start, frames)
        if targets is None:
            # Even the middle fails — emit a fully-masked item rather than crash a batch;
            # the mask zeroes its loss contribution.
            targets = {"delta": 0.0, "bar_period": 0.0,
                       "y": np.zeros(frames, dtype=np.float32),
                       "downbeat_times": np.empty(0), "anchors": np.empty(0)}
            valid = np.zeros(frames, dtype=np.float32)
        else:
            valid = np.ones(frames, dtype=np.float32)

        window = np.array(array[start:start + frames], dtype=np.float32)
        mask = valid
        pad = self.excerpt_frames - frames
        if pad > 0:                                            # song shorter than the window
            window = np.pad(window, [(0, pad)] + [(0, 0)] * (window.ndim - 1))
            targets["y"] = np.pad(targets["y"], (0, pad))
            mask = np.pad(mask, (0, pad))

        return {"input": window, "y": targets["y"], "mask": mask,
                "delta": np.float32(targets["delta"]),
                "bar_period": np.float32(targets["bar_period"]),
                "t0": np.float32(start / self.fps), "fps": np.float32(self.fps),
                "downbeat_times": targets["downbeat_times"],
                "anchors": targets["anchors"],
                "dataset": song.dataset, "stem": song.stem, "fold": song.fold}

    def _targets(self, downbeat_times, start: int, frames: int,
                 target_tol_frames: int = 1):
        """build_crop's target math on a [start, start+frames) window, or None.

        y is the ±1-frame-widened downbeat indicator; delta the constant per-frame
        phase advance 2π/(bar_period·fps); anchors BRACKET the window (one downbeat
        each side where available) so true_phase interpolation is defined at the edges.
        """
        lo_t, hi_t = start / self.fps, (start + frames) / self.fps
        period = bar_period(downbeat_times, lo_t, hi_t)
        if period is None or period <= 0:
            return None
        inside = downbeat_times[(downbeat_times >= lo_t) & (downbeat_times <= hi_t)]

        first = np.searchsorted(downbeat_times, lo_t, side="left")
        last = np.searchsorted(downbeat_times, hi_t, side="right")
        anchors = downbeat_times[max(0, first - 1):min(len(downbeat_times), last + 1)]

        y = np.zeros(frames, dtype=np.float32)
        for t in inside:
            centre = int(round(t * self.fps)) - start
            y[max(0, centre - target_tol_frames):centre + target_tol_frames + 1] = 1.0
        return {"delta": float(2.0 * np.pi / (period * self.fps)), "bar_period": period,
                "y": y, "downbeat_times": np.asarray(inside, dtype=np.float64),
                "anchors": np.asarray(anchors, dtype=np.float64)}


def materialize_crops(dataset: ExcerptDataset, frontend, batch_size: int = 16) -> list:
    """Deterministic excerpt items -> crop dicts for the EXISTING scorers.

    The bridge that lets the excerpt pipeline reuse every certified evaluation path
    (evaluate, preflight, Batches/collate) unchanged: features are computed ONCE
    through the frozen frontend, trimmed to the valid frames, with all scoring fields
    carried over. Val/test only -- training draws live windows.
    """
    assert dataset.deterministic, "materialize is for val/test; training draws live windows"
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         collate_fn=collate_excerpts)
    crops = []
    for batch in loader:
        h = frontend.forward_features(batch["input"]).cpu().numpy()
        for i in range(len(h)):
            valid = int(batch["mask"][i].sum())
            if valid == 0:                      # backstop item; nothing scorable in it
                continue
            crops.append({"h": np.asarray(h[i, :valid], dtype=np.float32),
                          "y": batch["y"][i, :valid].numpy(),
                          "delta": float(batch["delta"][i]),
                          "bar_period": float(batch["bar_period"][i]),
                          "t0": float(batch["t0"][i]), "fps": float(batch["fps"][i]),
                          "downbeat_times": batch["downbeat_times"][i],
                          "anchors": batch["anchors"][i],
                          "dataset": batch["dataset"][i], "stem": batch["stem"][i],
                          "fold": batch["fold"][i]})
    return crops


def collate_excerpts(batch: list) -> dict:
    """Fixed-size fields stack to tensors; variable/scoring fields stay python lists.

    The frontend consumes out["input"] via forward_features; the VAE consumes the
    stacked y/mask/delta exactly as the crop pipeline's collate provided them.
    """
    out = {}
    for key in ("input", "y", "mask"):
        out[key] = torch.from_numpy(np.stack([item[key] for item in batch]))
    for key in ("delta", "bar_period", "t0", "fps"):
        out[key] = torch.tensor([item[key] for item in batch])
    for key in ("downbeat_times", "anchors", "dataset", "stem", "fold"):
        out[key] = [item[key] for item in batch]
    return out
