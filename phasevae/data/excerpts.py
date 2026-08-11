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

The cache is one flat file per song (<cache>/<frontend>/<dataset>/<song_id>.npy).
Beat This's directory-per-song store existed to hold precomputed pitch/tempo
augmentation variants beside the plain spectrogram; that augmentation was tried on
the SMC mission and did not help, so the folder level is gone with it.

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

from .dataset import MIN_DOWNBEATS
from .features import atomic_save_npy

INPUT_CACHE_DIR = "/disk4/jaehoon/vbpm_input_cache"


def input_cache_path(frontend_name: str, song, cache_root: str = INPUT_CACHE_DIR):
    """<cache>/<frontend>/<dataset>/<song_id>.npy — one flat file per song."""
    return pathlib.Path(cache_root) / frontend_name / song.dataset / f"{song.song_id}.npy"


def _compute_input(frontend, song):
    """Audio file -> the frontend's model input (soundfile, mono mix, their recipe)."""
    import soundfile

    signal, sample_rate = soundfile.read(str(song.audio_path), dtype="float32")
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    return frontend.prepare_input(signal, sample_rate)


class ExcerptDataset(torch.utils.data.Dataset):
    """Per-epoch random windows of cached frontend input + framewise VAE targets.

    Holds NO reference to the frontend object (DataLoader workers pickle the dataset;
    a resident CUDA model must not ride along) — only its name and native frame rate,
    read off the instance at construction. Window draws use np.random (Beat This's
    convention), so DataLoader worker seeding governs reproducibility.

    Songs whose annotations can never yield a valid window (fewer than MIN_DOWNBEATS
    downbeats overall) are rejected at construction and counted in ``rejects``.

    A song missing from the input cache is computed HERE — construction is the only
    moment the live frontend is in hand (prepare_input is checkpoint-independent by
    contract, so this happens once per frontend, ever). A song whose audio cannot be
    read joins ``rejects`` rather than killing the run.
    """

    def __init__(self, songs, frontend, excerpt_seconds: float = 45.0,
                 deterministic: bool = False, cache_root: str = INPUT_CACHE_DIR):
        self.frontend_name = frontend.name
        self.fps = float(frontend.FPS)
        self.excerpt_frames = int(round(excerpt_seconds * self.fps))
        self.deterministic = deterministic
        self.cache_root = cache_root

        self.items, self.rejects = [], []
        computed = 0
        for song in songs:
            _beat_times, downbeat_times = song.beats()
            downbeat_times = np.asarray(downbeat_times, dtype=np.float64)
            if len(downbeat_times) < MIN_DOWNBEATS:
                self.rejects.append(song.song_id)
                continue
            path = input_cache_path(self.frontend_name, song, cache_root)
            if not path.exists():
                try:
                    array = _compute_input(frontend, song)
                except Exception as error:  # noqa: BLE001 — one bad file costs one song
                    print(f"  input FAILED for {song.song_id}: {error!r}", flush=True)
                    self.rejects.append(song.song_id)
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                atomic_save_npy(path, array)
                computed += 1
                if computed % 200 == 0:
                    print(f"  input cache: {computed} computed...", flush=True)
            self.items.append((song, downbeat_times, path))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        song, downbeat_times, path = self.items[index]
        array = np.load(path, mmap_mode="r")
        total = len(array)
        frames = min(self.excerpt_frames, total)
        longer = total - frames

        # Fresh random window per call (Beat This's policy); val/test take the middle
        # so every scored window is identical across runs.
        start = longer // 2 if self.deterministic else int(np.random.randint(0, longer + 1))
        targets = self._targets(downbeat_times, start, frames, target_tol_frames=3)

        window = np.array(array[start:start + frames], dtype=np.float32)
        labeled = len(targets["downbeat_times"]) > 0
        mask = np.full(frames, float(labeled), dtype=np.float32)

        pad = self.excerpt_frames - frames
        if pad > 0:                                            # song shorter than the window
            window = np.pad(window, [(0, pad)] + [(0, 0)] * (window.ndim - 1))
            targets["y"] = np.pad(targets["y"], (0, pad))
            mask = np.pad(mask, (0, pad))

        return {"input": window, "y": targets["y"], "mask": mask,
                "t0": np.float32(start / self.fps), "fps": np.float32(self.fps),
                "downbeat_times": targets["downbeat_times"],
                "anchors": targets["anchors"],
                "dataset": song.dataset, "song_id": song.song_id}

    def _targets(self, downbeat_times, start: int, frames: int,
                 target_tol_frames: int = 1):
        """build_crop's target math on a [start, start+frames) window, or None.

        y is the ±1-frame-widened downbeat indicator; delta the constant per-frame
        phase advance 2π/(bar_period·fps); anchors BRACKET the window (one downbeat
        each side where available) so true_phase interpolation is defined at the edges.
        """
        lo_t, hi_t = start / self.fps, (start + frames) / self.fps

        inside = downbeat_times[(downbeat_times >= lo_t) & (downbeat_times <= hi_t)]

        first = np.searchsorted(downbeat_times, lo_t, side="left")
        last = np.searchsorted(downbeat_times, hi_t, side="right")
        anchors = downbeat_times[max(0, first - 1):min(len(downbeat_times), last + 1)]

        y = np.zeros(frames, dtype=np.float32)
        for t in inside:
            centre = int(round(t * self.fps)) - start
            y[max(0, centre - target_tol_frames):centre + target_tol_frames + 1] = 1.0
        return {"y": y, "downbeat_times": np.asarray(inside, dtype=np.float64),
                "anchors": np.asarray(anchors, dtype=np.float64)}


def collate_excerpts(batch: list) -> dict:
    """Fixed-size fields stack to tensors; variable/scoring fields stay python lists.

    The frontend consumes out["input"] via forward_features; the VAE consumes the
    stacked y/mask/delta exactly as the crop pipeline's collate provided them.

    ``targets``/``valid`` are the ragged downbeat_times PADDED into [B, K] so a training
    objective can consume them -- see _downbeat_targets.
    """
    out = {}
    for key in ("input", "y", "mask"):
        out[key] = torch.from_numpy(np.stack([item[key] for item in batch]))
    for key in ("t0", "fps"):
        out[key] = torch.tensor([item[key] for item in batch])
    for key in ("downbeat_times", "anchors", "dataset", "song_id"):
        out[key] = [item[key] for item in batch]
    out["targets"], out["valid"] = _downbeat_targets(batch)
    return out


def _downbeat_targets(batch: list):
    """[B, K] annotated downbeat times as window-relative FRAME indices, and their mask.

    THE conversion an alignment objective needs: ``(downbeat_times - t0) * fps``. Two
    traps it exists to close, both of which have already cost this project a day.

      * downbeat_times are ABSOLUTE seconds while frame indices are window-relative.
        Omitting the ``- t0`` inverts every landscape built on top of it.
      * the events must come from downbeat_times, NEVER from nonzero(y > 0.5): y is the
        +-target_tol_frames WIDENED indicator, so that route yields 3 frames per downbeat
        (7 at tol=3) and inflates any derived rate by exactly that factor.

    Ragged by nature -- songs differ in downbeat count -- so it pads to the batch max and
    returns a validity mask. Padding is 0.0, which is a LEGAL frame index, hence the mask:
    a consumer that ignores it silently trains on phantom downbeats at frame 0.
    """
    times = [(np.asarray(item["downbeat_times"], dtype=np.float32)
              - float(item["t0"])) * float(item["fps"]) for item in batch]
    width = max((len(t) for t in times), default=0)
    targets = torch.zeros(len(batch), width, dtype=torch.float32)
    valid = torch.zeros(len(batch), width, dtype=torch.float32)
    for i, t in enumerate(times):
        targets[i, :len(t)] = torch.from_numpy(t)
        valid[i, :len(t)] = 1.0
    return targets, valid
