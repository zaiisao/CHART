"""Crops in, padded GPU batches out. Fold-honest features, one-time collation."""
from __future__ import annotations

import pathlib
import pickle
from collections import Counter

import numpy as np
import torch

from vbpm.data import iter_frontend_features

from .crops import song_crops

MAX_CROPS = 3         # per song; a SAMPLING cap, not a filter
VAL_FOLD = 7


def load_dataset(seed: int = 0, limit_per_fold=None, verbose: bool = True):
    """(crops, rejects) for every song whose fold-honest features are available."""
    rng = np.random.default_rng(seed)
    crops, rejects = [], Counter()
    for song, features in iter_frontend_features(limit_per_fold=limit_per_fold,
                                                 output="features+activations",
                                                 verbose=verbose):
        got, rej = song_crops(features, song, rng, MAX_CROPS)
        for crop in got:
            crop["fold"] = song.fold
        crops += got
        rejects.update(rej)
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


def load_gtzan_through(checkpoint: str):
    """Gtzan test crops through a FOLD checkpoint instead of the cache's final0.

    Any fold checkpoint is fold-honest for gtzan (none trained on it). final0 is an
    activation space the encoder never sees in training, which alone cost the supervised
    probe 0.75 -> 0.06.
    """
    rng = np.random.default_rng(0)
    crops, rejects = [], Counter()
    for song, features in iter_frontend_features(datasets=["gtzan"],
                                                 output="features+activations",
                                                 override_checkpoint=checkpoint):
        got, rej = song_crops(features, song, rng, MAX_CROPS)
        for crop in got:
            crop["fold"] = song.fold
        crops += got
        rejects.update(rej)
    return crops, rejects


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
