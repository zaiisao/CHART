"""Excerpt dataset: window draws, target math, padding, collate — frontend-free.

A stub frontend (name + FPS only) and a synthetic input cache exercise every
path that does not need a GPU model; the real frontends' forward paths are covered by
their own __main__ smoke checks.
"""
import numpy as np
import pytest
import torch

from phasevae.data.excerpts import (ExcerptDataset, collate_excerpts,
                                    input_cache_path, warm_input_cache)


class _StubFrontend:
    FPS = 50.0

    @property
    def name(self):
        return "stub"

    def prepare_input(self, signal, sample_rate):
        return np.zeros((int(len(signal) / sample_rate * self.FPS), 4),
                        dtype=np.float32)


class _Song:
    dataset = "toy"

    def __init__(self, stem, downbeats, fold=0):
        self.stem, self.fold = stem, fold
        self._downbeats = np.asarray(downbeats, dtype=np.float64)

    def beats(self):
        return self._downbeats, self._downbeats


def _make_cache(tmp_path, song, seconds=70.0, fps=50.0, width=4):
    path = input_cache_path("stub", song, str(tmp_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * fps)
    array = np.arange(frames, dtype=np.float32)[:, None].repeat(width, 1)
    np.save(path, array)
    return array


def test_window_is_a_contiguous_slice(tmp_path):
    song = _Song("long", np.arange(0.0, 70.0, 2.0))
    array = _make_cache(tmp_path, song)
    ds = ExcerptDataset([song], _StubFrontend(), cache_root=str(tmp_path), excerpt_seconds=45.0)
    item = ds[0]
    frames = ds.excerpt_frames
    assert item["input"].shape == (frames, 4)
    start = int(round(float(item["t0"]) * ds.fps))
    np.testing.assert_array_equal(item["input"], array[start:start + frames])
    assert item["mask"].all() and item["y"].shape == (frames,)


def test_deterministic_takes_the_middle(tmp_path):
    song = _Song("long", np.arange(0.0, 70.0, 2.0))
    _make_cache(tmp_path, song)
    ds = ExcerptDataset([song], _StubFrontend(), cache_root=str(tmp_path),
                        excerpt_seconds=45.0, deterministic=True)
    total, frames = int(70.0 * 50), ds.excerpt_frames
    expected_start = (total - frames) // 2
    assert int(round(float(ds[0]["t0"]) * ds.fps)) == expected_start
    assert int(round(float(ds[0]["t0"]) * ds.fps)) == expected_start   # stable


def test_fresh_windows_across_calls(tmp_path):
    song = _Song("long", np.arange(0.0, 70.0, 2.0))
    _make_cache(tmp_path, song)
    ds = ExcerptDataset([song], _StubFrontend(), cache_root=str(tmp_path), excerpt_seconds=45.0)
    starts = {float(ds[0]["t0"]) for _ in range(20)}
    assert len(starts) > 1, "non-deterministic dataset must draw fresh windows"


def test_short_song_padded_and_masked(tmp_path):
    song = _Song("short", np.arange(0.0, 30.0, 2.0))
    _make_cache(tmp_path, song, seconds=30.0)
    ds = ExcerptDataset([song], _StubFrontend(), cache_root=str(tmp_path), excerpt_seconds=45.0)
    item = ds[0]
    frames, real = ds.excerpt_frames, int(30.0 * 50)
    assert item["input"].shape == (frames, 4)
    assert item["mask"][:real].all() and not item["mask"][real:].any()
    assert not item["y"][real:].any()
    assert np.all(item["input"][real:] == 0)


def test_targets_match_annotations(tmp_path):
    period = 2.0
    song = _Song("long", np.arange(0.0, 70.0, period))
    _make_cache(tmp_path, song)
    ds = ExcerptDataset([song], _StubFrontend(), cache_root=str(tmp_path),
                        excerpt_seconds=45.0, deterministic=True)
    item = ds[0]
    assert float(item["bar_period"]) == pytest.approx(period)
    assert float(item["delta"]) == pytest.approx(2 * np.pi / (period * 50.0))
    start = int(round(float(item["t0"]) * ds.fps))
    for t in item["downbeat_times"]:
        centre = int(round(t * ds.fps)) - start
        assert item["y"][centre] == 1.0
    # anchors bracket the window: one downbeat at or before t0, one at or after the end
    t0, t1 = float(item["t0"]), float(item["t0"]) + 45.0
    assert item["anchors"][0] <= t0 + period and item["anchors"][-1] + period >= t1


def test_rejects_unannotated_and_uncached(tmp_path):
    cached_ok = _Song("ok", np.arange(0.0, 70.0, 2.0))
    too_few = _Song("sparse", [1.0, 3.0])
    uncached = _Song("missing", np.arange(0.0, 70.0, 2.0))
    _make_cache(tmp_path, cached_ok)
    _make_cache(tmp_path, too_few)
    ds = ExcerptDataset([cached_ok, too_few, uncached], _StubFrontend(), cache_root=str(tmp_path))
    assert len(ds) == 1
    assert sorted(ds.rejects) == ["missing", "sparse"]


def test_collate_stacks_and_lists(tmp_path):
    songs = [_Song(f"s{i}", np.arange(0.0, 70.0, 2.0)) for i in range(3)]
    for song in songs:
        _make_cache(tmp_path, song)
    ds = ExcerptDataset(songs, _StubFrontend(), cache_root=str(tmp_path), excerpt_seconds=45.0)
    batch = collate_excerpts([ds[i] for i in range(3)])
    frames = ds.excerpt_frames
    assert batch["input"].shape == (3, frames, 4)
    assert batch["y"].shape == (3, frames) and batch["mask"].shape == (3, frames)
    assert batch["delta"].shape == (3,) and torch.is_tensor(batch["delta"])
    assert isinstance(batch["downbeat_times"], list) and len(batch["stem"]) == 3


def test_warm_input_cache_writes_and_skips(tmp_path):
    frontend = _StubFrontend()
    song = _Song("tone", np.arange(0.0, 10.0, 2.0))
    import soundfile
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    song.audio_path = audio_dir / "tone.wav"
    soundfile.write(str(song.audio_path), np.zeros(22050 * 2, dtype=np.float32), 22050)

    done, failed = warm_input_cache([song], frontend, str(tmp_path), verbose=False)
    assert (done, failed) == (1, [])
    path = input_cache_path("stub", song, str(tmp_path))
    assert path.exists() and np.load(path).shape == (100, 4)
    done, failed = warm_input_cache([song], frontend, str(tmp_path), verbose=False)
    assert (done, failed) == (1, [])                     # second pass is a no-op

    broken = _Song("broken", np.arange(0.0, 10.0, 2.0))
    broken.audio_path = audio_dir / "nope.wav"
    done, failed = warm_input_cache([broken], frontend, str(tmp_path), verbose=False)
    assert done == 0 and len(failed) == 1
