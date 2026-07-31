"""§6 data path: real annotations + frontend -> training CROPS {h, y, m_true} for Stage 0.

y and m_true per §6.2, from the trusted beat/downbeat annotations via data/songs.py.
h per §6.1: live Beat This activations ([T,2] logits at 50 fps), computed through
frontends/ (no caches — user decision 2026-07-15), fold-honestly: each song's h comes
from the checkpoint that held that song out (final0 for gtzan, which is test-only).

Rejections are surfaced, never silent (§6.2): every excluded song is counted by reason.
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.songs import iter_songs  # noqa: E402

MIN_BARS = 3
DOWNBEAT_TOL_S = 0.02
VALUES = (2, 3, 4)
MIN_BEATS = 12          # §5: n >= MIN_BARS bars at the largest legal m
CROP_BARS = 8           # complete bars per crop: n = 16/24/32 beats at m = 2/3/4, all >= 12.
                        # One m per CROP (§4.1, forced by dropping phi) — a song is NOT one
                        # crop: a whole-song median m_true fabricates a label on the 4.1% of
                        # songs whose meter changes (§5, §10.9), concentrated in asap.


def derive_y(beat_times, downbeat_times, tol_s: float = DOWNBEAT_TOL_S):
    """§6.2: y_i = 1 iff beat i carries a downbeat within tol. Returns (y, n_unmatched).

    A downbeat matching no beat is a data error to surface, not a beat to insert.
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
    """§6.2: median over complete bars of beats in [d_k, d_{k+1}), half-up; None if < min_bars."""
    n_bars = len(downbeat_times) - 1
    if n_bars < min_bars:
        return None
    counts = [int(np.sum((beat_times >= downbeat_times[k] - tol_s)
                         & (beat_times < downbeat_times[k + 1] - tol_s)))
              for k in range(n_bars)]
    return int(math.floor(float(np.median(counts)) + 0.5))


def make_crops(beat_times, downbeat_times, crop_bars: int = CROP_BARS):
    """Bar-aligned crops: consecutive complete bars in chunks of ``crop_bars``.

    Yields (crop_beat_times, bar_bounds) with bar_bounds = the B+1 downbeat times delimiting
    the crop's complete bars (so §6.2's median runs over exactly those bars). Beats before
    the first downbeat and after the last complete bar are outside every crop (incomplete
    bars carry no §6.2 label). A tail shorter than MIN_BARS bars is dropped — a crop the
    spec rejects.
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


def load_crops(datasets=None, device: str = "cuda", limit_per_fold=None,
               values=VALUES, verbose: bool = True):
    """(crops, report). Each entry is one CROP: {h, y, m_true, dataset, fold, stem, crop}.

    h is sliced to the frames spanned by the crop's beats (§5: "T tracks n").
    """
    import soundfile
    from frontends.beat_this import BeatThisFrontend

    catalog = iter_songs(datasets=datasets)
    by_fold: dict = {}
    for s in catalog:
        by_fold.setdefault(s.fold, []).append(s)

    crops, rejects = [], Counter()
    total_unmatched = 0
    for fold, members in sorted(by_fold.items(), key=lambda kv: (kv[0] is None, kv[0])):
        checkpoint = "final0" if fold is None else f"fold{fold}"
        frontend = BeatThisFrontend(checkpoint=checkpoint, device=device)
        fps = frontend.fps
        if limit_per_fold is not None:
            members = members[:limit_per_fold]
        for s in members:
            beat_times, downbeat_times = s.beats()
            if len(downbeat_times) == 0:
                rejects["no_downbeat_annotation"] += 1
                continue
            if len(downbeat_times) - 1 < MIN_BARS:
                rejects[f"fewer_than_{MIN_BARS}_bars"] += 1
                continue
            song_crops = make_crops(beat_times, downbeat_times)
            if not song_crops:
                rejects["no_usable_crops"] += 1
                continue
            signal, sample_rate = soundfile.read(str(s.audio_path), dtype="float32")
            if signal.ndim > 1:
                signal = signal.mean(axis=1)
            h = frontend.get_features(signal, sample_rate).numpy()      # [T, 2] logits, 50 fps
            for crop_index, (crop_beats, bar_bounds) in enumerate(song_crops):
                # label the CROP, not the song (§4.1/§5): median over ITS complete bars
                m_true = derive_m_true(crop_beats, bar_bounds)
                if m_true is None:
                    rejects["crop_fewer_bars_than_min"] += 1
                    continue
                if m_true not in values:
                    rejects[f"crop_m_out_of_vocabulary({m_true})"] += 1
                    continue
                if len(crop_beats) < MIN_BEATS:
                    rejects["crop_fewer_than_12_beats"] += 1
                    continue
                # y against the crop's bar STARTS (bounds[:-1]); the closing bound is the
                # next crop's first downbeat, not a downbeat of this crop
                y, unmatched = derive_y(crop_beats, bar_bounds[:-1])
                total_unmatched += unmatched
                lo = max(0, int(math.floor(crop_beats[0] * fps)))
                hi = min(len(h), int(math.ceil(crop_beats[-1] * fps)) + 1)
                crops.append({"h": h[lo:hi], "y": y, "m_true": m_true,
                              "dataset": s.dataset, "fold": s.fold,
                              "stem": s.stem, "crop": crop_index})
        del frontend
        if verbose:
            print(f"  {checkpoint}: {sum(1 for c in crops if c['fold'] == fold)} crops loaded",
                  flush=True)

    report = {"usable": len(crops), "rejects": dict(rejects),
              "unmatched_downbeats": total_unmatched,
              "per_dataset": dict(Counter((c["dataset"], c["m_true"]) for c in crops))}
    return crops, report
