"""The madmom bar-pointer DBN as a meter baseline on the same activations.

The question (user, 2026-08-01): the field's standard decoder has selected meter from
exactly these beat/downbeat activations for a decade — so how much meter does SEQUENTIAL
inference (joint phase-meter decoding, accumulated over the whole song) extract where our
crop-level summaries could not? Its column lands next to peaks (0.512) and the trained
head (0.595); if it wins on asap, the deficit there is an extraction problem that the
bar-pointer dynamics of Stages 1-2 should fix — if it also fails, the evidence itself is
missing and the thaw-the-frontend argument gains force.

The DBN invocation replicates Beat This's own postprocessor.py exactly (sigmoid,
epsilon-squeeze 1e-5, the Boeck combined activation, 55-215 BPM, transition lambda 100),
with ONE deviation: beats_per_bar=[2, 3, 4] instead of [3, 4], so the decoder can express
every class in our vocabulary.

Two readings per crop, both from one whole-song decode:
    dbn-crop    median beats-per-bar over the DECODED bars overlapping the crop window
    dbn-song    the song-level median, applied to all its crops (maximum accumulation)

Run: CUDA_VISIBLE_DEVICES=1 /disk4/anaconda3/envs/vbpm/bin/python \
         experiments/stage0_madmom_meter.py
"""
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from vbpm.data import FPS, VALUES, extract_crops, iter_frontend_features
from vbpm.fitting import score, score_per_dataset

EPSILON = 1e-5          # Beat This postprocessor.py's DBN bounds


def build_dbn():
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor
    return DBNDownBeatTrackingProcessor(
        beats_per_bar=list(VALUES),   # the one deviation from Beat This's [3, 4]
        min_bpm=55.0, max_bpm=215.0, fps=FPS, transition_lambda=100)


def combined_activation(h_logits):
    """Beat This postprocessor.py's exact Boeck-style multiclass activation."""
    prob = 1.0 / (1.0 + np.exp(-h_logits.astype(np.float64)))
    prob = prob * (1 - EPSILON) + EPSILON / 2
    beat_prob, downbeat_prob = prob[:, 0], prob[:, 1]
    return np.vstack((np.maximum(beat_prob - downbeat_prob, EPSILON / 2),
                      downbeat_prob)).T


def decoded_bars(dbn_out):
    """[(start_s, end_s, beat_count), ...] for each complete decoded bar."""
    times, positions = dbn_out[:, 0], dbn_out[:, 1]
    downbeat_idx = np.where(positions == 1)[0]
    bars = []
    for a, b in zip(downbeat_idx[:-1], downbeat_idx[1:]):
        bars.append((times[a], times[b], b - a))
    return bars


def nearest_value(count):
    return min(VALUES, key=lambda m: (abs(m - count), m))


def crop_meter(bars, start_s, end_s, song_m):
    """Median decoded bar length over bars overlapping [start_s, end_s); song fallback."""
    overlapping = [c for (a, b, c) in bars if b > start_s and a < end_s]
    if not overlapping:
        return song_m
    return nearest_value(float(np.median(overlapping)))


def main():
    smoke = "--smoke" in sys.argv
    print("frontend pass (whole-song activations kept for the DBN)...", flush=True)
    songs = []
    for s, h in iter_frontend_features(limit_per_fold=2 if smoke else None):
        crops, _ = extract_crops(*s.beats())
        if crops:
            songs.append({"act": combined_activation(h), "crops": crops,
                          "dataset": s.dataset, "fold": s.fold})

    print(f"decoding {len(songs)} songs with the bar-pointer DBN...", flush=True)
    dbn = build_dbn()

    def decode(song):
        return decoded_bars(dbn(song["act"]))

    with ThreadPoolExecutor(max_workers=8) as pool:
        all_bars = list(pool.map(decode, songs))

    entries, crop_preds, song_preds = [], [], []
    n_no_bars = 0
    for song, bars in zip(songs, all_bars):
        if bars:
            song_m = nearest_value(float(np.median([c for (_, _, c) in bars])))
        else:
            song_m, n_no_bars = max(VALUES), n_no_bars + 1
        for c in song["crops"]:
            start_s, end_s = c["bounds"][0], c["bounds"][-1]
            entries.append({"m_true": c["m_true"], "dataset": song["dataset"],
                            "fold": song["fold"]})
            crop_preds.append(crop_meter(bars, start_s, end_s, song_m))
            song_preds.append(song_m)
    print(f"crops: {len(entries)}  songs without decoded bars: {n_no_bars}")

    cv_sel = [i for i, e in enumerate(entries) if e["fold"] is not None]
    test_sel = [i for i, e in enumerate(entries) if e["fold"] is None]
    for tag, preds in (("dbn-crop", crop_preds), ("dbn-song", song_preds)):
        print(f"\n######## {tag} ########")
        score_per_dataset([entries[i] for i in cv_sel], [preds[i] for i in cv_sel], VALUES)
        if test_sel:
            score("gtzan", [entries[i] for i in test_sel],
                  [preds[i] for i in test_sel], VALUES)


if __name__ == "__main__":
    main()
