"""Do meter-changing songs RETURN to previously-visited meters, or move to novel ones?

Annotation-only measurement (no model, no audio). If switches overwhelmingly return
(verse 4/4 -> chorus 3/4 -> verse 4/4 ...), a one-variable state augmentation ("the
song's alternate meter") earns a place in the Stage-2 dynamics; if switch targets look
like fresh draws, the plain sticky chain keeps its seat.

Method: per song, bar length in beats for every complete bar (consecutive annotated
downbeats); collapse into regimes, dropping regimes shorter than MIN_REGIME_BARS bars
(single-bar blips are pickups/notation artifacts, not meter changes). A switch at
regime k >= 2 (the first switch cannot return by definition) counts as a RETURN if its
meter appeared in an earlier regime. Null for comparison: the probability the switch
target would land on a visited meter if drawn from the corpus meter distribution
(restricted to bar lengths 2..12, excluding the meter being left).

Run: /disk4/anaconda3/envs/vbpm/bin/python experiments/meter_return_memory.py
"""
import numpy as np

from data.songs import iter_songs
from vbpm.data import DOWNBEAT_TOL_S

MIN_REGIME_BARS = 2
PLAUSIBLE = set(range(2, 13))


def bar_lengths(beat_times, downbeat_times):
    """Beats in each complete bar [d_k, d_{k+1})."""
    return [int(np.sum((beat_times >= downbeat_times[k] - DOWNBEAT_TOL_S)
                       & (beat_times < downbeat_times[k + 1] - DOWNBEAT_TOL_S)))
            for k in range(len(downbeat_times) - 1)]


def regimes(lengths):
    """Run-length collapse, then drop runs shorter than MIN_REGIME_BARS and re-merge."""
    runs = []
    for length in lengths:
        if length not in PLAUSIBLE:
            continue
        if runs and runs[-1][0] == length:
            runs[-1][1] += 1
        else:
            runs.append([length, 1])
    kept = [meter for meter, n in runs if n >= MIN_REGIME_BARS]
    merged = [kept[0]] if kept else []
    for meter in kept[1:]:
        if meter != merged[-1]:
            merged.append(meter)
    return merged


def main():
    corpus_counts: dict = {}
    songs_regimes = []
    for s in iter_songs():
        beat_times, downbeat_times = s.beats()
        if len(downbeat_times) < 2:
            continue
        lengths = bar_lengths(beat_times, downbeat_times)
        for length in lengths:
            if length in PLAUSIBLE:
                corpus_counts[length] = corpus_counts.get(length, 0) + 1
        r = regimes(lengths)
        if len(r) >= 2:
            songs_regimes.append((s.dataset, r))

    total_bars = sum(corpus_counts.values())
    corpus_p = {m: n / total_bars for m, n in corpus_counts.items()}

    returns, novels, null_return_prob = 0, 0, []
    pattern_counts: dict = {}
    for dataset, r in songs_regimes:
        visited = {r[0], r[1]}
        for k in range(2, len(r)):
            if r[k] in visited:
                returns += 1
            else:
                novels += 1
            # null: draw the switch target from corpus meters, excluding the one left
            candidates = {m: p for m, p in corpus_p.items() if m != r[k - 1]}
            z = sum(candidates.values())
            null_return_prob.append(
                sum(p for m, p in candidates.items() if m in visited) / z)
            visited.add(r[k])
        kind = ("alternation" if len(set(r)) == 2 else
                "multi-meter" if len(set(r)) > 2 else "?")
        pattern_counts[kind] = pattern_counts.get(kind, 0) + 1

    n_switchers = len(songs_regimes)
    multi = returns + novels
    print(f"songs with >=1 real switch (regimes >=2, blips <{MIN_REGIME_BARS} bars dropped): "
          f"{n_switchers}")
    print(f"regime patterns: {pattern_counts}")
    print(f"eligible switches (2nd onward, where 'return' is possible): {multi}")
    if multi:
        rate = returns / multi
        null = float(np.mean(null_return_prob))
        print(f"RETURN rate: {returns}/{multi} = {rate:.3f}   "
              f"(null if targets were corpus draws: {null:.3f})")
    per_ds: dict = {}
    for dataset, r in songs_regimes:
        per_ds[dataset] = per_ds.get(dataset, 0) + 1
    print(f"switching songs per dataset: {dict(sorted(per_ds.items()))}")


if __name__ == "__main__":
    main()
