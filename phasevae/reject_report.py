"""Crop accounting from annotations alone: no features, no GPU, seconds to run.

    PYTHONPATH=. python -m phasevae.reject_report

Separates the two exclusions that a single counter used to merge:
  * the CORPUS RESTRICTION (this song's meter is not m) -- one count per song,
  * the METER-VARIATION exclusion (this window has no single r_true) -- per window,
    reported against its own denominator, and only over songs already in m.
It also prints how many crops the song-level early return costs, because that early
return is a behaviour change and not only a relabelling: a song whose MEDIAN bar length
is not m can still contain windows that are internally consistent with m, and those
windows were kept before the split and are dropped after it.
"""
from __future__ import annotations

from collections import Counter

from data.songs import iter_songs
from vbpm.data import derive_y

from .crops import CROP_BARS, crop_starts, song_bar_length

M = 4


def main() -> None:
    """Print the split reject table and the cost of the song-level early return."""
    rejects: Counter = Counter()
    kept_after, kept_before = 0, 0
    songs_before_only = 0
    for song in iter_songs():
        beat_times, downbeat_times = song.beats()
        if len(downbeat_times) == 0 or len(beat_times) < CROP_BARS * M + 1:
            rejects["song_too_short_or_unannotated"] += 1
            continue
        y_song, unmatched = derive_y(beat_times, downbeat_times)
        rejects["unmatched_downbeats"] += unmatched
        bar_length = song_bar_length(y_song)
        starts, window_rejects = crop_starts(y_song, M, CROP_BARS)
        kept_before += len(starts)
        if bar_length is None:
            rejects["song_fewer_than_two_downbeats"] += 1
            continue
        if bar_length != M:
            rejects[f"song_meter_is_not_m(={bar_length})"] += 1
            songs_before_only += len(starts)
            continue
        rejects["songs_in_m"] += 1
        rejects.update(window_rejects)
        kept_after += len(starts)

    print("reject accounting (annotations only, m = %d, %d bars per crop)"
          % (M, CROP_BARS))
    for reason in sorted(rejects):
        print(f"  {reason:38s} {rejects[reason]:8d}")
    windows = rejects["candidate_windows"]
    varies = rejects["meter_varies_within_window"]
    print(f"\nwithin-song meter variation, over songs already in m: "
          f"{varies}/{windows} = {varies / max(windows, 1):.3%} of candidate windows")
    print(f"valid starts kept: {kept_after} (song-level early return costs "
          f"{songs_before_only} starts on {kept_before - kept_after == songs_before_only} "
          f"mixed-meter songs, i.e. {songs_before_only / max(kept_before, 1):.3%} "
          f"of what the pre-split rule kept)")


if __name__ == "__main__":
    main()
