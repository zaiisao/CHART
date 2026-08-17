import numpy as np
from vbpm.data.dataset import load_catalog

TRAIN = ["asap", "ballroom", "beatles", "candombe", "groove_midi", "guitarset",
         "hainsworth", "hjdb", "rwc", "simac", "tapcorrect"]

x = []
for ds, songs in load_catalog(TRAIN).items():
    for song in songs:
        try:
            beats, db = song.beats()
        except Exception:
            continue
        if len(db) < 5:
            continue
        for k in range(len(db) - 1):
            t0, t1 = db[k], db[k + 1]
            if not (0.5 < t1 - t0 < 8.0):
                continue
            inb = beats[(beats >= t0) & (beats <= t1)]
            d = np.diff(inb)
            d = d[(d > 0.1) & (d < 4.0)]
            if len(d) >= 2:
                x.extend(np.log(d[1:] / d[:-1]))
x = np.array(x); x = x[np.abs(x) < 0.7]
core = x[np.abs(x) < 3 * np.median(np.abs(x)) / 0.6745]
sb = float(np.std(core))
frames_per_beat = 61.0
sf = sb / np.sqrt(frames_per_beat)
print(f"WITHIN-bar beat-to-beat tempo log-ratio: n={len(x):,}  core sd {sb:.4f}")
print(f"  -> per-frame intra-bar walk sd ~ {sf:.5f}  (shipped mixture core was 0.00029)")
print("  inter (bar-to-bar, already measured): mix (0.646,0.354) x (0.0247,0.198) per crossing")
