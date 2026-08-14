import numpy as np
from vbpm.data.dataset import load_catalog

TRAIN = ["asap", "ballroom", "beatles", "candombe", "groove_midi", "guitarset",
         "hainsworth", "hjdb", "rwc", "simac", "tapcorrect"]
TWO_PI = 2 * np.pi

intra, inter = [], []
for ds, songs in load_catalog(TRAIN).items():
    for song in songs:
        try:
            beats, db = song.beats()
        except Exception:
            continue
        if len(db) < 5:
            continue
        d = np.diff(db)
        good = (d > 0.5) & (d < 8.0)
        for k in np.where(good)[0]:
            t0, t1 = db[k], db[k + 1]
            inb = beats[(beats > t0) & (beats < t1)]
            if len(inb) >= 2:
                frac = (inb - t0) / (t1 - t0)
                ideal = np.round(frac * (len(inb) + 1)) / (len(inb) + 1)
                intra.extend(TWO_PI * (frac - ideal))
        dg = d[good]
        inter.extend(TWO_PI * (dg[1:] - dg[:-1]) / dg[:-1])

intra = np.array(intra); intra = intra[np.abs(intra) < 1.0]
inter = np.array(inter); inter = inter[np.abs(inter) < 4.0]

def report(name, x):
    s_all = float(np.std(x))
    core = x[np.abs(x) < 3 * np.median(np.abs(x)) / 0.6745]
    s_core = float(np.std(core))
    print(f"{name}: n={len(x):,}  sd {s_all:.4f} rad (core {s_core:.4f})"
          f"  -> kappa ~ {1/s_core**2:,.0f}  ({s_core/TWO_PI*2454:.0f} ms on a 2.45s bar)")

print("phase innovation, measured from 11 training corpora:")
report("INTRA-bar (beat micro-timing)   ", intra)
report("INTER-bar (crossing innovation) ", inter)
print(f"\nshipped single kappa_physical: 100,000 (sd {1/np.sqrt(1e5):.4f} rad)"
      f"   the old 'honest' value: 383 (sd {1/np.sqrt(383):.4f} rad)")
