import numpy as np
from vbpm.data.dataset import load_catalog

TRAIN = ["asap", "ballroom", "beatles", "candombe", "groove_midi", "guitarset",
         "hainsworth", "hjdb", "rwc", "simac", "tapcorrect"]


def em_mix2(x, iters=300):
    w = np.array([0.6, 0.4]); s = np.array([np.std(x)*0.3, np.std(x)*2.0]) + 1e-12
    for _ in range(iters):
        lp = -0.5*(x[:,None]/s)**2 - np.log(s) + np.log(w)
        lp -= lp.max(1, keepdims=True)
        r = np.exp(lp); r /= r.sum(1, keepdims=True)
        w = r.mean(0)
        s = np.sqrt((r*x[:,None]**2).sum(0)/r.sum(0).clip(min=1e-9)) + 1e-12
    return w, s


xs = []
for ds, songs in load_catalog(TRAIN).items():
    for song in songs:
        try:
            _, db = song.beats()
        except Exception:
            continue
        if len(db) < 5: continue
        d = np.diff(db)
        good = (d > 0.5) & (d < 8.0)
        d = d[good]
        if len(d) < 4: continue
        xs.append(np.log(d[1:] / d[:-1]))
x = np.concatenate(xs)
x = x[np.abs(x) < 0.7]
w, s = em_mix2(x)
o = np.argsort(s); w, s = w[o], s[o]
print(f"bar-to-bar log-ratio: n = {len(x):,}")
print(f"  mixture w = ({w[0]:.3f}, {w[1]:.3f})  sigma = ({s[0]:.5f}, {s[1]:.5f})")
print(f"  core sigma {s[0]:.4f}  -> delta = 2*core = {2*s[0]:.4f}   (shipped default 0.05)")
