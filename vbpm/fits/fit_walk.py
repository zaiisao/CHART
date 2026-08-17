import numpy as np

from vbpm.data.dataset import load_catalog

FPS = 50.0
TRAIN = ["asap", "ballroom", "beatles", "candombe", "groove_midi", "guitarset",
         "hainsworth", "hjdb", "rwc", "simac", "tapcorrect"]


def em_mix2(x, iters=200):
    w = np.array([0.6, 0.4])
    s = np.array([np.std(x) * 0.3, np.std(x) * 2.0]) + 1e-12
    for _ in range(iters):
        logp = -0.5 * (x[:, None] / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi) + np.log(w)
        logp -= logp.max(1, keepdims=True)
        r = np.exp(logp); r /= r.sum(1, keepdims=True)
        w = r.mean(0)
        s = np.sqrt((r * x[:, None] ** 2).sum(0) / r.sum(0).clip(min=1e-9)) + 1e-12
    return w, s


def collect(scale):
    incs = []
    log_rates = []
    for ds, songs in load_catalog(TRAIN).items():
        for song in songs:
            try:
                _, db = song.beats()
            except Exception:
                continue
            if len(db) < 5:
                continue
            d = np.diff(db)
            good = (d > 0.5) & (d < 8.0)
            if good.sum() < 4:
                continue
            mids = (db[:-1] + db[1:]) / 2.0
            lr = np.log(2 * np.pi / (d * FPS))
            log_rates.append(lr[good])
            t0, t1 = db[0], db[-1]
            grid = np.arange(t0, t1, scale / FPS)
            if len(grid) < 3:
                continue
            lr_t = np.interp(grid, mids, lr)
            incs.append(np.diff(lr_t))
    return np.concatenate(incs), np.concatenate(log_rates)


for scale, name in ((1, "per-frame"), (25, "per-knot (stride 25)")):
    x, lr = collect(scale)
    x = x[np.abs(x) < 0.5]
    sg = float(np.std(x))
    b = float(np.mean(np.abs(x)))
    w, s = em_mix2(x)
    order = np.argsort(s)
    w, s = w[order], s[order]
    p5 = float(np.mean(np.abs(x) > 5 * 0.00104 * (scale ** 0.5)))
    print(f"\n=== {name}: n = {len(x):,} increments from {len(TRAIN)} corpora")
    print(f"  single Gaussian sigma {sg:.5f}   mean|d| {b:.5f}   Laplace b {b:.5f}")
    print(f"  2-Gauss mixture: w = ({w[0]:.3f}, {w[1]:.3f})  sigma = ({s[0]:.5f}, {s[1]:.5f})"
          f"   ratio {s[1]/s[0]:.1f}x")
    print(f"  P(|d| > 5*gauss-walk-sd at this scale) = {p5:.4f}"
          f"   (Gaussian predicts {2 * (1 - 0.9999997):.7f})")
    if scale == 1:
        print("  current shipped:  sigma_w 0.00104 (single), hand mix (0.95, 0.05) x (0.00104, 0.0104)")

print("\n=== initial-rate prior from bar periods")
_, lr = collect(1)
print(f"  measured: mu {float(np.mean(lr)):.4f}  sigma {float(np.std(lr)):.4f}")
import math
print(f"  shipped:  mu {math.log(2*math.pi/(1.91*50.0)):.4f}  sigma 0.370")
