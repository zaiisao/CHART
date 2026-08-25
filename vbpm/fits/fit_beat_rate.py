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
        r = np.exp(logp)
        r /= r.sum(1, keepdims=True)
        w = r.mean(0)
        s = np.sqrt((r * x[:, None] ** 2).sum(0) / r.sum(0).clip(min=1e-9)) + 1e-12
    return w, s


def collect(scale, level):
    incs, log_rates, at_bar = [], [], []
    for _ds, songs in load_catalog(TRAIN).items():
        for song in songs:
            try:
                bt, db = song.beats()
            except Exception:
                continue
            t = bt if level == "beat" else db
            if len(t) < 5 or len(db) < 4:
                continue
            d = np.diff(t)
            lo, hi = (0.1, 3.0) if level == "beat" else (0.5, 8.0)
            good = (d > lo) & (d < hi)
            if good.sum() < 4:
                continue
            mids = (t[:-1] + t[1:]) / 2.0
            lr = np.log(2 * np.pi / (d * FPS))
            log_rates.append(lr[good])
            grid = np.arange(t[0], t[-1], scale / FPS)
            if len(grid) < 3:
                continue
            incs.append(np.diff(np.interp(grid, mids, lr)))
            bar_lr = np.interp(db, mids, lr)
            at_bar.append(np.abs(np.diff(bar_lr)))
    return (np.concatenate(incs), np.concatenate(log_rates),
            np.concatenate(at_bar))


for level in ("bar", "beat"):
    x, lr, bar_step = collect(1, level)
    x = x[np.abs(x) < 0.5]
    w, s = em_mix2(x)
    order = np.argsort(s)
    w, s = w[order], s[order]
    rate = np.exp(lr)
    print(f"\n=== {level.upper()} rate   n = {len(lr):,} intervals, {len(x):,} increments")
    print(f"  prior      mu {np.mean(lr):+.4f}   sigma {np.std(lr):.4f}")
    print(f"  rate range p1 {np.percentile(rate, 1):.4f}   p99 {np.percentile(rate, 99):.4f}"
          f"   (span {np.percentile(rate,99)/np.percentile(rate,1):.1f}x)")
    print(f"  walk       sigma {np.std(x):.5f}   mix w ({w[0]:.3f}, {w[1]:.3f})"
          f"  sigma ({s[0]:.5f}, {s[1]:.5f})")
    print(f"  GAMMA      median |dlog rate| across a bar line {np.median(bar_step):.4f}")
