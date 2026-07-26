"""Shared data / Dirac / oracle helpers for the minimal-model ladder probes."""
import sys, glob, math
import numpy as np, torch
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")

CACHE = "/disk1/jaehoon/vbpm_mert_cache"
FPS = 50.0
TWO_PI = 2.0 * math.pi
H_DIM = 8


def load(split, cap=None):
    out = []
    for f in sorted(glob.glob(f"{CACHE}/{split}__*.npz")):
        d = np.load(f, allow_pickle=True)
        out.append(dict(key=f.split("/")[-1][:-4],
                        T=int(d["feats"].shape[1]),
                        beats=np.asarray(d["beats"], float),
                        downs=np.asarray(d["downs"], float)))
    return out[:cap] if cap else out


def est_meter(beats, downs):
    if len(downs) >= 2:
        bpb = np.median([np.sum((beats >= downs[i]) & (beats < downs[i + 1]))
                         for i in range(len(downs) - 1)])
        if bpb > 0:
            return max(2, min(int(round(bpb)), 4))
    return 4


def oracle_const_phase(song, T, start=0):
    """Best CONSTANT-tempo bar-pointer for this song: LS fit of unwrapped bar phase
    (2*pi*k at the k-th downbeat) as a*t + c.  Returns (phase[T], phidot_rad_per_frame, phi0).
    This is the exact functional form the deployed free_run mean-chain can express."""
    downs = song["downs"]
    if len(downs) >= 2:
        k = np.arange(len(downs), dtype=float)
        A = np.stack([downs, np.ones_like(downs)], 1)
        coef, *_ = np.linalg.lstsq(A, TWO_PI * k, rcond=None)
        a, c = float(coef[0]), float(coef[1])           # rad/sec, rad
    else:                                               # fall back on beat grid
        m = est_meter(song["beats"], downs)
        ibi = float(np.median(np.diff(song["beats"]))) if len(song["beats"]) > 2 else 0.5
        a = TWO_PI / (ibi * m); c = 0.0
    t = (np.arange(start, start + T) + 0.5) / FPS
    ph = (a * t + c) % TWO_PI
    return ph.astype(np.float32), a / FPS, float(ph[0])


def oracle_pw_phase(song, T, start=0):
    """Ideal PIECEWISE-linear bar phase (0 at each downbeat -> 2pi at the next)."""
    downs = song["downs"]; beats = song["beats"]
    anchors = downs if len(downs) >= 2 else (beats[::4] if len(beats) >= 8 else beats)
    if len(anchors) < 2:
        return None
    t = (np.arange(start, start + T) + 0.5) / FPS
    ph = np.zeros(T)
    for i in range(len(anchors) - 1):
        a, b = anchors[i], anchors[i + 1]
        msk = (t >= a) & (t < b)
        ph[msk] = TWO_PI * (t[msk] - a) / max(b - a, 1e-6)
    ph[t < anchors[0]] = 0.0
    ph[t >= anchors[-1]] = 0.0
    return ph.astype(np.float32)


def dirac_h(song, start, n, rng=None, noise=0.01):
    """h[:,0]=beat impulses, h[:,1]=downbeat impulses, small noise elsewhere."""
    r = rng if rng is not None else np.random
    h = (r.standard_normal((n, H_DIM)) if hasattr(r, "standard_normal")
         else r.randn(n, H_DIM)).astype(np.float32) * noise
    for t in song["beats"]:
        i = int(round(t * FPS)) - start
        if 0 <= i < n: h[i, 0] += 1.0
    for t in song["downs"]:
        i = int(round(t * FPS)) - start
        if 0 <= i < n: h[i, 1] += 1.0
    return h


def targets(song, start, n):
    b = np.zeros(n, np.float32); db = np.zeros(n, np.float32)
    for t in song["beats"]:
        i = int(round(t * FPS)) - start
        if 0 <= i < n: b[i] = 1.0
    for t in song["downs"]:
        i = int(round(t * FPS)) - start
        if 0 <= i < n: db[i] = 1.0
    return b, db
