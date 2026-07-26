"""Shared helpers for the ENCODER / amortized-posterior probes (Dirac setup)."""
import sys, glob, math, types
import numpy as np, torch
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")

CACHE = "/disk1/jaehoon/vbpm_mert_cache"
FPS = 50.0
TWO_PI = 2 * math.pi
H_DIM = 8


def load(split, cap=None):
    out = []
    for f in sorted(glob.glob(f"{CACHE}/{split}__*.npz")):
        d = np.load(f, allow_pickle=True)
        out.append(dict(key=f.split("__")[-1][:-4], T=int(d["feats"].shape[1]),
                        beats=np.asarray(d["beats"], float), downs=np.asarray(d["downs"], float)))
        if cap and len(out) >= cap:
            break
    return out


def dirac_h(beats, downs, start, n, rng=None, jitter=0.0):
    """EXACT copy of probe_dirac.dirac_h (h[:,0]=beat impulses, h[:,1]=downbeat impulses)."""
    r = rng if rng is not None else np.random
    h = r.standard_normal((n, H_DIM)).astype(np.float32) * 0.01 if rng is not None \
        else np.random.randn(n, H_DIM).astype(np.float32) * 0.01
    for t in beats:
        i = int(round((t + jitter) * FPS)) - start
        if 0 <= i < n:
            h[i, 0] += 1.0
    for t in downs:
        i = int(round((t + jitter) * FPS)) - start
        if 0 <= i < n:
            h[i, 1] += 1.0
    return h


def targets(beats, downs, start, n, jitter=0.0):
    b = np.zeros(n, np.float32); db = np.zeros(n, np.float32)
    for t in beats:
        i = int(round((t + jitter) * FPS)) - start
        if 0 <= i < n:
            b[i] = 1.0
    for t in downs:
        i = int(round((t + jitter) * FPS)) - start
        if 0 <= i < n:
            db[i] = 1.0
    return b, db


def oracle_barphase(beats, downs, start, n, jitter=0.0):
    """Ideal BAR phase over frames [start, start+n): 0 at each downbeat, linear to 2pi."""
    t = (np.arange(start, start + n) + 0.5) / FPS
    anchors = np.asarray(downs, float) + jitter
    if len(anchors) < 2:
        return None
    ph = np.zeros(n)
    ok = np.zeros(n, bool)
    for i in range(len(anchors) - 1):
        a, b = anchors[i], anchors[i + 1]
        msk = (t >= a) & (t < b)
        ph[msk] = TWO_PI * (t[msk] - a) / max(b - a, 1e-6)
        ok |= msk
    return ph, ok


def true_phidot(s):
    """True bar-advance rate in rad/frame from the GT downbeats (median bar length)."""
    if len(s["downs"]) >= 3:
        bar = float(np.median(np.diff(s["downs"])))
    else:
        bar = 4 * float(np.median(np.diff(s["beats"])))
    return TWO_PI / (bar * FPS)


def circ_diff(a, b):
    """(a - b) wrapped to (-pi, pi]."""
    return (a - b + math.pi) % TWO_PI - math.pi


def record_unpack(model):
    """Monkeypatch THIS INSTANCE's unpack to log every call (never touches vbpm/ source)."""
    log = []
    orig = type(model).unpack

    def patched(self, vec):
        out = orig(self, vec)
        log.append(tuple(x.detach() for x in out))
        return out
    model.unpack = types.MethodType(patched, model)
    return log
