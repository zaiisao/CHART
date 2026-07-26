"""P2: what is the CEILING of the phase_mu read-out, given P1 proved it is exactly a
constant-rate metronome (rate=exp(p_lv_mu), offset=p_ph_mu, both from mean-pooled h)?

 C1  oracle tempo + oracle offset  (best possible for a constant metronome)
 C2  oracle tempo + WORST-CASE / random offset (what an unaligned init gives)
 C3  ORACLE full bar phase (the known-answer reference, should be ~0.95)
Also: how much does a tempo error of x% cost?
"""
import sys, glob, math
import numpy as np
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
from vbpm.evaluate import beats_from_barphase, f_measure, metronome, _estimate_meter

CACHE = "/disk1/jaehoon/vbpm_mert_cache"; fps = 50.0; TWO_PI = 2 * math.pi


def load(split, cap=None):
    out = []
    for f in sorted(glob.glob(f"{CACHE}/{split}__*.npz")):
        d = np.load(f, allow_pickle=True)
        out.append(dict(key=f.split("__")[-1][:-4], T=int(d["feats"].shape[1]),
                        beats=np.asarray(d["beats"], float), downs=np.asarray(d["downs"], float)))
    return out[:cap] if cap else out


ev = load("eval", 30)
T_MAX = 1600


def const_chain(phi0, rate, T):
    return (phi0 + rate * np.arange(T)) % TWO_PI


def oracle_barphase(beats, downs, T):
    t = (np.arange(T) + 0.5) / fps
    anchors = downs if len(downs) >= 2 else beats
    if len(anchors) < 2: return None
    ph = np.zeros(T)
    for i in range(len(anchors) - 1):
        a, b = anchors[i], anchors[i + 1]
        msk = (t >= a) & (t < b)
        ph[msk] = TWO_PI * (t[msk] - a) / max(b - a, 1e-6)
    return ph


c1, c2, c3, cm, c1_off = [], [], [], [], []
tempo_curve = {k: [] for k in [0.0, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]}
for s in ev:
    T = min(s["T"], T_MAX)
    ref = s["beats"][s["beats"] < T / fps]; dref = s["downs"][s["downs"] < T / fps]
    if len(ref) < 3: continue
    m = _estimate_meter(ref, dref)
    ibi = float(np.median(np.diff(ref)))
    rate = TWO_PI / (ibi * m * fps)
    # C1: best offset by grid search
    best, bo = -1, 0
    for phi0 in np.linspace(0, TWO_PI, 200, endpoint=False):
        f = f_measure(ref, beats_from_barphase(const_chain(phi0, rate, T), m, fps))
        if f > best: best, bo = f, phi0
    c1.append(best); c1_off.append(bo)
    # C2: random offsets (mean over 20)
    rr = np.random.default_rng(0)
    c2.append(np.mean([f_measure(ref, beats_from_barphase(const_chain(rr.uniform(0, TWO_PI), rate, T), m, fps))
                       for _ in range(20)]))
    ph = oracle_barphase(s["beats"], s["downs"], T)
    c3.append(f_measure(ref, beats_from_barphase(ph, m, fps)))
    cm.append(f_measure(ref, metronome(T, fps)))
    # tempo error sensitivity at BEST offset each time
    for e in tempo_curve:
        r2 = rate * (1 + e)
        bb = max(f_measure(ref, beats_from_barphase(const_chain(p0, r2, T), m, fps))
                 for p0 in np.linspace(0, TWO_PI, 100, endpoint=False))
        tempo_curve[e].append(bb)

print("=" * 78)
print("P2  CEILING of the constant-metronome phase_mu read-out (n=%d eval songs)" % len(c1))
print("=" * 78)
print(f"  C3 ORACLE full bar phase          beat_F = {np.mean(c3):.3f}   <- read-out math is fine")
print(f"  C1 oracle tempo + BEST offset     beat_F = {np.mean(c1):.3f}   <- HARD CEILING of free_run's phase_mu")
print(f"  C2 oracle tempo + RANDOM offset   beat_F = {np.mean(c2):.3f}")
print(f"  CM 120-BPM metronome baseline     beat_F = {np.mean(cm):.3f}")
print("\n  tempo-error sensitivity (best offset each):")
for e in sorted(tempo_curve):
    print(f"    rate x (1+{e:.2f})  beat_F = {np.mean(tempo_curve[e]):.3f}")
print(f"\n  init model rate error = +1539% (16.39x)  -> off the end of this table -> F ~ 0.04")
