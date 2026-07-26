"""P1: numbers the ladder must be judged against.
 (1) oracle PIECEWISE bar phase  -> beats_from_barphase   (S1 replication, ~0.955 expected)
 (2) oracle CONSTANT-tempo ramp  -> the exact functional form the deployed free_run mean
     chain can express (P0). This is the TRUE ARCHITECTURE CEILING of the deploy read-out.
 (3) true phidot statistics (rad/frame) vs the model's init 1.032 rad/frame.
 (4) sensitivity of the ceiling to errors in the 2 deploy scalars (phi0, log phidot).
"""
import sys, math
import numpy as np
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_debug")
from common import load, est_meter, oracle_const_phase, oracle_pw_phase, FPS, TWO_PI
from vbpm.evaluate import beats_from_barphase, downbeats_from_barphase, metronome, f_measure

ev = load("eval"); tr = load("train")
MAXF = 1600
print(f"eval songs {len(ev)}  train songs {len(tr)}")

pw, cs, mt, dbpw, dbcs = [], [], [], [], []
phidots, meters = [], []
for s in ev:
    T = min(s["T"], MAXF)
    ref = s["beats"][s["beats"] < T / FPS]; dref = s["downs"][s["downs"] < T / FPS]
    if len(ref) < 2: continue
    m = est_meter(ref, dref)
    meters.append(m)
    php = oracle_pw_phase(s, T)
    phc, phidot, phi0 = oracle_const_phase(s, T)
    phidots.append(phidot)
    if php is not None:
        pw.append(f_measure(ref, beats_from_barphase(php, m, FPS)))
        if len(dref) >= 2: dbpw.append(f_measure(dref, downbeats_from_barphase(php, FPS)))
    cs.append(f_measure(ref, beats_from_barphase(phc, m, FPS)))
    if len(dref) >= 2: dbcs.append(f_measure(dref, downbeats_from_barphase(phc, FPS)))
    mt.append(f_measure(ref, metronome(T, FPS)))

print("=" * 78)
print(f"  (1) ORACLE PIECEWISE phase   beat_F = {np.mean(pw):.3f}   db_F = {np.mean(dbpw):.3f}   n={len(pw)}")
print(f"  (2) ORACLE CONSTANT-tempo    beat_F = {np.mean(cs):.3f}   db_F = {np.mean(dbcs):.3f}   n={len(cs)}")
print(f"      ^^^ THIS is the ceiling the deployed free_run mean-chain can reach (P0)")
print(f"  (3) 120BPM metronome floor   beat_F = {np.mean(mt):.3f}")
print()
pd = np.array(phidots)
print(f"  true phidot (BAR rad/frame): mean={pd.mean():.4f} median={np.median(pd):.4f} "
      f"min={pd.min():.4f} max={pd.max():.4f}")
print(f"  true log phidot            : mean={np.log(pd).mean():.3f} std={np.log(pd).std():.3f} "
      f"range=[{np.log(pd).min():.3f},{np.log(pd).max():.3f}]")
print(f"  VBPM init free-run log_tempo = +0.032 -> ratio {1.0324/pd.mean():.1f}x TOO FAST")
print(f"  meters: {np.bincount(meters, minlength=5)[2:5]} for m=2,3,4")

# (4) sensitivity: perturb the two deploy scalars
print()
print("=" * 78)
print("  (4) SENSITIVITY of the constant-tempo ceiling to the 2 deploy scalars")
print("      (how accurate must prior_init_head be?)")
def score(dphi0=0.0, dlogtempo=0.0):
    fs = []
    for s in ev:
        T = min(s["T"], MAXF)
        ref = s["beats"][s["beats"] < T / FPS]; dref = s["downs"][s["downs"] < T / FPS]
        if len(ref) < 2: continue
        m = est_meter(ref, dref)
        _, phidot, phi0 = oracle_const_phase(s, T)
        pd2 = phidot * math.exp(dlogtempo)
        ph = (phi0 + dphi0 + np.arange(T) * pd2) % TWO_PI
        fs.append(f_measure(ref, beats_from_barphase(ph.astype(np.float32), m, FPS)))
    return float(np.mean(fs))
print(f"      exact                          : {score():.3f}")
for e in [0.02, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 2.767]:
    print(f"      log-tempo error +{e:5.3f} ({math.exp(e):5.2f}x): {score(dlogtempo=e):.3f}   "
          f"-{e:5.3f} ({math.exp(-e):5.3f}x): {score(dlogtempo=-e):.3f}")
for e in [0.1, 0.3, math.pi/4, math.pi/2, math.pi]:
    print(f"      phase offset {e:5.3f} rad ({e/TWO_PI*100:4.1f}% of a bar): {score(dphi0=e):.3f}")
