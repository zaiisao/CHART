"""PROBE D: the scored read-out `evaluate.beats_from_barphase` applied to the constant-rate
sawtooth that free_run's phase_mu chain always is.

phase_mu[t] = (ph0 + t*r) mod 2pi  =>  psi[t] = (m*ph0 + t*m*r) mod 2pi, increment d = (m*r) mod 2pi.
A wrap is declared when diff(psi) < -pi. diff(psi) is either d (no wrap) or d-2pi (wrap).
  * d < pi  -> d-2pi < -pi  -> every true wrap IS detected  (correct behaviour)
  * d > pi  -> d-2pi > -pi  -> NO wrap is EVER detected     -> ZERO estimated beats -> F = 0
So for any phase rate r with (m*r mod 2pi) > pi the beat read-out returns an EMPTY beat list,
silently. Map the whole transfer function r -> #estimated beats and beat_F.
"""
import sys, math
import numpy as np
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
from vbpm.evaluate import beats_from_barphase, downbeats_from_barphase, f_measure, metronome

TWO_PI = 2 * math.pi; fps = 50.0; T = 1600
# a synthetic ground truth: 120 BPM, 4/4  -> beat every 25 frames, bar every 100
m = 4
ref = np.arange(0.0, T / fps, 0.5)
true_r = TWO_PI / (100.0)      # rad/frame bar advance

print("=" * 96)
print("D1  read-out transfer function: constant-rate sawtooth at rate r, m=4, truth=120BPM 4/4")
print("=" * 96)
print(f"{'r (rad/frame)':>14s} {'log r':>8s} {'BPM_implied':>12s} {'(m*r)mod2pi':>12s} {'#est':>6s} {'beat_F':>8s} {'#dbEst':>7s}")
for logr in [-4.0, -3.5, -3.0, -2.767, -2.5, -2.0, -1.5, -1.0, -0.5, -0.2, 0.0, 0.032, 0.3, 0.7, 1.0, 1.5, 2.0]:
    r = math.exp(logr)
    pm = (0.0 + np.arange(T) * r) % TWO_PI
    est = beats_from_barphase(pm, m, fps)
    dbest = downbeats_from_barphase(pm, fps)
    bpm = 60.0 * fps * m * r / TWO_PI
    print(f"{r:14.5f} {logr:8.3f} {bpm:12.1f} {(m*r)%TWO_PI:12.4f} {len(est):6d} {f_measure(ref,est):8.4f} {len(dbest):7d}")
print(f"  (truth: r={true_r:.5f}, log r={math.log(true_r):.3f}, 120.0 BPM, #ref beats={len(ref)})")
print(f"  metronome floor F = {f_measure(ref, metronome(T, fps)):.4f}")

print()
print("=" * 96)
print("D2  DEAD ZONES: fraction of log-rate space where the beat read-out returns ZERO beats")
print("=" * 96)
grid = np.linspace(-5, 2.5, 3000)
nest = []
for logr in grid:
    r = math.exp(logr)
    pm = (np.arange(400) * r) % TWO_PI
    nest.append(len(beats_from_barphase(pm, m, fps)))
nest = np.array(nest)
dead = nest == 0
print(f"  over log r in [-5, 2.5]: {dead.mean()*100:.1f}% of the range yields ZERO estimated beats")
print(f"  first dead region starts at log r = {grid[dead][0]:.3f} (r={math.exp(grid[dead][0]):.4f}); "
      f"threshold predicted by theory: m*r=pi -> r={math.pi/m:.4f} -> log r={math.log(math.pi/m):.3f}")
print(f"  model's UNTRAINED rate log r=0.032 -> (m*r)mod2pi = {(m*math.exp(0.032))%TWO_PI:.3f} "
      f"({'DEAD (>pi)' if (m*math.exp(0.032))%TWO_PI > math.pi else 'alive'})")

print()
print("=" * 96)
print("D3  same test with the m-subdivision read-out done SAFELY (unwrapped phase)")
print("=" * 96)


def beats_safe(phase, m, fps, min_dist_sec=0.10):
    """reference implementation: unwrap first, then find crossings of 2pi k / m."""
    ph = np.unwrap(np.asarray(phase, float) - math.pi) + math.pi
    psi = m * ph
    k0 = math.ceil(psi[0] / TWO_PI)
    ks = np.arange(k0, math.floor(psi[-1] / TWO_PI) + 1)
    if len(ks) == 0: return np.zeros(0)
    idx = np.searchsorted(psi, ks * TWO_PI)
    out, last = [], -1e9
    for i in idx:
        if i - last >= min_dist_sec * fps:
            out.append(i); last = i
    return np.asarray(out, float) / fps


for logr in [-2.767, -2.0, -1.0, 0.032]:
    r = math.exp(logr)
    pm = (np.arange(T) * r) % TWO_PI
    print(f"  log r={logr:7.3f}: buggy read-out #est={len(beats_from_barphase(pm,m,fps)):5d} F={f_measure(ref,beats_from_barphase(pm,m,fps)):.4f}"
          f"   | unwrap read-out #est={len(beats_safe(pm,m,fps)):5d} F={f_measure(ref,beats_safe(pm,m,fps)):.4f}")
print("  NOTE at log r=0.032 even the SAFE read-out cannot help: the sawtooth genuinely runs 16x too fast")
print("  and the 0.10 s min-distance filter caps the output at 10 beats/s.")
