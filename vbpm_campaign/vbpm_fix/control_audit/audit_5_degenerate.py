"""AUDIT 5 -- DEGENERATE-ESTIMATOR FLOORS (what a number can be worth without tracking).

Two ways a variant can look good without tracking anything:
  (i)  BEAT SPAM: emit a dense grid; mir_eval's 70 ms window is forgiving of recall.
       We measure F for grids at several rates, incl. the densest the read-out can
       produce (beats_from_activation/beats_from_barphase enforce min_dist 0.10-0.15 s).
  (ii) PERFECT OPEN LOOP: the deploy path is a metronome; so the *most* any fix that
       leaves it open-loop can achieve is a metronome with the ORACLE per-song tempo
       and the ORACLE start phase. That is the true ceiling of "not audio-responsive",
       and every claimed fix should be read against it, not against 120 BPM.
"""
import sys
import numpy as np

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
from audit_common import load_split, truncate, banner, metronome, f_measure, FPS

ev = load_split("eval")

banner("(i) BEAT-SPAM FLOOR: constant grid at rate r, scored with mir_eval F (70 ms)")
print(f"{'grid period':>14s} {'~BPM':>7s} {'beat_F':>8s} {'n_est/n_true':>13s}")
for per in [0.02, 0.10, 0.15, 0.25, 0.35, 0.50, 0.6]:
    Fs, ne, nt = [], 0, 0
    for s in ev:
        T, ref, _ = truncate(s, None)
        if len(ref) < 2: continue
        est = np.arange(0.0, T / FPS, per)
        Fs.append(f_measure(ref, est)); ne += len(est); nt += len(ref)
    print(f"{per:14.2f} {60/per:7.0f} {np.mean(Fs):8.3f} {ne/nt:13.2f}")

banner("(ii) PERFECT OPEN LOOP: metronome with ORACLE tempo / ORACLE tempo+phase")
res = {}
for name in ["120BPM fixed", "oracle tempo, phase=0", "oracle tempo, ORACLE phase",
             "oracle tempo, WORST phase", "oracle tempo, random phase (mean of 8)"]:
    res[name] = []
rng = np.random.default_rng(0)
for s in ev:
    T, ref, _ = truncate(s, None)
    if len(ref) < 3: continue
    dur = T / FPS
    ibi = float(np.median(np.diff(ref)))                     # oracle per-song beat period
    res["120BPM fixed"].append(f_measure(ref, metronome(T, FPS)))
    res["oracle tempo, phase=0"].append(f_measure(ref, np.arange(0.0, dur, ibi)))
    offs = np.linspace(0, ibi, 25, endpoint=False)
    scores = [f_measure(ref, np.arange(o, dur, ibi)) for o in offs]
    res["oracle tempo, ORACLE phase"].append(max(scores))
    res["oracle tempo, WORST phase"].append(min(scores))
    res["oracle tempo, random phase (mean of 8)"].append(
        float(np.mean([f_measure(ref, np.arange(rng.uniform(0, ibi), dur, ibi)) for _ in range(8)])))
for k, v in res.items():
    print(f"  {k:42s} beat_F = {np.mean(v):.3f}   (N={len(v)})")
print("\n  READING: a deploy path that stays open-loop cannot beat the 'oracle tempo, ORACLE phase'")
print("  row even with perfect tempo AND perfect start phase; and with a random start phase it")
print("  sits at the 'random phase' row. Any claimed fix landing between those two has")
print("  improved TEMPO ESTIMATION, not audio-responsiveness.")

banner("(iii) PERFECT OPEN LOOP vs EXCERPT LENGTH (drift makes the ceiling length-dependent)")
for label, songs, cap in [("eval[:30] <=1600 (32 s)", ev[:30], 1600),
                          ("ALL 79 <=1600 (32 s)", ev, 1600),
                          ("ALL 79 <=3000 (60 s)", ev, 3000),
                          ("ALL 79 FULL", ev, None)]:
    best, met = [], []
    for s in songs:
        T, ref, _ = truncate(s, cap)
        if len(ref) < 3: continue
        dur = T / FPS; ibi = float(np.median(np.diff(ref)))
        offs = np.linspace(0, ibi, 25, endpoint=False)
        best.append(max(f_measure(ref, np.arange(o, dur, ibi)) for o in offs))
        met.append(f_measure(ref, metronome(T, FPS)))
    print(f"  {label:26s} oracle-tempo+oracle-phase beat_F = {np.mean(best):.3f}   (120BPM {np.mean(met):.3f}, N={len(best)})")

banner("(iv) SPAM FLOOR AT THE READ-OUTS' OWN MINIMUM SPACING")
for per, who in [(0.10, "beats_from_barphase (min_dist 0.10 s)"),
                 (0.15, "beats_from_activation (min_dist 0.15 s)")]:
    Fs = []
    for s in ev:
        T, ref, _ = truncate(s, None)
        if len(ref) < 2: continue
        Fs.append(f_measure(ref, np.arange(0.0, T / FPS, per)))
    print(f"  {who:44s} spam beat_F = {np.mean(Fs):.3f}")
print("  -> ANY variant must clear ITS OWN spam floor, not just the 0.28-0.30 metronome.")
