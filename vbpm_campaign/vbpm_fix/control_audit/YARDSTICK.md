# VBPM CONTROL / ANTI-FOOLING YARDSTICK

Produced by `vbpm_fix/audit_*.py` (also copied to `vbpm_fix/control_audit/`).
Everything is scored through the OFFICIAL read-out
(`vbpm.evaluate.beats_from_barphase` / `beats_from_activation` + `mir_eval` F, 70 ms).
Nothing under `vbpm/` was modified.

Data: `/disk1/jaehoon/vbpm_mert_cache`, 147 train (folds 1-7) / 79 eval (fold 0),
stem overlap 0 -- the split itself is clean.

---
## A. Protocol warnings (read before comparing any two numbers)

1. **`eval[:30]` is 30/30 ballroom.** `sorted(glob("eval__*"))` orders
   ballroom(32) < beatles(23) < hainsworth(24), so the subset every prior probe
   used (`probe_linear.py`, `probe_dirac.py`, `train_mert.py`) is a single,
   metronomic, 4/4 genre.
2. **`max_frames=1600` truncates 46 of 79 eval songs** (median length 2461
   frames, max 10893). Open-loop drift grows with length, so this cap flatters
   any metronome-like model -- see C.
3. **The read-out is handed the meter from the ground truth**
   (`m = _estimate_meter(ref, dref)`). Worth **+0.076** beat_F over always-4
   (0.960 vs 0.883). Every variant gets the same gift, so it is not a
   per-variant cheat, but no number here is "from audio alone".

---
## B. Reference ceilings

| reference | protocol | beat_F | db_F |
|---|---|---|---|
| IDEAL bar phase (0 at each downbeat, linear to 2pi) | eval[:30] <=1600 | **0.955** | 0.970 |
| IDEAL bar phase | ALL 79, <=1600 | 0.966 | 0.983 |
| IDEAL bar phase | ALL 79, FULL | **0.960** | 0.981 |
| ideal BEAT-LINEAR phase (read-out code limit) | ALL 79, FULL | 0.968 | 0.978 |

The 0.955 previously reported is **reconfirmed exactly**. Per dataset (FULL):
ballroom 0.956, beatles 0.952, hainsworth 0.972. The read-out is not the problem.

---
## C. Floors -- scores obtainable with NO tracking

| floor | ALL 79 FULL | eval[:30] <=1600 |
|---|---|---|
| 120 BPM metronome | **0.281** | **0.295** |
| blind grid at 0.15 s (min spacing of `beats_from_activation`) | **0.433** | -- |
| blind grid at 0.10 s (min spacing of `beats_from_barphase`) | 0.326 | -- |
| blind grid at 3.9x true density | 0.380 | -- |
| perfect OPEN LOOP: oracle tempo + oracle start phase | **0.686** | **0.828** |
| open loop, oracle tempo, random start phase | 0.266 | -- |

Two consequences:

* **Beating 0.295 proves nothing.** A blind grid at the read-out's own minimum
  spacing scores 0.433. Always report `n_est/n_true` alongside beat_F.
* **On the 32 s protocol an entirely audio-blind deploy path can reach 0.828**
  if it merely picks the right tempo and start phase at frame 0. That is *above*
  the conv baseline. Under this cap, a high free-run score is NOT evidence that
  evidence reaches the state; only the roll control (E) can show that.

---
## D. Honest MERT baselines (same eval songs, same read-out)

| model | eval[:30] <=1600 | ALL 79 <=1600 | ALL 79 FULL | n_est/n_true (FULL) |
|---|---|---|---|---|
| 120 BPM metronome | 0.295 | 0.287 | 0.281 | 1.07 |
| linear probe + peak-pick (thr 0.5) | **0.725** | 0.735 | **0.736** | 1.35 |
| conv probe + peak-pick (thr 0.5) | **0.804** | 0.803 | **0.805** | 1.34 |
| conv probe, threshold picked on TRAIN (0.7) | 0.789 | -- | **0.814** | 1.11 |

0.725 and 0.804 are **reproduced exactly** (same arch/seed/steps as
`vbpm/probe_linear.py`). Per dataset (conv-equivalent, linear shown):
ballroom 0.720 / beatles 0.749 / hainsworth 0.745 -- the probes are not
genre-fragile; the ballroom-only subset happened not to distort them.

**The bar for any fixed VBPM is 0.74 / 0.81, not 0.31.**

---
## E. Dirac is an ORACLE -- the copy ceiling

DIRAC h puts a 1.0 at every true beat frame (channel 0) and every true downbeat
frame (channel 1). The input *is* the label track, including at eval time.

| copy path | eval[:30] <=1600 | ALL 79 FULL |
|---|---|---|
| ZERO parameters: peak-pick channel 0 | 0.999 | **1.000** |
| tiny linear (8->2), no latents, trained on train fold | 0.999 | **1.000** |
| tiny conv, no latents | 0.999 | 1.000 (db 0.999) |
| through the OFFICIAL bar-phase read-out (phase rebuilt from channel 1) | 0.955 | **0.960** |
| negative control: same nets, impulses removed | -- | 0.000 |

**A Dirac beat_F of anything up to 1.0 is fully explained by copying the input.**
Nothing below 0.960 through the phase read-out, or below 1.000 through an
activation read-out, is evidence about beat tracking -- only that the plumbing
conducts. Dirac results must be quoted with this ceiling attached.

---
## F. Time-roll leak control

Features slide +1000 frames (20 s); labels stay put; identical rng seed both arms.

| object | aligned | rolled | floor | verdict |
|---|---|---|---|---|
| ORACLE ideal phase (power check) | 0.960 | 0.296 | 0.281 | control HAS power |
| linear probe (ALL 79 FULL) | 0.736 | **0.319** | 0.281 | CLEAN |
| conv probe (ALL 79 FULL) | 0.805 | **0.314** | 0.281 | CLEAN |
| trained MERT VBPM `runs/mert_vbpm/best.pt` (ALL 79, <=1600) | 0.348 | **0.347** | 0.287 | **INVARIANT** (drop +0.000) |
| trained MERT VBPM (20-song subset) | 0.336 | 0.336 | 0.306 | INVARIANT (drop -0.001) |

The probes are clean: their gain is entirely in the feature-label alignment.

The trained VBPM never had a gain to lose. Two separate findings:

* Its score is **completely invariant** to a 20 s feature shift (0.348 -> 0.347).
  The within-song free-run phase increment is constant to **9.2e-07 rad/frame**:
  the deploy trajectory is a straight line. (A raw phase difference between the
  aligned and rolled runs, 1.55 rad, is NOT evidence of tracking -- audio picks
  ONE rate at t=0 via `prior_init_head(prior_ctx.mean(1))` and a ~0.1% change in
  that single constant integrates to >1 rad over 1600 frames. `roll_control.py`
  now reports |drop|<0.02 as AUDIO-BLIND rather than "clean".)
* Its aligned score is produced at `n_est/n_true = 3.66` and **downbeat_F = 0.000**.
  A blind constant grid at that density scores **0.380-0.396**. So
  **the previously quoted "VBPM free-run 0.31-0.34" is below the blind
  density-matched floor -- it is not a beat-tracking score at all.**

Reusable:
```python
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
from roll_control import roll_control
roll_control(my_phase_fn, my_feat_fn, eval_songs, roll=1000, cap=None, label="variant X")
```

---
## G. Noise floor (N = 79 songs)

| quantity | mean | sd | SEM |
|---|---|---|---|
| ideal ceiling | 0.960 | 0.081 | 0.009 |
| 120 BPM metronome | 0.281 | 0.068 | 0.008 |
| perfect open loop | 0.686 | 0.242 | 0.027 |
| spam grid 0.15 s | 0.433 | 0.091 | 0.010 |

Differences below ~0.05 need paired per-song reporting to be believable.

---
## H. Reporting checklist for any claimed fix

1. `beat_F`, `downbeat_F`, **and `n_estimated / n_true`**.
2. Which eval subset, which frame cap.
3. The metronome floor AND the density-matched spam floor under that protocol.
4. The perfect-open-loop reference under that protocol (0.686 FULL / 0.828 at 32 s).
   Landing below it does not show that evidence reaches the state.
5. The time-roll control result.
6. If DIRAC: state that the input contains the labels and quote the copy ceiling
   (1.000 activation / 0.960 bar-phase).

---
## I. Mechanistic corroboration (audit_9_increment.py)

Free-run of `runs/mert_vbpm/best.pt`, per song, cap 800 frames:

| statistic | value |
|---|---|
| within-song std of the phase increment | **8-11 x 1e-7 rad/frame** (a straight line) |
| the constant increment itself (wrapped) | mean -1.026, sd 0.143 across songs |
| unwrapped advance = 2pi - 1.026 | **~5.26 rad/frame** |
| true `phidot` for these songs | 0.040 - 0.090 rad/frame |
| ratio | **~60-130x too fast** -> the bar phase wraps every ~1.2 frames |
| relative change of that constant when features roll 20 s | 0.98 % |

So the deploy path emits *aliased* wraps that the read-out's 0.10 s minimum-distance
filter thins into a ~0.1-0.15 s grid. That is exactly the spam floor (0.326-0.396),
which is exactly what it scores (0.348), with `downbeat_F = 0.000`. The number is an
artifact of aliasing + min-distance filtering, not a tracking result.

---
## J. Still running at report time

`audit_4b_dirac_roll.py` (GPU 1, log `vbpm_fix/audit4b.log`) trains a Dirac VBPM
with `vbpm/probe_dirac.py`'s recipe (700 steps) and then applies the roll control
to it on eval[:30] and a stratified 30. Its purpose is supplementary: section E
already establishes the Dirac copy ceiling (1.000 / 0.960), which is the number a
Dirac result must be compared against regardless of what the roll shows.
