# innovq campaign — 2026-07-27

Source + logs from a full day on the innovation-space posterior (`vbpm_innovq`), copied out of
`/home/sogang/jaehoon/VBPM_reintegration` (which is NOT a git repo — that is why this exists).

## READ THIS FIRST: the corr convention

`probe()` logs **two different statistics** and the console prints the wrong one.

| name | formula | where |
|---|---|---|
| `corr` | `abs(mean over B*T of exp(i(phi-phi_true)))` — POOLED | printed to console |
| `corr_percrop` / `pc=` | `mean_crops abs(mean_t ...)` — PER-CROP | SPEC.md:220-225, abort criterion |

`isolate_tempo.py`'s `corr()` is **bit-identical to per-crop** (0.2451 vs 0.2451, diff 0.0), not to
pooled (0.0103) — 24x apart on the same trajectory. **There is no constant conversion:**
`pooled = percrop * R_between`, and R_between measured **0.042-0.534** on this codebase.
Per-crop constant phase offsets cost pooled everything and per-crop nothing.

The 0.998 "ceiling" looks convention-free (pooled 0.9955 / percrop 0.9979) only because that case
has zero offset error. It is also the wrong comparator: the reachable target is the **teacher,
per-crop 0.6335**.

Quote per-crop. Everything below is per-crop.

## What holds

1. **The tempogram level head works.** `--tg --tg_pre` reaches per-crop corr **0.57-0.64** at ELBO
   step 1 — at teacher level, vs a constant-tempo ceiling of 0.726. Pooled-GRU init reaches 0.19.
   Tempo train MAE 1.77% (target <2%) vs the pooled head's 4.40%.
2. **The ELBO then destroys it.** 0.6369 -> 0.0909 (s200) -> 0.0265 (s400). Level MAE vs
   `train['lt']` goes 10.8% -> 37% in 75 steps — **worse than predicting the corpus mean (28.1%)**,
   so it is not lazy collapse; the encoder learns an actively wrong tempo.
3. **corr is ~entirely a function of tempo error.** isolate: true-tempo 0.998, model-tempo 0.270;
   phase offset costs nothing (A==C, B==D). corr>=0.82 needs tempo within 2%.
4. **The innovation prior is ~90x too tight.** `gamma_phase=5.5e-4` vs physical microtiming
   ~0.049 rad. mu=0.005 rad/frame costs **4613 nats/crop** at 5.5e-4 vs **2.6** at 0.06, against a
   total recon benefit of correct phase of ~117 nats. Zero innovations was the CORRECT optimum.
5. **The decoder is phase-blind at initialization** (+0.00% recon rise to a full phase scramble);
   fit on oracle phase it reaches +38-45%.

## What was refuted (all mine)

- **Cramer / displacement reconstruction.** Verified far-field tempo slope 89250 vs BCE 63.7 on an
  IDEALIZED pulse train — but the decoder is LEARNED and adapts by going phase-insensitive, which
  flattens the landscape the loss was chosen to steepen. Through a trained decoder, adding Cramer
  *reduces* the tempo slope (38.1 -> 2.1) and decoder phase sensitivity (38.7% -> 3.3%).
  **lambda=0 (pure BCE) wins on both axes.** `lam_calib.py`. The gate held the emission fixed and
  therefore could not detect this.
- "corr stays at 0" for placement/handover: units error, was per-crop 0.688 train / 0.560 eval,
  ABOVE the pre-registered 0.60-0.635 bracket. (ELBO-stage collapse is real in both units.)
- "8.7% tempo representation limit": artifact of pooled-GRU features, not the input. Tempogram
  reaches 0.40% train on the same 2 channels.
- "`rollout_vec_s` is bit-identical": true only for `sample=False` and one small batch. See below.

## Known broken (found by adversarial workflow, NOT fixed)

- `--tg_freeze` only attenuates (mu_l1 moves 2.21e-02 vs 4.44e-01) — the shared GRU trunk keeps
  feeding the frozen head. Any conclusion resting on "tempo was held" is unsound.
- `--tg` + `--pre > 0` hard-crashes (`_TGInit.forward` broadcast, innovq_tg.py:43).
- `--dec_freeze` / `--tg_freeze` are silent no-ops unless their `--*_pre > 0` (nested inside).
- Console prints pooled only at HANDOVER; no per-crop eval number in any log.

## Fixed this commit

- `--gamma_phase` was a **wrong-op**: `GP1` and `R0` are derived from `RHO_P` at import and
  `main()` rebound only `RHO_P`, so `kl_phase_innov` mixed a new RHO_P with a stale GP1 (93x
  understated KL) and `R0` initialised rho_q at the old prior. Now rebinds all three;
  `rollout_vec_s` reads `IQ.R0` at call time instead of holding a from-import copy.
  **The 16-cell factorial (`F_*` logs) had gamma as its main axis and is therefore invalid.**
  The 14:54 `g_*` sweep predates the regression and stands.
- `rollout_vec_s` default `n_picard` 4 -> 8 (at 4, ~18% of crops get the wrong bar-crossing count).

## Speed

`rollout_vec_s` (Picard parallel-in-time, sampled, full ELBO outputs) turns a 30-min run into ~2
min (0.56 -> 26-33 it/s). **NOT exact at sampled T=1500**: ~1/24 crops settle on the other fixed
point of the bistable `cross = advance >= 2pi` threshold (dphi 4.75e-03, dn_cross 1, dkl_m 0.97
nats). Two fallback detectors were tried and both failed — documented in `src/rollout_vec_s.py`.
Below ELBO sampling variance (kl swings 51/36/348 across seeds) so acceptable for training, but
`--fast` runs must never be pooled with non-fast runs as replicates.

## Open question

The tempogram installs a working tempo; the ELBO removes it. Whether that is the tempo
side-channel (encoder paying KL to route information through log-tempo) has not been established
on this branch.
