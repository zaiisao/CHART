# VBPM — state of play, 2026-07-27

## 1. What the project is

**VBPM** = Variational Bar Pointer Model. A structured latent-variable model for beat and
downbeat tracking. Latents per frame: bar phase `φ` (wrapped, one bar = 2π), log-tempo `lt`,
meter `m`. Conditioned on `h` = frozen Beat This frontend activations, shape [T,2] (beat channel,
downbeat channel), 50 fps.

CVAE structure (Sohn 2015): `q_φ(z|h,b)` recognition, `p_ψ(z|h)` conditional prior,
`p_θ(b|z)` **latent-only** emission (h deliberately withheld from the beat emission to stop the
decoder shortcutting through the audio). There is a separate activation emission `p(h|z)`.

Goal: beat tracking competitive with madmom / Beat This, with an interpretable latent that also
permits **multi-task supervision** — tempo datasets → `lt`, time-signature datasets → `m` —
which a black-box tracker cannot do. Stated mission: improve SMC (a 217-piece corpus of
deliberately difficult/expressive music).

Posterior is "innovation-space": q emits an initial state plus **bounded per-frame corrections**
`ε` on top of the prior recursion, so trajectories are smooth by construction.
`φ_t = φ_{t-1} + exp(lt_{t-1}) + ε_t`, with `|mu_eps| ≤ s_phi = 0.05` rad.

## 2. THE ROOT CAUSE FOUND TODAY — infinite-variance latent dynamics

Both the phase and tempo random walks use heavy-tailed distributions with **infinite variance**,
so their sums scale as **n, not √n**.

**Phase:** innovations are **wrapped Cauchy** (tail index α=1). Measured, γ_q = 0.06 rad/frame:

| frames elapsed | median accumulated phase noise |
|---|---|
| 10 | 0.61 rad |
| 100 (2 seconds) | 6.16 rad (≈1 full rotation) |
| 500 | 30.20 rad |
| 1499 | 80.44 rad (≈13 rotations) |

Theory: sum of n Cauchy(0,γ) = Cauchy(0, n·γ) → 89.9 rad at T=1500. Simulation matches to 10%.
A beat is 1.571 rad. **The sampled phase trajectory is effectively uniform random within ~2 s.**

**Tempo:** level prior is **Student-t with DOF = 2** — also infinite variance. Measured: with the
level mean set *exactly* to the truth and innovations at prior scale, the sampled trajectory is
**43.67% off** (per-crop corr 0.151). Reaching ~2% requires shrinking `s_lv` 1.25e-3 → 2.5e-4 and
costs **+1203 nats** of level KL.

**Consequence:** `E_q[log p(b|z)]` is evaluated at essentially random phases. Gradient SNR on the
initial-phase parameter is **0.092** (noise 10× signal); the gradient sign is correct only 58% of
the time (chance = 50%). Deterministic evaluation uses the *means* and looks healthy (per-crop
corr 0.976); training uses *samples* and is noise. **The model we evaluate and the model we train
are not the same object.** This breaks the training AND inference distributions simultaneously,
which is why the usual CVAE train/test-gap diagnosis (Sohn §4.2) looked inapplicable — it was
pre-empted.

The spec (`docs/ELBO_for_DBN.md` §3, §5.2, Algorithm 2) specifies **von Mises** for phase.
Wrapped Cauchy is an **undocumented departure**. Heavy tails per se are defensible (real tempo
increments have kurtosis ~13 and Laplace fits best) — but Laplace and Student-t(ν>2) have
**finite** variance and √n accumulation. Cauchy (ν=1) is the single pathological point.

**Prescribed fix:** wrapped Student-t with ν ≥ 3 for phase, ν > 2 for tempo. Plus per-beat rather
than per-frame scale (see §4).

## 3. THE MEASUREMENT ERROR — pooled vs per-crop correlation

`probe()` logs two different statistics and **the console prints the wrong one**:

- `corr` = POOLED: `|mean over B*T of exp(i(φ−φ_true))|`
- `corr_percrop` / `pc=` = PER-CROP: `mean_crops |mean_t ...|` — the **pre-registered** convention
  (SPEC.md:220-225), and what the abort criterion uses.

`isolate_tempo.py`'s `corr()` is **bit-identical to per-crop** (0.2451 vs 0.2451, diff 0.0), not to
pooled (0.0103). **24× apart.** No constant conversion: `pooled = percrop × R_between`, and
R_between measured **0.042 – 0.534** on the same codebase. Per-crop constant phase offsets cost
pooled everything and per-crop nothing.

Consequence: a full day of "corr stays at 0" verdicts were pooled numbers read against a per-crop
ceiling. Placement/handover actually reached **per-crop 0.688 train / 0.560 eval** — *above* the
pre-registered 0.60–0.635 bracket. The ELBO-stage collapse is real in both conventions.

Also: the often-quoted 0.998 "ceiling" is the wrong comparator. The reachable target is the
**teacher**, per-crop **0.6335**.

## 4. BEAT F — the number that actually matters, and the offset

All of the above is `corr`, which is **invariant to a constant phase offset**. Beat F is not.
Measured (48 crops, T=1500, density-matched blind control mandatory):

| configuration | beat F | blind control | lift | per-crop corr |
|---|---|---|---|---|
| tempogram init, offset unset | 0.4561 | 0.5308 | **−0.0748** | 0.9763 |
| + TRUE phase offset | **0.8040** | 0.7181 | **+0.0859** | 0.9763 (unchanged) |
| ORACLE latent (ceiling) | 0.9737 | — | — | — |

corr is **identical** with and without the true offset while F moves 0.456 → 0.804. So the model
has the rhythm right and does not know where beat one is. **The remaining gap is one scalar per
song.** With it supplied, the model clears its blind control for the first time.

That scalar **is** `q(φ₁)` — the machinery exists (`mu_phi1 = atan2(init[:,K+1], init[:,K])`,
prior = wrapped Cauchy(π, 1e-6) ≈ uniform, correctly). And the reconstruction landscape over the
offset is a clean **305-nat** bowl with its minimum exactly at the truth. But `mu_phi1` **never
moves**: offset error 1.302 → 1.309 rad over 400 steps, and it still doesn't move with the
innovations frozen at zero and lr raised to 3e-2. Cause: the gradient estimator is noise (§2).

Structural note: `rho1_max = 0.9` caps the posterior's sharpness at a half-width of 0.451 rad ≈
2× the ±70 ms scoring tolerance (0.215 rad). The model cannot express a confident downbeat.

## 5. Other verified findings (2026-07-27)

- **Tempogram beats the pooled-GRU head.** Level MAE 1.77% vs 4.40%. Per-crop corr at ELBO step 1:
  **0.57–0.64** (teacher 0.6335; constant-tempo ceiling 0.726). The "8.7% representation limit" I
  previously claimed was an artifact of the pooled-GRU features, not the 2-channel input.
- **Innovation prior is ~90× too tight.** `gamma_phase = 5.5e-4` vs physical microtiming ~0.049 rad.
  An innovation of 0.005 rad/frame costs **4613 nats/crop** at 5.5e-4 vs **2.6** at 0.06, against a
  total reconstruction benefit of correct phase of ~117 nats. Zero innovations was the *correct*
  optimum. But loosening it alone makes the innovations cheap enough to abuse — both settings fail,
  for opposite reasons.
- **`s_phi = 0.05` is ~24× too loose** — it permits 77% of a frame's entire phase advance, every
  frame, where physical microtiming is ~0.049 rad *per beat*.
- **Beat-gating the tempo walk works and is free.** Restoring the doc's own between-beats-constant
  condition (Krebs 2015): tempo wobble **10.5% → 0.4%**, corr retention +43%, level MAE halved, and
  the ELBO is equal or better. Implemented with the same crossing/cummax machinery already used for
  meter, so the "differentiability" justification for the departure does not survive.
- **Decoder is phase-blind at initialization** (+0.00% recon rise to a full phase scramble),
  +38–45% once fit on the oracle trajectory.
- **The downbeat emission collapses to the base rate.** Downbeats are 1.12% of frames; the trained
  model's mean p_downbeat is 0.0129 vs a target rate of 0.0112, and it *never* exceeds 0.5. Under
  per-frame BCE the cheapest solution is silence, after which nothing in the likelihood depends on
  tempo. `stpool` (official ShiftTolerantBCELoss) already exists in `recon_terms` and is unused.
- **Tempo estimation vs madmom, identical activations, 312 eval crops:**

| estimator | Acc1 (within 4%) | Acc2 (octave-tolerant) | log-MAE |
|---|---|---|---|
| ours (tempogram, trained) | **83.0%** | 90.7% | 9.4% |
| madmom acf | 79.2% | 95.5% | 13.9% |
| madmom comb | 77.9% | 94.6% | 13.5% |
| madmom dbn | 73.4% | **95.8%** | 17.3% |

  We win exact accuracy (partly a learned corpus prior — we train, madmom doesn't) and **lose
  octave-tolerant by ~5 points**, meaning our raw periodicity detection is weaker. That gap is
  self-contained and actionable.

## 6. REFUTED TODAY (all of these were my own hypotheses)

- **Cramér / displacement reconstruction loss.** Its far-field tempo slope is 89,250 vs BCE's 63.7
  — but that was gated on a **fixed** emission. Through a **trained** decoder it *reduces* the slope
  (38.1 → 2.1) and decoder phase sensitivity (38.7% → 3.3%), because the learned emission adapts by
  going phase-insensitive. **λ = 0 (pure BCE) wins on both axes.** General lesson: fixed-emission
  loss-landscape analysis is not predictive when the likelihood is learned.
- **"The objective is informationally indifferent to tempo."** Overturned by the revised wire test:
  **+174 ± 9 nats** for a 10% level error; it orders errors up to ~5% and saturates beyond.
- **"free-q also degrades, therefore the objective is misspecified (H1)."** Invalid. The free
  per-crop family is **not a superset** of the encoder family — the encoder's innovations are
  *state-dependent feedback*, the free table is *open-loop*. The free arm ended **585 ± 8 nats worse**
  than the encoder on the identical objective. Wire-test verdict: **INCONCLUSIVE**.
- **H2 ("amortization is the breaker") is refuted in its stated form.** A free q started at the
  *exact* truth with the *oracle* offset still moved +5.12 pp (300 steps) / +8.56 pp (1500, monotone)
  while its ELBO **improved by 580 nats**. TEMPO_PREF = +1007.8 ± 8.2 toward the wrong level.
- Also refuted: "bar boundaries are being slid onto downbeats" (the degraded solution hits *fewer*
  downbeats, 0/12 vs 10/12); "the Cramér scales were too long" (sharp beats score 0.0 vs flat 3996.8).

## 7. Verified infrastructure

- `rollout_vec_s.py` — parallel-in-time (Picard) rollout with sampling and full ELBO outputs.
  **~50× speedup** (0.56 → 26–33 it/s at T=1500), gradient cosine 1.000000 vs the sequential loop
  under matched noise. **NOT exact** for sampled T=1500: ~1/24 crops settle on the other fixed point
  of the bistable `cross = advance ≥ 2π` threshold (dφ 4.75e-3, one bar-crossing, dkl_m 0.97 nats).
  Two fallback detectors were tried and both failed. Acceptable for training (below ELBO sampling
  variance), never for exactness claims; `--fast` runs must not be pooled with non-fast runs.
- `kl_t_mc` verified **unbiased** against scipy quadrature (0.3106 ± 0.0013 vs 0.308897). Its
  per-crop spread over 1499 steps is **35.2 nats**, which is why negative KL values are expected and
  why every ELBO comparison must be paired under common random numbers.
- **Fixed today:** `--gamma_phase` was a *wrong-op* — `GP1` and `R0` are derived from `RHO_P` at
  import and `main()` rebound only `RHO_P`, so `kl_phase_innov` mixed a new RHO_P with a stale GP1
  (93× understated KL). The 16-cell factorial had gamma as its main axis and is **invalid**.
- **Known broken, unfixed:** `--tg_freeze` only attenuates (the shared GRU trunk keeps feeding the
  frozen head); `--tg` with `--pre > 0` hard-crashes; `--dec_freeze`/`--tg_freeze` are silent no-ops
  without their `--*_pre`.

## 8. Krebs TASLP 2015 — what it actually claims (I previously mischaracterized this)

Krebs, Holzapfel, Cemgil, Widmer, *Inferring metrical structure in music using particle filters*,
IEEE TASLP 2015. Verbatim:

> "the largest HMM3 **slightly outperforms** the AMPF on all three datasets. However, this comes at
> a high price in terms of runtimes... it confirms that the **HMM on a sufficiently dense grid is
> able to perform accurate inference that cannot be outperformed using approximate methods (such as
> the PF) using the same underlying model.**"

Their SMC results (the **same 217 pieces** we use) — FM / AMLt / runtime:

| system | FM | AMLt | runtime |
|---|---|---|---|
| AMPF (their PF) | 40.8 | 35.8 | 18.1 min |
| HMM1 | 39.6 | 31.7 | 11.1 min |
| HMM2 | 40.5 | 35.1 | 42.1 min |
| HMM3 | 42.7 | 38.7 | **164.0 min** |

Ours on the same 217 songs: `final_gated` (PF stack) **0.6369**, peak-pick **0.6333**,
madmom DBN 55-215 λ100 **0.5927**, DBN 40-215 **0.6072**.

**We have NOT disproved them.** Their claim is essentially a correctness statement — exact
inference on a dense grid ≥ any approximation *of the same model* — and their own table shows a PF
beating a poorly-parameterized HMM (AMPF 40.8 > HMM1 39.6). Our comparison is not same-model
(different observation model, different state space; MASK2 is PF-posterior masking + peak-picking,
not a plain PF). And our 0.63 vs their 0.42 on identical audio is the **frontend** (Beat This),
not the inference.

**What we do have:** with a modern neural frontend, madmom's *deployed default* DBN falls **below
naive peak-picking** (0.5927 vs 0.6333) — the structured decoder actively costs accuracy on hard
material — while PF-based decoding does not (0.6369, though only +0.0036 over peak-pick and
marginally worse on CMLt/AMLt, so it is a tie with peak-pick, not a win). Framed as *extending*
Krebs, not refuting him.

## 9. The professor's new tutorial (received today)

A 48-page derivation: the **normalized** bar-pointer model, where Böck's and Krebs's *unnormalized*
scores sit relative to it, EM for the two λ's, and two neural extensions.

- **Version A** (§8.8): input-conditioned λ = g_θ(x), λ_v = f_θ(x). **= our R3, which lost** to the
  global λ (over-stiffened).
- **Version B** (§9): neural emission `p_θ(b|z,x)` with **exact** forward–backward (latent is
  discrete, so no encoder, no ELBO, no amortized inference). **= our R4.5, which hit "emission
  domination"** — the 256-dim Gaussian emission swamped the transition.
- **§9.5 provides the fix we lacked:** factor the emission so the phase term reads *only* z:
  `logit p(b|z,x) = α(φ,m) + β_θ(x)`. β never receives φ, so a powerful audio channel can modulate
  but cannot replace the phase term. Principle: *limit what the decoder conditions on, not how big
  it is.* Plus a monitoring list (emission variance over z, posterior sharpening, transition
  identifiability, latent ablation) and the positional-encoding trap (absolute indices let a
  transformer compute `k mod 16` = metrical phase; use RoPE/relative, no absolute embedding).

Professor's stated plan: **implement and experiment on this interpretable ladder first, then extend
to the CVAE framework and compare.** So VBPM-as-VAE is explicitly deferred to a comparison study.

Caveat: §9.5's collapse taxonomy assumes a *powerful decoder over x*. Ours collapsed with a weak
decoder and no x at all, driven by sparse-BCE class imbalance — a mechanism not in his taxonomy,
and one that would survive the factored emission.

## 10. Publication landscape

| paper | status |
|---|---|
| **PF vs DBN under modern frontends** | data ready (n=217, verified); needs reframing per §8 |
| **Dithering artifact** | ready. Unsupervised EM returns λ≈40 vs madmom's hand-set 100; cause is the integer-interval lattice manufacturing spurious ±1-frame transitions; a two-component mixture kernel recovers ≈100. It is a *correction to the professor's own §8.5*. |
| **Infinite-variance latent dynamics** | conditional on the repair working; the general-ML candidate (α≤1 → n-scaling breaks sequential latents; plus the learned-emission-flattens-the-loss result). Needs a second model — the Kalman-VAE (linear-Gaussian, finite variance) scored 0.878 where the heavy-tailed VBPM never trained, which is a natural experiment already half-run but confounded. |
| **Neural DBN (professor's tutorial)** | the thesis paper. Journal, not ICASSP — the derivation *is* the contribution and 4 pages would gut it. TISMIR or TASLP. §10 (online tracking / driving a dancer) should be spun out separately. |
| **VBPM thesis** | not writable. No positive result; multi-task plumbing does not exist. |

## 11. WEDNESDAY — the first test

**Swap wrapped Cauchy for Student-t (ν ≥ 3) on the phase innovations and ν > 2 on the tempo level,
then run one training arm from the tempogram initialization.**

Prediction: accumulated phase drift falls from ~90 rad to a few rad, the gradient estimator becomes
usable, and **per-crop corr holds instead of collapsing** — which would be the first non-degrading
training run in this project's history. If it still degrades, the objective is misspecified
independently and the repair list is longer.

Second task, cheap and decision-relevant: **date when wrapped Cauchy replaced von Mises in the
codebase**, then sort past negative verdicts into "inside the defect window" (need re-running before
they constrain anything) and "outside it" (still stand). If the defect spans the failing campaigns,
most of the evidence behind "VBPM is fundamentally flawed" is uninformative rather than wrong.

Repair order after that: (1) finite-variance innovations — nothing else sticks without it;
(2) the initial-phase distribution / offset; (3) `stpool` shift-tolerant emission; (4) beat-gated
tempo (already demonstrated).
