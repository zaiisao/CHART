# VBPM v2 — Technical Specification, Stage 0 (z = m)

> **SUPERSEDED by `docs/SPEC.md`, 2026-07-30.** This document was written in a session whose
> output is not trusted, and it is not traceable to either authoritative source (the
> `ELBO_for_DBN` paper or the professor's tutorial). Do not cite it. Kept only so the
> superseding document's provenance claims can be checked against it.

Status: DRAFT for review. Nothing implemented. Written 2026-07-30.

## 0. Why this document exists

VBPM v1 produced a day of results that reversed ~16 times. Post-mortem, the causes were not
modelling errors but **unverifiable plumbing**: a meter index stored as `bpb-1` and consumed as a
count (oracle ceiling 0.31 instead of 0.97), a `p(h|z)` term absent from the spec, 45% of parameters
receiving zero gradient, a metric invariant to the error that mattered, and a gradient that pointed
at the truth in 0.000 of crops for reasons still unresolved.

This spec is therefore written to a single standard:

> **Every number a human reads must be traceable to an assertion in code.**

Stage 0 deliberately models ONE latent (meter). It is not a simplification for tractability — it is
so that when something breaks there are few enough moving parts to bisect in minutes.

---

## 1. Scope of Stage 0

| in scope | out of scope (Stage 1+) |
|---|---|
| latent `m` = beats per bar, one value per crop | bar phase `phi` |
| exact enumeration of the posterior over `m` | log-tempo `phidot` |
| learned conditional prior `p_psi(m | h)` | continuous latents, reparameterisation, sampling |
| emission on the BEAT INDEX domain | frame-level emission, tolerance windows |
| both `h` variants (2-ch and rich) | end-to-end frontend training |

**Expansion contract.** Stage 1 adds latents by extending `LatentSpec` (§4.1) and adding factors to
`Emission` (§4.4). No file in §6 changes signature. This is checked by a test (§8, T-EXP-1).

---

## 2. Conventions — normative, asserted at runtime

Every past metric disaster in v1 traces to one of these being implicit. All are asserted by
`conventions.assert_all()`, called at the top of every script.

| # | quantity | convention | assertion |
|---|---|---|---|
| C1 | `m` | the **COUNT** of beats per bar (2, 3, 4, ...). **Never** an index. Any 0-based index is named `m_idx` and only ever appears inside a `Categorical` | `assert m in MeterSpec.values` |
| C2 | `MeterSpec.values` | explicit tuple, e.g. `(2, 3, 4)`. `K = len(values)`. Mapping both ways via `to_idx` / `from_idx` only | round-trip test T-CONV-1 |
| C3 | time | **seconds** in all public interfaces. Frames only inside `FrameGrid`, which owns `fps` | T-CONV-2 |
| C4 | frame centres | frame `i` spans `[i/fps, (i+1)/fps)`, centre `(i+0.5)/fps`. **One** definition, in `FrameGrid` | T-CONV-3 |
| C5 | beat list | strictly increasing `np.ndarray[float]`, seconds, no duplicates within 10 ms | T-CONV-4 |
| C6 | downbeats ⊆ beats | every downbeat time matches a beat time within `DOWNBEAT_TOL_S = 0.02` | T-CONV-5 |
| C7 | `y` (per-beat downbeat label) | `y[i] = 1` iff beat `i` is a downbeat. `len(y) == len(beats)` | T-CONV-6 |
| C8 | probabilities | natural log everywhere. Functions returning log-probs are suffixed `_logp` | naming lint |
| C9 | "deployable" | a quantity computable without ground-truth `beats`/`downbeats`. Enforced by `@deployable`, which raises if the call stack touched an annotation field | T-LEAK-1 |

**C9 is the one that caught v1 out repeatedly.** In v1 the encoder read the ground-truth beat train
and the headline metric used the oracle meter, both silently.

---

## 3. Statistical model

### 3.1 Observation

Stage 0 models the **per-beat downbeat pattern**, not the frame-level train. Given the beat times,
the only remaining question is which beats are downbeats — which is exactly what meter answers.

    y in {0,1}^n,  n = number of beats in the crop

This is a deliberate reduction. It removes frame jitter, tolerance windows, and beat-detection error
from Stage 0 entirely, so a failure is attributable to meter inference and nothing else. Stage 1
reintroduces the frame domain.

### 3.2 Generative model

    m       ~ p_psi(m | h)          Categorical over MeterSpec.values, K categories
    y | m   ~ p_theta(y | m)        exact, with the bar offset MARGINALISED

Emission, with offset `r` marginalised because it is a nuisance parameter of Stage 0:

    p_theta(y | m) = (1/m) * sum_{r=0}^{m-1} prod_{i=0}^{n-1} Bern( y_i ; pi_i )
    pi_i = sigmoid(alpha)   if (i - r) mod m == 0      # this beat is a downbeat
           sigmoid(beta)    otherwise

Two learnable scalars, `alpha` and `beta`. That is the ENTIRE emission. It is interpretable
(`sigmoid(alpha)` = P(labelled downbeat | truly a downbeat)), it has a closed form, and its
gradient can be verified by hand.

Rationale for marginalising `r` rather than making it a latent: `r` is not of scientific interest at
Stage 0 and marginalising it is exact and costs `sum_m m <= 9` terms. When `phi` arrives in Stage 1,
`r` is subsumed by `phi` and this marginalisation is replaced by the phase factor.

### 3.3 Inference

    q_phi(m | h, y) = Categorical(softmax(g_phi(h, y)))

### 3.4 Objective — exact, no sampling

Because `K <= 4`, the ELBO is computed by **enumeration**:

    ELBO = sum_m q_phi(m) * log p_theta(y | m)  -  KL( q_phi(m | h,y) || p_psi(m | h) )

**There is no reparameterisation, no Gumbel-Softmax, and no Monte Carlo anywhere in Stage 0.**
This is the single most important design decision in the document. It makes structurally impossible:
single-sample gradient variance, Gumbel temperature pathologies, straight-through bias, and
saturation-induced vanishing gradients — four hypotheses that consumed a day in v1.

### 3.5 The exact posterior — a first-class diagnostic

Because everything is enumerable, the true posterior is available in closed form:

    p(m | y, h) = p_theta(y|m) p_psi(m|h) / sum_m' p_theta(y|m') p_psi(m'|h)

This is not a nicety. It gives the **amortization gap** as a directly measured quantity:

    gap = KL( p(m | y,h) || q_phi(m | h,y) )     nats, per crop

v1 spent a day inferring the amortization gap indirectly. Here it is a logged scalar. If `gap ~ 0`
and accuracy is still poor, the model is wrong. If `gap` is large, the encoder is wrong. These two
failure modes were conflated for weeks in v1.

### 3.6 What Stage 0 can and cannot be blamed for

| observation | verdict |
|---|---|
| exact posterior accurate, `q` accurate | model and inference both fine |
| exact posterior accurate, `q` poor, `gap` large | **encoder/amortization** fault |
| exact posterior poor | **emission or evidence** fault; encoder is irrelevant |
| both poor but the peak-count baseline is high | the model discards information present in the input |

---

## 4. Classes

### 4.1 `MeterSpec` (frozen dataclass) — `vbpm/spec.py`
    values: tuple[int, ...] = (2, 3, 4)
    K -> int
    to_idx(m: int) -> int          # raises on unknown m
    from_idx(i: int) -> int
Single source of truth for C1/C2. **No integer meter appears anywhere without passing through this.**

### 4.2 `FrameGrid` (frozen dataclass) — `vbpm/spec.py`
    fps: float = 50.0
    n_frames: int
    centre_s(i) -> float           # (i+0.5)/fps, the ONLY definition (C4)
    to_frame(t_s) -> int
Owns every seconds<->frames conversion. Nothing else may divide by `fps`.

### 4.3 `Crop` (frozen dataclass) — `vbpm/data/crop.py`
    stem: str
    dataset: str
    h: np.ndarray            # [T, C] float32, the conditioning evidence
    beats_s: np.ndarray      # [n] float64, ANNOTATION
    downs_s: np.ndarray      # [n_d] float64, ANNOTATION
    y: np.ndarray            # [n] uint8, per-beat downbeat label (C7), derived
    m_true: int              # COUNT (C1), derived
    grid: FrameGrid
`__post_init__` asserts C5, C6, C7 and that `m_true in MeterSpec.values`. `beats_s`, `downs_s`,
`y` and `m_true` are tagged `ANNOTATION` for the `@deployable` guard (C9).

`m_true` derivation is explicit and single-source: median over bars of the count of beats in
`[d_k, d_{k+1})`. Rejects the crop if fewer than `MIN_BARS = 3` complete bars.

### 4.4 Models — `vbpm/model/`

    Emission(nn.Module)                     # p_theta(y | m)
        alpha, beta : nn.Parameter (scalars)
        logp(y, m) -> Tensor                # exact, offset marginalised (§3.2)
        logp_all(y) -> Tensor[K]            # all meters at once
    ConditionalPrior(nn.Module)             # p_psi(m | h)
        forward(h) -> Tensor[K] log-probs
    Encoder(nn.Module)                      # q_phi(m | h, y)
        forward(h, y) -> Tensor[K] log-probs
    VBPM(nn.Module)
        emission, prior, encoder
        elbo(batch) -> ElboTerms            # exact enumeration (§3.4)
        exact_posterior(batch) -> Tensor[K] # §3.5
        predict(h) -> Tensor[K]             # DEPLOYABLE: prior only, no y

**Note `predict` uses the prior, not the encoder.** At test time `y` is unavailable, so the
deployable predictor is `p_psi(m|h)`. In v1 the reported metric came from `q(z|h,b)` with `b` the
ground-truth beat train — a leak that survived weeks. Here the two are different methods with
different names and the guard enforces it.

### 4.5 Feature backends — `vbpm/features/`
    Backend(Protocol): name; n_channels; __call__(audio_or_cached) -> np.ndarray[T, C]
    BeatThis2Ch(Backend)      # n_channels = 2   (beat, downbeat activations)
    BeatThisRich(Backend)     # n_channels = 512 (transformer_blocks output)
    SyntheticPulses(Backend)  # n_channels = 2, for the bench (§7)
Selected by config string. **`h` is never mutated after construction**; any learned transform of `h`
is a module inside `Encoder`/`ConditionalPrior`, so `h` in a `Crop` always means the raw backend
output.

### 4.6 `ElboTerms` (frozen dataclass)
    recon: Tensor[B]      # sum_m q(m) log p(y|m)
    kl: Tensor[B]
    elbo: Tensor[B]
    q_logp: Tensor[B, K]
    exact_logp: Tensor[B, K]
    gap: Tensor[B]        # §3.5
`__post_init__` asserts `elbo == recon - kl` to 1e-5 and that `q_logp` normalises. A whole class of
v1 bugs (a term silently absent, a term double-counted) cannot survive this.

---

## 5. Datasets

| id | source | n | purpose |
|---|---|---|---|
| `synth` | generated (§7) | configurable, balanced by construction | the bench; must reach ceiling |
| `real_small` | `act_train.npz` / `act_eval.npz` | 147 / 79 | v1 parity check |
| `real_big` | + `fit_corpora_bt_acts.npz` x `labeled_data/*/label/*.beats` | 770 | the real experiment |

`real_big` construction is already validated: activation-annotation alignment `z = +3.2` against a
trusted-cache baseline of `+3.9` and a deliberately-mismatched control of `-0.0`.

**Alignment guard, mandatory.** `data/build_real.py` computes that `z` per song and **rejects**
any song below `ALIGN_Z_MIN = 1.0`, writing the rejects to a manifest. v1 trained on a
self-sourced corpus that turned out misaligned in ~75% of crops and the numbers stood for days.

**Splits are fold-honest and stored, not recomputed.** `data/splits.json` maps stem -> fold. No
script may derive a split. Known bpb distributions must be recorded in the manifest so imbalance is
never a surprise: `real_big` is `{2: 8, 3: 174, 4: 588}`.

---

## 6. Scripts — each does one thing and prints a verdict

    scripts/build_synth.py     --n-per-class --seed          -> data/synth/*.npz + manifest
    scripts/build_real.py      --which {small,big}           -> data/real_*/*.npz + manifest + reject list
    scripts/check_conventions.py                             -> runs §2 assertions over a dataset
    scripts/ceilings.py        --data                        -> §9 baselines table
    scripts/train.py           --config                      -> checkpoint + metrics.jsonl
    scripts/evaluate.py        --ckpt --data                 -> §9 metrics table
    scripts/diagnose.py        --ckpt --data                 -> §10 diagnostics
    scripts/bisect.py          --ckpt --data                 -> §10.3 probe ladder

Every script writes a JSON with `git_sha`, full config, and the convention-assertion result. A
metrics file without a passing convention block is invalid and `evaluate.py` refuses to read it.

---

## 7. The synthetic bench — a first-class citizen, not a toy

`build_synth.py` generates, per class in `MeterSpec.values`:
- bar period drawn from `U(bar_min_s, bar_max_s)` **independently of `m`**, so tempo carries no
  meter information (verified: `corr(m, bar_len) = -0.022`)
- equidistant beats at `bar/m`; downbeats every `bar`; **downbeats are beats** (C6)
- `h` channel 0 = Gaussian bumps at all beats, channel 1 = bumps at downbeats only,
  `sigma_s = 0.06` (≈140 ms FWHM at 50 fps)
- optional `--noise`, `--jitter-s`, `--miss-prob`, `--extra-prob` to add realism one axis at a time

Hard requirements, enforced as tests:
- classes exactly balanced
- every song distinct (v1's bench generated 160 bit-identical songs and reported "mean over 64
  eval songs" that was one number repeated — effective n = 1, fully in-sample)
- eval stems disjoint from train stems
- the peak-count baseline (§9) reaches balanced accuracy 1.000 on the noise-free setting

**Gate: no result on real data is reported until the model reaches >= 0.95 balanced accuracy on
noise-free `synth`.** v1 spent a day debugging on real data a defect that reproduces here in
four minutes.

---

## 8. Tests — `tests/`, all must pass in CI before any training run

Conventions: T-CONV-1..6 per §2.
Leakage: T-LEAK-1 `@deployable` raises when an annotation field is touched; T-LEAK-2 `predict()`
never reads `y`.
Emission: T-EM-1 `logp_all` normalises over `y` for `n <= 6` by brute force; T-EM-2 `logp` is
invariant to a cyclic shift of `y` by `m` (offset marginalisation); T-EM-3 analytic gradient of
`alpha`/`beta` matches autograd; T-EM-4 with `alpha -> +inf, beta -> -inf`, `argmax_m logp` equals
the true `m` for synthetic `y`.
ELBO: T-ELBO-1 `elbo == recon - kl`; T-ELBO-2 `gap >= 0`; T-ELBO-3 with `q` set to the exact
posterior, `gap == 0` and the ELBO equals `log p(y|h)`; T-ELBO-4 no `torch.randn`/`rand`/`gumbel`
is reachable from `elbo()` (AST check — enforces §3.4).
Gradients: T-GRAD-1 every parameter in `named_parameters()` receives a non-zero gradient from
`elbo().backward()`, else the test names it. **v1 had 45% of parameters at exactly zero gradient
for weeks.** T-GRAD-2 the gradient on `q_logp` increases the true meter's log-prob for synthetic
data with a converged emission (v1's was aligned in 0.000 of crops and nobody noticed).
Data: T-DATA-1 `m_true` derivation matches a hand-computed fixture; T-DATA-2 the alignment guard
rejects a deliberately shifted crop; T-DATA-3 balanced-by-construction holds for `synth`.
Expansion: T-EXP-1 adding a dummy latent to `LatentSpec` leaves all §6 signatures unchanged.

---

## 9. Metrics and baselines — reported together, always

Primary: **balanced accuracy** of `argmax predict(h)` (deployable). Raw accuracy is reported but
never primary: `real_big` is 76% one class, so raw accuracy has ~0.24 of headroom while balanced has
0.67, and in v1 a uniform posterior was laundered into "87% correct".

Also reported, every time, on both train and eval:
- `NLL(m_true)` under `q` and under `predict`
- amortization `gap` (§3.5), mean and 90th percentile
- confusion matrix and the count of distinct predicted classes (v1's model predicted **one** class
  for all 79 eval songs while scoring 0.873 raw)
- **train-eval gap** on every metric. v1's evidence head hit 1.000 train / 0.340 eval and I reported
  only eval for an hour.

Mandatory baselines in the same table, from `ceilings.py`:
| baseline | what it establishes |
|---|---|
| chance = 1/K | floor |
| majority class | the shortcut a collapsed model finds |
| peak-count estimator on `h` | information present in the evidence, no learning |
| exact posterior with a converged emission | ceiling of the model class |
| supervised classifier on `h` (labelled as supervised) | ceiling of the feature set |

A result is only reported as progress if it **exceeds the majority baseline on balanced accuracy**.

---

## 10. Diagnostics — `diagnose.py`

10.1 Collapse detectors, logged every eval: distinct predicted classes; `sd` across crops of
`q_logp`; mean max softmax; and the **offset-to-signal ratio** — between-class variation of the
logits vs the class-independent offset. v1's was 167x at init and 12,000,000x after training, and
that single number would have found the bug on day one.

10.2 Term accounting: `recon`, `kl`, `gap` in nats, plus `d recon / d m` — the reconstruction gain
from the true meter versus the most confusable alternative, against the KL cost of committing. v1
measured 89 nats gain vs 1.386 cost and still collapsed, which is what made the collapse
interesting rather than expected.

10.3 Probe ladder (`bisect.py`): a linear probe's balanced accuracy at each stage —
`h` -> encoder trunk -> pooled statistic -> `q_logp`. Localises where class information dies, or
proves it does not. In v1 this took minutes and overturned three days of hypotheses.

---

## 11. Open questions for review

1. **Is the per-beat `y` reduction too strong?** It removes beat-detection error, which is real. I
   claim it belongs in Stage 1; a reviewer may reasonably want it at Stage 0.
2. **`predict(h)` uses the prior only.** That is the honest deployable path, but it means the encoder
   is trained and never deployed — the aggregated-posterior gap the tutorial calls structural. An
   alternative is `q_phi(m | h, y_hat)` with `y_hat` from the frontend; that is deployable but
   introduces frontend error into Stage 0. Not decided.
3. **`MeterSpec.values = (2,3,4)`** excludes 6/8 and 7/4, which exist in the corpora (one eval song
   is bpb 6). v1 silently clamped them to 4. Options: extend `values`, or record them as rejects in
   the manifest. I favour rejecting, visibly.
4. **One `m` per crop** assumes meter is constant within a crop. Real corpora have within-song
   changes (~1,963 files in Lakh). Fine for Stage 0; needs stating.
5. **Class imbalance.** Balanced resampling changes `p_data`, not the objective, so it is faithful —
   but on `real_big` it oversamples 8 bpb=2 songs heavily. I suggest reporting balanced-sampled and
   natural-sampled runs side by side rather than choosing.
6. **Rich vs 2-channel `h`.** The 2-channel downbeat activation has a half-bar artifact measured at
   0.695 of the true peak, which caps hand-written estimators. The rich 512-dim output may not. This
   is the first experiment I would run after the bench gate passes.
