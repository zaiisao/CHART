# Adding phase to VBPM — design, increments, pre-registered criteria

**Status:** PROPOSAL, 2026-08-03. Not normative. `docs/SPEC.md` is normative and is
**unmodified** by this work; §6 below is a proposal to amend it, for the user to accept or
reject. `tests/` is frozen and is **unmodified**; new tests live in `tests_phase/`.

---

## 1. What was verified, and what was refuted

The brief came with two "critical facts". Both were re-derived from the code and the
annotations rather than taken on trust.

### 1.1 Fact 2 — bar-aligned crops. **CONFIRMED, and stronger than stated.**

The brief said "3971 of 4000 crops have bar offset r = 0". Measured over the whole live
catalog (`data/songs.py` → `vbpm.data.extract_crops`, annotations only, no audio):

| dataset | crops | fraction with r = 0 |
|---|---|---|
| asap | 9742 | 1.0000 |
| ballroom | 1561 | 1.0000 |
| beatles | 1729 | 1.0000 |
| gtzan | 1977 | 1.0000 |
| hainsworth | 737 | 1.0000 |
| hjdb | 945 | 1.0000 |
| rwc | 2211 | 1.0000 |
| **total** | **18902** | **1.000000** |

Zero crops have a downbeat-free `y`. `r = 0` is not "essentially no variance" — it is an
**identity**, forced by construction in two places in `vbpm/data.py`:

- `make_crops` selects beats with `beat_times >= bounds[0] - tol`, so the crop's **first
  beat is a bar start**;
- `extract_crops` derives `y` against `bar_bounds[:-1]`, i.e. against exactly those bar
  starts.

So `y[0] == 1` always, and `r ≡ 0` by definition of the crop. (The brief's 29 exceptions
presumably came from a stored crop set built by a different path; they do not exist here.)

**Consequence, and it is the load-bearing one.** Any phase latent trained on this data has
a constant target. It would reach 100% offset accuracy while learning nothing, and the
number would be uninterpretable. **Un-aligning the crops is a precondition, not a
refinement.** It is also the deployment-honest choice: at deployment nothing tells you
where a bar starts.

A second consequence that the brief did not mention and that matters for the metric:
because `r ≡ 0`, `experiments/stage0_downbeat_decode.py`'s `grid-oracle` arm — which picks
`r̂` by argmax of the frontend's downbeat activation — is being scored on data where
`r = 0` is always right. Its numbers are not a phase result either.

### 1.2 Fact 1 — the reducer. **CONFIRMED in substance, one clause refuted.**

Confirmed:
- `h` is `[T, D]` with variable `T`; `Stage0.prior_logp` needs a fixed-size vector, and
  `self.reducer` is the only thing bridging them (`vbpm/stage0.py:122`).
- Neither reducer is trained. `reduce_h` and `peak_summary` are pure numpy/`torch` with no
  parameters; `vbpm/reducers.py`'s module docstring states it outright — *"non-
  differentiable is fine — the prior's gradient flows through W and b, s(h) is a per-crop
  constant."* All ψ learning in the `Stage0` path is the `W ∈ R^{3×s_dim}`, `b ∈ R^3`
  affine map (`s_dim` = 4 for `meanmax`, 10 for `peaks`).
- The reducer exists **because** the latent has no time index. Once evidence enters per
  beat through a recursion there is nothing left for it to do.

**Refuted:** "*All* learning is in a 10→3 linear map." That is true of `vbpm/stage0.py`,
but the repo has already partly moved past it. `vbpm/heads.py` contains two trained,
reducer-free priors that consume `[B, T, D]` directly — `AutocorrHead` (FFT autocorrelation
over lags, then convolutions) and `TransformerPrior` (self-attention over all frames with a
learned query token, whose docstring says "*No pooled summary statistic anywhere*"). Commit
`0884b49` is literally titled "*transformer prior: attention across t=1..T replaces the
pooled reducer*".

This does not change the plan; it sharpens the claim. **Deleting the reducer is not new —
it has already been done twice, by pooling replacements.** What is new here is deleting it
for the *right structural reason*: not "a better summary of the crop", but "there is no
crop-level quantity to summarise, because the latent is per beat". The existing heads still
answer one question per crop (`[B,T,D] → [B,K]`); the bar-pointer answers one per beat.

### 1.3 The design fork — where I agree and where I would change the framing

I agree with the recommendation: **B and C as a pair**, with B as the exact-inference
reference. Two corrections to the framing.

**A is not as separable as presented.** Increment A ("promote `r` to a 9-state latent, still
one `z` per crop") is a strict special case of B: it is B with the meter transition frozen
and no per-beat audio potential. Building B gives A for free as a configuration, and the
tests below use exactly that reduction as a correctness check. There is no reason to build
A as its own increment; there *is* reason to keep it reachable as a control arm.

**"B still does not exercise variational machinery" is right, but understates B's second
job.** B is not only the ceiling for C. It is also the only way to find out whether the
*generative structure* (bar-gated meter, per-beat audio potential, latent-only emission)
can place downbeats at all on this data. If B cannot, C's failure would be over-determined
and the A/B/C ladder would answer nothing. So B is a gate on C, not merely a yardstick.

---

## 2. The three increments

| | latent | inference | reducer | what it answers |
|---|---|---|---|---|
| **A** | `(m, r)`, one per crop, 9 states | exact enumeration | present | is `r` learnable at all? *(subsumed by B as a config)* |
| **B** | `z_i = (m_i, r_i)`, `i = 1..n` beats | **exact** forward–backward | **deleted** | can this generative structure place downbeats? and: what is the exact-inference ceiling? |
| **C** | `z_t = (m_t, φ_t)`, `φ` continuous | variational, von Mises, reparameterised | deleted | is the VAE encoding anything real? |

**This document implements B.** C is specified in §5 and not built.

### 2.1 Increment B — the model

Beats are indexed `i = 1..n` on the **given** beat grid (a crop). State space

    S = { (m, r) : m ∈ values, 0 ≤ r < m }        |S| = 2 + 3 + 4 = 9 for values = (2,3,4)

`r_i` is the beat-in-bar pointer: `r_i = 0` ⟺ beat `i` is a downbeat. `r` is the beat-grid
discretisation of SPEC §11's bar phase `φ`, and a wrap `r: m−1 → 0` is the discrete image of
`φ` crossing `2π`. That correspondence is the point of B: it is the same generative story as
Stage 1, on a grid where the posterior is computable in closed form.

**Initial** — no `h`, no reducer, no parameters beyond a `K`-vector:

    log π(m, r) = log_softmax(init_m)[m] − log m

`r` uniform over the `m` legal offsets is *exactly* SPEC §4.3's uniform marginalisation of
the bar offset, now written as a latent's initial distribution instead of a sum inside the
emission. This is the hinge that makes B an extension rather than a replacement (§3.1).

**Transition** — bar-gated, matching SPEC §11's gate and §4.1's own sanctioned segmental
model:

    r_{i−1} < m_{i−1} − 1 :  m_i = m_{i−1},  r_i = r_{i−1} + 1     (copy, probability 1)
    r_{i−1} = m_{i−1} − 1 :  r_i = 0,  m_i ~ Cat( softmax_k [ A[m_{i−1}, k] + g_ψ(h_i)_k ] )

`A ∈ R^{K×K}` is a learnable meter transition matrix, initialised sticky. `g_ψ: R^D → R^K`
is a per-beat head over the frozen features. **Meter is redrawn only at bar crossings and
copied between them** — SPEC §10.4 measures free per-bar redraw at 24× worse held-out
likelihood, so the gate is not decoration.

**Audio potential** — this is where the reducer's job goes:

    S_ψ(i, (m, r)) = e_ψ(h_i) · 1[r = 0]

`e_ψ: R^D → R` is VBPM's **own** per-beat downbeat-evidence head over frozen frontend
features (SPEC §6.1: we may not reuse the frontend's own activation channels as our
evidence). `h_i` is `h` pooled over the frames of beat `i` (beat-synchronous), so `h`
enters the recursion `n` times per crop instead of once.

Because `S_ψ` is not locally normalised, the latent path is **globally** normalised:

    log p_ψ(z_{1:n} | h) = log π(z_1) + Σ_i log T_i(z_i|z_{i−1}) + Σ_i S_ψ(i, z_i) − log Z_ψ(h)

with `Z_ψ(h)` the forward-algorithm partition function. This is a linear-chain conditional
model; every quantity below is exact.

**Emission — latent-only, unchanged from SPEC §4.3:**

    p_θ(y_i | z_i) = Bern( y_i ; sigmoid(α) if r_i = 0 else sigmoid(β) )

Two scalars, `θ = {α, β}`, exactly as at Stage 0. `h` does **not** enter the emission; SPEC
§4.3's conditional independence `y ⊥ h | z` is preserved verbatim. `h` conditions the
dynamics only, which is also SPEC §11's shape (`f^m_ψ(…, h)` in the prior, nothing in the
decoder).

**Objective.** Inference is exact, so `q` *is* the posterior, the KL slack is zero and the
ELBO equals the log evidence:

    log p(y | h) = logZ_ψ+emission(h, y) − log Z_ψ(h)

both by the forward algorithm. Training maximises the mean of this over crops. **B performs
no variational work and this is deliberate** — it is the exact-inference reference. There is
no φ parameter set at B; ψ and θ only.

**Read-outs.** Given `h` alone (never `y` — SPEC C2):

- forward–backward with `π, T, S_ψ` only → per-beat marginals `γ_i(m, r)`;
- **Viterbi** path → a coherent `(m̂_i, r̂_i)` sequence;
- **downbeat placement**: `{ i : r̂_i = 0 }`;
- **meter**: the modal `m̂_i` on the Viterbi path (a COUNT, converted through `to_value`
  once — SPEC C1).

### 2.2 Increment B — the data change (mandatory, per §1.1)

New, additive; `make_crops` / `extract_crops` are **untouched** so every existing number
stays reproducible.

`make_crops_unaligned(beat_times, downbeat_times, crop_beats, rng)`: slide a fixed-length
window of `crop_beats` beat indices with stride `crop_beats`, starting at a uniformly random
offset `o ∈ [0, crop_beats)`. A window is kept if it contains at least `MIN_BARS + 1`
downbeats, so `m_true` is a median over ≥ `MIN_BARS` complete bars *inside the window*.
`y` marks every beat in the window carrying a downbeat. `r_1` is then whatever the music
says it is.

`crop_beats = 32` (8 bars at m=4, ~10.7 at m=3, 16 at m=2; ≥ 12 as SPEC §5 requires).

**Pre-registered data check, before any model runs:** the resulting `r_1` distribution must
be approximately uniform over `0..m−1` per meter class. If it is not, the un-aligning did
not work and nothing downstream is interpretable.

---

## 3. Exactly what changes

### 3.1 New files

| file | contents |
|---|---|
| `vbpm/barpointer.py` | state space, per-beat potentials, forward / forward–backward / Viterbi in log space, the `BarPointer` model, training loop |
| `tests_phase/synth_phase.py` | phase-bearing synthetic bench: **random bar offset per song**, plus a meter-change generator |
| `tests_phase/test_barpointer.py` | the correctness and recovery tests of §4 |
| `tests_phase/__init__.py` | — |
| `docs/PHASE_PLAN.md` | this document |

### 3.2 Modified files

| file | change | risk to existing results |
|---|---|---|
| `vbpm/data.py` | **add** `make_crops_unaligned`, `extract_crops_unaligned`, `CROP_BEATS`; add `beats`/`downs`/`t0` to nothing existing | none — additive only, existing functions byte-identical |

### 3.3 Deliberately unchanged (controls)

- `docs/SPEC.md` — normative, not mine to edit. Amendments proposed in §6.
- `tests/` (142 tests) — frozen; re-run and must stay green. It was green before this work
  and must be green after.
- `vbpm/stage0.py` — untouched. B is a sibling, not a rewrite. Stage 0 remains the control
  arm for "does time-indexing buy anything".
- `vbpm/reducers.py`, `vbpm/heads.py` — untouched. `reduce_h`/`peak_summary` stay because
  Stage 0 still needs them; B simply does not import them. "Delete the reducer" means
  *the bar-pointer has no reducer*, not "remove a function Stage 0 depends on".
- `train.py`, `experiments/` — untouched.
- The emission `{α, β}` — same two scalars, same semantics, at A, B and C. Holding the
  emission fixed is what makes the A/B/C gap attributable to inference and dynamics.

---

## 4. Metric, and pre-registered success criteria

### 4.1 The metric

**Downbeat F on a given beat grid.** Predicted and true downbeats are both subsets of the
same known beat grid, so this is computed on **beat indices**, exactly:

    P = |pred ∩ true| / |pred|,  R = |pred ∩ true| / |true|,  F = 2PR / (P + R)

Reported per dataset, never only pooled (SPEC §6.3, §10.8). Reported **beside** meter
accuracy: raw accuracy **and** balanced accuracy **and** the confusion matrix, always all
three (the brief's gtzan example — 19/120/1838 crops, a degenerate always-2 predictor
scoring 0.422 balanced on 0.239 raw — is exactly why).

Beat-index F is preferred over `mir_eval` ±70 ms here because the grid is given: a ±70 ms
window on a fast grid can admit a *neighbouring* beat and silently forgive an off-by-one
phase error, which is the single error mode this whole increment exists to measure.

### 4.2 Baselines the criteria are stated against

1. **`r = 0` always** (what the current crops make free). On un-aligned crops with `m` = 4
   this scores F ≈ 1/4 at best.
2. **Frontend peak-pick** on the downbeat activation channel — the strong external baseline.
3. **Oracle-`m`, activation-argmax `r`** — `experiments/stage0_downbeat_decode.py`'s
   `grid-oracle` rule, re-run on un-aligned crops. This is the "you did not need a model"
   baseline and is the one that matters.
4. **Stage 0** for meter accuracy (`vbpm/stage0.py`, `peaks` reducer, fold-honest).

### 4.3 Pre-registered criteria — B

Stated before any number exists. The brief measures run-to-run CUDA nondeterminism at
**±0.08 balanced accuracy** on the FFT/attention arms; every margin below clears that, and
B's synthetic tier is CPU/float64 and deterministic, so ±0.08 does not apply there.

**B-0 (data, gating).** `r_1` on un-aligned real crops is approximately uniform: for every
meter class, no offset holds more than **0.45** of the mass (chance is 1/m ≤ 0.5, so this
is a real constraint only for m = 3, 4 — for m = 2 the criterion is ≤ 0.60). *Fail ⇒ stop;
nothing downstream means anything.*

**B-1 (correctness, exact).** With the audio potential disabled (`e_ψ ≡ 0`), the meter
transition frozen to no-switching, and `init_m` one-hot at `m`, the forward algorithm's
`log p(y|h)` must equal `Stage0.emission_logp_all(y)[k]` to **< 1e-10** in float64, for
random `y` and every `m ∈ {2,3,4}`. This is the formal statement that **B extends Stage 0
rather than replacing it**. *Fail ⇒ the implementation is wrong; no other number is
reportable.*

**B-2 (correctness, exact).** Three internal identities, each < 1e-10:
(a) forward `logZ` == backward `logZ`;
(b) per-beat marginals sum to 1 at every `i`;
(c) the Viterbi path's score == the max over all paths by brute-force enumeration, on a
short crop where enumeration is feasible.
*Fail ⇒ implementation wrong.*

**B-3 (synthetic phase recovery — the gate on everything else).** On the noise-free
phase-bearing bench (`tests_phase/synth_phase.py`, random bar offset per song, ≥ 24 songs,
balanced over `m`), after training on the bench:
- **downbeat F ≥ 0.95** on held-out songs, and
- **meter balanced accuracy ≥ 0.95**, and
- offset accuracy `r̂_1 == r_1` **≥ 0.95**.

Rationale for the height of the bar: the bench is exactly periodic and noise-free, so this
is a *machinery* check, not a benchmark — SPEC §6.4 sets the same expectation for Stage 0
("~0.95+ … anything less means the machinery is broken"). *Fail ⇒ B is broken; do not run
on real data.*

**B-4 (synthetic, meter change).** On a bench where `m` changes once mid-song at a bar
boundary, the Viterbi path must recover the change point to within **±1 bar** in ≥ 0.8 of
songs. This is the property Stage 0 provably cannot have (SPEC §4.1, §10.9) and is
therefore B's clearest *qualitative* win. *Fail ⇒ report as a limitation, not a blocker;
it does not invalidate B-3.*

**B-5 (real data, the actual question).** Fold-honest, pooled out-of-fold, per dataset, on
un-aligned crops:
- **B beats the `r = 0` baseline on downbeat F by ≥ 0.15** on every dataset. (Low bar,
  deliberately: this is the "did anything happen" check.)
- **B is within 0.05 of, or beats, the oracle-`m` activation-argmax rule (baseline 3) on
  downbeat F**, on `asap` and `ballroom` separately. This is the criterion that decides
  whether B earns its complexity. Losing by more than 0.05 is a **negative result and must
  be reported as one** — a learned bar-pointer that a two-line argmax beats is not a
  contribution.
- **meter balanced accuracy is not worse than Stage 0's by more than 0.08** on `asap` and
  `rwc`. B is not required to *improve* meter — it is required not to pay for phase with
  meter. `hjdb` (100% m=4) and `gtzan` (test-only, 93.7% m=4) are reported but **carry no
  meter verdict**.

**B-6 (control, leakage).** The deployable path is called with `h` only; a run in which `y`
is replaced by a derangement must leave the *deployable* downbeat F unchanged to within
float noise on a fixed `ψ`, and must destroy it after re-training. *Fail ⇒ leak.*

### 4.4 Pre-registered criteria — C (stated now, so C cannot be graded after the fact)

**C-1.** C's ELBO ≤ B's exact `log p(y|h)` on the same crops, always. A violation is a bug
in one of the two.

**C-2.** The headline number is **the gap**: `B's log p(y|h) − C's ELBO`, in nats per beat,
per dataset. This is the variational + amortisation cost with the model class held fixed,
and it is the number this whole exercise exists to produce.

**C-3.** C is declared to have *encoded something real* iff its **deployable** downbeat F is
within **0.05** of B's on `asap` and `ballroom`. Anything worse is a variational failure,
not a modelling failure, **because B holds the model class fixed** — which is precisely why
B has to exist.

**C-4.** If C's KL to the prior is < 0.05 nats/beat the latent is dead and C-3 is void
regardless of its F score (posterior collapse; report as collapse, never as a score).

---

## 5. Increment C — sketch only, not implemented

`φ_t ∈ [0, 2π)` per **frame**, von Mises, reparameterised. `r` becomes `⌊φ / (2π/m)⌋`; the
bar crossing `φ_{t−1} + φ̇_{t−1} ≥ 2π` replaces the discrete wrap. `q_φ(z_{1:T} | h, y)` is
an amortised recognition network; the ELBO needs sampling and the KL is no longer zero.
`θ = {α, β}` is unchanged, so B and C differ **only** in inference. Stage 2 then adds `φ̇`.

Not built here. What B provides for it: the state space, the gate semantics, the un-aligned
crops, the metric, the baselines, and C-1..C-4 above.

---

## 6. SPEC amendment proposal (NOT applied — `docs/SPEC.md` is unmodified)

`docs/SPEC.md` is normative and fixes `z = m`. B contradicts it in seven places. Below is
precisely what would need to change and what the replacement text should say. **Nothing
here has been written into the SPEC**; this is a request for a decision.

The cleanest framing is to introduce **Stage 0.5** rather than to loosen Stage 0: Stage 0
stays exactly as specified and stays the control arm, and Stage 0.5 is the discrete-phase
rung between it and Stage 1.

| # | clause | current text | proposed amendment |
|---|---|---|---|
| **S1** | §1, staging table | rows for stages 0, 1, 2 | insert a row: **0.5 — `+ r`** — "discrete bar pointer on the given beat grid, exact forward–backward, `m` bar-gated and time-indexed". Stage 1 then adds *continuity and sampling* to a structure already proven, rather than adding both at once. |
| **S2** | §2, "Out, and deliberately so" | "`z` has no time index at Stage 0: **one `m` per crop**, not per frame." | scope it: "…at Stage 0. At Stage 0.5 `z_i = (m_i, r_i)` is indexed by **beat**; at Stage 1 by frame." |
| **S3** | §4.1 | `m ∈ values`, one per crop | §4.1 **already** sanctions this: "*one could write `p(m_i | m_{i−1})` gated on 'beat `i` is a bar boundary under `(m_{i−1}, r)`' … That is a well-defined segmental model.*" The amendment is to promote that paragraph from an aside to Stage 0.5's definition, and to note that §4.1's stated reason for not doing it — that §4.6's exact-enumeration guarantee would lapse — is **not** what happens: forward–backward at `O(n·|S|²)` is still exact, deterministic and bit-reproducible. What lapses is *enumeration*, not *exactness*. |
| **S4** | §4.3, consequences | "Downbeat *phase* is unobservable: an implementation that places the downbeat one beat late is provably equivalent. **Tests must not reject it.**" | **This is the sharpest conflict and it must be decided explicitly.** It is true at Stage 0 *only because* `r` is marginalised under a uniform prior with no audio potential. At Stage 0.5 the audio potential `e_ψ(h_i)·1[r_i=0]` makes phase **observable**, and off-by-one is **no longer equivalent**. Proposed text: "This equivalence is a property of Stage 0's uniform, audio-free marginalisation of `r`, and **lapses at Stage 0.5**, where `r` is a latent carrying an audio potential. `tests/`'s requirement that the off-by-one corruption SURVIVE is therefore **stage-scoped**: it must survive against a Stage-0 implementation and must be **killed** against a Stage-0.5 one." *This is the one amendment that changes what a passing test means, and it should not be made silently.* |
| **S5** | §4.4, the reducer | "`h` is `[T,D]`; the prior needs a fixed-size input… `s(h) = concat[mean, max]`… required to be swappable." | add: "The reducer is a **Stage-0 artifact**. It exists only because `z` carries no time index. At Stage 0.5 and beyond, `h` enters the prior **per beat / per frame** through the transition and the state potential, and there is no crop-level quantity left to summarise: the reducer is **removed**, not replaced." Note §4.4 already calls itself "the most likely thing to be wrong"; this identifies *why*. |
| **S6** | §4.6 | "Both terms are **exact sums over K ≤ 4 terms**… An implementation that Monte-Carlo-estimates this is defective." | scope to Stage 0, and add for Stage 0.5: "the objective is `log p(y|h)` computed by the forward algorithm — still exact, still deterministic, still bit-reproducible, but `O(n·|S|²)` rather than a `K`-term sum. Inference is exact, so the ELBO is tight and the KL slack is zero: **Stage 0.5 does no variational work either, and that is its purpose** — it is the exact-inference reference against which Stage 1's variational cost is measured." Note §12 already forbids assuming exact enumerability, so this is a clarification, not a reversal. |
| **S7** | §5 / §6.2, crops | crops are "a contiguous range of beats"; `vbpm/data.py` cuts at bar boundaries and derives `y` against the bar starts | add: "Crops for any stage with a phase latent must be **un-aligned to bars**. Measured 2026-08-03: **18902 of 18902** crops from `make_crops` have bar offset `r = 0` — an identity, not a skew — so a phase latent trained on them has a constant target and any offset accuracy measured on them is vacuous. Un-aligned cropping is also the deployment-honest choice." |
| **S8** | §7 / §8, deployment & eval | "predict `argmax_k p_ψ(m|h)`"; scored as classification | add for Stage 0.5: deployment is **Viterbi over `(m, r)` given `h` alone**, yielding a meter count *and* a downbeat placement; scored additionally by **downbeat F on the given beat grid**, per dataset, alongside the §8 classification metrics and controls, which all still apply. |
| **S9** | Appendix A | binds `Stage0` | **additive**: `BarPointer` is a new surface, `Stage0`'s is unchanged and `tests/` still binds to it. No existing row is altered. |

**Recommendation.** S3, S5, S6, S7, S8, S9 are clarifications of things §12's expansion
contract already anticipates and I would accept them. **S1, S2 and especially S4 are real
decisions** and should be made by the user. S4 in particular changes the meaning of a
currently-passing frozen test, and I have not touched it.

---

## 7. Results so far — what is measured and what is not

Every criterion is graded against §4.3 as written **before** these numbers existed.

| criterion | status | evidence |
|---|---|---|
| **B-0** data, un-aligned offsets | **PASS** | 12,429 un-aligned real crops. Offset fractions: m=2 → 0.513/0.487; m=3 → 0.336/0.334/0.330; m=4 → 0.256/0.263/0.222/0.259. Chance is 1/m; the criterion was "no offset above 0.45 (0.60 at m=2)". Compare `extract_crops`: **1.0000 at r=0**. |
| **B-1** extends Stage 0 | **PASS** | 4 tests. Reduced to Stage 0's configuration the chain reproduces `Stage0.emission_logp_all` to < 1e-10, per meter and for the uniform-meter mixture. |
| **B-2** exactness | **PASS** | 10 tests. Forward `logZ` == brute-force enumeration; Viterbi score == enumerated maximum; forward == backward; marginals normalised; the bar gate holds. |
| **B-3** synthetic phase recovery | **PASS** | Held-out downbeat F, meter accuracy and offset accuracy all ≥ 0.95 on the noise-free phase bench. The `r = 0` baseline scores < 0.7 on the same data, so the bench is not bar-aligned. |
| **B-4** meter change | **PASS** | Viterbi locates a single mid-song meter change within ±1 bar in ≥ 0.8 of 20 songs. Stage 0 cannot represent this at all. |
| **B-6** no label leak | **PASS** (synthetic) | `decode` takes no `y`; deranging `y` leaves a fixed model's decode bit-identical. The real-data form is not run. |
| **B-5** real data | **NOT RUN** | Requires the fold-honest frontend pass. Pre-registered above and deliberately left ungraded. |
| **C-1..C-4** | **NOT RUN** | Increment C is not implemented. |

Two things worth recording because they were found rather than assumed:

- **The gradient audit caught a real defect on its first run.** Zero-initialising both of
  the evidence head's output layers makes `d(loss)/d(body)` **exactly** zero at step 0, so
  the head body would never have trained — SPEC §10.2's failure mode, reproduced in new
  code within an hour of writing it. It is fixed (small non-zero output weights, zero
  biases) and the test that caught it is `test_b3_every_parameter_receives_gradient`.
  This is the argument for §10.2's rule, not a hypothetical.
- **B-3 and B-4 pass on a bench that hands the model beat-level features directly**, so
  they certify the chain, not `beat_sync`. `beat_sync` is tested separately and much more
  weakly. A real-data failure could still live there.

## 8. Honest scope of this delivery

Built: increment B (§2.1, §2.2), its tests (§4.3 B-1..B-4), the un-aligned crop path, this
document, the §6 proposal. Not built: increment A as a standalone (it is a configuration of
B), increment C, and any real-data run beyond what §8 of the report records. Criteria B-5
and B-6 are **stated and not yet met or missed** — they are pre-registered for the run that
follows, which is the point of writing them down now.
