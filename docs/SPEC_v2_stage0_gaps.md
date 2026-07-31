# SPEC_v2_stage0 — completeness audit

> **SUPERSEDED, 2026-07-30.** This audits `SPEC_v2_stage0.md`, which is no longer
> authoritative: it was written in a session that was hallucinating, and the 197
> spec-derived contract tests it was a to-do list for have been deleted. The north star is
> the professor's tutorial (`VAEBPM_fin.pdf`). Read the A- and C-group entries below as
> "gaps in a document we are not going to finish", not as a work plan.
>
> **What survives is the B group**, because those are mathematical results about the
> objective rather than commentary on the text, and each is now pinned by an oracle test
> that runs on every invocation:
>
> | finding | test |
> |---|---|
> | G-08: §3.5's `gap` is KL(p‖q); the bound's slack is KL(q‖p) — different numbers | `test_spec_gap_is_not_the_elbo_gap` |
> | T-EM-2 is false unless `m` divides `n` | `test_TEM2_as_written_is_false_when_m_does_not_divide_n` |
> | §10.1's collapse formula has no unique reading | `test_m_true_even_bar_count_tiebreak_is_a_choice_not_a_spec` |
>
> Those tests are the durable form of this document. If the rest of the file is ever
> deleted, nothing checkable is lost.

Companion to `docs/SPEC_v2_stage0.md` (DRAFT, 2026-07-30). Written while turning §8 into
an executable suite (`tests/v2/`), so every entry below is something that blocked writing
a test or forced the test to invent an API.

**Verdict: the model is fully specified; the plumbing is not.** §3 is implementable from
the text alone — `tests/v2/reference.py` is a working transcription of it and its 104
self-tests pass. Everything the ELBO touches on the way in and out (batching, the
annotation guard, the alignment statistic, configs, checkpoints) is named but not defined.

Three statements are not merely missing but **wrong or ambiguous as written** (B-group). A
faithful implementation of the text would produce a test that fails on correct code, a
diagnostic that is not the quantity its own §3.6 table interprets, and a collapse detector
with no unique formula.

| group | meaning | count |
|---|---|---|
| A | blocks implementation: a decision is required before code can be written | 8 |
| B | specified incorrectly or ambiguously: implementing the text gives a wrong result | 4 |
| C | referenced but given no API: mechanical to add once decided | 13 |

---

## A. Blocking

**G-01 · `LatentSpec` is never defined.** §1's expansion contract and T-EXP-1 are both
written against it; §4.1 defines `MeterSpec` and the document names `LatentSpec` nowhere
else. Since the expansion contract is the whole justification for Stage 0 modelling one
latent, this is the most load-bearing omission. *Needs:* the class, its relationship to
`MeterSpec` (does `MeterSpec` become an entry in it?), and the extension operation
T-EXP-1 exercises.

**G-02 · The `@deployable` mechanism is unimplementable as described.** C9 says it "raises
if the call stack touched an annotation field". A decorator cannot observe attribute
access in frames below it without either a proxy object or an audit hook. §4.3 says the
annotation fields are "tagged ANNOTATION" but gives no machine-readable tag. *Needs:* the
tagging mechanism (class-level set or dataclass field metadata) and the enforcement
mechanism (guarded proxy is the workable option). Tests assume either tag form and require
the guard to catch a leak one frame down.

**G-03 · No batch type.** §4.4 has `elbo(batch)` and `exact_posterior(batch)`; `batch` is
never defined. This is not cosmetic: crops have different beat counts `n`, and §4.6's
`ElboTerms` is batched (`Tensor[B]`, `Tensor[B, K]`) while §4.4's `Emission.logp_all(y)`
returns `Tensor[K]` with no batch dimension. *Needs:* a `Batch` dataclass or `collate`,
the padding convention for ragged `y`, and the mask that keeps padding out of the
likelihood. A wrong mask here silently adds fake beats to every crop.

**G-05 · The alignment statistic `z` is never defined.** §5 mandates
`ALIGN_Z_MIN = 1.0` and quotes `z = +3.2`, `+3.9`, `-0.0`, but not what `z` measures or
against which null. This is the guard standing between the project and a repeat of the
misaligned-corpus incident, so it cannot be left to the implementer's taste. *Needs:* the
statistic and its null. `reference.py` pins one candidate (mean beat-channel activation at
annotated beat frames, standardised against random circular shifts) purely so T-DATA-2 can
run.

**G-06 · `m_true`'s median has no tie-break.** §4.3 says "median over bars of the count of
beats"; an even number of bars can give 3.5, which C1 forbids as a meter. *Needs:* a stated
rule. The oracle rounds half up and `test_m_true_even_bar_count_tiebreak_is_a_choice_not_a_spec`
pins that choice so it is visible rather than incidental.

**G-18 · No config schema.** §6 has `train.py --config` and §6 requires the "full config"
in every JSON sidecar, but nothing says what is in it: optimiser, learning rate, epochs,
batch size, stopping rule, device, seed, backend selection, and — per open question 5 —
whether sampling is balanced or natural. Two runs cannot be compared until this is fixed.

**G-19 · No checkpoint format.** `evaluate.py`, `diagnose.py` and `bisect.py` all take
`--ckpt`; nothing says what `train.py` writes. At minimum it must carry the `MeterSpec`
values and the backend name, or a checkpoint can be silently evaluated under a different
meter set than it was trained with — a C1-class failure with a new coat of paint.

**G-21 · Backends and datasets are two unconnected stories.** §4.5 backends map
`audio_or_cached -> [T, C]`; §5 datasets are `.npz` files (`act_train.npz`,
`fit_corpora_bt_acts.npz`). Which one populates `Crop.h`? Where do `fps` and `n_frames`
come from for a cached array? Under the project's no-caches decision this matters: the
frontends are meant to be the only activation producer, and an `.npz` path reintroduces a
second, uncertified pipeline unless the spec says how the two relate.

---

## B. Wrong or ambiguous as written

**G-07 · T-EM-2 is false unless `m` divides `n`.** The test says `logp` is invariant to a
cyclic shift of `y` by `m`. A cyclic shift by `m` preserves each beat's residue class mod
`m` only when `m | n`; otherwise wrap-around remaps residues and the likelihood changes.
Counterexample in `test_TEM2_as_written_is_false_when_m_does_not_divide_n` (n=8, m=3).
*Fix:* add the precondition. The stronger true statement is worth having instead — with
`m | n`, marginalising the offset makes `logp` invariant to **any** cyclic shift, which is
a sharper probe of the same property and is what the suite tests.

**G-08 · §3.5's `gap` is not the ELBO's gap.** §3.5 defines
`gap = KL(p(m|y,h) ‖ q_phi)`, the forward KL. The bound's slack is the reverse KL:
`log p(y|h) − ELBO = KL(q ‖ p)`. Both vanish together and both are defensible
amortization diagnostics, but they are different numbers — a factor of ~4 apart in the
test case — and §3.6 reads "gap large" as a threshold. Forward KL is mass-covering and
reverse is mode-seeking, so they disagree precisely when `q` is bad, which is when the
diagnostic gets consulted. *Fix:* say which one, or log both (the oracle returns both).

**G-09 · The offset-to-signal ratio has no formula.** §10.1 says "between-class variation
of the logits vs the class-independent offset" and quotes 167× and 1.2e7. A
class-independent per-crop offset does not affect the softmax at all, so read literally the
denominator is a quantity with no effect on predictions. The collapse it is meant to catch
is a *crop-independent class bias* dominating the *crop-dependent* signal. The oracle pins
that reading; the spec should confirm it, since the number is offered as the one that
"would have found the bug on day one".

**G-24 · "Converged emission" is undefined.** It appears in T-GRAD-2 and again in §9's
ceiling baseline ("exact posterior with a converged emission"). Fitted how, on which
split, to what tolerance? As a *ceiling* this matters: fitting `alpha`/`beta` on the eval
split makes the ceiling optimistic, and nothing in §9 forbids it.

---

## C. Referenced but given no API

| id | where | what is missing |
|---|---|---|
| G-04 | T-ELBO-3 | no way to override `q` in `elbo()`, so "set `q` to the exact posterior" cannot be expressed against the API |
| G-10 | §10.3 | `Encoder` exposes no intermediate taps, so the probe ladder (`trunk`, `pooled`) cannot be built |
| G-11 | §2, §4.3, §5 | `MIN_BARS`, `DOWNBEAT_TOL_S`, `ALIGN_Z_MIN` and the `conventions` module itself have no home in §4's file list |
| G-12 | §9 | no `@deployable`-guarded function on the reported-metric path; the guard exists but nothing says the headline number goes through it |
| G-13 | §4.5 | "selected by config string" implies a registry; none is specified |
| G-14 | §7 | the ≥0.95 bench gate is prose; nothing mechanically stops real-data numbers being printed |
| G-15 | §5 | `data/splits.json` format (stem → fold) is named but not shown |
| G-16 | §9 | the "exceeds majority to count as progress" rule is prose only |
| G-17 | §1 | `Emission` has no factor seam for Stage 1 to extend |
| G-20 | §4.4 | pooling over variable `T` in `Encoder`/`ConditionalPrior` is unspecified — and it is exactly what §10.3 probes |
| G-22 | §4.5 | `__call__(audio_or_cached)` — the union is not defined |
| G-23 | §6 | no `Dataset`/iteration contract between the `.npz` files and `train.py` |
| G-25 | §4.4, §9 | `predict(h) -> Tensor[K]` is single-crop; the batched path behind balanced accuracy is undefined (follows from G-03) |

---

## Non-gap notes

- **Package collision.** The spec's paths (`vbpm/spec.py`, `vbpm/model/`) name a package
  that already exists as v1 at `vbpm_campaign/vbpm/`. Two importable `vbpm` packages whose
  precedence depends on the working directory is a bug waiting to happen; pick a distinct
  name or make v1's non-importable.
- **§3 is genuinely complete.** The emission, the enumerated ELBO, the exact posterior and
  the marginalised offset are all implementable from the text with no guesswork — that is
  what `reference.py` demonstrates. The contrast with the plumbing is stark and worth
  preserving in the next revision.
- **Open questions 1–6 are decisions, not gaps.** They are excluded from the counts above.
  Question 3 (6/8 and 7/4) does interact with G-06: whichever way it is decided,
  `derive_m_true` needs the corresponding branch.
