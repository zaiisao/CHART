# tests_stage_p — the acceptance suite for Stage P (`docs/SPEC_phase.md`)

    PYTHONPATH=$PWD /disk4/anaconda3/envs/vbpm/bin/python -m pytest tests_stage_p -q

**102 tests, all green — against an oracle, not against a model.** No Stage-P
implementation exists yet. That is deliberate: the suite and the reference were authored
from the spec *before* the model, so a later implementation is judged by a contract it did
not get to write.

`tests/` (Stage 0) is frozen and is **not** edited by anything here. Where Stage P and
Stage 0 disagree, the disagreement is stage-scoped and lives in this directory's own
registry — `SPEC_phase.md` §9 requires exactly that.

## Layout

| file | role |
|---|---|
| `reference_p.py` | numpy oracle; brute force, **never imports `vbpm`** |
| `bench_p.py` | the §6.5 synthetic bench: known phase, slip, mid-crop meter change |
| `subject_p.py` | the `StageP` interface + `oracle()`, the known-correct subject |
| `properties_p.py` | 29 properties of the fitted system, cheap-to-expensive |
| `mutants_p.py` | 25 named corruptions + the provably-`EQUIVALENT` set |
| `test_reference_p_selftest.py` | 43 tests making the oracle trustworthy *before* it judges |
| `test_properties_p.py` | the "proper ⇒ passes" half |
| `test_mutation_registry_p.py` | the "passes ⇒ proper" half |

## The iff, and how each half is earned

**proper ⇒ passes** — all 29 properties run against `subject_p.oracle()`. A failure there
means the *test* is wrong. The 4 `EQUIVALENT` mutants guard the same direction: they must
**survive**.

**passes ⇒ proper** — you cannot get this by writing more assertions, because the set of
wrong programs is not enumerable. What you can do is fix a set of wrongnesses and prove
each is caught. 21 must-be-caught mutants, each killed by a named property:

| mutant | first killed by |
|---|---|
| `downbeat_off_by_one` | `check_offset_is_the_latent_not_a_marginalised_nuisance` (+5 more phase properties) |
| `emission_marginalises_r` | `check_offset_is_the_latent_not_a_marginalised_nuisance` |
| `emission_fixed_at_zero` | `check_offset_is_the_latent_not_a_marginalised_nuisance` |
| `emission_inert` | `check_offset_is_the_latent_not_a_marginalised_nuisance` |
| `readout_uses_pointer_not_offset` | `check_shift_consistency_of_the_readout` |
| `readout_argmax_over_emission` | `check_shift_consistency_of_the_readout` |
| `psi_ignores_h` | `check_every_parameter_receives_gradient` |
| `psi_shift_invariant_summary` | `check_every_parameter_receives_gradient` (+3 phase properties) |
| `psi_offset_bias` | `check_p2_with_zero_slip_reduces_exactly_to_p1` (+4 phase properties) |
| `psi_frozen` | `check_every_parameter_receives_gradient` |
| `predict_leaks_y` | `check_predict_is_annotation_blind` |
| `predict_constant` | `check_shift_consistency_of_the_readout` |
| `kl_flipped` | `check_elbo_lower_bounds_the_evidence` |
| `elbo_sign` | `check_elbo_lower_bounds_the_evidence` |
| `elbo_sampled` | `check_bound_is_tight_exactly_when_q_is_the_posterior` |
| `q_is_prior` | `check_every_parameter_receives_gradient` |
| `q_ignores_h` | `check_every_parameter_receives_gradient` |
| `posterior_ignores_prior` | `check_posterior_is_bayes` |
| `slip_ignored` | `check_p2_slip_parameters_receive_gradient` |
| `transition_unnormalised` | `check_p2_transition_normalises` |
| `p2_forgets_prior_partition` | `check_p2_with_zero_slip_reduces_exactly_to_p1` |

## The one decision that matters

`downbeat_off_by_one` is in Stage 0's `EQUIVALENT` set and `tests/` requires it to
**survive**. Here it must be **killed**. Both are right, and the difference *is* the stage:
Stage 0 marginalises the bar offset, which makes downbeat phase unobservable by design, so
pinning it there would be over-specification. Stage P promotes that offset to the latent
and to the deployable output, so an emission whose downbeat lands one beat late is simply
wrong (§4.3, §11 A2). It has its own test, `test_downbeat_off_by_one_is_killed_at_stage_p`,
which additionally requires it to die for a **phase** reason rather than incidentally.

The control on that decision is `global_phase_offset_one_bar` — a shift by exactly one
*bar*, which is the identity at both stages. A suite that killed both would be
pattern-matching on source text, not detecting a phase error.

## What this suite does NOT prove

- **That phase is recoverable from real music.** Everything runs on the synthetic bench,
  where `h` carries downbeat bumps by construction. §10.6 is explicit: such a control
  proves the crop/label/prior/read-out chain is sound and **cannot** separate "the frontend
  is deaf" from "the annotation's metrical level differs from the audio's".
- **That shift-invariant heads score at chance** (§4.4's table, §8.3's P-0). Position leaks
  through real "shift-invariant" heads at small amplitude, and crops at different offsets
  are different audio *windows*, not cyclic shifts of one signal — so shift-invariance is
  not even the governing property. P-0 is an empirical expectation to be measured, not a
  theorem. The `psi_shift_invariant_summary` mutant is a claim about *that pooling on this
  bench*, enforced against the oracle; it is not a proof about `AutocorrHead`.
- **Anything about P2's loss.** §4.5 defines an encoder for the static P1 latent only, and
  no chain-valued encoder is defined anywhere in the spec. P2 is asserted here as exact
  **inference** only — forward recursion, brute-force agreement, transition normalisation,
  and the ε = 0 reduction. Building a P2 objective would mean inventing a design decision
  inside the acceptance suite.
- **That the amortization gap is zero.** Genuinely present in this variant; asserting it
  away would fail correct code.

## Where the spec was ambiguous, and what was chosen

**`r` is the index of the first downbeat, not "the bar pointer at the crop's first beat".**
§4.1 says both, and they are negations of each other. The formula `i ≡ r (mod m)`, §6.2's
`r_true`, and `vbpm/fitting.py:emission_counts` (`slots = arange(r, n, m)`) all agree on
the first reading; the prose is the odd one out, three sources to one. The confusion has
its own mutant (`readout_uses_pointer_not_offset`) because it is invisible on bar-aligned
data — the two agree exactly when `r = 0`, which was ~99% of Stage-0 crops.

**Slip is two free parameters.** §4.1 writes "with probability 1 − ε" while naming
`ε_hold` and `ε_skip`, and never relates them. Implemented as advance mass
`1 − ε_hold − ε_skip`, the only reading under which the three branches normalise, and
normalisation is asserted rather than assumed.

**ψ is a per-beat potential, summed along each offset's comb to give P1's logits.** §4.4
mandates the per-beat form at P2; extending it to P1 makes the ε = 0 reduction exact *in ψ
as well as in the likelihood*, so "P2 reduces to P1" becomes a statement about the whole
model. It also makes the read-out shift-equivariant, which is what §8.4 demands.

**ψ carries no per-offset bias.** Not stated by the spec, forced by it: §1 says `r` is
uniform over `0..m−1` by construction, so a learned offset bias models nothing and breaks
equivariance. `psi_offset_bias` is the mutant.

**Crops must span whole bars** (`n % m == 0`) — see the next section.

## Where `SPEC_phase.md` is wrong

1. **§8.3 P-0's chance level is wrong unless `n % m == 0`.** The number of downbeat slots
   under offset `r` is `len(range(r, n, m))`, which *depends on r* otherwise. At `m = 4`:
   `n = 13 → [4,3,3,3]`, `n = 14 → [4,4,3,3]`, `n = 15 → [4,4,4,3]`. In each case a
   completely position-blind summary that sees only the slot count scores **0.500, not
   0.250, with no leak at all** — so P-0, the gate every other Stage-P number depends on,
   would fire on a clean model. `count_partition_chance()` derives the real chance level and
   `assert_whole_bars()` enforces the precondition at the point crops are cut.
2. **§4.1's "bar pointer at the crop's first beat" contradicts its own formula.** See above.
3. **§4.1's slip kernel does not normalise as written.** See above.
4. **§8.2/§8.3 P-3's uniform-`r` null cannot be scored on placement F.** A model that
   marginalises the offset emits no downbeat *locations*, so it has no placement. Majority-`r`
   and random-`r` are used instead; the uniform-`r` null remains correct for NLL, where it
   is exactly Stage 0's emission.
5. **§8.3 P-4 is incoherent.** At Stage P the prior *is* the deployable path (C2), so
   "replace the posterior with the prior and measure degradation" measures the train/deploy
   gap, not latent use — a large value is the failure mode, not evidence of success. No
   property implements it; it needs redefinition.
6. **§4.4 contradicts itself.** "No pooled crop-level summary anywhere on the generative
   path" is false at P1, where `logits_ψ(h)` is necessarily crop-level. The reducer dies
   structurally only at P2.
7. **§4.4's "provably shift-invariant" table is overstated**, and §5's "P1 is
   deterministic" holds for the closed-form path only (attention arms carry ±0.08 of CUDA
   nondeterminism).
8. **§10.2's ~64% is wrong**: the `m_true == 4` restriction retains 9,932/18,902 = **52.5%**,
   and retains 93%/100%/85% of gtzan/hjdb/beatles rather than "all of" them. §10.1's figures
   do reproduce.

## Spec clauses with no property behind them

Gaps, listed so they are legible rather than assumed covered: §6.1's
`extract_crops_unaligned` audit (this suite cuts its own crops and never calls it), §6.3
rich-feature checkpoint fragility, §6.4 fold-honest splits and per-dataset reporting, §8.2's
Stage-0-on-un-aligned-crops and peak-picking baselines, §8.3's P-0/P-2/P-4/P-5 (all require
real data or a redefinition), §5's crop-length and free-bits prohibitions, and §2.1's
corpus-restriction condition. All of these are properties of a *training campaign on real
data*, not of a model given a crop, and none can be settled by an oracle.
