"""The "only if" half: every named Stage-P wrongness must be caught by a named property.

``test_properties_p.py`` shows the suite passes on a correct implementation. That alone is
worthless -- ``assert True`` also passes on a correct implementation. This file supplies
the missing direction by running the same properties against deliberately broken subjects
and requiring each to die, and it reports a survivor BY NAME.

Equivalent mutants are the opposite control: provably indistinguishable from correct code,
so they must SURVIVE. Killing one means a property is asserting an implementation detail
the model quotients out, which breaks the other half of the iff.
"""
from __future__ import annotations

import pytest

import mutants_p as M
import properties_p as P

# Properties that are phase-SPECIFIC: they can only fail on a model that gets bar phase
# wrong, as opposed to one that gets the plumbing wrong. Used to assert that
# ``downbeat_off_by_one`` dies for the right reason and not by accident.
PHASE_PROPERTIES = {
    "check_offset_is_the_latent_not_a_marginalised_nuisance",
    "check_emission_matches_independent_oracle",
    "check_posterior_recovers_the_true_offset",
    "check_deployable_readout_recovers_known_phase",
    "check_placement_beats_the_nulls",
    "check_shift_consistency_of_the_readout",
}


def _factory(mutant_name):
    def make(m=None, capacity=None):
        return M.build(mutant_name, m, capacity or "full")
    return make


def _first_kill(make):
    """Run the properties cheap-first and stop at the first that fires."""
    for name, fn in P.CHECKS.items():
        try:
            fn(make)
        except P.CapacityUnsupportedError:
            continue                       # not-run, not a kill
        except AssertionError as exc:
            return name, str(exc).split("\n")[0]
        except Exception as exc:           # a crash is detection too
            return name, f"{type(exc).__name__}: {exc}"
    return None, None


def _all_kills(make):
    """Run every property and return the names of all that fire."""
    killers = []
    for name, fn in P.CHECKS.items():
        try:
            fn(make)
        except P.CapacityUnsupportedError:
            continue
        except AssertionError:
            killers.append(name)
        except Exception:
            killers.append(name)
    return killers


@pytest.mark.mutation
@pytest.mark.parametrize("mutant", [m for m in M.MUTANTS if m not in M.EQUIVALENT])
def test_mutant_is_killed(mutant, record_property):
    """Assert a corruption that MUST be caught is caught by at least one property."""
    killer, why = _first_kill(_factory(mutant))
    record_property("killed_by", killer or "")
    assert killer is not None, (
        f"MUTANT SURVIVED: {mutant!r} -- {M.describe(mutant)}\n"
        f"Every property passed on an implementation known to be broken, so passing this "
        f"suite does not imply the implementation is correct. Add a property that "
        f"distinguishes it.")


@pytest.mark.mutation
@pytest.mark.parametrize("mutant", sorted(M.EQUIVALENT))
def test_equivalent_mutant_survives(mutant):
    """Assert a provably-equivalent corruption is NOT rejected."""
    killer, why = _first_kill(_factory(mutant))
    assert killer is None, (
        f"OVER-SPECIFIED: {mutant!r} is provably equivalent to correct code "
        f"({M.describe(mutant)}), but {killer!r} rejected it:\n  {why}\n"
        f"A correct implementation that made this choice would fail the suite.")


@pytest.mark.gap
def test_downbeat_off_by_one_is_killed_at_stage_p():
    """SS11 A2: the mutant Stage 0 requires to SURVIVE must be KILLED here.

    This is the single most consequential decision in the Stage-P suite, so it gets its own
    test rather than living anonymously inside the parametrised sweep, and it asserts the
    mutant dies for a PHASE reason. Dying only through, say, a gradient audit would mean
    the suite catches this bug by luck and would stop catching it after an unrelated
    refactor.

    Stage 0 marginalises the bar offset, which makes downbeat phase unobservable by design;
    ``tests/mutants.py`` proves the equivalence and ``tests/`` stays frozen. Stage P
    promotes the offset to the latent AND to the deployable read-out, so the same code is
    now simply wrong.
    """
    assert "downbeat_off_by_one" not in M.EQUIVALENT, (
        "downbeat_off_by_one is listed as EQUIVALENT at Stage P. It is equivalent at "
        "Stage 0 only, because Stage 0 marginalises the offset. Listing it here licenses "
        "exactly the error this stage exists to detect (SPEC_phase.md SS4.3, SS11 A2).")

    killers = _all_kills(_factory("downbeat_off_by_one"))
    assert killers, "downbeat_off_by_one survived the Stage-P suite"
    phase_killers = set(killers) & PHASE_PROPERTIES
    assert phase_killers, (
        f"downbeat_off_by_one died, but only via {sorted(killers)} -- none of them a "
        f"phase-specific property. It must be caught BECAUSE the phase is wrong.")


@pytest.mark.gap
def test_stage0_and_stage_p_disagree_only_about_phase():
    """The equivalence that Stage 0 asserts and Stage P denies must be exactly one mutant.

    Guards the boundary in both directions. ``global_phase_offset_one_bar`` is a shift by a
    whole bar and is the identity at BOTH stages, so a suite that kills it is not detecting
    a phase error but pattern-matching on source text. If the two sets ever differ by more
    than ``downbeat_off_by_one``, the stages disagree about something other than phase and
    the SS4.6 reductions are in doubt.
    """
    stage0_equivalent = {"downbeat_off_by_one"}       # tests/mutants.py, frozen
    only_stage0 = stage0_equivalent - M.EQUIVALENT
    assert only_stage0 == {"downbeat_off_by_one"}, (
        f"expected exactly downbeat_off_by_one to lose its equivalence at Stage P, got "
        f"{sorted(only_stage0)}")


def test_registry_covers_the_named_spec_candidates():
    """SS9 names specific mutants for both lists; each must have an entry."""
    must_be_killed = {
        "downbeat_off_by_one",              # SS9 "off-by-one in r"
        "readout_argmax_over_emission",     # SS9 "argmax over the emission"
        "psi_ignores_h",                    # SS9 "ignoring h"
        "psi_shift_invariant_summary",      # SS9 "a shift-invariant summary for psi"
    }
    must_survive = {
        "relabel_pointer_states",           # SS9 "a consistent relabelling of r states"
        "global_phase_offset_one_bar",      # SS9 "a global phase offset of one bar"
    }
    missing = (must_be_killed | must_survive) - set(M.MUTANTS)
    assert not missing, f"no mutant represents: {sorted(missing)}"
    wrong_side = must_be_killed & M.EQUIVALENT
    assert not wrong_side, f"SS9 says these must be KILLED but they are EQUIVALENT: {wrong_side}"
    assert must_survive <= M.EQUIVALENT, (
        f"SS9 says these must SURVIVE but they are not in EQUIVALENT: "
        f"{sorted(must_survive - M.EQUIVALENT)}")


def test_registry_covers_the_v1_post_mortem():
    """Each failure v1 actually shipped must keep a mutant standing for it."""
    required = {
        "psi_frozen",             # 45.3% / 50.88% of parameters at zero gradient
        "q_is_prior",             # latent dead
        "predict_leaks_y",        # deployable path reading annotations
        "emission_inert",         # latent present but decorative
        "predict_constant",       # collapse the metric was invariant to
        "elbo_sampled",           # seed noise mistaken for signal
    }
    missing = required - set(M.MUTANTS)
    assert not missing, f"no mutant represents: {sorted(missing)}"
