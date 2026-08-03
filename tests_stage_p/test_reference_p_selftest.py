"""Self-tests making ``reference_p`` trustworthy BEFORE it judges any implementation.

These take plain numpy arguments and never touch a subject. An oracle that is only checked
against the thing it is meant to be checking is not an oracle.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

import bench_p as B
import reference_p as R


# ----------------------------------------------------------------------------------
# numerics
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("x", [-40.0, -3.0, -0.5, 0.0, 0.5, 3.0, 40.0])
def test_log_sigmoid_matches_naive_where_naive_is_safe(x):
    """The stable form must agree with the definition in the range the definition works."""
    if abs(x) < 30:
        assert abs(R.log_sigmoid(x) - math.log(1 / (1 + math.exp(-x)))) < 1e-12
    assert abs(math.exp(R.log_sigmoid(x)) + math.exp(R.log1m_sigmoid(x)) - 1.0) < 1e-12


def test_logsumexp_handles_all_negative_infinity():
    """An all-zero-mass vector must give -inf rather than a nan."""
    assert R.logsumexp([-math.inf, -math.inf]) == -math.inf
    assert abs(R.logsumexp([0.0, 0.0]) - math.log(2)) < 1e-12


# ----------------------------------------------------------------------------------
# SS4.1 conventions
# ----------------------------------------------------------------------------------
def test_offset_and_pointer_are_mutual_inverses():
    """The P1/P2 seam: the two representations must round-trip for every state."""
    for m in (2, 3, 4, 5):
        for r in range(m):
            assert R.offset_from_pointer(R.pointer_from_offset(r, m), m) == r


def test_downbeat_convention_matches_emission_counts_slots():
    """``is_downbeat`` must agree with ``vbpm/fitting.py``'s ``slots = arange(r, n, m)``.

    This pins the convention against the SHIPPED code rather than against the spec prose,
    which contradicts itself (SS4.1 calls r the pointer at the first beat, then writes
    ``i == r (mod m)``, which is its negation).
    """
    for m in (2, 3, 4):
        for n in (m * 2, m * 3):
            for r in range(m):
                slots = set(range(r, n, m))
                mine = {i for i in range(n) if R.is_downbeat(i, r, m)}
                assert mine == slots, f"m={m} n={n} r={r}: {sorted(mine)} != {sorted(slots)}"


# ----------------------------------------------------------------------------------
# SS4.3 emission
# ----------------------------------------------------------------------------------
def test_conditional_emission_normalises_over_y(theta):
    """``sum_y p(y|r) == 1`` for every offset, brute force over ``{0,1}^n``."""
    m = R.STAGE_P_M
    for n in (m, 2 * m):
        for r in range(m):
            total = sum(math.exp(R.offset_loglik(y, r, m, **theta))
                        for y in itertools.product((0, 1), repeat=n))
            assert abs(total - 1.0) < 1e-9, f"n={n} r={r}: {total}"


def test_emission_peaks_at_the_generating_offset(theta):
    """A y generated at offset r must score highest at r. The sensitivity half."""
    m = R.STAGE_P_M
    for r in range(m):
        y = R.downbeats_from_offset(4 * m, r, m)
        lp = R.offset_loglik_all(y, m, alpha=4.0, beta=-4.0)
        assert int(lp.argmax()) == r, f"y at offset {r} peaked at {int(lp.argmax())}"


def test_marginalised_emission_is_shift_invariant_when_m_divides_n(theta):
    """Contrast the two stages' behaviour under a cyclic shift.

    Stage 0's marginalised emission is INVARIANT to any cyclic shift when ``m | n``; Stage
    P's conditional emission is EQUIVARIANT, permuting rather than staying fixed. Both
    halves are asserted, because it is the pair that distinguishes the stages.
    """
    m = R.STAGE_P_M
    n = 4 * m
    y = R.downbeats_from_offset(n, 1, m)
    base = R.stage0_emission_logp(y, m, **theta)
    for k in (1, 2, 3, 7):
        shifted = np.roll(y, k)
        assert abs(R.stage0_emission_logp(shifted, m, **theta) - base) < 1e-9
        rolled = R.offset_loglik_all(shifted, m, **theta)
        want = np.roll(R.offset_loglik_all(y, m, **theta), k)
        assert np.allclose(rolled, want, atol=1e-9), (
            "the conditional emission must PERMUTE under a cyclic shift, not stay fixed")


# ----------------------------------------------------------------------------------
# the crop-length arithmetic behind SS8.3's chance level
# ----------------------------------------------------------------------------------
@pytest.mark.gap
@pytest.mark.parametrize("n,expected", [(12, 0.25), (13, 0.5), (14, 0.5), (15, 0.5),
                                        (16, 0.25), (20, 0.25)])
def test_count_partition_chance_is_not_always_one_over_m(n, expected):
    """SS8.3's 0.25 is WRONG unless the crop spans whole bars.

    A summary seeing only the number of downbeat slots per offset already separates the
    offsets into groups when ``n % m != 0``: at ``m = 4``, ``n = 13`` gives ``[4, 3, 3, 3]``
    and such a summary scores 0.500 with no leak whatsoever. Pinned here so a Stage-P
    leak-detector gate can never silently hardcode ``1/m``.
    """
    assert R.count_partition_chance(n, R.STAGE_P_M) == expected
    counts = R.downbeat_slot_counts(n, R.STAGE_P_M)
    assert (len(set(counts)) == 1) == (n % R.STAGE_P_M == 0)


def test_assert_whole_bars_rejects_partial_bars():
    """The precondition must actually raise, not merely warn."""
    R.assert_whole_bars(16, 4)
    with pytest.raises(ValueError, match="whole bars"):
        R.assert_whole_bars(13, 4)


# ----------------------------------------------------------------------------------
# SS4.1 / SS9 chain inference
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("eps", [(0.0, 0.0), (0.1, 0.05), (0.03, 0.0), (0.25, 0.25)])
def test_transition_matrix_is_row_stochastic(eps):
    """The three slip branches must form a distribution from every state."""
    T = R.transition_matrix(R.STAGE_P_M, *eps)
    assert np.allclose(T.sum(axis=1), 1.0, atol=1e-12)
    assert (T >= 0).all()


def test_transition_matrix_rejects_illegal_slip():
    """Normalisation is asserted, not assumed -- SS4.1 never states the relation."""
    with pytest.raises(ValueError):
        R.transition_matrix(4, 0.7, 0.7)
    with pytest.raises(ValueError):
        R.transition_matrix(4, -0.1, 0.0)


@pytest.mark.parametrize("eps", [(0.0, 0.0), (0.12, 0.06), (0.3, 0.1)])
def test_forward_backward_matches_brute_force_exactly(eps, rng, theta):
    """SS9: forward-backward against enumeration for ``n <= 8``, on evidence AND marginals.

    The subject-facing property compares only the log evidence, because the pointer basis
    is a free internal choice there. Here both implementations are the reference's own, so
    the marginals and the pairwise marginals are compared too -- which is what makes the
    recursion itself trustworthy.
    """
    m = R.STAGE_P_M
    for _ in range(4):
        n = int(rng.integers(2, 9))
        y = rng.integers(0, 2, n)
        init = R.log_normalise(rng.normal(size=m))
        fb = R.p2_forward_backward(y, init, m, eps_hold=eps[0], eps_skip=eps[1], **theta)
        bf = R.p2_brute_force(y, init, m, eps_hold=eps[0], eps_skip=eps[1], **theta)
        assert abs(fb["log_evidence"] - bf["log_evidence"]) < 1e-9
        assert np.allclose(np.exp(fb["marginals"]), np.exp(bf["marginals"]), atol=1e-9)
        assert np.allclose(np.exp(fb["pairwise"]), np.exp(bf["pairwise"]), atol=1e-9)


def test_chain_logz_brute_matches_forward_backward(rng, theta):
    """The potential-based enumeration used by the subject property is itself correct."""
    m = R.STAGE_P_M
    with np.errstate(divide="ignore"):
        logT = np.log(R.transition_matrix(m, 0.1, 0.05))
    for _ in range(4):
        n = int(rng.integers(2, 8))
        node = rng.normal(size=(n, m))
        brute = R.chain_logz_brute(node, logT)
        # forward recursion, written out here independently of reference_p's own
        a = -math.log(m) + node[0]
        for i in range(1, n):
            a = np.array([R.logsumexp(a + logT[:, b]) for b in range(m)]) + node[i]
        assert abs(brute - R.logsumexp(a)) < 1e-9


def test_p2_with_zero_slip_equals_p1(rng, theta):
    """SS4.6's reduction, at the reference level, with psi carried across correctly."""
    m = R.STAGE_P_M
    for _ in range(5):
        n = int(rng.integers(1, 4)) * m
        y = rng.integers(0, 2, n)
        prior = R.log_normalise(rng.normal(size=m))                  # over OFFSETS
        init = np.array([prior[R.offset_from_pointer(s, m)] for s in range(m)])
        p2 = R.p2_forward_backward(y, init, m, eps_hold=0.0, eps_skip=0.0, **theta)
        assert abs(p2["log_evidence"] - R.p1_log_evidence(y, prior, m, **theta)) < 1e-9


def test_viterbi_recovers_a_clean_path(theta):
    """With a sharp emission and no slip, Viterbi must return the generating trajectory."""
    m = R.STAGE_P_M
    for r in range(m):
        y = R.downbeats_from_offset(4 * m, r, m)
        init = np.full(m, -math.log(m))
        path = R.p2_viterbi(y, init, m, alpha=6.0, beta=-6.0, eps_hold=0.01, eps_skip=0.01)
        assert R.offset_from_pointer(int(path[0]), m) == r
        assert np.array_equal((path == 0).astype(np.uint8), np.asarray(y).astype(np.uint8))


# ----------------------------------------------------------------------------------
# SS4.6 identities at the reference level
# ----------------------------------------------------------------------------------
def test_elbo_is_tight_at_the_posterior_and_slack_elsewhere(rng, theta):
    """``ELBO <= log p(y|h)``, equality iff q is the posterior, slack == reverse KL."""
    m = R.STAGE_P_M
    for _ in range(6):
        n = int(rng.integers(1, 4)) * m
        y = rng.integers(0, 2, n)
        prior = R.log_normalise(rng.normal(size=m))
        lik = R.offset_loglik_all(y, m, **theta)
        ev = R.p1_log_evidence(y, prior, m, **theta)
        post = R.p1_posterior(y, prior, m, **theta)

        assert abs(R.elbo_from(post, lik, prior) - ev) < 1e-9

        q = R.log_normalise(rng.normal(size=m) * 1.5)
        elbo = R.elbo_from(q, lik, prior)
        assert elbo <= ev + 1e-12
        assert abs((ev - elbo) - R.reverse_kl(q, post)) < 1e-9


# ----------------------------------------------------------------------------------
# SS6.2 exclusions
# ----------------------------------------------------------------------------------
def test_derive_r_true_recovers_a_clean_offset():
    """Every offset must be recoverable from a slip-free crop."""
    m = R.STAGE_P_M
    for r in range(m):
        assert R.derive_r_true(R.downbeats_from_offset(4 * m, r, m), m) == r


def test_derive_r_true_refuses_a_crop_with_a_meter_change():
    """SS6.2: no single offset fits, so the answer must be None -- not a best guess."""
    m = R.STAGE_P_M
    y = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], dtype=np.uint8)   # a bar of 3
    assert R.derive_r_true(y, m) is None
    assert R.derive_r_true(np.zeros(8, dtype=np.uint8), m) is None


def test_label_crops_counts_exclusions_and_invents_nothing():
    """Every crop is either labelled correctly or counted as an exclusion."""
    m = R.STAGE_P_M
    crops = B.make_meter_change_dataset(n_crops=40, n_beats=16, m=m)
    labelled, n_excluded = R.label_crops(crops, m)
    assert n_excluded > 0
    assert len(labelled) + n_excluded == len(crops)
    for c in labelled:
        assert np.array_equal(np.asarray(c["y"]).astype(np.uint8),
                              R.downbeats_from_offset(len(c["y"]), c["r_true"], m))


# ----------------------------------------------------------------------------------
# SS8.1 metrics
# ----------------------------------------------------------------------------------
def test_placement_f_is_one_only_on_an_exact_match():
    """No tolerance window (SS8.1): the grid is given."""
    m = R.STAGE_P_M
    y = R.downbeats_from_offset(4 * m, 0, m)
    assert R.placement_f(y, y) == 1.0
    assert R.placement_f(y, R.downbeats_from_offset(4 * m, 1, m)) == 0.0
    assert R.placement_f(np.zeros(4), np.zeros(4)) == 1.0
    assert R.placement_f(y, np.zeros_like(y)) == 0.0


def test_offset_metrics_agree_with_the_confusion_matrix():
    """P4: raw accuracy and the confusion matrix must be consistent by construction."""
    m = R.STAGE_P_M
    true = [0, 1, 2, 3, 0, 1]
    pred = [0, 1, 2, 0, 0, 2]
    C = R.offset_confusion(true, pred, m)
    assert C.sum() == len(true)
    assert C.trace() / C.sum() == R.offset_accuracy(true, pred)


def test_majority_null_is_a_collapse_detector():
    """The majority null must pick the modal offset, deterministically."""
    assert R.majority_r_null([0, 0, 0, 1, 2], R.STAGE_P_M) == 0
    assert R.majority_r_null([], R.STAGE_P_M) == 0


# ----------------------------------------------------------------------------------
# SS6.1 / SS6.5 the bench itself
# ----------------------------------------------------------------------------------
def test_bench_crops_are_not_bar_aligned():
    """SS6.1/P2: the bench must NOT put r=0 in ~99% of crops, as Stage 0's did (SS10.1).

    This is the single change SS6.1 calls "the only change that makes phase learnable
    rather than vacuous", so the bench is checked for it directly.
    """
    crops = B.make_labelled_dataset(n_crops=32, n_beats=16)
    offsets = [c["r_true"] for c in crops]
    assert set(offsets) == set(range(R.STAGE_P_M)), (
        f"the bench only produced offsets {sorted(set(offsets))}: phase is not being varied")
    share_zero = offsets.count(0) / len(offsets)
    assert abs(share_zero - 1 / R.STAGE_P_M) < 0.1, (
        f"{share_zero:.1%} of crops start on a downbeat; SS6.1 requires crop starts uniform "
        f"over the bar, and Stage 0's ~99% is what made phase unstudiable")


def test_bench_h_is_consistent_with_its_own_labels():
    """The downbeat channel must peak at the beats the labels call downbeats.

    SS10.6's caveat applies and is the point: the bumps sit AT the annotations, so a high
    score on this bench proves the chain is wired, never that a real frontend hears phase.
    """
    crops = B.make_labelled_dataset(n_crops=8, n_beats=16)
    for c in crops:
        at_beats = np.asarray(c["h"])[np.asarray(c["beat_frames"]), 1]
        y = np.asarray(c["y"]).astype(bool)
        assert at_beats[y].min() > at_beats[~y].max(), (
            "the bench's downbeat channel does not separate downbeats from other beats")


def test_bench_shift_moves_r_true_by_exactly_one():
    """SS8.4's control is only meaningful if the bench's own shift does what it claims."""
    m = R.STAGE_P_M
    track = B.make_track(n_beats=96, m=m)
    for start in (0, 5, 13, 27):
        a = B.crop_at(track, start, 16)
        b = B.crop_at(track, start + 1, 16)
        assert (a["r_true"] - 1) % m == b["r_true"]


def test_bench_rejects_a_crop_that_does_not_span_whole_bars():
    """The precondition is enforced at the point crops are cut, not merely documented."""
    track = B.make_track(n_beats=96)
    with pytest.raises(ValueError, match="whole bars"):
        B.crop_at(track, 0, 13)


def test_meter_change_bench_produces_both_labelled_and_excluded_crops():
    """SS6.5: the P2 bench needs slip AND a mid-crop meter change actually present."""
    crops = B.make_meter_change_dataset(n_crops=48, n_beats=16)
    labelled = [c for c in crops if c["r_true"] is not None]
    excluded = [c for c in crops if c["r_true"] is None]
    assert labelled and excluded, (
        f"{len(labelled)} labelled / {len(excluded)} excluded: the bench must contain both")
