"""Independent reference for Stage P (`docs/SPEC_phase.md`) -- bar phase as the latent.

This file deliberately does NOT import ``vbpm``, and does not import Stage 0's
``tests/reference.py`` either. It must be derivable from the spec text alone, because it
is the oracle every property is checked against. Everything here is brute force and slow;
correctness is the only goal. Self-tested by ``test_reference_p_selftest.py``.

The one place ``vbpm`` legitimately appears in this suite is the Stage-0 reduction check
(SS4.6), which is *about* agreement with the shipped Stage-0 emission; it lives in
``properties_p.py`` and imports the package there, not here.

Two conventions this file pins, because SS4.1 states them in two mutually inconsistent
ways (see README, "Where the spec was ambiguous"):

``r`` -- the crop OFFSET
    ``r`` is the index of the first downbeat in the crop, so beat ``i`` is a downbeat iff
    ``i == r (mod m)``. This is what SS4.1's formula says, what SS6.2's ``r_true`` says,
    and what ``vbpm/fitting.py:emission_counts`` does (``slots = arange(r, n, m)``). SS4.1's
    prose "``r`` is the bar pointer at the crop's first beat" describes the *negation* of
    this quantity and is the odd one out; three sources against one.

``s_i`` -- the per-beat POINTER (P2 only)
    ``s_i = (i - r) mod m``, so beat ``i`` is a downbeat iff ``s_i == 0`` -- which is
    exactly what SS4.3's emission and SS7's P2 read-out are written against. It advances
    ``s_{i+1} = s_i + 1``, matching SS4.1's transition.

The two are the same object seen from different ends: ``s_0 = (-r) mod m``. Keeping both
explicit is not pedantry -- confusing them is a real off-by-one at the P1/P2 seam, and it
has its own mutant (``readout_uses_pointer_not_offset``).
"""
from __future__ import annotations

import itertools
import math

import numpy as np

# SS2.1: m is FIXED at Stage P but is a named constant and a parameter of every function
# below, never an inlined literal. Condition 1 and condition 3 of the C3 exemption.
STAGE_P_M = 4

DEFAULT_FPS = 50.0

# SS8.1 says chance on offset accuracy is 1/m = 0.25. That is true ONLY when a crop spans
# whole bars -- see :func:`count_partition_chance`. Never hardcode this constant into a
# gate; derive the chance level from the crop length instead.
CHANCE_OFFSET_ACC = 1.0 / STAGE_P_M


# --------------------------------------------------------------------------------------
# numerics
# --------------------------------------------------------------------------------------
def log_sigmoid(x: float) -> float:
    """Return log(sigmoid(x)), stable for large |x|.

    Args:
        x: A real logit.

    Returns:
        ``log(1 / (1 + exp(-x)))``.
    """
    return -math.log1p(math.exp(-abs(x))) + (0.0 if x >= 0 else x)


def log1m_sigmoid(x: float) -> float:
    """Return log(1 - sigmoid(x)), which equals ``log_sigmoid(-x)``.

    Args:
        x: A real logit.

    Returns:
        ``log(1 - sigmoid(x))``.
    """
    return log_sigmoid(-x)


def logsumexp(vals) -> float:
    """Return log(sum(exp(v))) over an iterable, stable, with -inf handled.

    Args:
        vals: An iterable of log-space values.

    Returns:
        The log of the sum of the exponentials, or -inf if every entry is -inf.
    """
    vals = [float(v) for v in vals]
    mx = max(vals)
    if mx == -math.inf:
        return -math.inf
    return mx + math.log(sum(math.exp(v - mx) for v in vals))


def log_normalise(vals) -> np.ndarray:
    """Normalise a vector of log-weights into log-probabilities.

    Args:
        vals: Unnormalised log-weights.

    Returns:
        A float64 array of log-probabilities summing (in probability space) to one.
    """
    v = np.asarray(vals, dtype=np.float64)
    return v - logsumexp(v)


# --------------------------------------------------------------------------------------
# SS4.1 the latent, and the offset <-> pointer relabelling
# --------------------------------------------------------------------------------------
def is_downbeat(i: int, r: int, m: int) -> bool:
    """Report whether beat ``i`` is a downbeat under crop offset ``r``.

    Args:
        i: Beat index within the crop, zero-based.
        r: Crop offset -- the index of the first downbeat.
        m: Beats per bar.

    Returns:
        True iff ``i == r (mod m)``.
    """
    return ((int(i) - int(r)) % int(m)) == 0


def downbeat_slot_counts(n: int, m: int) -> list[int]:
    """Return ``[m]`` of how many downbeat slots each offset produces in an n-beat crop.

    ``count(r) = len(range(r, n, m))``. This is constant in ``r`` iff ``m`` divides ``n``.

    Args:
        n: Number of beats in the crop.
        m: Beats per bar.

    Returns:
        A list of slot counts indexed by offset.
    """
    return [len(range(r, n, m)) for r in range(m)]


def count_partition_chance(n: int, m: int) -> float:
    """Return the offset accuracy reachable by a summary that sees ONLY slot counts.

    This is the real chance level for SS8.3's P-0 leak detector, and SS8.3's stated 0.25 is
    wrong whenever ``n % m != 0``. The number of downbeat slots under offset ``r`` is
    ``len(range(r, n, m))``, which depends on ``r`` unless ``m | n``. Verified: at ``m = 4``,
    ``n = 13`` gives counts ``[4, 3, 3, 3]``, ``n = 14`` gives ``[4, 4, 3, 3]``, ``n = 15``
    gives ``[4, 4, 4, 3]``. In each case a completely shift-invariant, position-blind count
    summary partitions the offsets into two distinguishable groups and scores 0.500 -- with
    no leak whatsoever. Only ``n % m == 0`` gives ``[4, 4, 4, 4]`` and a true 0.25.

    A gate that hardcodes ``1/m`` would therefore report a phantom leak (or a phantom
    success) on any crop that does not span whole bars. Stage-P crops are required to span
    whole bars for exactly this reason; see :func:`assert_whole_bars`.

    Args:
        n: Number of beats in the crop.
        m: Beats per bar.

    Returns:
        The best offset accuracy achievable from slot counts alone: the number of distinct
        count values divided by ``m``.
    """
    return len(set(downbeat_slot_counts(n, m))) / float(m)


def assert_whole_bars(n: int, m: int) -> None:
    """Assert an un-aligned crop spans whole bars, so that chance is genuinely ``1/m``.

    Args:
        n: Number of beats in the crop.
        m: Beats per bar.

    Raises:
        ValueError: If ``m`` does not divide ``n``.
    """
    if n % m != 0:
        raise ValueError(
            f"crop of n={n} beats does not span whole bars at m={m}: downbeat-slot counts "
            f"are {downbeat_slot_counts(n, m)}, so a position-blind count summary reaches "
            f"{count_partition_chance(n, m):.3f} rather than {1 / m:.3f}. Un-aligned crops "
            f"must satisfy n % m == 0 or every chance-level gate is wrong.")


def pointer_from_offset(r: int, m: int) -> int:
    """Convert a crop offset into the P2 pointer state at beat zero.

    Args:
        r: Crop offset (index of the first downbeat).
        m: Beats per bar.

    Returns:
        ``(-r) mod m``, the value of ``s_0``.
    """
    return (-int(r)) % int(m)


def offset_from_pointer(s0: int, m: int) -> int:
    """Convert a P2 pointer state at beat zero back into a crop offset.

    Args:
        s0: Pointer state at beat zero.
        m: Beats per bar.

    Returns:
        ``(-s0) mod m``, the crop offset. The inverse of :func:`pointer_from_offset`.
    """
    return (-int(s0)) % int(m)


# --------------------------------------------------------------------------------------
# SS4.3 emission -- theta is exactly {alpha, beta}
# --------------------------------------------------------------------------------------
def beat_logp(y_i: int, on_downbeat: bool, alpha: float, beta: float) -> float:
    """Return log p_theta(y_i | r_i) for a single beat.

    Args:
        y_i: The observed downbeat indicator for this beat, 0 or 1.
        on_downbeat: Whether the latent places a downbeat on this beat.
        alpha: Downbeat-slot logit.
        beta: Other-slot logit.

    Returns:
        The Bernoulli log-probability under the appropriate logit.
    """
    x = alpha if on_downbeat else beta
    return log_sigmoid(x) if y_i else log1m_sigmoid(x)


def offset_loglik(y, r: int, m: int, alpha: float, beta: float) -> float:
    """Return log p_theta(y | r) for a static offset -- the P1 conditional likelihood.

    Args:
        y: Per-beat downbeat indicators, length n.
        r: Crop offset.
        m: Beats per bar.
        alpha: Downbeat-slot logit.
        beta: Other-slot logit.

    Returns:
        The summed per-beat Bernoulli log-likelihood.
    """
    return float(sum(beat_logp(int(yi), is_downbeat(i, r, m), alpha, beta)
                     for i, yi in enumerate(y)))


def offset_loglik_all(y, m: int, alpha: float, beta: float) -> np.ndarray:
    """Return ``[m]`` of log p_theta(y | r) for every offset r.

    Args:
        y: Per-beat downbeat indicators.
        m: Beats per bar.
        alpha: Downbeat-slot logit.
        beta: Other-slot logit.

    Returns:
        A float64 array indexed by offset.
    """
    return np.array([offset_loglik(y, r, m, alpha, beta) for r in range(m)],
                    dtype=np.float64)


def stage0_emission_logp(y, m: int, alpha: float, beta: float) -> float:
    """Return Stage 0's emission: log[(1/m) sum_r p_theta(y|r)], r marginalised uniformly.

    This is the SS4.6 reduction target. Stage 0 treats the offset as a uniform nuisance and
    integrates it out; Stage P promotes the same quantity's summands to a latent. Equality
    of this function with Stage 0's shipped emission is what "Stage P extends Stage 0"
    means formally.

    Args:
        y: Per-beat downbeat indicators.
        m: Beats per bar.
        alpha: Downbeat-slot logit.
        beta: Other-slot logit.

    Returns:
        The marginal log-likelihood of y given m.
    """
    return logsumexp(offset_loglik_all(y, m, alpha, beta)) - math.log(m)


# --------------------------------------------------------------------------------------
# SS4.4 / SS4.6 P1 inference
# --------------------------------------------------------------------------------------
def p1_posterior(y, prior_logp, m: int, alpha: float, beta: float) -> np.ndarray:
    """Return the exact log posterior over the offset, ``[m]``.

    Args:
        y: Per-beat downbeat indicators.
        prior_logp: ``[m]`` log p_psi(r | h).
        m: Beats per bar.
        alpha: Downbeat-slot logit.
        beta: Other-slot logit.

    Returns:
        ``log p(r | y, h)``, normalised.
    """
    return log_normalise(offset_loglik_all(y, m, alpha, beta)
                         + np.asarray(prior_logp, dtype=np.float64))


def p1_log_evidence(y, prior_logp, m: int, alpha: float, beta: float) -> float:
    """Return log p(y | h) for P1 by exact enumeration over the m offsets.

    Args:
        y: Per-beat downbeat indicators.
        prior_logp: ``[m]`` log p_psi(r | h).
        m: Beats per bar.
        alpha: Downbeat-slot logit.
        beta: Other-slot logit.

    Returns:
        The scalar log evidence.
    """
    return logsumexp(offset_loglik_all(y, m, alpha, beta)
                     + np.asarray(prior_logp, dtype=np.float64))


def elbo_from(q_logp, lik, prior_logp) -> float:
    """Return ``E_q[log p(y|r)] - KL(q || p_psi)`` from three ``[m]`` vectors.

    The single composition of SS4.6's objective in this reference. Both P1 and P2 route
    through it, so the KL's direction is stated once.

    Args:
        q_logp: ``[m]`` log q_phi(r | h, y).
        lik: ``[m]`` log p_theta(y | r).
        prior_logp: ``[m]`` log p_psi(r | h).

    Returns:
        The scalar ELBO.
    """
    q = np.exp(np.asarray(q_logp, dtype=np.float64))
    lik = np.asarray(lik, dtype=np.float64)
    prior_logp = np.asarray(prior_logp, dtype=np.float64)
    recon = float((q * lik).sum())
    kl = float((q * (np.asarray(q_logp, dtype=np.float64) - prior_logp)).sum())
    return recon - kl


def reverse_kl(q_logp, p_logp) -> float:
    """Return ``KL(q || p)`` from two log-probability vectors.

    SS4.6 pins the ELBO's slack as the REVERSE KL -- against the posterior, in this
    direction. SPEC.md SS3.5's diagnostic uses the forward direction and is a different
    number; the frozen Stage-0 suite makes the same distinction.

    Args:
        q_logp: ``[m]`` log q.
        p_logp: ``[m]`` log p.

    Returns:
        The scalar KL divergence from q to p.
    """
    q_logp = np.asarray(q_logp, dtype=np.float64)
    p_logp = np.asarray(p_logp, dtype=np.float64)
    return float((np.exp(q_logp) * (q_logp - p_logp)).sum())


# --------------------------------------------------------------------------------------
# SS4.1 P2 transition, and exact inference over the chain
# --------------------------------------------------------------------------------------
def transition_matrix(m: int, eps_hold: float, eps_skip: float) -> np.ndarray:
    """Return the ``[m, m]`` pointer transition, entry ``(a, b) = p(s_{i+1}=b | s_i=a)``.

    SS4.1's slip kernel is malformed as written: it uses ``eps`` both as a single parameter
    ("with probability ``1 - eps``") and as the pair ``{eps_hold, eps_skip}``, and never
    states the relation between them. Treated here as TWO free parameters with advance mass
    ``1 - eps_hold - eps_skip``, which is the only reading under which the three branches
    form a distribution. Normalisation is asserted, not assumed.

    Args:
        m: Beats per bar.
        eps_hold: Probability of holding (a locally long bar).
        eps_skip: Probability of skipping (a dropped beat or pickup).

    Returns:
        A row-stochastic float64 matrix.

    Raises:
        ValueError: If the slip probabilities do not leave non-negative mass to advance.
    """
    if eps_hold < 0 or eps_skip < 0 or eps_hold + eps_skip > 1:
        raise ValueError(f"illegal slip ({eps_hold}, {eps_skip}): must be >= 0 and sum <= 1")
    adv = 1.0 - eps_hold - eps_skip
    T = np.zeros((m, m), dtype=np.float64)
    for a in range(m):
        T[a, (a + 1) % m] += adv
        T[a, a] += eps_hold
        T[a, (a + 2) % m] += eps_skip
    return T


def p2_emission_matrix(y, m: int, alpha: float, beta: float) -> np.ndarray:
    """Return ``[n, m]`` of log p_theta(y_i | s_i), the per-beat emission.

    Args:
        y: Per-beat downbeat indicators, length n.
        m: Beats per bar.
        alpha: Downbeat-slot logit.
        beta: Other-slot logit.

    Returns:
        A float64 array indexed by (beat, pointer state).
    """
    n = len(y)
    E = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for s in range(m):
            E[i, s] = beat_logp(int(y[i]), s == 0, alpha, beta)
    return E


def p2_forward_backward(y, init_logp, m: int, alpha: float, beta: float,
                        eps_hold: float, eps_skip: float) -> dict:
    """Run exact forward-backward over the pointer chain. O(n m^2), nothing sampled.

    Args:
        y: Per-beat downbeat indicators, length n.
        init_logp: ``[m]`` log p(s_0), the initial pointer distribution.
        m: Beats per bar.
        alpha: Downbeat-slot logit.
        beta: Other-slot logit.
        eps_hold: Hold probability.
        eps_skip: Skip probability.

    Returns:
        A dict with ``log_evidence`` (scalar), ``marginals`` (``[n, m]`` log p(s_i | y)),
        and ``init_posterior`` (``[m]``, the smoothed distribution over ``s_0``).
    """
    n = len(y)
    E = p2_emission_matrix(y, m, alpha, beta)
    with np.errstate(divide="ignore"):
        logT = np.log(transition_matrix(m, eps_hold, eps_skip))
    init_logp = np.asarray(init_logp, dtype=np.float64)

    alpha_f = np.zeros((n, m), dtype=np.float64)
    alpha_f[0] = init_logp + E[0]
    for i in range(1, n):
        for b in range(m):
            alpha_f[i, b] = logsumexp(alpha_f[i - 1] + logT[:, b]) + E[i, b]

    beta_b = np.zeros((n, m), dtype=np.float64)
    for i in range(n - 2, -1, -1):
        for a in range(m):
            beta_b[i, a] = logsumexp(logT[a, :] + E[i + 1] + beta_b[i + 1])

    log_ev = logsumexp(alpha_f[n - 1])
    marg = np.stack([log_normalise(alpha_f[i] + beta_b[i]) for i in range(n)])

    # pairwise marginals xi[i, a, b] = log p(s_i = a, s_{i+1} = b | y), i = 0..n-2.
    # SS4.6's KL is between two CHAIN distributions, and a chain's entropy is not a
    # function of its per-beat marginals alone -- the pairwise term is what makes the P2
    # ELBO computable at all, so it is produced here rather than reconstructed downstream.
    xi = np.full((max(n - 1, 0), m, m), -math.inf, dtype=np.float64)
    for i in range(n - 1):
        for a in range(m):
            for b in range(m):
                xi[i, a, b] = (alpha_f[i, a] + logT[a, b] + E[i + 1, b]
                               + beta_b[i + 1, b] - log_ev)
    return {"log_evidence": float(log_ev), "marginals": marg, "pairwise": xi,
            "init_posterior": marg[0].copy()}


def p2_brute_force(y, init_logp, m: int, alpha: float, beta: float,
                   eps_hold: float, eps_skip: float) -> dict:
    """Enumerate every state sequence in ``{0..m-1}^n`` -- the check on forward-backward.

    Exponential in n by construction, so callers must keep ``n <= 8`` (SS9). It shares no
    code path with :func:`p2_forward_backward` beyond the emission and the transition, so
    agreement between the two is real evidence rather than a tautology.

    Args:
        y: Per-beat downbeat indicators, length n.
        init_logp: ``[m]`` log p(s_0).
        m: Beats per bar.
        alpha: Downbeat-slot logit.
        beta: Other-slot logit.
        eps_hold: Hold probability.
        eps_skip: Skip probability.

    Returns:
        A dict with the same keys as :func:`p2_forward_backward`.
    """
    n = len(y)
    E = p2_emission_matrix(y, m, alpha, beta)
    with np.errstate(divide="ignore"):
        logT = np.log(transition_matrix(m, eps_hold, eps_skip))
    init_logp = np.asarray(init_logp, dtype=np.float64)

    weights, paths = [], []
    for path in itertools.product(range(m), repeat=n):
        w = init_logp[path[0]] + E[0, path[0]]
        for i in range(1, n):
            w += logT[path[i - 1], path[i]] + E[i, path[i]]
        if w == -math.inf:
            continue
        weights.append(w)
        paths.append(path)

    log_ev = logsumexp(weights)
    marg = np.full((n, m), -math.inf, dtype=np.float64)
    xi = np.full((max(n - 1, 0), m, m), -math.inf, dtype=np.float64)
    for w, path in zip(weights, paths):
        for i, s in enumerate(path):
            marg[i, s] = logsumexp([marg[i, s], w])
        for i in range(n - 1):
            xi[i, path[i], path[i + 1]] = logsumexp([xi[i, path[i], path[i + 1]], w])
    marg = marg - log_ev
    xi = xi - log_ev
    return {"log_evidence": float(log_ev), "marginals": marg, "pairwise": xi,
            "init_posterior": marg[0].copy()}


def chain_logz_brute(node_logpot, logT) -> float:
    """Return the chain's log partition by enumerating every path in ``{0..m-1}^n``.

    The independent check on any forward recursion (SS9: forward-backward against brute
    force for ``n <= 8``). Exponential in n by construction, so callers must keep n small.
    The initial state is uniform, matching SS4.4's chain: the audio potentials carry the
    phase information, so no separate initial distribution is needed.

    Args:
        node_logpot: ``[n, m]`` additive per-beat log-potentials.
        logT: ``[m, m]`` log transition.

    Returns:
        The scalar log partition function.
    """
    node = np.asarray(node_logpot, dtype=np.float64)
    logT = np.asarray(logT, dtype=np.float64)
    n, m = node.shape
    weights = []
    for path in itertools.product(range(m), repeat=n):
        w = -math.log(m) + node[0, path[0]]
        for i in range(1, n):
            w += logT[path[i - 1], path[i]] + node[i, path[i]]
        if w != -math.inf:
            weights.append(w)
    return logsumexp(weights)


def p2_viterbi(y, init_logp, m: int, alpha: float, beta: float,
               eps_hold: float, eps_skip: float) -> np.ndarray:
    """Return the MAP pointer path ``[n]`` -- SS7's P2 deployment read-out.

    Args:
        y: Per-beat downbeat indicators, length n.
        init_logp: ``[m]`` log p(s_0).
        m: Beats per bar.
        alpha: Downbeat-slot logit.
        beta: Other-slot logit.
        eps_hold: Hold probability.
        eps_skip: Skip probability.

    Returns:
        An int array of pointer states; beat i is predicted a downbeat iff entry i is 0.
    """
    n = len(y)
    E = p2_emission_matrix(y, m, alpha, beta)
    with np.errstate(divide="ignore"):
        logT = np.log(transition_matrix(m, eps_hold, eps_skip))
    delta = np.asarray(init_logp, dtype=np.float64) + E[0]
    back = np.zeros((n, m), dtype=int)
    for i in range(1, n):
        nxt = np.zeros(m, dtype=np.float64)
        for b in range(m):
            scores = delta + logT[:, b]
            back[i, b] = int(np.argmax(scores))
            nxt[b] = float(np.max(scores)) + E[i, b]
        delta = nxt
    path = np.zeros(n, dtype=int)
    path[n - 1] = int(np.argmax(delta))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


# --------------------------------------------------------------------------------------
# SS6.2 r_true, and the crops where it does not exist
# --------------------------------------------------------------------------------------
def derive_r_true(y, m: int):
    """Return the crop offset that exactly explains y, or None when none does.

    SS6.2: ``r_true`` is well-defined only when the meter is constant across the crop. When
    the meter varies internally no single offset fits, and the crop must be EXCLUDED and
    COUNTED rather than given a fictional label. Returning None is how this reference
    refuses to invent one.

    "Exactly explains" is the strict reading: the offset comb must reproduce y beat for
    beat. The alternative -- best-fitting offset by Hamming distance -- always returns
    something and so cannot express exclusion at all, which is the failure SS6.2 legislates
    against. SS10.1 reports both statistics ("fits perfectly" 98.03% vs "best-fitting"
    99.24%) and it is the perfect-fit reading that defines the ~2% exclusions.

    Args:
        y: Per-beat downbeat indicators.
        m: Beats per bar.

    Returns:
        The unique offset in ``0..m-1`` reproducing y exactly, else None. If several
        offsets fit (possible only on degenerate y, e.g. all zeros) the label is not
        unique and None is returned.
    """
    y = np.asarray(y).astype(int)
    n = len(y)
    fits = [r for r in range(m)
            if all(int(y[i]) == int(is_downbeat(i, r, m)) for i in range(n))]
    return int(fits[0]) if len(fits) == 1 else None


def label_crops(crops, m: int) -> tuple[list, int]:
    """Split crops into those with a well-defined ``r_true`` and a count of exclusions.

    Args:
        crops: An iterable of dicts each carrying a ``y``.
        m: Beats per bar.

    Returns:
        A tuple ``(labelled, n_excluded)`` where ``labelled`` carries an ``r_true`` key.
    """
    labelled, excluded = [], 0
    for c in crops:
        r = derive_r_true(c["y"], m)
        if r is None:
            excluded += 1
            continue
        labelled.append({**c, "r_true": r})
    return labelled, excluded


# --------------------------------------------------------------------------------------
# SS8.1 metrics
# --------------------------------------------------------------------------------------
def downbeats_from_offset(n: int, r: int, m: int) -> np.ndarray:
    """Return the ``[n]`` 0/1 downbeat placement implied by a crop offset.

    Args:
        n: Number of beats.
        r: Crop offset.
        m: Beats per bar.

    Returns:
        A uint8 array with ones on the offset comb.
    """
    return np.array([1 if is_downbeat(i, r, m) else 0 for i in range(n)], dtype=np.uint8)


def placement_f(true_y, pred_y) -> float:
    """Return downbeat placement F on the given beat grid -- SS8.1's primary metric.

    The grid is given, so a predicted downbeat is correct iff it lands on an annotated
    downbeat beat. No tolerance window.

    Args:
        true_y: Annotated per-beat downbeat indicators.
        pred_y: Predicted per-beat downbeat indicators.

    Returns:
        The F-measure, with the convention F = 0 when both sets are empty is avoided by
        returning 1.0 only if both are empty.
    """
    t = np.asarray(true_y).astype(bool)
    p = np.asarray(pred_y).astype(bool)
    tp = float((t & p).sum())
    if t.sum() == 0 and p.sum() == 0:
        return 1.0
    if tp == 0:
        return 0.0
    prec = tp / float(p.sum())
    rec = tp / float(t.sum())
    return 2 * prec * rec / (prec + rec)


def majority_r_null(train_r, m: int) -> int:
    """Return the most frequent offset in a training split -- SS8.2's collapse detector.

    Also serves as a PLACEMENT null. SS8.2 nominates the uniform-``r`` null (Stage 0's
    marginalised emission) for that role, but a model that marginalises ``r`` emits no
    downbeat locations at all, so it has no placement to score and cannot be compared on
    SS8.1's primary metric. Majority-``r`` and :func:`random_r_null` are the nulls that
    actually place downbeats. The uniform-``r`` null remains meaningful on NLL, where it is
    exactly Stage 0's emission.

    Args:
        train_r: Iterable of offsets seen in training.
        m: Beats per bar.

    Returns:
        The modal offset, ties broken toward the smaller offset.
    """
    counts = {r: 0 for r in range(m)}
    for r in train_r:
        counts[int(r)] += 1
    return max(range(m), key=lambda r: (counts[r], -r))


def random_r_null(n_crops: int, m: int, seed: int = 0) -> np.ndarray:
    """Return uniformly random offsets -- the placement null whose expectation is 1/m.

    Args:
        n_crops: How many offsets to draw.
        m: Beats per bar.
        seed: Seed for the draw.

    Returns:
        An int array of offsets.
    """
    return np.random.default_rng(seed).integers(0, m, size=n_crops)


def offset_accuracy(true_r, pred_r) -> float:
    """Return the fraction of crops whose offset is exactly right. Chance is 1/m.

    Args:
        true_r: Iterable of true offsets.
        pred_r: Iterable of predicted offsets.

    Returns:
        Raw accuracy in [0, 1].
    """
    t = np.asarray(list(true_r))
    p = np.asarray(list(pred_r))
    return float((t == p).mean()) if len(t) else float("nan")


def offset_confusion(true_r, pred_r, m: int) -> np.ndarray:
    """Return the ``[m, m]`` confusion matrix over offsets. Mandatory beside every figure.

    SS3 P4: every reported figure carries raw accuracy AND the confusion matrix, because a
    degenerate predictor once scored 0.422 balanced with 0.239 raw (SS10.4).

    Args:
        true_r: Iterable of true offsets.
        pred_r: Iterable of predicted offsets.
        m: Beats per bar.

    Returns:
        An integer array with rows indexed by the true offset.
    """
    C = np.zeros((m, m), dtype=int)
    for t, p in zip(true_r, pred_r):
        C[int(t), int(p)] += 1
    return C
