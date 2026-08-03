"""Named corruptions of the Stage-P oracle -- the machinery behind "only if".

You cannot get "the tests pass ONLY IF the implementation is proper" by writing more
assertions, because the set of wrong programs is not enumerable. What you CAN do is fix a
set of wrongnesses you care about and prove mechanically that each is caught.

``test_mutation_registry_p.py`` runs the whole property suite against each entry and
asserts at least one property dies. A surviving mutant is reported BY NAME: a legible hole,
not a silent pass.

The one that matters most
-------------------------
``downbeat_off_by_one`` is listed in Stage 0's ``EQUIVALENT`` set and its
``test_equivalent_mutant_survives`` requires it to SURVIVE. At Stage P it must be KILLED.
Both are correct, and the difference is the whole point of the stage: Stage 0 marginalises
the bar offset, which makes downbeat phase unobservable BY DESIGN, so pinning it there
would be over-specification. Stage P promotes that offset to the latent and to the
deployable output, so an emission whose downbeat lands one beat late is simply wrong.
``SPEC_phase.md`` SS11 A2 flags this as "a real decision, not a clarification"; SS9 makes
the registry stage-scoped so that ``tests/`` can stay frozen while Stage P disagrees with
it. Getting this backwards would silently license the exact bug the stage exists to detect.
"""
from __future__ import annotations

import math

import numpy as np
import torch

import reference_p as R
import subject_p as S


# ------------------------------------------------------------------------------------
# emission corruptions
# ------------------------------------------------------------------------------------
def _downbeat_off_by_one(s, y):
    """``((i - r + 1) % m == 0)``: the downbeat lands one beat late.

    EQUIVALENT at Stage 0 (``tests/mutants.py`` proves it: marginalising over r enumerates
    the same SET of masks in a different order, and logsumexp is symmetric). KILLED at
    Stage P, because r is no longer summed over -- ``emission_logp_all[r]`` now names a
    specific, deployable, scored hypothesis, and this one names the wrong beat.
    """
    y_t = torch.as_tensor(np.asarray(y, dtype=np.float64), dtype=torch.float64)
    n = len(y_t)
    lsig = torch.nn.functional.logsigmoid
    on = y_t * lsig(s.alpha) + (1 - y_t) * lsig(-s.alpha)
    off = y_t * lsig(s.beta) + (1 - y_t) * lsig(-s.beta)
    idx = torch.arange(n, dtype=torch.float64)
    out = []
    for r in range(s.m):
        mask = (((idx - r + 1) % s.m) == 0).to(torch.float64)
        out.append((mask * on + (1 - mask) * off).sum())
    return torch.stack(out)


def _global_phase_offset_one_bar(s, y):
    """``((i - r + m) % m == 0)``: a global offset of EXACTLY one bar.

    PROVABLY EQUIVALENT and required to survive. Adding ``m`` inside a modulo-``m``
    comparison is the identity map on the comb, so this is the correct emission written
    with redundant arithmetic. SS9 names it as an equivalent candidate. It is the control
    on ``downbeat_off_by_one``: a suite that kills BOTH is not detecting a phase error, it
    is pattern-matching on the source text.
    """
    y_t = torch.as_tensor(np.asarray(y, dtype=np.float64), dtype=torch.float64)
    n = len(y_t)
    lsig = torch.nn.functional.logsigmoid
    on = y_t * lsig(s.alpha) + (1 - y_t) * lsig(-s.alpha)
    off = y_t * lsig(s.beta) + (1 - y_t) * lsig(-s.beta)
    idx = torch.arange(n, dtype=torch.float64)
    out = []
    for r in range(s.m):
        mask = (((idx - r + s.m) % s.m) == 0).to(torch.float64)
        out.append((mask * on + (1 - mask) * off).sum())
    return torch.stack(out)


def _emission_via_counts(s, y):
    """The same emission through the counts linearisation ``vbpm/fitting.py`` uses.

    PROVABLY EQUIVALENT and required to survive. ``log p(y|r)`` is linear in
    ``[lsig(a), lsig(-a), lsig(b), lsig(-b)]`` with integer coefficients, so accumulating
    the four counts and taking an inner product is exact algebra, not an approximation.

    Kept because the suite must not pin an implementation PATH. The shipped package
    computes its emission this way for speed and asserts the two agree
    (``fitting.verify_vectorized``); a Stage-P suite that rejected the fast form would
    reject the package's own arithmetic.
    """
    y = np.asarray(y, dtype=np.float64)
    n, total_ones = len(y), float(y.sum())
    lsig = torch.nn.functional.logsigmoid
    v = torch.stack([lsig(s.alpha), lsig(-s.alpha), lsig(s.beta), lsig(-s.beta)])
    out = []
    for r in range(s.m):
        slots = np.arange(r, n, s.m, dtype=int)
        on_ones = float(y[slots].sum())
        counts = torch.tensor([on_ones, len(slots) - on_ones, total_ones - on_ones,
                               (n - len(slots)) - (total_ones - on_ones)],
                              dtype=torch.float64)
        out.append(counts @ v)
    return torch.stack(out)


def _emission_marginalises_r(s, y):
    """Stage 0's emission wearing Stage P's signature: the same value for every offset.

    The headline failure mode of a stage that says it promoted a nuisance to a latent and
    did not. Every phase number such a model produces is chance by construction, and it
    passes any check that only looks at likelihood values in aggregate.
    """
    lp = S._emission_logp_all(s, y)
    return (torch.logsumexp(lp, -1) - math.log(s.m)).repeat(s.m)


def _emission_fixed_at_zero(s, y):
    """Assumes every crop starts on a downbeat -- SS3 P2's forbidden assumption.

    Realistic because it is what you write if you forget crops are un-aligned, and it was
    true of ~99% of Stage-0 crops (SS10.1), so it would have gone unnoticed for the whole
    of Stage 0.
    """
    lp = S._emission_logp_all(s, y)
    return lp[0].repeat(s.m)


# ------------------------------------------------------------------------------------
# read-out and psi corruptions -- the P1/P2 seam
# ------------------------------------------------------------------------------------
def _readout_uses_pointer_not_offset(s, obs):
    """Returns ``(-r) mod m``: the pointer state confused with the crop offset.

    SS4.1 invites this directly by describing ``r`` as "the bar pointer at the crop's first
    beat" one line after writing ``i == r (mod m)``, which is its negation. The two agree
    at ``r = 0`` -- which is ~99% of Stage-0 crops (SS10.1) -- so on the old bar-aligned
    data this bug would have been invisible.
    """
    return int((-int(s.predict(obs).argmax())) % s.m)


def _readout_argmax_over_emission(s, obs):
    """SS9's named mutant: argmax over the emission instead of over the posterior.

    Not merely wrong but a LEAK: the emission reads y, so this read-out is not deployable
    at all. It will often score WELL, which is what makes it dangerous.
    """
    y = getattr(s, "_stashed_y", None)
    if y is None:
        return int(s.predict(obs).argmax())
    return int(s.emission_logp_all(y).argmax())


def _elbo_stashing(s, obs, y):
    """Stash y during training so the read-out above can quietly reuse it."""
    s._stashed_y = y
    return S._elbo(s, obs, y)


def _predict_leaks_y(s, obs):
    """The realistic leak: state stashed during training, reused at deployment."""
    y = getattr(s, "_stashed_y", None)
    if y is None:
        return s.prior_logp(obs)
    return s.exact_posterior(obs, y)


def _psi_ignores_h(s, obs):
    """SS9's named mutant: the prior drops h, so the deployable path can learn nothing."""
    return torch.zeros(s.m, dtype=torch.float64) * s.w_beat.sum() - math.log(s.m)


def _psi_shift_invariant_summary(s, obs):
    """SS9's named mutant: a POOLED summary substituted for the per-beat potential.

    ``mean`` and ``max`` over time are permutation-invariant, so this psi is genuinely
    blind to position and must score at chance. SS4.4 predicts exactly this for
    ``mean_max`` / ``peak_summary`` / ``AutocorrHead``.

    Note the scope. This is a claim about THIS pooling of THIS synthetic bench, where the
    crops really are shifts of one signal, and it is enforced as a mutant of the oracle --
    not as a proof about any real head. Real crops at different offsets are different audio
    WINDOWS rather than cyclic shifts, and real "shift-invariant" heads leak position at
    small amplitude through boundary effects, so SS8.3's P-0 remains something to measure
    on real data, not something this suite asserts.
    """
    h = torch.as_tensor(np.asarray(obs["h"], dtype=np.float64), dtype=torch.float64)
    pooled = torch.cat([h.mean(0), h.max(0).values])          # [4], position-blind
    logits = (pooled[:2] @ s.w_beat).repeat(s.m)
    return logits - torch.logsumexp(logits, -1)


def _psi_offset_bias(s, obs):
    """A learned per-offset bias added to the prior logits.

    Kills shift-equivariance, and models nothing: Stage P's latent is uniform over
    ``0..m-1`` BY CONSTRUCTION (SS1 -- a crop may begin anywhere in the bar), so there is
    no marginal for a bias to learn. Realistic because it is the first thing anyone writes
    when copying a classifier head across from Stage 0, where a class bias IS meaningful
    (SS10.2: gtzan is 93% m=4).

    The magnitude is deliberately comparable to the fitted audio potential rather than
    token, and that is itself a finding. A small per-offset bias on this bench changes no
    prediction at all -- the synthetic potentials are enormous -- so it is caught only by
    the P1/P2 consistency identity, which is a fragile place to catch it. A bias big enough
    to compete with the audio is caught by the read-out properties that are supposed to
    catch it. So "psi must carry no offset bias" is enforceable here only at the scale
    where the bias actually competes; on real data, where logit scales are far smaller, a
    bias of any size is a live risk and the equivariance argument is the reason to forbid
    it outright.
    """
    base = S._prior_logits(s, obs)
    bias = torch.linspace(0.0, 40.0, s.m, dtype=torch.float64)
    return base + bias


def _psi_frozen(m, capacity):
    """Freeze psi: registered but never CALLED, v1's shipped bug.

    45.3% / 50.88% of parameters sat at exactly zero gradient for weeks while the prior's
    scales were effectively constants.
    """
    s = S.oracle(m=m, capacity=capacity)
    s.name = "mutant"
    s.w_beat = torch.tensor([0.3, -0.1], dtype=torch.float64, requires_grad=False)
    return s


def _predict_constant(s, obs):
    """Total collapse: the same offset for every crop, regardless of h.

    v1's collapse detector read 1.2e7 after training and nobody looked (SS10.4 records the
    metric that was invariant to it).
    """
    out = torch.full((s.m,), -math.log(s.m), dtype=torch.float64)
    out[0] = 0.0
    return out - torch.logsumexp(out, -1)


# ------------------------------------------------------------------------------------
# objective corruptions
# ------------------------------------------------------------------------------------
def _kl_flipped(s, obs, y):
    """``KL(p_psi || q)`` in place of ``KL(q || p_psi)``. SS4.6 pins the direction."""
    q_logp = s.q_logp(obs, y)
    prior = s.prior_logp(obs)
    recon = (q_logp.exp() * s.emission_logp_all(y)).sum()
    kl = (prior.exp() * (prior - q_logp)).sum()
    return recon - kl


def _elbo_sign(s, obs, y):
    """``recon + kl``: the KL stops being a penalty, so nothing restrains q."""
    q_logp = s.q_logp(obs, y)
    q = q_logp.exp()
    recon = (q * s.emission_logp_all(y)).sum()
    kl = (q * (q_logp - s.prior_logp(obs))).sum()
    return recon + kl


def _elbo_sampled(s, obs, y):
    """A one-sample estimate of an expectation that has exactly m = 4 terms.

    Unbiased, so every identity still holds IN EXPECTATION -- which is precisely why it is
    dangerous: it looks correct on paper and turns every downstream comparison into a coin
    flip. This project has already lost 17 A/B results to seed noise that turned out to be
    float dust. Spelled with numpy so an AST scan for ``torch.randn`` would miss it.
    """
    q_logp = s.q_logp(obs, y)
    q = q_logp.exp()
    p = q.detach().numpy().astype(np.float64)
    k = int(np.random.default_rng().choice(s.m, p=p / p.sum()))
    recon = s.emission_logp_all(y)[k] * q[k] / q[k].detach()
    kl = (q * (q_logp - s.prior_logp(obs))).sum()
    return recon - kl


def _q_is_prior(s, obs, y):
    """Q ignores y: the latent is dead and q carries nothing the prior lacks."""
    return s.prior_logp(obs)


def _q_ignores_h(s, obs, y):
    """Q drops h: nothing is amortized, so the deployable path can learn nothing."""
    lik = s.emission_logp_all(y)
    return lik - torch.logsumexp(lik, -1)


def _posterior_ignores_prior(s, obs, y):
    """Bayes without the prior: ``p(r|y,h)`` proportional to the likelihood alone."""
    lp = s.emission_logp_all(y)
    return lp - torch.logsumexp(lp, -1)


def _emission_inert(m, capacity):
    """Tie ``beta`` to ``alpha``, so y carries no phase information at all.

    The offset then cannot matter however well everything else is wired: the 'latent is
    decorative' failure.
    """
    s = S.oracle(m=m, capacity=capacity)
    s.name = "mutant"
    s.beta = s.alpha
    return s


# ------------------------------------------------------------------------------------
# P2 corruptions
# ------------------------------------------------------------------------------------
def _slip_ignored(s):
    """The transition always advances, whatever ``eps`` says -- P1 with extra steps.

    SS4.1: "an implementation of P2 without slip has implemented P1 with extra steps, and a
    test must assert the reduction". This is that implementation. It passes the ``eps = 0``
    reduction trivially, which is why the reduction alone is not enough and
    ``check_p2_slip_actually_changes_the_model`` exists.
    """
    T = torch.zeros((s.m, s.m), dtype=torch.float64)
    for a in range(s.m):
        T[a, (a + 1) % s.m] = 1.0
    return torch.log(T) + 0.0 * s.eps_logits.sum()


def _transition_unnormalised(s):
    """Slip mass added on top of a full advance, so rows sum to more than one.

    Not a distribution, so the 'evidence' the recursion returns is not a log-probability
    and the P1 reduction silently acquires an offset proportional to n.
    """
    eh, es = s.slip()
    T = torch.zeros((s.m, s.m), dtype=torch.float64)
    for a in range(s.m):
        T[a, (a + 1) % s.m] = T[a, (a + 1) % s.m] + 1.0
        T[a, a] = T[a, a] + eh
        T[a, (a + 2) % s.m] = T[a, (a + 2) % s.m] + es
    return torch.log(T)


def _p2_forgets_prior_partition(s, obs, y):
    """Omits ``- log Z(potentials)``, so P2's 'evidence' is not normalised in y.

    SS4.4 puts audio potentials on the chain, which makes the prior unnormalised over
    paths; its partition function has to be divided out. Forgetting it is invisible on any
    single crop -- the term is constant in y -- and breaks the P1 reduction, which is
    exactly the kind of bug that only a cross-stage identity catches.
    """
    return s._chain_logz(s.node_potentials(obs) + s.p2_emission(y))


def _p2_pointer_advances_backwards(s):
    """The pointer advances by ``-1`` instead of ``+1``.

    PROVABLY EQUIVALENT, and this entry is here because the first draft of this registry
    asserted the opposite and was wrong -- the mutant survived the sweep, and the survival
    was correct.

    The reflection ``s -> (-s) mod m`` maps advance ``+1`` to advance ``-1``, maps skip
    ``+2`` to ``-2``, and fixes state 0. Both the node potentials and the emission are
    supported on state 0 alone (SS4.4: the audio potential says "is this beat a downbeat"),
    and the initial pointer distribution is uniform, so the reflection is a symmetry of
    every term in the path weight. Every path's weight is therefore permuted, not changed:
    the partition function, the P2 log evidence, the ``eps = 0`` reduction to P1, and the
    SS7 read-out ``{i : s_i == 0}`` are all identical. Verified numerically at
    ``8.9e-16`` across ``eps`` settings from 0 to 0.25.

    So the DIRECTION of the bar pointer is a gauge freedom at P2, exactly as the pointer
    basis is. What is NOT free is the relation between the pointer and the crop offset at
    the deployable interface -- that is pinned by SS6.2, and getting it wrong is
    ``readout_uses_pointer_not_offset``, which is killed.
    """
    eh, es = s.slip()
    adv = 1.0 - eh - es
    T = torch.zeros((s.m, s.m), dtype=torch.float64)
    for a in range(s.m):
        T[a, (a - 1) % s.m] = T[a, (a - 1) % s.m] + adv
        T[a, a] = T[a, a] + eh
        T[a, (a - 2) % s.m] = T[a, (a - 2) % s.m] + es
    return torch.log(T)


def _relabel_pointer_states(rot: int):
    """Rotate the P2 pointer BASIS consistently, inverting nowhere it is observed.

    PROVABLY EQUIVALENT and required to survive -- SS9 names "a consistent relabelling of r
    states together with the emission" as an equivalent candidate. The pointer basis is a
    genuinely free internal choice: nothing in the spec fixes whether ``s_i`` counts up
    from the downbeat or toward it. Rotating the node potentials and the emission by the
    same amount permutes every path weight identically, so the partition function -- and
    therefore the P2 log evidence, and therefore the P1 reduction -- is unchanged.

    Contrast ``downbeat_off_by_one``, which shifts the emission WITHOUT shifting anything
    else, and ``_p2_pointer_advances_backwards``, which changes the transition's direction
    rather than the labelling. Those are observable; this is not.
    """
    def node(s, obs):
        g = s.beat_potential(obs)
        out = torch.zeros((g.shape[0], s.m), dtype=torch.float64)
        out[:, rot % s.m] = g
        return out

    def emis(s, y):
        y_t = torch.as_tensor(np.asarray(y, dtype=np.float64), dtype=torch.float64)
        lsig = torch.nn.functional.logsigmoid
        on = y_t * lsig(s.alpha) + (1 - y_t) * lsig(-s.alpha)
        off = y_t * lsig(s.beta) + (1 - y_t) * lsig(-s.beta)
        E = off.unsqueeze(1).repeat(1, s.m)
        E[:, rot % s.m] = on
        return E

    return node, emis


def _relabelled_subject(m, capacity):
    """Build an oracle whose P2 pointer basis is rotated by one, consistently."""
    s = S.oracle(m=m, capacity=capacity)
    s.name = "mutant"
    node, emis = _relabel_pointer_states(1)
    s.hooks["node_potentials"] = node
    s.p2_emission = lambda y, _s=s, _e=emis: _e(_s, y)
    return s


# ------------------------------------------------------------------------------------
# builders
# ------------------------------------------------------------------------------------
def _with(hooks, **kw):
    def build(m, capacity):
        s = S.oracle(m=m, capacity=capacity)
        s.hooks.update(hooks)
        s.name = "mutant"
        for k, v in kw.items():
            setattr(s, k, v)
        return s
    return build


MUTANTS = {
    # emission / phase
    "downbeat_off_by_one": (_with({"emission_logp_all": _downbeat_off_by_one}),
                            "downbeat lands one beat late (EQUIVALENT at Stage 0, WRONG here)"),
    "emission_marginalises_r": (_with({"emission_logp_all": _emission_marginalises_r}),
                                "Stage 0's emission with Stage P's signature: r not a latent"),
    "emission_fixed_at_zero": (_with({"emission_logp_all": _emission_fixed_at_zero}),
                               "assumes every crop starts on a downbeat (SS3 P2)"),
    "emission_inert": (_emission_inert,
                       "alpha == beta: y carries no phase information"),
    # read-out / psi
    "readout_uses_pointer_not_offset": (_with({"predict_offset": _readout_uses_pointer_not_offset}),
                                        "pointer state returned as the crop offset (SS4.1 seam)"),
    "readout_argmax_over_emission": (_with({"predict_offset": _readout_argmax_over_emission,
                                            "elbo": _elbo_stashing}),
                                     "argmax over the emission, not the posterior (SS9)"),
    "psi_ignores_h": (_with({"prior_logits": _psi_ignores_h}),
                      "the prior drops h (SS9)"),
    "psi_shift_invariant_summary": (_with({"prior_logits": _psi_shift_invariant_summary}),
                                    "a pooled, position-blind summary substituted for psi (SS9)"),
    "psi_offset_bias": (_with({"prior_logits": _psi_offset_bias}),
                        "a learned per-offset bias breaks shift-equivariance"),
    "psi_frozen": (_psi_frozen,
                   "psi gets no gradient: registered but never called (v1, 50.88%)"),
    "predict_leaks_y": (_with({"predict": _predict_leaks_y, "elbo": _elbo_stashing}),
                        "deployable predict() reuses a y stashed during training"),
    "predict_constant": (_with({"predict": _predict_constant}),
                         "the deployed read-out collapses to one offset"),
    # objective
    "kl_flipped": (_with({"elbo": _kl_flipped}),
                   "KL(p||q) instead of KL(q||p) in the bound (SS4.6)"),
    "elbo_sign": (_with({"elbo": _elbo_sign}),
                  "recon + kl: the KL stops being a penalty"),
    "elbo_sampled": (_with({"elbo": _elbo_sampled}),
                     "an enumerable m=4 expectation estimated by one sample"),
    "q_is_prior": (_with({"q_logp": _q_is_prior}),
                   "q ignores y: the latent is dead"),
    "q_ignores_h": (_with({"q_logp": _q_ignores_h}),
                    "q ignores h: nothing is amortized"),
    "posterior_ignores_prior": (_with({"exact_posterior": _posterior_ignores_prior}),
                                "Bayes rule without the prior"),
    # P2
    "slip_ignored": (_with({"log_transition": _slip_ignored}),
                     "transition always advances: P1 with extra steps (SS4.1)"),
    "transition_unnormalised": (_with({"log_transition": _transition_unnormalised}),
                                "slip added on top of a full advance: rows exceed one"),
    "p2_forgets_prior_partition": (_with({"p2_log_evidence": _p2_forgets_prior_partition}),
                                   "omits -log Z(potentials): P2 evidence not normalised"),
    # provably equivalent -- these must SURVIVE
    "global_phase_offset_one_bar": (_with({"emission_logp_all": _global_phase_offset_one_bar}),
                                    "a global offset of exactly one bar: the identity (SS9)"),
    "emission_via_counts": (_with({"emission_logp_all": _emission_via_counts}),
                            "the emission through the counts linearisation: exact algebra"),
    "relabel_pointer_states": (_relabelled_subject,
                               "the P2 pointer basis rotated consistently (SS9)"),
    "p2_pointer_advances_backwards": (_with({"log_transition": _p2_pointer_advances_backwards}),
                                      "the bar pointer runs backwards: the reflection s -> -s"),
}


# Mutants provably indistinguishable from correct code. They are EXPECTED to survive;
# killing one means a property asserts an implementation detail the model quotients out,
# which breaks the "proper => passes" half of the iff.
#
# ``downbeat_off_by_one`` is deliberately NOT here, though it IS in Stage 0's equivalent
# set. See this module's docstring, and SPEC_phase.md SS11 A2.
EQUIVALENT = {"global_phase_offset_one_bar", "emission_via_counts", "relabel_pointer_states",
              "p2_pointer_advances_backwards"}


def build(name: str, m: int = None, capacity: str = "full"):
    """Construct a named mutant.

    Args:
        name: A key of :data:`MUTANTS`.
        m: Beats per bar; defaults to the Stage-P constant.
        capacity: Encoder capacity.

    Returns:
        A corrupted :class:`subject_p.StageP`.
    """
    builder, _ = MUTANTS[name]
    return builder(R.STAGE_P_M if m is None else int(m), capacity)


def describe(name: str) -> str:
    """Return the one-line description of a named mutant.

    Args:
        name: A key of :data:`MUTANTS`.

    Returns:
        The description string.
    """
    return MUTANTS[name][1]
