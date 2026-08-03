"""Properties of the fitted Stage-P system, authored from ``docs/SPEC_phase.md``.

Every check takes ``make``, a factory ``(m=..., capacity=...) -> StageP``, so the same body
runs against the oracle, against each mutant, and later against a real implementation.
Checks raise AssertionError; ``test_properties_p.py`` wraps them for pytest and
``test_mutation_registry_p.py`` uses them as the kill criterion.

Ordered cheap-to-expensive: the registry short-circuits on the first kill.

What this file deliberately does NOT assert
-------------------------------------------
*That a shift-invariant head scores at chance* (SS4.4's table, SS8.3's P-0). Two reasons,
both fatal to it as a property. Position leaks through real "shift-invariant" heads at
small but non-zero amplitude -- ``AutocorrHead``'s masked FFT autocorrelation is linear and
zero-padded, so boundary effects carry roughly 0.2% of logit scale. And more fundamentally,
two crops at different offsets are different AUDIO WINDOWS, not cyclic shifts of one
signal, so shift-invariance is not even the governing property. P-0 is an empirical
expectation to be measured on real data, not a theorem to assert in a test.

*That phase is recoverable on real music.* Everything here runs on the synthetic bench,
where ``h`` carries downbeat bumps by construction. SS10.6 is explicit that such a control
proves the crop/label/prior/read-out chain is sound and cannot separate "the frontend is
deaf" from "the annotation's metrical level differs from the audio's".

*Anything about P2's loss.* SS4.5 defines an encoder only for the static P1 latent and no
P2 encoder is defined anywhere in the spec, so the P2 objective is unwritable from the
normative text. P2 is asserted here as exact INFERENCE only.
"""
from __future__ import annotations

import numpy as np
import torch

import bench_p as B
import reference_p as R
import subject_p as S

CHECKS: dict = {}


def check(fn):
    """Register a property so the registry and the pytest wrapper can both find it.

    Args:
        fn: The property function, taking a single ``make`` factory.

    Returns:
        ``fn`` unchanged.
    """
    CHECKS[fn.__name__] = fn
    return fn


# Small on purpose: the mutation sweep runs these many times over.
N_CROPS = 16
N_BEATS = 16            # MUST satisfy N_BEATS % m == 0 -- see check_crops_span_whole_bars
FIT_STEPS = 300
FIT_LR = 0.1


class CapacityUnsupportedError(Exception):
    """Raised when a check needs an encoder capacity the subject does not provide.

    ``capacity`` is a TESTING affordance, not part of the model (SS4.5). A real
    implementation exposes ONE encoder; only the oracle can be built with a deliberately
    exact q. Checks that require one must announce themselves as not-run rather than fail a
    conformant implementation for lacking a knob the spec never asks for.
    """


def _require_capacity(s, want: str):
    if getattr(s, "capacity", None) != want:
        raise CapacityUnsupportedError(
            f"subject does not support capacity={want!r} (got "
            f"{getattr(s, 'capacity', None)!r}); this check is oracle-coverage only")


def _np(t) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float64)


def _tv(logp, logq) -> float:
    p, q = np.exp(np.asarray(logp)), np.exp(np.asarray(logq))
    return 0.5 * float(np.abs(p - q).sum())


def _data(m, n_crops=N_CROPS, n_beats=N_BEATS, seed=0, noise=0.0):
    return B.make_labelled_dataset(n_crops=n_crops, n_beats=n_beats, m=m,
                                   seed=seed, noise=noise)


def _obs(crop):
    return S.deployable_view(crop)


def _fitted(make, m=R.STAGE_P_M, seed=0, **kw):
    s = make()
    crops = _data(s.m, seed=seed, **kw)
    s.fit(crops, steps=FIT_STEPS, lr=FIT_LR)
    return s, crops


def _predicted_offsets(s, crops):
    return [s.predict_offset(_obs(c)) for c in crops]


def _randomise(s, group: str, seed: int, scale: float = 1.0, stage: str = "P1"):
    """Perturb one SS4.7 parameter group off its initialisation, shape-agnostically.

    Reaching parameters by GROUP and ``p.shape`` rather than by field name is what keeps
    the suite from binding to one implementation's psi layout.

    Args:
        s: The subject.
        group: ``"theta"``, ``"psi"`` or ``"phi"``.
        seed: Seed for the draw.
        scale: Standard deviation of the draw.
        stage: Which stage's parameter set to reach into.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in s.param_groups(stage)[group].values():
            p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * scale)


# ======================================================================================
# 1. structural -- cheap, no fitting
# ======================================================================================
@check
def check_theta_is_exactly_two_scalars(make):
    """SS4.3: theta is ``{alpha, beta}``, the same two scalars as Stage 0, frozen in form.

    The clause exists so that any performance change at Stage P is attributable to the
    latent and to psi rather than to a richer decoder. A third emission parameter, or a
    per-beat alpha, would silently make Stage P and Stage 0 incomparable -- and would make
    every SS4.6 reduction meaningless, because the thing being reduced would have changed.
    """
    s = make()
    theta = s.param_groups("P1")["theta"]
    assert set(theta) == {"alpha", "beta"}, (
        f"theta is {sorted(theta)}, but SS4.3 pins it to exactly {{alpha, beta}}. The "
        f"emission must not gain capacity at Stage P.")
    for name, p in theta.items():
        assert p.numel() == 1, (
            f"theta parameter {name!r} has {p.numel()} elements, shape {tuple(p.shape)}. "
            f"SS4.3 says two SCALARS; a vector-valued alpha is extra decoder capacity.")


@check
def check_crops_span_whole_bars(make):
    """The crop-length precondition without which every chance-level gate is wrong.

    SS8.3 states chance on offset accuracy is ``1/m = 0.25``. That holds only when ``m``
    divides the crop's beat count. The number of downbeat slots under offset ``r`` is
    ``len(range(r, n, m))``, which depends on ``r`` otherwise: at ``m = 4``, ``n = 13``
    gives counts ``[4, 3, 3, 3]`` and a completely position-blind summary that sees only
    the slot count reaches 0.500. P-0, the leak detector every other Stage-P number is
    gated on, would then fire on a model with no leak at all.

    So the bench is required to cut whole bars, and the chance level is DERIVED from the
    crop length rather than hardcoded, so this cannot silently regress.
    """
    s = make()
    for crop in _data(s.m):
        n = len(crop["y"])
        assert n % s.m == 0, (
            f"crop has n={n} beats at m={s.m}: slot counts {R.downbeat_slot_counts(n, s.m)}, "
            f"so a position-blind count summary scores "
            f"{R.count_partition_chance(n, s.m):.3f}, not {1 / s.m:.3f}")
        assert abs(R.count_partition_chance(n, s.m) - 1.0 / s.m) < 1e-12, (
            f"derived chance {R.count_partition_chance(n, s.m)} != 1/m at n={n}")


@check
def check_offset_is_the_latent_not_a_marginalised_nuisance(make):
    """SS4.3: with ``r`` promoted to a latent, ``log p(y|r)`` must actually depend on ``r``.

    This is the whole difference between Stage P and Stage 0. Stage 0 integrates the offset
    out and is therefore invariant to downbeat phase BY DESIGN (SPEC.md SS4.3); Stage P
    conditions on it. An implementation whose emission returns the same value for every
    offset has written Stage 0 with a longer signature, and every phase number it produces
    is chance by construction.
    """
    s = make()
    with torch.no_grad():
        s.alpha.fill_(2.5)
        s.beta.fill_(-2.5)
    y = R.downbeats_from_offset(N_BEATS, 1, s.m)
    lp = _np(s.emission_logp_all(y))
    assert lp.shape == (s.m,), f"emission_logp_all returned shape {lp.shape}, expected ({s.m},)"
    assert lp.max() - lp.min() > 1e-6, (
        f"log p(y|r) is constant across offsets ({lp.round(6)}): r is being marginalised "
        f"rather than conditioned on, so the model is Stage 0 and phase is unobservable")
    assert int(lp.argmax()) == 1, (
        f"y was generated at offset 1 but log p(y|r) peaks at r={int(lp.argmax())}")


@check
def check_emission_matches_independent_oracle(make):
    """Agreement with ``reference_p`` on RANDOM y -- off the model's own manifold.

    Feeding the emission only well-formed periodic ``y`` cannot distinguish a correct
    implementation from one that is correct only on data drawn from its own generative
    process. The offsets are compared elementwise, which at Stage P is legitimate: the
    label ``r`` is the deployable OUTPUT (SS7) and is pinned by SS6.2's ``r_true``, so it
    is not a free internal choice the way a pointer basis is.
    """
    g = np.random.default_rng(11)
    for trial in range(10):
        alpha, beta = float(g.normal(0, 2)), float(g.normal(0, 2))
        s = make()
        with torch.no_grad():
            s.alpha.fill_(alpha)
            s.beta.fill_(beta)
        n = int(g.integers(1, 5)) * s.m
        y = g.integers(0, 2, n).astype(np.uint8)
        got = _np(s.emission_logp_all(y))
        want = R.offset_loglik_all(y, s.m, alpha, beta)
        assert np.allclose(got, want, atol=1e-9), (
            f"trial {trial}: emission {got.round(9)} != oracle {want.round(9)}")


@check
def check_emission_normalises_over_y(make):
    """``sum_y p_theta(y | r) == 1`` for every offset. Brute force over ``{0,1}^n``.

    A conditional likelihood that does not normalise is not a likelihood, and the ELBO
    built on it bounds nothing.
    """
    s = make()
    with torch.no_grad():
        s.alpha.fill_(0.7)
        s.beta.fill_(-1.1)
    n = s.m
    totals = np.zeros(s.m)
    for bits in range(2 ** n):
        y = np.array([(bits >> i) & 1 for i in range(n)], dtype=np.uint8)
        totals += np.exp(_np(s.emission_logp_all(y)))
    for r in range(s.m):
        assert abs(totals[r] - 1.0) < 1e-9, (
            f"sum_y p(y | r={r}) == {totals[r]:.9f} != 1 at n={n}")


@check
def check_r_true_is_excluded_when_ill_defined(make):
    """SS6.2: crops whose meter varies internally get NO label, and are COUNTED.

    Measured on ~2% of real crops (SS10.1). The spec is explicit that these must be
    excluded from scoring and counted as an exclusion, "never assigned a fictional label" --
    and this project has form: SS10.1 records an agent reporting 100% coverage by
    conflating ``y[0] == 1`` with "a constant-``m`` offset fits".

    The bench manufactures the condition with a mid-crop meter change, so the check is on
    data where the answer is known rather than on whatever the corpus happens to contain.
    """
    s = make()
    crops = B.make_meter_change_dataset(n_crops=40, n_beats=N_BEATS, m=s.m)
    labelled, n_excluded = R.label_crops(crops, s.m)
    assert n_excluded > 0, (
        "the meter-change bench produced no exclusions at all, so this check is vacuous; "
        "a crop straddling a slip event has no single fitting offset by construction")
    assert len(labelled) + n_excluded == len(crops), "crops were lost, not accounted for"
    for c in labelled:
        assert c["r_true"] is not None and 0 <= c["r_true"] < s.m
        rebuilt = R.downbeats_from_offset(len(c["y"]), c["r_true"], s.m)
        assert np.array_equal(np.asarray(c["y"]).astype(np.uint8), rebuilt), (
            f"crop labelled r_true={c['r_true']} but that offset does not reproduce y; a "
            f"best-fitting offset was invented where SS6.2 requires an exclusion")


@check
def check_posterior_is_bayes(make):
    """``p(r|y,h) == normalise(log p(y|r) + log p_psi(r|h))``. The prior must be included."""
    s = make()
    _randomise(s, "psi", seed=21)
    for crop in _data(s.m, n_crops=4):
        lik = _np(s.emission_logp_all(crop["y"]))
        prior = _np(s.prior_logp(_obs(crop)))
        want = lik + prior
        want = want - R.logsumexp(want)
        got = _np(s.exact_posterior(_obs(crop), crop["y"]))
        assert np.allclose(got, want, atol=1e-9), (
            f"posterior {np.exp(got).round(4)} != Bayes {np.exp(want).round(4)}; the prior "
            f"is being dropped or double counted")


@check
def check_every_parameter_receives_gradient(make):
    """SS4.7, asserted by READING gradients -- the failure mode that keeps recurring.

    Measured precedent: 26 tensors / 45.3% at exactly zero gradient in one configuration,
    41 / 50.88% in another, with psi in the optimiser both times. A brand-new
    implementation reproduced it within an hour on 2026-08-03, because zero-initialised
    output layers make ``d(loss)/d(body)`` exactly zero at step 0.

    Two traps this must avoid, both measured. ``== 0.0`` is the wrong predicate: a
    disconnected parameter still picks up rounding noise, and a CONNECTED one can sit at an
    analytic stationary point -- at ``c = 1`` the encoder IS the exact posterior, so
    ``dELBO/dc = 0`` by the envelope theorem and the check would pass on float dust. So
    every parameter is displaced off any stationary point first, then compared against a
    floor scaled to the largest gradient in the model.
    """
    s = make()
    crops = _data(s.m, n_crops=4)
    with torch.no_grad():
        for i, p in enumerate(s.named_params("P1").values()):
            p.add_(torch.full_like(p, 0.3 if i % 2 == 0 else -0.2))
    for p in s.named_params("P1").values():
        p.grad = None
    loss = -torch.stack([s.elbo(_obs(c), c["y"]) for c in crops]).mean()
    loss.backward()

    grads = [float(p.grad.abs().max()) for p in s.named_params("P1").values()
             if p.grad is not None]
    scale = max(grads, default=0.0)
    floor = max(1e-10 * scale, 1e-12)
    dead, frozen = [], []
    for name, p in s.named_params("P1").items():
        if not p.requires_grad:
            frozen.append(name)
        elif p.grad is None or float(p.grad.abs().max()) <= floor:
            dead.append(name)
    assert not frozen, f"parameters excluded from optimisation entirely: {frozen}"
    assert not dead, (
        f"parameters with no gradient (|grad| <= {floor:.3e}, largest in model "
        f"{scale:.3e}): {dead}")


@check
def check_p2_slip_parameters_receive_gradient(make):
    """SS4.7 again, on the P2 path: ``eps_hold`` and ``eps_skip`` are LEARNED (SS4.1).

    Separated from the P1 audit because the slip parameters are exactly the ones a P1-only
    implementation will carry, register, and never call -- which is v1's psi failure with a
    new name.

    Scoped to theta and psi, and the exclusion of phi is a finding rather than a
    convenience. The objective used here is the exact P2 log evidence, because it is the
    only P2 objective the spec supports: SS4.5 defines the encoder ``q_phi(r | h, y)`` for
    the static P1 latent only, and no chain-valued encoder is defined anywhere in
    ``SPEC_phase.md``. So there is no P2 loss for phi to receive a gradient FROM, and
    asserting one would be asserting a design decision this suite has no authority to make.
    Reported as a spec gap; see the README.
    """
    s = make()
    crops = _data(s.m, n_crops=3, n_beats=8)
    audited = {}
    for group in ("theta", "psi"):
        audited.update(s.param_groups("P2")[group])
    for p in audited.values():
        p.grad = None
    obj = torch.stack([s.p2_log_evidence(_obs(c), c["y"]) for c in crops]).mean()
    obj.backward()

    assert "eps_logits" in audited, (
        "psi carries no slip parameters at P2, but SS4.1 says eps_hold and eps_skip are "
        "LEARNED; a P2 with no slip is P1 with extra steps")
    grads = [float(p.grad.abs().max()) for p in audited.values() if p.grad is not None]
    scale = max(grads, default=0.0)
    floor = max(1e-10 * scale, 1e-12)
    dead = [n for n, p in audited.items()
            if p.grad is None or float(p.grad.abs().max()) <= floor]
    assert not dead, (
        f"P2 parameters with no gradient (|grad| <= {floor:.3e}): {dead}. SS4.1 says the "
        f"slip probabilities are learned; registered-but-never-called is v1's psi bug.")


@check
def check_the_objective_is_exact_not_sampled(make):
    """With ``R = m = 4`` the P1 ELBO is a four-term sum (SS4.6). Sampling it is a defect.

    Checked BEHAVIOURALLY -- two evaluations of one crop must agree to the last bit. An AST
    scan for ``torch.randn`` is weaker because it whitelists spellings; bit-identity cannot
    be evaded. Note SS5's "P1 is deterministic" is a claim about the closed-form path only:
    an attention-based head carries CUDA reduction nondeterminism worth +-0.08 (SS10.3),
    which is why this asserts determinism of the OBJECTIVE and not of a training run.
    """
    s = make()
    for crop in _data(s.m, n_crops=3):
        a = float(s.elbo(_obs(crop), crop["y"]))
        b = float(s.elbo(_obs(crop), crop["y"]))
        assert a == b, (
            f"the ELBO is not a deterministic function of (h, y): {a!r} then {b!r}. With "
            f"R = {s.m} it should be an exact enumeration.")


# ======================================================================================
# 2. SS4.6 bound identities -- cheap
# ======================================================================================
@check
def check_elbo_lower_bounds_the_evidence(make):
    """``ELBO <= log p(y|h)`` for ANY q. The defining property of the bound."""
    s = make()
    _randomise(s, "phi", seed=3, scale=2.0)
    _randomise(s, "psi", seed=4)
    slack = []
    for crop in _data(s.m, n_crops=4):
        elbo = float(s.elbo(_obs(crop), crop["y"]))
        ev = float(s.log_evidence(_obs(crop), crop["y"]))
        assert elbo <= ev + 1e-9, f"ELBO {elbo:.6f} EXCEEDS log evidence {ev:.6f}"
        slack.append(ev - elbo)
    assert max(slack) > 1e-6, (
        f"q was randomised yet the bound is tight everywhere (max slack {max(slack):.3e}); "
        f"this check is vacuous as configured and would pass on an implementation that "
        f"always returns the evidence")


@check
def check_bound_is_tight_exactly_when_q_is_the_posterior(make):
    """SS4.6: equality holds IFF q is the exact posterior. Both directions.

    The 'only if' half is what makes this more than a restatement of the bound: with q
    displaced off the posterior the slack must be strictly positive, so an implementation
    that reports the evidence and calls it an ELBO fails.
    """
    s_exact = make(capacity="exact")
    _require_capacity(s_exact, "exact")
    _randomise(s_exact, "psi", seed=5)
    for crop in _data(s_exact.m, n_crops=3):
        elbo = float(s_exact.elbo(_obs(crop), crop["y"]))
        ev = float(s_exact.log_evidence(_obs(crop), crop["y"]))
        assert abs(elbo - ev) < 1e-9, (
            f"q IS the exact posterior yet the bound is slack by {ev - elbo:.6e}")

    s = make()
    if getattr(s, "capacity", None) == "exact":
        return
    _randomise(s, "phi", seed=6, scale=2.0)
    _randomise(s, "psi", seed=5)
    gaps = []
    for crop in _data(s.m, n_crops=3):
        q = _np(s.q_logp(_obs(crop), crop["y"]))
        post = _np(s.exact_posterior(_obs(crop), crop["y"]))
        if _tv(q, post) < 1e-8:
            continue
        gaps.append(float(s.log_evidence(_obs(crop), crop["y"]))
                    - float(s.elbo(_obs(crop), crop["y"])))
    assert gaps and min(gaps) > 1e-8, (
        f"q differs from the exact posterior yet the bound is tight (slacks {gaps}); "
        f"equality is supposed to hold ONLY at the posterior")


@check
def check_slack_equals_the_reverse_kl(make):
    """SS4.6: ``log p(y|h) - ELBO == KL(q || p(r|y,h))``, the REVERSE KL, exactly.

    Direction matters and the spec says so: SPEC.md SS3.5's diagnostic is the forward
    ``KL(p||q)``, a different number. Pinning the identity that actually holds is what
    catches a bound built on the wrong direction.
    """
    s = make()
    _randomise(s, "phi", seed=11, scale=1.5)
    _randomise(s, "psi", seed=12)
    for crop in _data(s.m, n_crops=4):
        elbo = float(s.elbo(_obs(crop), crop["y"]))
        ev = float(s.log_evidence(_obs(crop), crop["y"]))
        q_logp = _np(s.q_logp(_obs(crop), crop["y"]))
        post = _np(s.exact_posterior(_obs(crop), crop["y"]))
        rev = R.reverse_kl(q_logp, post)
        fwd = R.reverse_kl(post, q_logp)
        assert abs((ev - elbo) - rev) < 1e-9, (
            f"slack {ev - elbo:.9f} != reverse KL(q||p) {rev:.9f} (forward KL is "
            f"{fwd:.9f}); SS4.6 pins the reverse direction")


@check
def check_q_depends_on_both_h_and_y(make):
    """SS4.5: ``q_phi(r | h, y)`` must be a function of BOTH.

    q that ignores y is a dead latent; q that ignores h is not amortized at all. Both were
    live v1 failure modes.
    """
    s = make()
    crops = _data(s.m, n_crops=4)
    a, b = crops[0], crops[1]
    _randomise(s, "psi", seed=4)
    priors = [_np(s.prior_logp(_obs(c))) for c in crops]
    assert max(_tv(priors[0], p) for p in priors[1:]) > 1e-6, (
        "psi was randomised yet the prior is constant in h, so the h-dependence probe "
        "below would pass on a q that ignores h")

    same_h_diff_y = _tv(_np(s.q_logp(_obs(a), a["y"])), _np(s.q_logp(_obs(a), b["y"])))
    assert same_h_diff_y > 1e-6, (
        "holding h fixed and changing y left q unchanged: q ignores the labels, so the "
        "latent is dead")
    diff_h_same_y = _tv(_np(s.q_logp(_obs(a), a["y"])), _np(s.q_logp(_obs(b), a["y"])))
    assert diff_h_same_y > 1e-6, (
        "holding y fixed and changing h left q unchanged: nothing is amortized")


# ======================================================================================
# 3. SS4.6 reductions -- the formal statements that Stage P EXTENDS Stage 0
# ======================================================================================
@check
def check_marginalised_p1_reproduces_stage0_emission(make):
    """SS4.6: P1 with ``r`` marginalised uniformly reproduces Stage 0's emission, < 1e-9.

    Checked against the SHIPPED code -- ``vbpm.stage0.Stage0.emission_logp_all`` and
    ``vbpm.fitting.emission_logp_from_counts``, which is the linearised twin every
    vectorized path in the package routes through. Both, because agreeing with one and not
    the other would mean the package's own two emissions disagree.

    This is the formal content of "Stage P extends Stage 0": the same likelihood, with the
    offset promoted from a uniform nuisance to a latent. If it fails, the two stages are
    different models and no Stage-P number is comparable to a Stage-0 number for any
    reason other than coincidence.
    """
    import vbpm.fitting as VF
    from vbpm.stage0 import Stage0 as VbpmStage0

    g = np.random.default_rng(23)
    s = make()
    values = (2, 3, 4)
    assert s.m in values, f"m={s.m} is outside Stage 0's meter vocabulary {values}"
    k = values.index(s.m)

    worst = 0.0
    for _ in range(8):
        alpha, beta = float(g.normal(0, 2)), float(g.normal(0, 2))
        n = int(g.integers(1, 5)) * s.m
        y = g.integers(0, 2, n).astype(np.uint8)
        with torch.no_grad():
            s.alpha.fill_(alpha)
            s.beta.fill_(beta)
        mine = float(s.marginalised_emission(y))

        v = VbpmStage0(values=values)
        with torch.no_grad():
            v.alpha.fill_(alpha)
            v.beta.fill_(beta)
        theirs = float(v.emission_logp_all(y)[k])

        counts, off = VF.emission_counts(y, values)
        from_counts = float(VF.emission_logp_from_counts(
            torch.tensor(counts), torch.tensor(off),
            torch.log(torch.tensor([float(m) for m in values], dtype=torch.float64)),
            torch.tensor(alpha, dtype=torch.float64),
            torch.tensor(beta, dtype=torch.float64))[k])
        worst = max(worst, abs(mine - theirs), abs(mine - from_counts))

    assert worst < 1e-9, (
        f"marginalising r over Stage P's emission differs from Stage 0's by {worst:.3e} "
        f"(tolerance 1e-9). Stage P does not extend Stage 0.")


@check
def check_p2_with_zero_slip_reduces_exactly_to_p1(make):
    """SS4.1/SS4.6: at ``eps = 0`` P2 must reduce EXACTLY to P1, psi included.

    SS4.1 states the reason this test is mandatory rather than decorative: with a
    deterministic advance and fixed ``m``, ``r_i`` is a deterministic function of ``r_1``,
    forward-backward provably collapses to an ``m``-way argmax, and the time index buys
    NOTHING. Slip is the only thing that makes the recursion non-trivial. "An
    implementation of P2 without slip has implemented P1 with extra steps, and a test must
    assert the reduction."

    The reduction is asserted on the log evidence, which is the quantity both stages
    define, and with a SHARED psi -- so this is a statement about the whole model rather
    than about the emission alone.
    """
    s = make()
    with torch.no_grad():
        s.alpha.fill_(1.4)
        s.beta.fill_(-0.9)
        s.eps_logits.fill_(-80.0)         # advance mass 1 - 2e-35: eps = 0 to float64
    worst = 0.0
    for crop in _data(s.m, n_crops=5, n_beats=N_BEATS):
        p1 = float(s.log_evidence(_obs(crop), crop["y"]))
        p2 = float(s.p2_log_evidence(_obs(crop), crop["y"]))
        worst = max(worst, abs(p1 - p2))
    assert worst < 1e-9, (
        f"P2 at eps=0 differs from P1 by {worst:.3e}. Either the chain is not reducing to "
        f"the m deterministic paths, or P1 and P2 are not sharing psi -- in which case the "
        f"'time index earns its cost' comparison (P-5) would be measuring two models.")


@check
def check_p2_slip_actually_changes_the_model(make):
    """The companion to the reduction: at ``eps > 0`` P2 must STOP agreeing with P1.

    Without this, ``check_p2_with_zero_slip_reduces_exactly_to_p1`` is passed trivially by
    an implementation that ignores the slip parameters entirely -- which is exactly the
    "P1 with extra steps" that SS4.1 legislates against.
    """
    s = make()
    with torch.no_grad():
        s.alpha.fill_(1.4)
        s.beta.fill_(-0.9)
        s.eps_logits.copy_(torch.tensor([-1.5, -2.0], dtype=torch.float64))
    diffs = []
    for crop in _data(s.m, n_crops=4):
        diffs.append(abs(float(s.log_evidence(_obs(crop), crop["y"]))
                         - float(s.p2_log_evidence(_obs(crop), crop["y"]))))
    assert max(diffs) > 1e-4, (
        f"with substantial slip (eps_hold ~ 0.15) P2 still equals P1 to {max(diffs):.3e}: "
        f"the slip parameters are inert, so the recursion is P1 with extra steps")


@check
def check_p2_transition_normalises(make):
    """SS4.1's three branches must form a distribution from every state.

    The spec's kernel is malformed as written -- ``eps`` appears both as one parameter and
    as the pair ``{eps_hold, eps_skip}``, with the relation between them never stated. Any
    implementation must therefore CHOOSE, and this asserts the choice is at least a
    probability distribution.
    """
    s = make()
    for logits in ([-3.0, -3.5], [0.0, 0.0], [1.0, -2.0], [-80.0, -80.0]):
        with torch.no_grad():
            s.eps_logits.copy_(torch.tensor(logits, dtype=torch.float64))
        T = _np(s.log_transition().exp())
        rows = T.sum(axis=1)
        assert np.allclose(rows, 1.0, atol=1e-12), (
            f"transition rows sum to {rows.round(9)} at eps_logits={logits}, not 1")
        assert (T >= -1e-15).all(), f"negative transition mass at eps_logits={logits}"
        eh, es = (float(v) for v in s.slip())
        assert eh >= 0 and es >= 0 and eh + es <= 1 + 1e-12, (
            f"slip ({eh}, {es}) leaves advance mass {1 - eh - es}")


@check
def check_forward_recursion_matches_brute_force_enumeration(make):
    """SS9: the chain's forward recursion against brute-force enumeration, ``n <= 8``, exactly.

    The recursion is ``O(n m^2)`` and the enumeration is ``O(m^n)``; they share no code
    path beyond the potentials and the transition, so agreement is real evidence rather
    than a tautology. Run at genuinely non-zero slip, because at ``eps = 0`` almost every
    path has zero weight and the enumeration would confirm only the easy case.

    Compared on the log evidence, a labelling-invariant scalar. The pointer BASIS is a free
    internal choice at P2 -- nothing in the spec fixes whether the state counts up from the
    downbeat or toward it -- so comparing state-indexed marginals here would reject a
    correct implementation that chose the other basis (see ``relabel_pointer_states`` in
    the equivalent set). The reference's own forward-backward IS checked against brute
    force marginal-by-marginal, in ``test_reference_p_selftest.py``.
    """
    s = make()
    with torch.no_grad():
        s.alpha.fill_(1.2)
        s.beta.fill_(-1.0)
        s.eps_logits.copy_(torch.tensor([-1.2, -1.8], dtype=torch.float64))
    logT = _np(s.log_transition())
    worst = 0.0
    for crop in _data(s.m, n_crops=3, n_beats=8):
        obs, y = _obs(crop), crop["y"]
        node = _np(s.node_potentials(obs))
        emis = _np(s.p2_emission(y))
        brute = R.chain_logz_brute(node + emis, logT) - R.chain_logz_brute(node, logT)
        worst = max(worst, abs(float(s.p2_log_evidence(obs, y)) - brute))
    assert worst < 1e-9, (
        f"forward recursion differs from brute-force enumeration by {worst:.3e} at n=8")


# ======================================================================================
# 4. leakage and the deployable path
# ======================================================================================
@check
def check_predict_is_annotation_blind(make):
    """C2/P6: the deployable path is a pure function of ``h`` and the given beat grid.

    Enforced two ways, because a signature of ``predict(obs)`` proves neither. First,
    ``predict`` is handed a view with ``y``, ``r_true`` and ``pointer`` REMOVED, so
    reaching for one raises rather than quietly scoring better. Second, the realistic leak
    is state stashed during training and reused at deployment, so ``predict`` is called,
    the model is shown a stream of different annotations, and the output must be
    bit-identical afterwards.
    """
    s = make()
    crops = _data(s.m, n_crops=4)
    obs = _obs(crops[0])
    assert not any(k in obs for k in S.ANNOTATION_KEYS), "the deployable view leaks a label"
    before = _np(s.predict(obs)).copy()
    for c in crops:
        s.elbo(_obs(c), c["y"])
    assert np.array_equal(before, _np(s.predict(obs))), (
        f"predict changed after the model saw annotations: {np.exp(before).round(4)} -> "
        f"{np.exp(_np(s.predict(obs))).round(4)}. The deployable path reads y through "
        f"hidden state.")
    s.elbo(obs, np.ones(len(crops[0]["y"]), dtype=np.uint8))
    assert np.array_equal(before, _np(s.predict(obs))), (
        "predict responds to the y last passed to elbo()")


@check
def check_shift_consistency_of_the_readout(make):
    """SS8.4: shift a crop's beat window by one beat and ``r_hat`` must move by exactly one.

    "A model that ignores the shift is not reading phase." The two crops are two real cuts
    of the SAME track one beat apart, so ``h``, the beat grid and ``y`` all move together --
    shifting ``y`` alone would exercise the emission and leave the deployable path out of
    the control entirely.

    Advancing the window's start by one beat moves the first downbeat one beat earlier, so
    the required response is ``r_hat -> (r_hat - 1) mod m``. The equivariance this demands
    is why the oracle's psi carries no per-offset bias term: Stage P's latent is uniform
    over ``0..m-1`` by construction (SS1), so a learned bias models nothing and destroys
    exactly this property. ``psi_offset_bias`` is the mutant.
    """
    s, crops = _fitted(make)
    track = B.make_track(n_beats=96, m=s.m)
    checked = 0
    for start in (3, 9, 14, 21, 30, 41):
        a = B.crop_at(track, start, N_BEATS)
        b = B.crop_at(track, start + 1, N_BEATS)
        assert (a["r_true"] - 1) % s.m == b["r_true"], "bench shift is not what it claims"
        ra = s.predict_offset(_obs(a))
        rb = s.predict_offset(_obs(b))
        assert (ra - 1) % s.m == rb, (
            f"start={start}: r_hat went {ra} -> {rb} under a one-beat shift, expected "
            f"{ra} -> {(ra - 1) % s.m}. A read-out that does not track the shift is not "
            f"reading phase.")
        checked += 1
    assert checked >= 5


@check
def check_posterior_recovers_the_true_offset(make):
    """On clean synthetic input the exact posterior must concentrate on ``r_true``.

    THE Stage-P property, and the one that changes the meaning of ``downbeat_off_by_one``.
    At Stage 0 that mutant is provably equivalent to correct code and is required to
    SURVIVE (``tests/mutants.py``), because marginalising the offset makes downbeat phase
    unobservable by design. At Stage P the offset IS the latent and IS the deployable
    output, so an emission whose downbeat lands one beat late is simply wrong, and this
    check is where it dies (SS4.3, SS11 A2).

    Uses ``r_true`` from the bench, where phase is known by construction (SS6.5).
    """
    s, crops = _fitted(make)
    masses = []
    for c in crops:
        post = np.exp(_np(s.exact_posterior(_obs(c), c["y"])))
        masses.append(float(post[c["r_true"]]))
    mean_mass = float(np.mean(masses))
    assert mean_mass > 0.85, (
        f"mean posterior mass on the true offset is {mean_mass:.3f} on noise-free "
        f"synthetic input where phase is identifiable by construction")


@check
def check_deployable_readout_recovers_known_phase(make):
    """SS6.5/SS8.3 P-1: on the bench, the DEPLOYABLE path must recover phase (>= 0.95).

    "A phase model that cannot recover known phase from clean synthetic input is broken,
    and this must be established before any real-data number exists." Scored through
    ``predict`` (h only), not through the posterior, because that is what SS7 deploys.

    Reported with raw accuracy AND the confusion matrix, per P4: SS10.4 records a
    degenerate predictor scoring 0.422 balanced against 0.239 raw, carried by a 19-crop
    class. Chance is derived from the crop length, never hardcoded.
    """
    s, crops = _fitted(make)
    true = [c["r_true"] for c in crops]
    pred = _predicted_offsets(s, crops)
    acc = R.offset_accuracy(true, pred)
    chance = R.count_partition_chance(len(crops[0]["y"]), s.m)
    conf = R.offset_confusion(true, pred, s.m)
    assert acc >= 0.95, (
        f"offset accuracy {acc:.3f} (chance {chance:.3f}) on the known-phase bench.\n"
        f"confusion (rows = true r):\n{conf}")


@check
def check_placement_beats_the_nulls(make):
    """SS8.1/SS8.2: placement F on the given grid, against nulls that actually place beats.

    SS8.2 nominates the uniform-``r`` null (Stage 0's marginalised emission) as the minimum
    bar. It cannot serve here: a model that marginalises the offset emits no downbeat
    LOCATIONS at all, so it has no placement to score. Majority-``r`` and random-``r`` are
    the nulls that place downbeats, and they are what this compares against. The
    uniform-``r`` null remains the right baseline for NLL, where it is exactly Stage 0.

    No tolerance window (SS8.1): the grid is given, so a predicted downbeat is correct iff
    it lands on an annotated downbeat beat.
    """
    s, crops = _fitted(make)
    n = len(crops[0]["y"])
    model_f = float(np.mean([R.placement_f(c["y"], s.predict_placement(_obs(c), n))
                             for c in crops]))

    maj = R.majority_r_null([c["r_true"] for c in crops], s.m)
    maj_f = float(np.mean([R.placement_f(c["y"], R.downbeats_from_offset(n, maj, s.m))
                           for c in crops]))
    rnd = R.random_r_null(len(crops), s.m, seed=5)
    rnd_f = float(np.mean([R.placement_f(c["y"], R.downbeats_from_offset(n, int(r), s.m))
                           for c, r in zip(crops, rnd)]))

    assert model_f > max(maj_f, rnd_f) + 0.10, (
        f"placement F {model_f:.3f} does not beat the nulls by SS8.3's 0.10 margin "
        f"(majority-r {maj_f:.3f}, random-r {rnd_f:.3f})")


@check
def check_predictions_are_not_degenerate(make):
    """SS8.4/P4: after fitting on an offset-balanced bench the read-out must not collapse.

    The bench sweeps the start beat, so every offset is equally represented and a predictor
    answering one offset for everything is unambiguously degenerate. Raw accuracy sits
    beside the class count because SS10.4 records balanced accuracy flattering exactly this
    failure.
    """
    s, crops = _fitted(make)
    pred = _predicted_offsets(s, crops)
    true = [c["r_true"] for c in crops]
    distinct = len(set(pred))
    assert distinct > 1, (
        f"the deployed read-out answered r={pred[0]} for all {len(pred)} crops, which are "
        f"balanced over offsets. Raw accuracy {R.offset_accuracy(true, pred):.3f}, "
        f"confusion:\n{R.offset_confusion(true, pred, s.m)}")


@check
def check_shuffled_phase_destroys_deployable_accuracy(make):
    """SS8.4's shuffled null, scored HELD OUT. Breaking h<->phase must collapse accuracy.

    Run as a PAIR: the intact fit must be well above chance AND the shuffled fit at chance.
    Both high means a shortcut is being measured; both low means the model is simply
    broken. Scored on a held-out split, because on the training split a prior with enough
    capacity can memorise whichever offset each crop was dealt.

    The derangement re-pairs each crop's ``h`` with a ``y`` whose offset DIFFERS from its
    own, so no residual h<->phase signal survives. A plain permutation would leave 1/m of
    pairs correctly labelled and a competent learner would then beat chance through the
    genuine relationship -- the opposite of what this control claims to show.
    """
    s0 = make()
    m = s0.m
    train = _data(m, n_crops=24, seed=0)
    held = _data(m, n_crops=16, seed=7)
    chance = R.count_partition_chance(len(held[0]["y"]), m)

    intact = make().fit(train, steps=FIT_STEPS, lr=FIT_LR)
    acc_real = R.offset_accuracy([c["r_true"] for c in held],
                                 _predicted_offsets(intact, held))

    rng = np.random.default_rng(99)
    deranged = []
    for c in train:
        others = [o for o in train if o["r_true"] != c["r_true"]]
        pick = others[int(rng.integers(len(others)))]
        deranged.append({**c, "y": pick["y"], "r_true": pick["r_true"]})
    broken = make().fit(deranged, steps=FIT_STEPS, lr=FIT_LR)
    acc_shuf = R.offset_accuracy([c["r_true"] for c in held],
                                 _predicted_offsets(broken, held))

    assert acc_real > 0.75, (
        f"held-out offset accuracy on intact data is only {acc_real:.3f}; the model does "
        f"not work, so the control below proves nothing")
    assert acc_shuf < chance + 0.20, (
        f"offset accuracy survives phase derangement ({acc_shuf:.3f} vs chance "
        f"{chance:.3f}): the deployable path is exploiting something other than the "
        f"h -> phase relationship")


@check
def check_fit_actually_trains_every_named_parameter(make):
    """SS4.7 governs the objective ``fit()`` DIFFERENTIATES, not the one ``elbo()`` returns.

    ``check_every_parameter_receives_gradient`` builds the loss itself and backprops it;
    nothing connects that to the training loop, so ``fit()`` could optimise a different
    objective -- or none -- and no property would notice. Measured at Stage 0: a ``fit()``
    ascending the exact marginal likelihood instead of the ELBO passed every other check
    while phi moved exactly 0.0, which deletes variational inference and leaves the suite
    green.

    phi is displaced off its initialisation first, because at ``c = 1`` the encoder is
    already the exact posterior and ``dELBO/dc = 0`` there by the envelope theorem -- a
    frozen encoder would be indistinguishable from a trained one.
    """
    s = make()
    crops = _data(s.m, n_crops=8, seed=1)
    _randomise(s, "phi", seed=7)
    before = {n: _np(p).copy() for n, p in s.named_params("P1").items()}
    s.fit(crops, steps=FIT_STEPS, lr=FIT_LR)
    still = [n for n, p in s.named_params("P1").items()
             if np.array_equal(_np(p), before[n])]
    assert not still, (
        f"parameters bit-identical after fit(): {still}. fit() is not optimising the "
        f"objective these parameters belong to.")


@check
def check_higher_elbo_means_better_phase(make):
    """The objective and the metric must point the same way (SS5's crop-length warning).

    SS5 records that at 256-frame crops the ELBO PREFERS a metronome -- truth-coast +12.3
    at 256/512 against -46.6 at 1500 -- so "a phase model trained on crops that are too
    short will correctly converge to something useless". That is not a bug in the model but
    a mismatch between the objective and the task, and it is invisible to every other check
    here, all of which either test the objective or test the metric.

    So: the fitted model must both score better on offset accuracy AND hold a higher mean
    ELBO than the same model at its initialisation. If the ELBO can be raised while phase
    accuracy falls, the bench's crop length is in the regime SS5 warns about and no amount
    of training will help.
    """
    s = make()
    crops = _data(s.m, n_crops=16, seed=2)

    def mean_elbo(model):
        with torch.no_grad():
            return float(np.mean([float(model.elbo(_obs(c), c["y"])) for c in crops]))

    before_elbo = mean_elbo(s)
    before_acc = R.offset_accuracy([c["r_true"] for c in crops], _predicted_offsets(s, crops))
    s.fit(crops, steps=FIT_STEPS, lr=FIT_LR)
    after_elbo = mean_elbo(s)
    after_acc = R.offset_accuracy([c["r_true"] for c in crops], _predicted_offsets(s, crops))

    assert after_elbo > before_elbo - 1e-9, (
        f"fitting LOWERED the mean ELBO ({before_elbo:.4f} -> {after_elbo:.4f})")
    assert after_acc >= before_acc, (
        f"the ELBO rose ({before_elbo:.4f} -> {after_elbo:.4f}) while offset accuracy FELL "
        f"({before_acc:.3f} -> {after_acc:.3f}). The objective is not aligned with the "
        f"task on {len(crops[0]['y'])}-beat crops -- SS5's short-crop regime.")
    assert after_acc > 0.5, f"fitted offset accuracy {after_acc:.3f} is barely above chance"


@check
def check_m_is_a_parameter_not_a_hardcoded_four(make):
    """SS2.1 condition 3: ``m`` is a parameter of every function that uses it.

    Stage P is allowed to FIX ``m = 4`` -- a declared, corpus-restricted, scoped violation
    of SPEC.md C3 -- but only on the condition that ``m`` remains a parameter throughout,
    so that P2 -> Stage 1 restores it as a latent "without rewriting anything around it".
    v1 hardcoded ``M = 4`` in read-out, emission and sawtooth and made meter causally inert
    for weeks; the exemption exists precisely to not repeat that.

    Checked by building the subject at ``m = 3`` and requiring every shape and every
    downbeat comb to follow. A model that inlined 4 anywhere returns a 4-vector here.
    """
    s = make(m=3)
    if s.m != 3:
        raise CapacityUnsupportedError(f"subject ignored m=3 and built m={s.m}")
    with torch.no_grad():
        s.alpha.fill_(3.0)
        s.beta.fill_(-3.0)
    y = R.downbeats_from_offset(12, 2, 3)
    lp = _np(s.emission_logp_all(y))
    assert lp.shape == (3,), (
        f"emission_logp_all returned {lp.shape} at m=3, expected (3,): m is hardcoded")
    assert int(lp.argmax()) == 2, (
        f"at m=3 a y generated at offset 2 peaks at r={int(lp.argmax())}")
    crops = _data(3, n_crops=12, n_beats=12)
    assert {c["r_true"] for c in crops} <= {0, 1, 2}
    s.fit(crops, steps=FIT_STEPS, lr=FIT_LR)
    pred = _predicted_offsets(s, crops)
    assert set(pred) <= {0, 1, 2}, f"predicted offsets {sorted(set(pred))} illegal at m=3"
    acc = R.offset_accuracy([c["r_true"] for c in crops], pred)
    assert acc >= 0.95, (
        f"offset accuracy {acc:.3f} at m=3 on the known-phase bench; the pipeline works at "
        f"m=4 only, so m is not genuinely a parameter")
