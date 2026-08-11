"""Behavioural tests for the phasevae package, derived from docstrings and math only.

Every expected value here comes from a function's stated contract (docstring, signature,
or the mathematical definition it names), never from running the implementation first.
Tests that expose a contract violation keep their honest assertion and carry a strict=False
xfail with the violation spelled out.
"""
from __future__ import annotations

import math
import pathlib
import types

import numpy as np
import pytest
import torch

from phasevae import run as run_mod
from phasevae.config import load_config
from phasevae.scoring import controls as controls_mod
from phasevae.scoring.evaluation import f_measure, null_times, peak_times
from phasevae.model import (BarPhaseVAE, Encoder, KAPPA_PHYSICAL, MAX_KAPPA, TWO_PI,
                            bounded_kappa, downbeat_frames, inverse_softplus,
                            vonmises_entropy)

from phasevae.data.features import FPS, atomic_save_npy

assert FPS == 50.0, "synthetic fixtures below are built around 50 fps"

# ---------------------------------------------------------------------------- helpers


def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


class _Song:
    dataset = "toy"
    song_id = "toy_song"
    fold = 0

    def __init__(self, downbeats):
        self._db = np.asarray(downbeats, dtype=np.float64)

    def beats(self):
        return self._db, self._db  # every beat a downbeat: pure downbeat data


# ======================================================================= metrics_db


def test_f_measure_perfect_match():
    """Identical predicted and annotated times -> f = precision = recall = 1.
    Perfect agreement is the fixed point of any matching metric.
    """
    times = np.array([1.0, 2.0, 3.0])
    assert f_measure(times, times) == (1.0, 1.0, 1.0)


def test_f_measure_empty_predicted():
    """No predictions against real annotations -> recall 0 and f 0; nothing was found,
    so the score must not reward silence.
    """
    f, precision, recall = f_measure(np.array([]), np.array([1.0, 2.0]))
    assert f == 0.0 and recall == 0.0


def test_f_measure_empty_annotated():
    """Empty annotations: predicting nothing is exactly right (f=1); predicting anything
    is a pure false positive (f=0). Both edges follow from the definitions.
    """
    assert f_measure(np.array([]), np.array([]))[0] == 1.0
    assert f_measure(np.array([1.0]), np.array([]))[0] == 0.0


def test_f_measure_tolerance_edge_inclusive():
    """A prediction exactly `tolerance` from the annotation must count as a hit
    ("inside tolerance" with <= in the matcher), and one just beyond must not.
    Uses tolerance=0.5 so the edge is exactly float-representable.
    """
    f_at, _, _ = f_measure(np.array([1.5]), np.array([1.0]), tolerance=0.5)
    f_beyond, _, _ = f_measure(np.array([1.5000001]), np.array([1.0]), tolerance=0.5)
    assert f_at == 1.0
    assert f_beyond == 0.0


def test_f_measure_greedy_one_to_one():
    """Two predictions both within tolerance of ONE annotation: one-to-one matching
    allows only a single hit, so precision = 1/2, recall = 1. Double-counting would
    let a spike train score perfectly.
    """
    f, precision, recall = f_measure(np.array([1.00, 1.01]), np.array([1.0]))
    assert precision == 0.5 and recall == 1.0
    assert f == pytest.approx(2 * 0.5 * 1.0 / 1.5)


def test_peak_times_relative_threshold():
    """The threshold is RELATIVE to the curve's own max (per the docstring): a curve
    topping out at 0.16 must still yield its peak, and a frame under threshold*max
    must not be picked.
    """
    probs = np.zeros(200)
    probs[50] = 0.16          # global max -> normalised to 1.0, always picked
    probs[150] = 0.16 * 0.4   # 0.4 relative < default 0.5 threshold -> rejected
    times = peak_times(probs, fps=50.0, period_s=1.0)
    assert 50 / 50.0 in times
    assert 150 / 50.0 not in times


def test_peak_times_min_gap_half_bar():
    """Two equal-height peaks 10 frames apart with a bar of 2 s (half-bar = 50 frames):
    only one may survive, because peaks are separated by at least half a bar so one
    bar contributes one downbeat.
    """
    probs = np.zeros(300)
    probs[100] = 1.0
    probs[110] = 0.9   # within half a bar of the first -> suppressed
    probs[200] = 0.9   # 100 frames away -> kept
    times = peak_times(probs, fps=50.0, period_s=2.0)
    assert len(times) == 2
    assert set(times) == {2.0, 4.0}


def test_null_times_rate_correctness():
    """kind='zero' on a crop of duration D with bar period P emits a grid starting at t0
    with spacing P: ceil(D/P) times, all t0 + k*P. The null's whole point is the right
    RATE with no learned phase.
    """
    crop = {"bar_period": 2.0, "y": np.zeros(500), "fps": 50.0, "t0": 3.0,
            "downbeat_times": np.array([4.0, 6.0])}
    rng = np.random.default_rng(0)
    times = null_times(crop, "zero", rng)
    duration = 500 / 50.0
    assert len(times) == math.ceil(duration / 2.0)
    np.testing.assert_allclose(times, 3.0 + 2.0 * np.arange(len(times)))


def test_null_times_random_offset_within_period():
    """kind='random' shifts the same grid by a uniform offset in [0, period): the first
    emitted time is in [t0, t0 + period) and the spacing is still exactly one period.
    """
    crop = {"bar_period": 2.0, "y": np.zeros(500), "fps": 50.0, "t0": 0.0,
            "downbeat_times": np.array([4.0, 6.0])}
    rng = np.random.default_rng(1)
    times = null_times(crop, "random", rng)
    assert 0.0 <= times[0] < 2.0
    np.testing.assert_allclose(np.diff(times), 2.0)


# ============================================================================ model


def test_bounded_kappa_identity_and_bound():
    """MAX*tanh(x/MAX) is the identity for x << MAX and never exceeds MAX_KAPPA; both
    follow from tanh's series and its range.
    """
    small = torch.tensor([1.0, 100.0, 2000.0], dtype=torch.float64)
    assert torch.allclose(bounded_kappa(small), small, rtol=1e-3)

    big = bounded_kappa(torch.tensor([1e6, 1e7], dtype=torch.float64))
    # mathematically tanh < 1 always; in floats tanh saturates to exactly 1.0 once
    # 1 - tanh underflows (raw >~ 19*MAX), so assert strictness where representable
    # and never-exceeds everywhere
    assert torch.all(big <= MAX_KAPPA)
    assert big[0] < MAX_KAPPA          # tanh(10) < 1 is float64-representable
    assert big[0] > 0.99 * MAX_KAPPA


def test_bounded_kappa_strictly_positive_gradient():
    """Tanh never saturates exactly, so d(bounded_kappa)/dx must be strictly positive
    even at huge inputs -- the docstring's stated advantage over a hard clamp.
    """
    x = torch.tensor([0.0, 1e3, 1e5, 1e6], dtype=torch.float64,
                     requires_grad=True)
    bounded_kappa(x).sum().backward()
    assert torch.all(x.grad > 0)


def test_inverse_softplus_roundtrip():
    """softplus(inverse_softplus(v)) == v for v below 30, and the function is the
    identity above 30 where softplus is linear to machine precision.
    """
    for v in (0.5, 1.0, 5.0, 29.0):
        assert torch.nn.functional.softplus(
            torch.tensor(inverse_softplus(v))).item() == pytest.approx(v, rel=1e-6)
    assert inverse_softplus(2000.0) == 2000.0


def test_vonmises_entropy_uniform_limit_and_scipy():
    """H(vM) at kappa=0 is log(2*pi) (the uniform circle), and at general kappa must
    match scipy.stats.vonmises.entropy -- both direct consequences of the formula in
    the docstring.
    """
    assert vonmises_entropy(torch.tensor(0.0)).item() == pytest.approx(
        math.log(TWO_PI), abs=1e-6)
    scipy_stats = pytest.importorskip("scipy.stats")
    for k in (0.5, 2.0, 50.0, 2000.0):
        ours = vonmises_entropy(torch.tensor(k, dtype=torch.float64)).item()
        assert ours == pytest.approx(float(scipy_stats.vonmises(kappa=k).entropy()),
                                     abs=1e-5)


def test_encoder_trajectory_rotates_monotonically_by_construction():
    """mu = offset + cumsum(exp(pooled log-rate)): rotation is STRUCTURAL, not learned.

    Replaces a test that asserted the opposite -- that mu was emitted freely per frame.
    That parameterisation was measured to collapse: from a random init the second
    differences are of order 1 rad, the KL opens at ~151k against a reconstruction of
    ~286, and the steepest descent is to flatten mu. The trace showed the endpoint
    precisely -- increments sign-balanced at frac>0 = 0.50 with mean +0.00001, i.e. two
    parts in a thousand of the phase motion contributing net rotation.

    The three properties below are what the new parameterisation guarantees so that the
    objective never has to be persuaded of them.
    """
    _seed()
    enc = Encoder(input_dim=4, d_model=8, pool_span=50)
    h = torch.randn(2, 300, 4)
    mu, kappa = enc(h, torch.ones(2, 300))

    inc = mu[:, 1:] - mu[:, :-1]

    # 1. strictly increasing: the rate is exp(...) so every step is positive. A frozen or
    #    sign-balanced trajectory -- the measured collapse -- is unrepresentable.
    assert torch.all(inc > 0), "phase is not monotonically advancing"

    # 2. constant within a pooling span: within-bar increment variance is 100% of what the
    #    collapsed model produced and 0% of what the annotations contain, so the degrees of
    #    freedom are DELETED here rather than taxed by the prior.
    for start in (0, 50, 100):
        block = inc[:, start:start + 49]
        assert torch.allclose(block, block[:, :1].expand_as(block), atol=1e-6), \
            "increment is not constant inside a pooling span"

    # 3. the rate lands in the physical band at initialisation (a 0.6-12 s bar), which is
    #    what puts a fresh model inside the +-3% basin where the reconstruction gradient
    #    on the rate is coherent at all.
    assert 0.01 <= float(inc.mean()) <= 0.2


def test_encoder_offset_head_is_reachable_from_the_loss():
    """mu[0] must equal the offset, and channels 0,1 must receive gradient.

    Dropping the `offset +` term leaves mu[0] = 0 for every window and silently starves
    the offset head -- measured at exactly 0.000e+00 gradient on both channels. That is
    fatal rather than wasteful: F on this model is the anchor-within-tolerance rate, so
    the offset IS the score. It is also the dead-subnetwork failure the Encoder's own
    init comment says shipped three times.
    """
    _seed()
    enc = Encoder(input_dim=4, d_model=8, pool_span=50)
    h = torch.randn(2, 200, 4)
    mu, kappa = enc(h, torch.ones(2, 200))

    out = enc.out(enc.features(h, torch.ones(2, 200)))
    offset = torch.atan2(out[:, 0, 0], out[:, 0, 1])
    assert torch.allclose(mu[:, 0], offset, atol=0), "mu[0] is not the offset, bitwise"

    enc.zero_grad()
    (mu.sum() + kappa.sum()).backward()
    grads = [float(enc.out.weight.grad[c].abs().sum()) for c in range(4)]
    for c, g in enumerate(grads):
        assert g > 0.0, f"output channel {c} receives no gradient (dead head)"


def test_encoder_target_blind():
    """Point 2: the base encoder reads AUDIO ONLY -- structurally. Its forward
    has no target parameter at all (the psi variant's posterior subclass adds one),
    so target-blindness is a property of the signature, not a flag to keep off.
    """
    _seed()
    enc = Encoder(input_dim=4, d_model=8)
    named = enc.forward.__code__.co_varnames[:enc.forward.__code__.co_argcount]
    assert "y" not in named
    assert not getattr(enc, "reads_target", False)


def test_emission_logits_cosine_shape():
    """A + b*cos(phi): peak at phi=0, trough at phi=pi, even symmetry, period 2*pi --
    all properties of the cosine the docstring names.
    """
    _seed()
    model = BarPhaseVAE(input_dim=4, d_model=8, emission="cosine")
    phi = torch.linspace(-math.pi, math.pi, 101)[None]

    logits = model.emission_logits(phi)[0]
    peak = model.emission_logits(torch.zeros(1, 1))[0, 0]
    trough = model.emission_logits(torch.full((1, 1), math.pi))[0, 0]
    assert torch.all(logits <= peak + 1e-6)
    assert torch.all(logits >= trough - 1e-6)

    a, b = model.emission_a.item(), model.emission_b.item()
    assert peak.item() == pytest.approx(a + b, abs=1e-5)
    assert trough.item() == pytest.approx(a - b, abs=1e-5)

    sym = model.emission_logits(-phi)[0]
    assert torch.allclose(logits, sym, atol=1e-6)
    period = model.emission_logits(phi + TWO_PI)[0]
    assert torch.allclose(logits, period, atol=1e-5)


def test_emission_logits_triangle_shape():
    """Tent: logit = a + b*(1 - 2|phi|/pi) on the wrapped angle -- value a+b at 0, a-b
    at pi, LINEAR in |phi| in between, even, and continuous across the wrap at +-pi.
    """
    _seed()
    model = BarPhaseVAE(input_dim=4, d_model=8, emission="triangle")
    a, b = model.emission_a.item(), model.emission_b.item()

    def at(p):
        return model.emission_logits(torch.tensor([[p]]))[0, 0].item()

    assert at(0.0) == pytest.approx(a + b, abs=1e-5)
    assert at(math.pi) == pytest.approx(a - b, abs=1e-4)

    # linearity: value at phi is a + b*(1 - 2 phi/pi) for phi in (0, pi)
    for p in (0.3, 1.0, 2.5):
        assert at(p) == pytest.approx(a + b * (1 - 2 * p / math.pi), abs=1e-4)
        assert at(-p) == pytest.approx(at(p), abs=1e-6)          # symmetry

    # continuity at the wrap: pi - eps and pi + eps (== -pi + eps) agree
    eps = 1e-3
    assert at(math.pi - eps) == pytest.approx(at(math.pi + eps), abs=5e-3)


def test_emission_b_floor_semantics():
    """emission_b == emission_b_floor + softplus(emission_b_raw), with the floor
    defaulting to 0 -- exactly the property's docstring ('never below the scheduled
    floor').
    """
    _seed()
    model = BarPhaseVAE(input_dim=4, d_model=8, emission="cosine")
    assert model.emission_b_floor == 0.0
    sp = torch.nn.functional.softplus(model.emission_b_raw).item()
    assert model.emission_b.item() == pytest.approx(sp)

    model.emission_b_floor.fill_(5.0)   # a BUFFER: mutate in place, never rebind
    assert model.emission_b.item() == pytest.approx(5.0 + sp)
    assert model.emission_b.item() >= 5.0


def test_emission_b_floor_survives_state_dict_roundtrip():
    """Set the scheduled floor, save state_dict, load into a fresh model: emission_b
    must be preserved, because the floor is part of the likelihood the checkpoint
    claims to represent. (Was an xfail: the floor used to be a plain attribute and
    silently reset to 0.0 on reload; it is a registered buffer now.)
    """
    _seed()
    model = BarPhaseVAE(input_dim=4, d_model=8, emission="cosine")
    model.emission_b_floor.fill_(5.0)
    b_before = model.emission_b.item()
    state = model.state_dict()
    fresh = BarPhaseVAE(input_dim=4, d_model=8, emission="cosine")
    fresh.load_state_dict(state)
    assert fresh.emission_b.item() == pytest.approx(b_before), \
        "scheduled emission floor lost across save/load"


def test_kl_to_physical_prior_nonnegative():
    """KL(q||p) >= 0 for any q and p; the closed form must respect this up to float
    tolerance across many random (mu, kappa, delta) draws.
    """
    _seed()
    model = BarPhaseVAE(input_dim=4, d_model=8)
    for _ in range(20):
        B, T = 3, 12
        mu = (torch.rand(B, T) * 2 - 1) * math.pi
        kappa = torch.rand(B, T) * 3000 + 1.0
        mask = torch.ones(B, T)
        kl = model.kl_to_physical_prior(mu, kappa, mask)
        assert torch.all(kl > -1e-3), f"negative KL {kl.min().item()}"


def test_kl_to_physical_prior_analytic_three_frame():
    """T=3 hand computation against the docstring's constant-rate prior.

    phi_1, phi_2 are unconditioned (the second difference needs two predecessors), so
    each contributes log p = -log 2pi. Only t=3 has a chain term, whose prior mean is
    2*phi_2 - phi_1, giving

        E log p(phi_3 | phi_2, phi_1)
          = kappa_p * A_1(k3) * A_2(k2) * A_1(k1) * cos(mu3 - 2 mu2 + mu1)
            - log 2pi - log I0(kappa_p)

    The MIDDLE frame carries coefficient 2, hence A_2 = I_2/I_0 and not A_1^2 -- the
    factor this test exists to pin. Every piece below comes from scipy independently.
    """
    scipy_stats = pytest.importorskip("scipy.stats")
    scipy_special = pytest.importorskip("scipy.special")
    model = BarPhaseVAE(input_dim=4, d_model=8)
    mu = torch.tensor([[0.30, 0.36, 0.43]], dtype=torch.float64)
    kappa = torch.tensor([[800.0, 1200.0, 950.0]], dtype=torch.float64)
    mask = torch.ones(1, 3, dtype=torch.float64)
    got = model.kl_to_physical_prior(mu, kappa, mask).item()

    def A1(k):
        return scipy_special.ive(1, k) / scipy_special.ive(0, k)

    def A2(k):
        return scipy_special.ive(2, k) / scipy_special.ive(0, k)

    entropies = [float(scipy_stats.vonmises(kappa=float(k)).entropy())
                 for k in (800.0, 1200.0, 950.0)]
    log_i0_p = float(np.log(scipy_special.ive(0, KAPPA_PHYSICAL)) + KAPPA_PHYSICAL)
    accel = 0.43 - 2 * 0.36 + 0.30
    cross = (KAPPA_PHYSICAL * A1(950.0) * A2(1200.0) * A1(800.0) * math.cos(accel)
             - math.log(TWO_PI) - log_i0_p)
    expected = -sum(entropies) - (-2 * math.log(TWO_PI) + cross)

    # tolerance limited by scipy's own vonmises.entropy precision at large kappa
    assert got == pytest.approx(expected, abs=1e-3)


def test_kl_to_physical_prior_second_order_needs_a2_not_a1_squared():
    """A_2(kappa) != A_1(kappa)^2, so the closed form must not use the square.

    At a moderate kappa the two differ by tens of percent (A_2 = 0.302 vs A_1^2 = 0.487
    at kappa = 2), which is large enough that substituting the square shifts the KL well
    outside float tolerance. Guards the one algebraic step that is easy to get wrong.
    """
    scipy_special = pytest.importorskip("scipy.special")
    model = BarPhaseVAE(input_dim=4, d_model=8)
    mu = torch.tensor([[0.1, 0.2, 0.35]], dtype=torch.float64)
    kappa = torch.full((1, 3), 2.0, dtype=torch.float64)
    mask = torch.ones(1, 3, dtype=torch.float64)
    got = model.kl_to_physical_prior(mu, kappa, mask).item()

    a1 = float(scipy_special.ive(1, 2.0) / scipy_special.ive(0, 2.0))
    a2 = float(scipy_special.ive(2, 2.0) / scipy_special.ive(0, 2.0))
    accel = 0.35 - 2 * 0.2 + 0.1
    log_i0_p = float(np.log(scipy_special.ive(0, KAPPA_PHYSICAL)) + KAPPA_PHYSICAL)

    def kl_with(middle):
        cross = (KAPPA_PHYSICAL * a1 * middle * a1 * math.cos(accel)
                 - math.log(TWO_PI) - log_i0_p)
        return -3 * float(vonmises_entropy(torch.tensor(2.0, dtype=torch.float64))) \
            - (-2 * math.log(TWO_PI) + cross)

    # rel rather than abs: the KL is ~1.7e3 here because log I0(kappa_p) is ~kappa_p
    assert got == pytest.approx(kl_with(a2), rel=1e-7)

    # The wrong factor is not subtle -- but its size scales with kappa_physical, so state
    # the threshold in units of kappa_p rather than as a constant that goes stale the next
    # time the prior is recalibrated (2000 -> 383 broke a hardcoded 100 here).
    wrong = abs(kl_with(a1 * a1) - kl_with(a2))
    assert wrong > 0.05 * KAPPA_PHYSICAL, f"A_1^2 vs A_2 differ by only {wrong:.1f} nats"


def test_kl_to_physical_prior_invariant_to_shift_and_rate():
    """The prior penalises rate CHANGE only, so the KL must be unchanged by (a) adding a
    constant to the whole trajectory and (b) adding a constant RATE ramp. Both leave the
    second difference identical -- the docstring's phase-blindness and tempo-scale-
    freedom, which together are why only the emission can locate or size a bar.
    """
    _seed()
    model = BarPhaseVAE(input_dim=4, d_model=8)
    T = 9
    mu = torch.cumsum(torch.rand(1, T, dtype=torch.float64) * 0.02 + 0.05, dim=1)
    kappa = torch.rand(1, T, dtype=torch.float64) * 500 + 50
    mask = torch.ones(1, T, dtype=torch.float64)
    base = model.kl_to_physical_prior(mu, kappa, mask).item()

    shifted = model.kl_to_physical_prior(mu + 1.234, kappa, mask).item()
    ramp = torch.arange(T, dtype=torch.float64)[None] * 0.031
    rerated = model.kl_to_physical_prior(mu + ramp, kappa, mask).item()

    assert shifted == pytest.approx(base, rel=1e-9)
    assert rerated == pytest.approx(base, rel=1e-9)


def test_kl_to_physical_prior_masked_frames_contribute_zero():
    """A frame with mask 0 must contribute nothing: the KL of a [1,5] sequence whose
    last two frames are masked equals the KL of the first three frames alone. Three
    real frames because the chain term needs two predecessors.
    """
    model = BarPhaseVAE(input_dim=4, d_model=8)
    mu5 = torch.tensor([[0.30, 0.36, 0.43, 9.0, -9.0]], dtype=torch.float64)
    kappa5 = torch.tensor([[800.0, 1200.0, 950.0, 7.0, 7.0]], dtype=torch.float64)
    mask5 = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]], dtype=torch.float64)
    kl_masked = model.kl_to_physical_prior(mu5, kappa5, mask5).item()
    kl_short = model.kl_to_physical_prior(
        mu5[:, :3], kappa5[:, :3], torch.ones(1, 3, dtype=torch.float64)).item()
    assert kl_masked == pytest.approx(kl_short, rel=1e-9)


def test_forward_elbo_identity_and_unweighted_recon(monkeypatch):
    """Forward's returned dict must satisfy elbo == recon - kl exactly, and with
    pos_weight=1 recon must equal the plain masked Bernoulli log-likelihood of y under
    the emission at the sampled phase (computed by hand with sampling frozen to the
    mean).
    """
    _seed()
    import phasevae.model as model_mod
    monkeypatch.setattr(model_mod, "sample_vonmises", lambda k: torch.zeros_like(k))
    model = BarPhaseVAE(input_dim=4, d_model=8, emission="cosine")

    B, T = 2, 12
    h = torch.randn(B, T, 4)
    mask = torch.ones(B, T)
    mask[1, 8:] = 0.0
    y = torch.zeros(B, T)
    y[0, 3] = 1.0
    y[1, 5] = 1.0

    out = model(h, mask, y, samples=1, pos_weight=1.0)
    assert torch.allclose(out["elbo"], out["recon"] - out["kl"], atol=1e-5)

    with torch.no_grad():
        logits = model.emission_logits(out["mu"])
        ll = (y * torch.nn.functional.logsigmoid(logits)
              + (1 - y) * torch.nn.functional.logsigmoid(-logits))
        expected = (ll * mask).sum(1)
    assert torch.allclose(out["recon"], expected, atol=1e-4)


def test_forward_pos_weight_scales_positive_frames(monkeypatch):
    """pos_weight=w multiplies exactly the downbeat-labelled frames: the change in
    recon must equal (w-1) times the positive frames' log-likelihood.
    """
    _seed()
    import phasevae.model as model_mod
    monkeypatch.setattr(model_mod, "sample_vonmises", lambda k: torch.zeros_like(k))
    model = BarPhaseVAE(input_dim=4, d_model=8, emission="cosine")

    B, T = 1, 10
    h = torch.randn(B, T, 4)
    mask = torch.ones(B, T)
    y = torch.zeros(B, T)
    y[0, 4] = 1.0

    out1 = model(h, mask, y, pos_weight=1.0)
    out3 = model(h, mask, y, pos_weight=3.0)

    with torch.no_grad():
        logits = model.emission_logits(out1["mu"])
        pos_ll = torch.nn.functional.logsigmoid(logits[0, 4])
    assert (out3["recon"] - out1["recon"]).item() == pytest.approx(
        (2.0 * pos_ll).item(), abs=1e-4)


def test_phase_ablation_gap_zero_when_phase_ignored():
    """An emission that returns the same logits regardless of phi must score a gap of
    exactly 0 -- the docstring defines the gap as the mean |shift| when phi is
    frozen.
    """
    _seed()
    model = BarPhaseVAE(input_dim=4, d_model=8, emission="cosine")
    model.emission_logits = lambda phi, mask=None: torch.full_like(phi, -3.0)
    phi = torch.rand(2, 20) * TWO_PI
    assert model.phase_ablation_gap(phi) == 0.0


def test_phase_ablation_gap_positive_for_cosine():
    """The genuine cosine emission depends on phi, so the gap must be strictly
    positive on a non-constant phase.
    """
    _seed()
    model = BarPhaseVAE(input_dim=4, d_model=8, emission="cosine")
    phi = torch.linspace(0, TWO_PI, 40)[None]
    assert model.phase_ablation_gap(phi) > 0.01


def test_downbeat_frames_marks_zero_crossings():
    """Rule g: a linear ramp crossing phi = 0 (mod 2*pi) exactly at frames 50, 100, 150
    must produce crossings there and nowhere else; the crossing is flagged on the frame
    AFTER the wrap of the [0, 2pi) representation.
    """
    t = torch.arange(200, dtype=torch.float64)
    mu = (t * (TWO_PI / 50.0) - 3.0 * TWO_PI / 2.0)[None]   # wraps every 50 frames
    crossing = downbeat_frames(mu)[0]
    idx = torch.nonzero(crossing).flatten() + 1              # diff index -> frame index
    # mu(t) = 2*pi*t/50 - 3*pi is 0 mod 2*pi at t = 25, 75, 125, 175
    expected = {25, 75, 125, 175}
    assert set(idx.tolist()) == expected


def test_infer_phase_requires_eval_mode():
    """The deployment path asserts eval mode; calling it in train mode must raise."""
    _seed()
    model = BarPhaseVAE(input_dim=4, d_model=8)
    model.train()
    with pytest.raises(AssertionError):
        model.infer_phase(torch.randn(1, 5, 4))

    model.eval()
    out = model.infer_phase(torch.randn(1, 5, 4))
    assert out.shape == (1, 5)


# ====================================================================== run helpers


def _args(beta_start=0.0, beta_end=1.0, beta_warmup=4):
    return types.SimpleNamespace(beta_start=beta_start, beta_end=beta_end,
                                 beta_warmup=beta_warmup)


def test_beta_at_schedule_edges():
    """Linear annealing: epoch 0 -> beta_start, epoch >= warmup -> beta_end, midpoint
    exactly halfway; warmup 0 disables annealing and returns beta_end always.
    """
    a = _args(0.0, 1.0, 4)
    assert run_mod.beta_at(0, a) == 0.0
    assert run_mod.beta_at(2, a) == pytest.approx(0.5)
    assert run_mod.beta_at(4, a) == 1.0
    assert run_mod.beta_at(100, a) == 1.0
    assert run_mod.beta_at(0, _args(0.3, 0.9, 0)) == 0.9


def _blindness_batch(input_dim: int, batch_size: int = 2, frames: int = 10):
    """A synthetic batch for the target-blindness control: no real audio needed.

    The control reads only signatures and determinism, so random h is equivalent to a
    frontend batch -- which is why this is a unit test and not a per-run assertion.
    """
    return {"h": torch.randn(batch_size, frames, input_dim),
            "delta": torch.full((batch_size, frames), 0.06),
            "mask": torch.ones(batch_size, frames),
            "y": torch.zeros(batch_size, frames)}


def test_target_blindness_control_detects_leak():
    """The control must PASS the real model (whose deployed net never receives y) and
    FAIL a model whose deployed phase moves with anything but h -- otherwise a passing
    control proves nothing.
    """
    _seed()
    batch = _blindness_batch(input_dim=4)
    controls_mod.assert_encoder_is_target_blind(BarPhaseVAE(input_dim=4, d_model=8), batch)

    leaky = BarPhaseVAE(input_dim=4, d_model=8)
    calls = {"n": 0}
    real = leaky.infer_phase

    def cheating(h, delta=None):
        calls["n"] += 1
        return real(h, delta) + (0.1 if calls["n"] > 2 else 0.0)
    leaky.infer_phase = cheating

    with pytest.raises(AssertionError):
        controls_mod.assert_encoder_is_target_blind(leaky, batch)


def test_deployed_net_reads_no_target_in_any_shipped_config():
    """EVERY config's deployed net must read h and delta only.

    This is what the removed per-run control covered: the signature contract holds for
    the model each recipe actually builds, not just for a bare BarPhaseVAE. A variant
    that starts reading y at deploy time is unusable at test time, and the failure is
    silent -- scores would simply be too good.
    """
    configs = sorted((pathlib.Path(__file__).resolve().parent.parent / "configs")
                     .glob("*.yaml"))
    assert configs, "no configs found: this test would pass vacuously"

    for path in configs:
        _seed()
        cfg, hooks = load_config(str(path))
        model = hooks.build_model(cfg, input_dim=4)
        controls_mod.assert_encoder_is_target_blind(
            model, _blindness_batch(input_dim=4))


# ============================================================== vbpm cache write


def _cache_write(group_dir: pathlib.Path, stem: str, features: np.ndarray):
    """Exercises vbpm.data.atomic_save_npy, the single write-then-rename authority."""
    cache_path = group_dir / f"{stem}.npy"
    group_dir.mkdir(parents=True, exist_ok=True)
    atomic_save_npy(cache_path, features)
    return cache_path


def test_cache_write_atomic_rename_and_roundtrip(tmp_path):
    """After the write-then-rename block, exactly <stem>.npy exists (no stray .partial
    or .partial.npy from np.save's silent suffix-appending), and np.load returns the
    array bit-exactly in float32 -- the atomicity the comment promises.
    """
    features = np.arange(12, dtype=np.float64).reshape(3, 4)
    path = _cache_write(tmp_path, "songA", features)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["songA.npy"], f"stray files: {names}"
    np.testing.assert_array_equal(np.load(path), features.astype(np.float32))


def test_cache_write_dotted_stem(tmp_path):
    """A stem containing dots (e.g. 'song.v2') must still produce exactly
    'song.v2.npy': with_suffix replaces only the final '.npy', and the open-handle
    np.save never appends a second one.
    """
    features = np.ones((2, 2), dtype=np.float32)
    path = _cache_write(tmp_path, "song.v2", features)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["song.v2.npy"]
    np.testing.assert_array_equal(np.load(path), features)


def test_cache_write_overwrites_existing(tmp_path):
    """A rewrite replaces the previous array atomically: the final file holds the new
    contents and no partial remains.
    """
    _cache_write(tmp_path, "s", np.zeros((2, 2), dtype=np.float32))
    path = _cache_write(tmp_path, "s", np.ones((2, 2), dtype=np.float32))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["s.npy"]
    np.testing.assert_array_equal(np.load(path), np.ones((2, 2), dtype=np.float32))


# ================================================== trajectory health diagnostics


def test_trajectory_health_separates_the_recorded_failure_modes():
    """The four numbers must distinguish trajectories that F cannot.

    Each row is a failure this project actually measured, with its published signature:
      oracle              advance 2*pi/(P*fps), phase_err 0, full circle
      oracle + half bar   SAME advance and circle, phase_err pi -- a pure offset error,
                          which is what F punishes and the KL is provably blind to
      spike train         advance 0.715 (the 2026-06 measurement), phase_err ~= chance
      frozen phase        advance 0, phase_err = chance, circle = one bin
    A diagnostic that cannot tell these apart cannot read a training run.
    """
    from phasevae.scoring.evaluation import scoring_records, trajectory_health

    n, fps, period = 600, 50.0, 2.0
    db = np.arange(1.0, 12.0, period)
    y = np.zeros(n, dtype=np.float32)
    for t in db:
        y[int(t * fps) - 1:int(t * fps) + 2] = 1.0
    raw = {"y": torch.tensor(y)[None].repeat(2, 1), "mask": torch.ones(2, n),
           "t0": torch.zeros(2), "fps": torch.full((2,), fps),
           "downbeat_times": [db] * 2,
           "anchors": [np.concatenate([[db[0] - period], db, [db[-1] + period]])] * 2,
           "dataset": ["toy"] * 2, "song_id": ["a", "b"]}
    crops = scoring_records(raw)
    t = torch.arange(n, dtype=torch.float32)
    rate = TWO_PI / (period * fps)
    kappa = torch.full((2, n), 2000.0)

    def wrap(x):
        return torch.atan2(torch.sin(x), torch.cos(x))

    def health(mu):
        return trajectory_health(mu[None].repeat(2, 1), kappa, torch.ones(2, n), crops)

    adv, kap, err, cov = health(wrap(rate * (t - db[0] * fps)))
    assert adv == pytest.approx(rate, rel=1e-3)
    assert err < 1e-3 and cov == 1.0 and kap == pytest.approx(2000.0)

    adv_o, _, err_o, cov_o = health(wrap(rate * (t - db[0] * fps) + math.pi))
    assert adv_o == pytest.approx(rate, rel=1e-3)      # rate right, offset wrong
    assert err_o == pytest.approx(math.pi, abs=1e-3)   # worse than chance, not random
    assert cov_o == 1.0

    adv_s, _, err_s, _ = health(wrap(0.715 * t))
    assert adv_s == pytest.approx(0.715, rel=1e-3)
    assert err_s > 1.5                                  # indistinguishable from chance

    adv_f, _, err_f, cov_f = health(torch.zeros(n))
    assert adv_f == pytest.approx(0.0, abs=1e-6)
    assert err_f == pytest.approx(math.pi / 2, rel=0.02)
    assert cov_f == pytest.approx(1 / 16)
