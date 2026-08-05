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

from phasevae import controls as controls_mod
from phasevae import run as run_mod
from phasevae.crops import bar_period, build_crop, song_crops, true_phase, MIN_DOWNBEATS
from phasevae.metrics_db import f_measure, null_times, peak_times
from phasevae.model import (BarPhaseVAE, Encoder, KAPPA_PHYSICAL, MAX_KAPPA, TWO_PI,
                            bounded_kappa, downbeat_frames, inverse_softplus,
                            vonmises_entropy)

FPS = 50.0

# ---------------------------------------------------------------------------- helpers


def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


def _pin_or_skip(fn, *a, **k):
    """collate() pins memory; skip cleanly on hosts where pinning is unavailable."""
    try:
        return fn(*a, **k)
    except RuntimeError as e:  # pragma: no cover - CPU-only torch builds
        pytest.skip(f"pinned memory unavailable: {e}")


def _oracle_crop(period=2.0, n_seconds=70.0, lo_t=3.0, crop_seconds=45.0, dim=4):
    """A synthetic song: metronomic downbeats every `period` s, random features."""
    rng = np.random.default_rng(0)
    features = rng.standard_normal((int(n_seconds * FPS), dim)).astype(np.float32)
    downbeats = np.arange(0.0, n_seconds, period)
    crop = build_crop(features, downbeats, lo_t, FPS, crop_seconds)
    assert crop is not None
    return crop


class _Song:
    dataset = "toy"
    stem = "toy_song"
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


# ============================================================================ crops


def test_bar_period_median_of_diffs():
    """Downbeats at 0,2,4,6.5,8.5 inside the window: diffs [2,2,2.5,2], median 2.0.
    The docstring promises one constant = the median interval over the window.
    """
    db = np.array([0.0, 2.0, 4.0, 6.5, 8.5])
    assert bar_period(db, 0.0, 10.0) == pytest.approx(2.0)


def test_bar_period_none_when_too_few_downbeats():
    """Fewer than MIN_DOWNBEATS inside [lo, hi] -> None: the docstring says the period
    is undefined without enough phase evidence.
    """
    db = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    assert MIN_DOWNBEATS == 4
    assert bar_period(db, 0.0, 5.0) is None       # only 3 inside
    assert bar_period(db, 0.0, 6.0) is not None   # exactly 4 inside


def test_build_crop_delta_formula():
    """Delta must equal 2*pi/(bar_period*fps) -- the per-frame advance that completes
    one full turn per bar, exactly as the key's docstring states.
    """
    crop = _oracle_crop(period=2.0)
    assert crop["delta"] == pytest.approx(2.0 * np.pi / (2.0 * FPS))
    assert crop["bar_period"] == pytest.approx(2.0)


def test_build_crop_target_widened():
    """Y is 1 on each downbeat frame AND +-target_tol_frames around it (default 1),
    0 elsewhere: the docstring widens the spike because the emission is smooth.
    """
    crop = _oracle_crop(period=2.0, lo_t=3.0)
    y = crop["y"]
    for t in crop["downbeat_times"]:
        centre = int(round(t * FPS)) - int(round(3.0 * FPS))
        if centre >= len(y):        # the boundary downbeat at exactly hi_t
            continue
        assert y[centre] == 1.0
        assert y[max(0, centre - 1)] == 1.0 and y[min(len(y) - 1, centre + 1)] == 1.0
    # a frame far from any downbeat is 0
    assert y[centre - 25] == 0.0
    # widening is tight: 2 frames away from a mid-crop downbeat is 0
    mid = int(round(crop["downbeat_times"][2] * FPS)) - int(round(3.0 * FPS))
    assert y[mid + 2] == 0.0 and y[mid - 2] == 0.0


def test_build_crop_anchors_bracket_window():
    """Anchors must include one downbeat at or before lo_t and one at or after hi_t
    (when they exist), so interpolation has a left and right anchor at every frame.
    """
    crop = _oracle_crop(period=2.0, lo_t=3.0, crop_seconds=45.0)
    anchors = crop["anchors"]
    assert anchors[0] <= 3.0
    assert anchors[-1] >= 3.0 + 45.0
    inside = crop["downbeat_times"]
    assert anchors[0] < inside[0] and anchors[-1] > inside[-1]


def test_true_phase_zero_at_downbeats():
    """The phase must be 0 (mod 2*pi) at each annotated downbeat frame: the docstring
    pins phase 0 exactly on every annotated downbeat.
    """
    crop = _oracle_crop(period=2.0, lo_t=3.0)
    phase, valid = true_phase(crop)
    for t in crop["downbeat_times"]:
        idx = int(round((t - crop["t0"]) * FPS))
        if idx >= len(phase):       # the boundary downbeat at exactly hi_t
            continue
        assert valid[idx]
        circ = min(phase[idx], TWO_PI - phase[idx])
        assert circ < 1e-4, f"phase {phase[idx]} at downbeat {t}"


def test_true_phase_monotone_between_anchors():
    """Linear interpolation of increasing turn counts is strictly increasing, so the
    UNWRAPPED phase must be monotone over the valid span.
    """
    crop = _oracle_crop(period=2.0, lo_t=3.0)
    phase, valid = true_phase(crop)
    unwrapped = np.unwrap(phase[valid].astype(np.float64))
    assert np.all(np.diff(unwrapped) > 0)


def test_true_phase_invalid_outside_annotated_span():
    """Frames before the first anchor or after the last must be flagged invalid, not
    extrapolated -- the docstring forbids extrapolation.
    """
    rng = np.random.default_rng(0)
    features = rng.standard_normal((int(60 * FPS), 4)).astype(np.float32)
    # downbeats only in [10, 40): frames of the crop before 10 s have no left anchor
    downbeats = np.arange(10.0, 40.0, 2.0)
    crop = build_crop(features, downbeats, 5.0, FPS, 40.0)
    assert crop is not None
    phase, valid = true_phase(crop)
    assert not valid[0]                       # 5.0 s < first downbeat 10.0 s
    assert not valid[-1]                      # 45.0 s > last downbeat 38.0 s
    assert valid[int(round((20.0 - 5.0) * FPS))]


def test_song_crops_no_duplicate_t0():
    """song_crops promises 'never emit two crops with the same start': all t0 values
    (rounded to ms as the code does) must be distinct.
    """
    rng_np = np.random.default_rng(0)
    features = rng_np.standard_normal((int(80 * FPS), 4)).astype(np.float32)
    song = _Song(np.arange(0.0, 80.0, 2.0))
    got, rejects = song_crops(features, song, np.random.default_rng(7), max_crops=3)
    assert len(got) >= 2
    t0s = [round(c["t0"], 3) for c in got]
    assert len(set(t0s)) == len(t0s)


def test_song_crops_sampling_cap():
    """max_crops is a SAMPLING cap: a long fully-annotated song must yield at most (and
    here exactly) max_crops crops.
    """
    rng_np = np.random.default_rng(0)
    features = rng_np.standard_normal((int(80 * FPS), 4)).astype(np.float32)
    song = _Song(np.arange(0.0, 80.0, 2.0))
    got, _ = song_crops(features, song, np.random.default_rng(3), max_crops=2)
    assert len(got) == 2
    for c in got:
        assert c["dataset"] == "toy" and c["stem"] == "toy_song"


def test_song_crops_rejects_unannotated():
    """A song with fewer than MIN_DOWNBEATS annotated downbeats yields no crops and a
    counted rejection -- rejections are surfaced, never silent.
    """
    rng_np = np.random.default_rng(0)
    features = rng_np.standard_normal((int(80 * FPS), 4)).astype(np.float32)
    song = _Song([1.0, 3.0])
    got, rejects = song_crops(features, song, np.random.default_rng(0))
    assert got == []
    assert sum(rejects.values()) >= 1


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


def test_encoder_drift_bound_confines_advance():
    """With drift_bound=eps>0 the docstring builds mu = offset + cumsum(delta + drift)
    with |drift| <= eps, so every per-frame advance mu[t]-mu[t-1] must lie in
    [delta-eps, delta+eps].
    """
    _seed()
    enc = Encoder(input_dim=4, hidden=8, drift_bound=0.05)
    h = torch.randn(2, 30, 4)
    delta = torch.full((2, 30), 0.0628)
    mu, kappa = enc(h, delta)
    adv = mu[:, 1:] - mu[:, :-1]
    assert torch.all((adv - 0.0628).abs() <= 0.05 + 1e-6)
    assert torch.all(kappa > 0) and torch.all(kappa < MAX_KAPPA)


def test_encoder_drift_bound_first_frame_is_offset():
    """mu[t=0] is the crop's phase offset read from h at frame 0 ('ONE anchor per
    crop'): it must not depend on delta, since cumsum contributes nothing at t=0.
    """
    _seed()
    enc = Encoder(input_dim=4, hidden=8, drift_bound=0.05)
    h = torch.randn(1, 20, 4)
    mu_a, _ = enc(h, torch.full((1, 20), 0.05))
    mu_b, _ = enc(h, torch.full((1, 20), 0.10))
    assert torch.equal(mu_a[:, 0], mu_b[:, 0])
    assert (-math.pi <= mu_a[0, 0].item() <= math.pi)   # atan2 range


def test_encoder_free_mode_per_frame_unconstrained():
    """With drift_bound=0 the mean is emitted per frame through atan2: every mu is in
    (-pi, pi] and delta plays no role at all (the free parameterisation ignores it).
    """
    _seed()
    enc = Encoder(input_dim=4, hidden=8, drift_bound=0.0)
    h = torch.randn(2, 25, 4)
    mu_none, _ = enc(h, None)
    mu_delta, _ = enc(h, torch.full((2, 25), 0.5))
    assert torch.equal(mu_none, mu_delta)
    assert torch.all(mu_none > -math.pi - 1e-6) and torch.all(mu_none <= math.pi + 1e-6)


def test_encoder_target_blind():
    """Point 2: the encoder reads AUDIO ONLY. Its signature admits nothing but h and
    delta, and its output is a deterministic function of them (bit-identical repeat).
    """
    _seed()
    enc = Encoder(input_dim=4, hidden=8)
    code = enc.forward.__code__
    named = set(code.co_varnames[:code.co_argcount])
    assert named <= {"self", "h", "delta"}
    h = torch.randn(1, 10, 4)
    mu1, k1 = enc(h)
    mu2, k2 = enc(h)
    assert torch.equal(mu1, mu2) and torch.equal(k1, k2)


def test_emission_logits_cosine_shape():
    """A + b*cos(phi): peak at phi=0, trough at phi=pi, even symmetry, period 2*pi --
    all properties of the cosine the docstring names.
    """
    _seed()
    model = BarPhaseVAE(input_dim=4, hidden=8, emission="cosine")
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
    model = BarPhaseVAE(input_dim=4, hidden=8, emission="triangle")
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
    model = BarPhaseVAE(input_dim=4, hidden=8, emission="cosine")
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
    model = BarPhaseVAE(input_dim=4, hidden=8, emission="cosine")
    model.emission_b_floor.fill_(5.0)
    b_before = model.emission_b.item()
    state = model.state_dict()
    fresh = BarPhaseVAE(input_dim=4, hidden=8, emission="cosine")
    fresh.load_state_dict(state)
    assert fresh.emission_b.item() == pytest.approx(b_before), \
        "scheduled emission floor lost across save/load"


def test_kl_to_physical_prior_nonnegative():
    """KL(q||p) >= 0 for any q and p; the closed form must respect this up to float
    tolerance across many random (mu, kappa, delta) draws.
    """
    _seed()
    model = BarPhaseVAE(input_dim=4, hidden=8)
    for _ in range(20):
        B, T = 3, 12
        mu = (torch.rand(B, T) * 2 - 1) * math.pi
        kappa = torch.rand(B, T) * 3000 + 1.0
        delta = torch.rand(B, 1).expand(B, T) * 0.1
        mask = torch.ones(B, T)
        kl = model.kl_to_physical_prior(mu, kappa, delta, mask)
        assert torch.all(kl > -1e-3), f"negative KL {kl.min().item()}"


def test_kl_to_physical_prior_analytic_two_frame():
    """T=2 hand computation. KL = -H(k1)-H(k2) - [log p(phi1) + E log p(phi2|phi1)]
    with log p(phi1) = -log 2pi and E log p(phi2|phi1) =
    kappa_p*A(k1)*A(k2)*cos(mu2-mu1-delta) - log 2pi - log I0(kappa_p),
    assembled here from independent scipy pieces.
    """
    scipy_stats = pytest.importorskip("scipy.stats")
    scipy_special = pytest.importorskip("scipy.special")
    model = BarPhaseVAE(input_dim=4, hidden=8)
    mu = torch.tensor([[0.3, 0.5]], dtype=torch.float64)
    kappa = torch.tensor([[800.0, 1200.0]], dtype=torch.float64)
    delta = torch.tensor([[0.06, 0.06]], dtype=torch.float64)
    mask = torch.ones(1, 2, dtype=torch.float64)
    got = model.kl_to_physical_prior(mu, kappa, delta, mask).item()

    def A(k):
        return scipy_special.i1e(k) / scipy_special.i0e(k)

    h1 = float(scipy_stats.vonmises(kappa=800.0).entropy())
    h2 = float(scipy_stats.vonmises(kappa=1200.0).entropy())
    log_i0_p = float(np.log(scipy_special.i0e(KAPPA_PHYSICAL)) + KAPPA_PHYSICAL)
    cross = (KAPPA_PHYSICAL * A(800.0) * A(1200.0) * math.cos(0.5 - 0.3 - 0.06)
             - math.log(TWO_PI) - log_i0_p)
    expected = -(h1 + h2) - (-math.log(TWO_PI) + cross)
    # tolerance limited by scipy's own vonmises.entropy precision at large kappa
    assert got == pytest.approx(expected, abs=1e-3)


def test_kl_to_physical_prior_masked_frames_contribute_zero():
    """A frame with mask 0 must contribute nothing: the KL of a [1,4] sequence whose
    last two frames are masked equals the KL of the first two frames alone.
    """
    model = BarPhaseVAE(input_dim=4, hidden=8)
    mu4 = torch.tensor([[0.3, 0.5, 9.0, -9.0]], dtype=torch.float64)
    kappa4 = torch.tensor([[800.0, 1200.0, 7.0, 7.0]], dtype=torch.float64)
    delta4 = torch.full((1, 4), 0.06, dtype=torch.float64)
    mask4 = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float64)
    kl_masked = model.kl_to_physical_prior(mu4, kappa4, delta4, mask4).item()
    kl_short = model.kl_to_physical_prior(mu4[:, :2], kappa4[:, :2], delta4[:, :2],
                                          torch.ones(1, 2).double()).item()
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
    model = BarPhaseVAE(input_dim=4, hidden=8, emission="cosine")
    B, T = 2, 12
    h = torch.randn(B, T, 4)
    delta = torch.full((B, T), 0.06)
    mask = torch.ones(B, T)
    mask[1, 8:] = 0.0
    y = torch.zeros(B, T)
    y[0, 3] = 1.0
    y[1, 5] = 1.0
    out = model(h, delta, mask, y, samples=1, pos_weight=1.0)
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
    model = BarPhaseVAE(input_dim=4, hidden=8, emission="cosine")
    B, T = 1, 10
    h = torch.randn(B, T, 4)
    delta = torch.full((B, T), 0.06)
    mask = torch.ones(B, T)
    y = torch.zeros(B, T)
    y[0, 4] = 1.0
    out1 = model(h, delta, mask, y, pos_weight=1.0)
    out3 = model(h, delta, mask, y, pos_weight=3.0)
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
    model = BarPhaseVAE(input_dim=4, hidden=8, emission="cosine")
    model.emission_logits = lambda phi, mask=None: torch.full_like(phi, -3.0)
    phi = torch.rand(2, 20) * TWO_PI
    assert model.phase_ablation_gap(phi) == 0.0


def test_phase_ablation_gap_positive_for_cosine():
    """The genuine cosine emission depends on phi, so the gap must be strictly
    positive on a non-constant phase.
    """
    _seed()
    model = BarPhaseVAE(input_dim=4, hidden=8, emission="cosine")
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
    model = BarPhaseVAE(input_dim=4, hidden=8)
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


def _tiny_crops():
    rng = np.random.default_rng(0)
    out = []
    for t_len, delta in ((10, 0.05), (14, 0.07), (6, 0.03), (12, 0.06)):
        out.append({"h": rng.standard_normal((t_len, 3)).astype(np.float32),
                    "y": (rng.random(t_len) < 0.2).astype(np.float32),
                    "delta": delta})
    return out


def test_collate_padding_mask_dtype_delta():
    """Collate pads to the longest crop: h is fp16 with zeros beyond each length, mask
    sums to the true lengths, y matches, and delta is the crop's constant broadcast
    over its valid frames (0 in the padding).
    """
    batch = _tiny_crops()
    out = _pin_or_skip(run_mod.collate, batch)
    length = max(len(c["y"]) for c in batch)
    assert out["h"].shape == (4, length, 3) and out["h"].dtype == torch.float16
    for i, crop in enumerate(batch):
        t = len(crop["y"])
        assert out["mask"][i].sum().item() == t
        assert torch.all(out["h"][i, t:] == 0)
        assert torch.all(out["delta"][i, :t] == crop["delta"])
        assert torch.all(out["delta"][i, t:] == 0)
        np.testing.assert_allclose(out["y"][i, :t].numpy(), crop["y"], atol=1e-6)
        np.testing.assert_allclose(out["h"][i, :t].float().numpy(), crop["h"],
                                   atol=1e-2)


def test_batches_length_sorted_buckets():
    """Batches sorts crops by length once: within the concatenated bucket order the
    crop lengths must be non-decreasing, and every crop appears exactly once.
    """
    crops = _tiny_crops()
    loader = _pin_or_skip(run_mod.Batches, crops, batch_size=2, device=torch.device("cpu"))
    flat = [c for chunk in loader.chunks for c in chunk]
    lengths = [len(c["y"]) for c in flat]
    assert lengths == sorted(lengths)
    assert {id(c) for c in flat} == {id(c) for c in crops}


def test_batches_shuffle_permutes_bucket_order_only():
    """Shuffling must reorder which bucket comes first but never change any bucket's
    membership -- 'bucket order, not membership' per the docstring.
    """
    crops = _tiny_crops() * 3   # 12 crops -> 6 buckets of 2
    loader = _pin_or_skip(run_mod.Batches, crops, batch_size=2, device=torch.device("cpu"))
    base = [tuple(id(c) for c in chunk) for chunk, _ in loader()]
    rng = np.random.default_rng(0)
    shuffled = [tuple(id(c) for c in chunk) for chunk, _ in loader(shuffle=True, rng=rng)]
    assert sorted(base) == sorted(shuffled)       # same buckets, membership intact
    assert len(base) == len(shuffled) == 6


def test_readout_oracle_control_passes_on_truth():
    """assert_readout_recovers_oracle run on clean synthetic crops must succeed with a
    score near 1.0: the true phase read through the shipped rule g recovers the
    annotated downbeats.
    """
    crops = [_oracle_crop(period=2.0, lo_t=lo) for lo in (3.0, 5.0, 8.0)]
    value = run_mod.assert_readout_recovers_oracle(crops)
    assert value > 0.95


def test_readout_oracle_control_raises_on_broken_readout(monkeypatch):
    """Replace the read-out with the documented historical bug (differencing mu
    directly, whose discontinuity sits at phi = pi, half a bar off): the control must
    raise, because a read-out that cannot score the truth cannot score a model.
    """
    def broken(mu, mask=None):
        return torch.diff(mu, dim=-1) < -math.pi
    monkeypatch.setattr(controls_mod, "downbeat_frames", broken)
    crops = [_oracle_crop(period=2.0, lo_t=lo) for lo in (3.0, 5.0, 8.0)]
    with pytest.raises(AssertionError, match="READ-OUT BROKEN"):
        run_mod.assert_readout_recovers_oracle(crops)


def test_target_blindness_control_detects_leak():
    """assert_encoder_is_target_blind must PASS for the real model (whose encoder never
    receives y) and FAIL for a model whose deployed phase moves with the target.
    """
    _seed()
    model = BarPhaseVAE(input_dim=4, hidden=8)
    batch = {"h": torch.randn(2, 10, 4), "delta": torch.full((2, 10), 0.06),
             "mask": torch.ones(2, 10), "y": torch.zeros(2, 10)}
    run_mod.assert_encoder_is_target_blind(model, batch)   # must not raise

    leaky = BarPhaseVAE(input_dim=4, hidden=8)
    calls = {"n": 0}
    real = leaky.infer_phase

    def cheating(h, delta=None):
        calls["n"] += 1
        return real(h, delta) + (0.1 if calls["n"] > 2 else 0.0)
    leaky.infer_phase = cheating
    with pytest.raises(AssertionError):
        run_mod.assert_encoder_is_target_blind(leaky, batch)


# ============================================================== vbpm cache write


def _cache_write(group_dir: pathlib.Path, stem: str, features: np.ndarray):
    """Replicates vbpm.data.iter_frontend_features' write-then-rename block verbatim."""
    cache_path = group_dir / f"{stem}.npy"
    group_dir.mkdir(parents=True, exist_ok=True)
    partial = cache_path.with_suffix(".npy.partial")
    with open(partial, "wb") as fh:
        np.save(fh, features.astype(np.float32))
    partial.replace(cache_path)
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
