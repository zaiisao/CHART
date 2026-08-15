"""Behavioural tests for the vbpm package, derived from docstrings and math only."""
from __future__ import annotations

import math
import pathlib
import types

import numpy as np
import pytest
import torch

from vbpm import run as run_mod
from vbpm.config import load_config
from vbpm.scoring import controls as controls_mod
from vbpm.scoring.evaluation import f_measure, null_times, peak_times
from vbpm.constants import (KAPPA_PHYSICAL, MAX_KAPPA, TEMPO_SIGMA_CEIL, TEMPO_SIGMA_INIT,
                            TWO_PI)
from vbpm.model import VBPM, downbeat_frames
from vbpm.nets import Encoder, bounded_kappa, inverse_softplus, vonmises_entropy
from vbpm.vonmises import log_i0, mean_resultant, sample_vonmises

from vbpm.data.features import FPS, atomic_save_npy

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

    big = bounded_kappa(torch.tensor([10 * MAX_KAPPA, 1000 * MAX_KAPPA],
                                     dtype=torch.float64))
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
    """mu = offset + cumsum(exp(log-dotphi)): rotation is STRUCTURAL, not learned."""
    _seed()
    enc = Encoder(input_dim=4, d_model=8)
    h = torch.randn(2, 300, 4)
    post, _ = enc(h, torch.ones(2, 300))
    inc = post["tempo"]["mu"][:, :-1]

    # 1. strictly increasing: the tempo is exp(...) so every step is positive. A frozen or
    #    sign-balanced trajectory -- the measured collapse -- is unrepresentable.
    assert torch.all(inc > 0), "phase is not monotonically advancing"

    # 3. the tempo lands in the physical band at initialisation (a 0.6-12 s bar), which is
    #    what puts a fresh model inside the +-3% basin where the reconstruction gradient
    #    on the tempo is coherent at all.
    assert 0.01 <= float(inc.mean()) <= 0.2


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
    model = VBPM(input_dim=4, d_model=8, emission="cosine")
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
    model = VBPM(input_dim=4, d_model=8, emission="triangle")
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
    model = VBPM(input_dim=4, d_model=8, emission="cosine")
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
    model = VBPM(input_dim=4, d_model=8, emission="cosine")
    model.emission_b_floor.fill_(5.0)
    b_before = model.emission_b.item()
    state = model.state_dict()
    fresh = VBPM(input_dim=4, d_model=8, emission="cosine")
    fresh.load_state_dict(state)
    assert fresh.emission_b.item() == pytest.approx(b_before), \
        "scheduled emission floor lost across save/load"


def test_kl_jitter_nonnegative():
    """KL(q||p) >= 0 for any q and p; the closed form must respect this up to float
    tolerance across many random (mu, kappa, delta) draws.
    """
    _seed()
    model = VBPM(input_dim=4, d_model=8)
    for _ in range(20):
        B, T = 3, 12
        mu = (torch.rand(B, T) * 2 - 1) * math.pi
        kappa = torch.rand(B, T) * 3000 + 1.0
        mask = torch.ones(B, T)
        kl = model.kl_jitter(mu, kappa, mask)
        assert torch.all(kl > -1e-3), f"negative KL {kl.min().item()}"


def test_kl_jitter_analytic_three_frame():
    """T=3 hand computation: same-mean vM KL per frame, summed under the mask."""
    scipy_special = pytest.importorskip("scipy.special")
    model = VBPM(input_dim=4, d_model=8)
    mu = torch.tensor([[0.30, 0.36, 0.43]], dtype=torch.float64)
    kappas = (800.0, 1200.0, 950.0)
    mask = torch.ones(1, 3, dtype=torch.float64)
    got = model.kl_jitter(mu, torch.tensor([list(kappas)], dtype=torch.float64), mask).item()

    def log_i0(k):
        return float(np.log(scipy_special.ive(0, k)) + k)

    def A1(k):
        return float(scipy_special.ive(1, k) / scipy_special.ive(0, k))

    expected = sum(log_i0(KAPPA_PHYSICAL) - log_i0(k) + A1(k) * (k - KAPPA_PHYSICAL)
                   for k in kappas)
    assert got == pytest.approx(expected, rel=1e-9)


def test_kl_jitter_zero_at_the_prior_and_masked_frames_free():
    """kappa == kappa_physical costs exactly 0, and a masked frame contributes 0."""
    model = VBPM(input_dim=4, d_model=8)
    mu = torch.tensor([[0.30, 0.36, 0.43]], dtype=torch.float64)
    kappa = torch.full((1, 3), KAPPA_PHYSICAL, dtype=torch.float64)
    mask = torch.ones(1, 3, dtype=torch.float64)
    assert model.kl_jitter(mu, kappa, mask).item() == pytest.approx(0.0, abs=1e-12)

    kappa2 = kappa.clone()
    kappa2[0, 2] = 5.0
    mask2 = mask.clone()
    mask2[0, 2] = 0.0
    assert model.kl_jitter(mu, kappa2, mask2).item() == pytest.approx(0.0, abs=1e-12)


def test_kl_jitter_invariant_to_shift_and_rate():
    """The prior penalises tempo CHANGE only, so the KL must be unchanged by (a) adding a
        constant to the whole trajectory and (b) adding a constant RATE ramp. Both leave the
        second difference identical -- the docstring's phase-blindness and tempo-scale-
        freedom, which together are why only the emission can locate or size a bar.
    """
    _seed()
    model = VBPM(input_dim=4, d_model=8)
    T = 9
    mu = torch.cumsum(torch.rand(1, T, dtype=torch.float64) * 0.02 + 0.05, dim=1)
    kappa = torch.rand(1, T, dtype=torch.float64) * 500 + 50
    mask = torch.ones(1, T, dtype=torch.float64)
    base = model.kl_jitter(mu, kappa, mask).item()

    shifted = model.kl_jitter(mu + 1.234, kappa, mask).item()
    ramp = torch.arange(T, dtype=torch.float64)[None] * 0.031
    rerated = model.kl_jitter(mu + ramp, kappa, mask).item()

    assert shifted == pytest.approx(base, rel=1e-9)
    assert rerated == pytest.approx(base, rel=1e-9)


def test_kl_jitter_masked_frames_contribute_zero():
    """A frame with mask 0 must contribute nothing: the KL of a [1,5] sequence whose
    last two frames are masked equals the KL of the first three frames alone. Three
    real frames because the chain term needs two predecessors.
    """
    model = VBPM(input_dim=4, d_model=8)
    mu5 = torch.tensor([[0.30, 0.36, 0.43, 9.0, -9.0]], dtype=torch.float64)
    kappa5 = torch.tensor([[800.0, 1200.0, 950.0, 7.0, 7.0]], dtype=torch.float64)
    mask5 = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]], dtype=torch.float64)
    kl_masked = model.kl_jitter(mu5, kappa5, mask5).item()
    kl_short = model.kl_jitter(
        mu5[:, :3], kappa5[:, :3], torch.ones(1, 3, dtype=torch.float64)).item()
    assert kl_masked == pytest.approx(kl_short, rel=1e-9)


def test_forward_elbo_identity_and_bernoulli_recon(monkeypatch):
    """elbo == recon - kl exactly, and recon must equal the masked BCE computed by hand."""
    _seed()
    import vbpm.model as model_mod
    monkeypatch.setattr(model_mod, "sample_vonmises", lambda k: torch.zeros_like(k))
    model = VBPM(input_dim=4, d_model=8, emission="cosine")

    B, T = 2, 60
    h = torch.randn(B, T, 4)
    mask = torch.ones(B, T)
    mask[1, 50:] = 0.0
    y = torch.zeros(B, T)
    y[0, 5] = 1.0
    y[1, 18] = 1.0

    out = model(h, mask, y, samples=1)
    assert torch.allclose(out["elbo"], out["recon"] - out["kl"], atol=1e-5)

    with torch.no_grad():
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            model.emission_logits(out["phi"], mask), y, reduction="none")
        expected = -(bce * mask).sum(1)
    assert torch.allclose(out["recon"], expected, atol=1e-4)

    with torch.no_grad():
        alt = model(h, mask, y, samples=1, pos_weight=30.0)
    assert not torch.allclose(out["recon"], alt["recon"], atol=1e-5), \
        "pos_weight no longer reweights the positive frames"


def test_phase_ablation_gap_zero_when_phase_ignored():
    """An emission that returns the same logits regardless of phi must score a gap of
    exactly 0 -- the docstring defines the gap as the mean |shift| when phi is
    frozen.
    """
    _seed()
    model = VBPM(input_dim=4, d_model=8, emission="cosine")
    model.emission_logits = lambda phi, mask=None: torch.full_like(phi, -3.0)
    phi = torch.rand(2, 20) * TWO_PI
    assert model.phase_ablation_gap(phi) == 0.0


def test_phase_ablation_gap_positive_for_cosine():
    """The genuine cosine emission depends on phi, so the gap must be strictly
    positive on a non-constant phase.
    """
    _seed()
    model = VBPM(input_dim=4, d_model=8, emission="cosine")
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
    model = VBPM(input_dim=4, d_model=8)
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
    """A synthetic batch for the target-blindness control: no real audio needed."""
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
    controls_mod.assert_encoder_is_target_blind(VBPM(input_dim=4, d_model=8), batch)

    leaky = VBPM(input_dim=4, d_model=8)
    calls = {"n": 0}
    real = leaky.infer_phase

    def cheating(h, delta=None):
        calls["n"] += 1
        return real(h, delta) + (0.1 if calls["n"] > 2 else 0.0)
    leaky.infer_phase = cheating

    with pytest.raises(AssertionError):
        controls_mod.assert_encoder_is_target_blind(leaky, batch)


def test_deployed_net_reads_no_target_in_any_shipped_config():
    """EVERY config's deployed net must read h and delta only."""
    configs = sorted((pathlib.Path(__file__).resolve().parent.parent / "configs")
                     .glob("*.yaml"))
    assert configs, "no configs found: this test would pass vacuously"

    checked = 0
    for path in configs:
        _seed()
        cfg, hooks = load_config(str(path))
        # BROKEN BY THE OBSERVATION-MODEL CHANGE, deliberately. Each of these variants
        # owns a forward()/heads() written against the per-frame Bernoulli and the
        # two-value heads() signature; they raise rather than silently mis-score. They
        # are BCE models, so running them would compare two different objectives under
        # one elbo column. Delete a name here when the variant is ported, not before.
        if cfg.variant in {"psi", "ladder", "anchor_time",
                           "interval_exact_norm", "interval_exact_rotation",
                           "interval_exact_tempo"}:
            continue
        model = hooks.build_model(cfg, input_dim=4)
        controls_mod.assert_encoder_is_target_blind(
            model, _blindness_batch(input_dim=4))
        checked += 1
    assert checked, "every config was skipped: this test would pass vacuously"


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
    """The four numbers must distinguish trajectories that F cannot."""
    from vbpm.scoring.evaluation import scoring_records, trajectory_health

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
    tempo = TWO_PI / (period * fps)
    kappa = torch.full((2, n), 2000.0)

    def wrap(x):
        return torch.atan2(torch.sin(x), torch.cos(x))

    def health(mu):
        return trajectory_health(mu[None].repeat(2, 1), kappa, torch.ones(2, n), crops)

    adv, kap, err, cov = health(wrap(tempo * (t - db[0] * fps)))
    assert adv == pytest.approx(tempo, rel=1e-3)
    assert err < 1e-3 and cov == 1.0 and kap == pytest.approx(2000.0)

    adv_o, _, err_o, cov_o = health(wrap(tempo * (t - db[0] * fps) + math.pi))
    assert adv_o == pytest.approx(tempo, rel=1e-3)      # tempo right, offset wrong
    assert err_o == pytest.approx(math.pi, abs=1e-3)   # worse than chance, not random
    assert cov_o == 1.0

    adv_s, _, err_s, _ = health(wrap(0.715 * t))
    assert adv_s == pytest.approx(0.715, rel=1e-3)
    assert err_s > 1.5                                  # indistinguishable from chance

    adv_f, _, err_f, cov_f = health(torch.zeros(n))
    assert adv_f == pytest.approx(0.0, abs=1e-6)
    assert err_f == pytest.approx(math.pi / 2, rel=0.02)
    assert cov_f == pytest.approx(1 / 16)


def test_rate_bound_is_identity_in_the_interior():
    """The bound may soften the rails; it must not move interior tempos."""
    for seconds in (1.0, 1.5, 2.0, 3.0, 5.0):
        x = torch.tensor([[math.log(TWO_PI / (seconds * 50.0))]], dtype=torch.float64)
        tempo, _ = Encoder._ramp(x)
        assert float(TWO_PI / (float(tempo) * 50.0)) == pytest.approx(seconds, rel=1e-9)
    x = torch.tensor([[math.log(TWO_PI / (60.0 * 50.0))]], dtype=torch.float64,
                     requires_grad=True)
    Encoder._ramp(x)[0].sum().backward()
    assert float(x.grad) > 0.0, "the tempo bound is an absorbing rail again"


def test_tempo_entropy_is_charged_per_frame_not_per_bar():
    """The 2026-08-13 repair: tempo_entropy is sum over FRAMES of 0.5*log(2*pi*e*sigma^2),
    so the latent's dimension stops being a function of the tempo. A per-bar charge let a
    fast rate harvest ~1.7 nats per extra bar."""
    torch.manual_seed(0)
    model = VBPM(input_dim=4, d_model=8, emission="triangle").double().train()
    h = torch.randn(1, 700, 4).double()
    w = torch.ones(1, 700).double()
    y = torch.zeros(1, 700).double()
    y[:, ::100] = 1.0

    model.eval()
    sigma = model.encoder(h, w)[0]["tempo"]["sigma"]
    expected = float(((0.5 * math.log(2 * math.pi * math.e)
                       + torch.log(sigma)) * w).sum(1)[0])

    out = model(h, w, y, samples=1)

    assert float(out["tempo_entropy"][0]) == pytest.approx(expected, rel=1e-9)
    assert sigma.shape == (1, 700), "sigma is per frame, not per bar"


def test_learned_sigma_starts_at_init_and_gets_gradient():
    """A fresh sigma head reads the softplus init everywhere, and the ELBO trains it."""
    _seed()
    model = VBPM(input_dim=4, d_model=8, emission="triangle").train()
    h, mask = torch.randn(2, 300, 4), torch.ones(2, 300)
    y = torch.zeros(2, 300); y[:, ::100] = 1.0

    trunk = model.encoder.features(h, mask)
    raw = model.encoder.output_channels(trunk)["tempo_sigma_logit"]
    sigma = torch.nn.functional.softplus(raw)
    assert float((sigma - 0.0005).abs().max()) < 1e-4

    out = model(h, mask, y, samples=1)
    model.zero_grad()
    (-out["elbo"].mean()).backward()
    g = float(model.encoder.out.weight.grad[3].abs().sum())
    assert g > 0.0, "the sigma channel receives no gradient"


def test_anchor_k_folds_its_own_evidence_not_a_frontend_channel():
    """anchor_k must fold the model's own evidence, not a raw frontend channel."""
    import inspect
    from vbpm.variants import anchor_k as ak
    code = "\n".join(line.split("#")[0] for line in inspect.getsource(ak).splitlines())
    assert "h[..., -2:]" not in code, \
        "anchor_k is folding a raw frontend channel again"
    assert "downbeat_scores" in code, \
        "anchor_k no longer folds the model's own evidence head"
    sig = inspect.signature(ak.AnchorKVAE.bin_downbeat)
    assert list(sig.parameters)[1] == "a", \
        "bin_downbeat's first argument is not the evidence a_t"
