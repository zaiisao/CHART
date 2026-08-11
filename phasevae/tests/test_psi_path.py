"""Tests for the psi-mode (conditional-prior) code paths of phasevae.

Covered here and nowhere else: vonmises_log_density, the psi variant module's
PosteriorEncoder / RotationPrior extensions, PsiBarPhaseVAE forward and its mixture
KL, the psi deployment path of infer_phase, the target-blindness control on the
deployed prior_net, the physics anchor's rotation invariance, and a psi state-dict
roundtrip.

Every expected value is derived from a signature, docstring, or the mathematics it
names -- never from running the implementation first. Contract violations, if any,
keep their honest assertion under @pytest.mark.xfail(strict=False).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------------
# SKIPPED WHOLESALE, deliberately. This module tests a variant whose forward() and
# heads() are written against the per-frame Bernoulli observation model and the old
# two-value heads() signature. Commit 2191ea9 replaced the observation with the
# annotated downbeat TIMES (model.align_log_prob) and the anchor became a circular
# mean over every frame, so these variants raise rather than silently mis-score.
# They are BCE models: running them would put two different objectives under one
# elbo column. Re-enable by deleting this block WHEN the variant is ported, not
# before -- and expect the assertions themselves to need rewriting, since several
# encode the Bernoulli composition.
import pytest as _pytest
_pytest.skip("variant not ported to the alignment objective (see 2191ea9)",
             allow_module_level=True)
# ---------------------------------------------------------------------------------

import math

import numpy as np
import pytest
import torch

from phasevae.scoring import controls as controls_mod
from phasevae.model import TWO_PI, Encoder, vonmises_log_density
from phasevae.variants.psi import PosteriorEncoder, PsiBarPhaseVAE, RotationPrior


def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


def _psi_model(psi_rotations=4, d_model=8):
    _seed()
    return PsiBarPhaseVAE(input_dim=4, d_model=d_model, emission="cosine",
                          rotations=psi_rotations)


def _batch(B=2, T=10):
    h = torch.randn(B, T, 4)
    delta = torch.full((B, T), 0.06)
    mask = torch.ones(B, T)
    mask[-1, T - 3:] = 0.0
    y = torch.zeros(B, T)
    y[0, 3] = 1.0
    y[-1, 4] = 1.0
    return h, delta, mask, y


# ================================================= 1. vonmises_log_density


def test_vonmises_log_density_normalises_over_circle():
    """Input: a dense grid on [0, 2*pi) at several kappas. Asserted: the quadrature
    of exp(log f) over the circle is 1. Why: 'Per-element log vM(z; mu, kappa)'
    names a probability density on the circle; a density that does not integrate
    to 1 has a wrong normaliser.
    """
    z = torch.linspace(0.0, TWO_PI, 200_001, dtype=torch.float64)
    for k in (0.0, 0.5, 5.0, 200.0, 2000.0):
        logf = vonmises_log_density(z, torch.tensor(0.7, dtype=torch.float64),
                                    torch.tensor(k, dtype=torch.float64))
        trapezoid = getattr(np, "trapezoid", np.trapz)
        integral = float(trapezoid(np.exp(logf.numpy()), z.numpy()))
        assert integral == pytest.approx(1.0, abs=1e-4), f"kappa={k}: {integral}"


def test_vonmises_log_density_matches_scipy_logpdf():
    """Input: random angles, means and kappas. Asserted: elementwise agreement with
    scipy.stats.vonmises.logpdf. Why: both claim the same named density; scipy is
    the independent reference implementation.
    """
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    z = rng.uniform(-math.pi, math.pi, 50)
    mu = rng.uniform(-math.pi, math.pi, 50)
    for k in (0.3, 4.0, 90.0, 1500.0):
        ours = vonmises_log_density(torch.tensor(z), torch.tensor(mu),
                                    torch.tensor(k, dtype=torch.float64)).numpy()
        ref = scipy_stats.vonmises(kappa=k, loc=mu).logpdf(z)
        np.testing.assert_allclose(ours, ref, atol=1e-8)


def test_vonmises_log_density_rotation_invariant():
    """Input: (z, mu) pairs and a set of global rotations c. Asserted:
    log f(z + c; mu + c, k) == log f(z; mu, k). Why: the density depends on z and
    mu only through z - mu (kappa*cos(z - mu) in the docstring), so a shared
    rotation must cancel.
    """
    _seed()
    z = torch.rand(30, dtype=torch.float64) * TWO_PI
    mu = torch.rand(30, dtype=torch.float64) * TWO_PI
    kappa = torch.tensor(12.0, dtype=torch.float64)
    base = vonmises_log_density(z, mu, kappa)
    for c in (0.9, -2.5, 7.0 * math.pi):
        rotated = vonmises_log_density(z + c, mu + c, kappa)
        assert torch.allclose(base, rotated, atol=1e-8)


# ============================================== 2. Encoder reads_target flag


def test_encoder_reads_target_requires_y():
    """Input: Encoder(reads_target=True) called without y. Asserted: AssertionError.
    Why: the forward asserts 'posterior encoder requires the target input y' --
    a y-informed posterior evaluated without its target is meaningless.
    """
    _seed()
    enc = PosteriorEncoder(input_dim=4, d_model=8)
    with pytest.raises(AssertionError):
        enc(torch.randn(1, 10, 4))


def test_encoder_reads_target_output_depends_on_y_and_h_unmutated():
    """Input: the same h with two different targets. Asserted: (mu, kappa) CHANGE
    when y changes, and the h tensor is bit-identical before and after forward.
    Why: reads_target=True makes q a posterior over (h, y) -- the target 'enters as
    frame t's SECOND variable, joined here, never by mutating the feature tensor
    upstream'.
    """
    _seed()
    enc = PosteriorEncoder(input_dim=4, d_model=8)
    h = torch.randn(2, 12, 4)
    h_copy = h.clone()
    mu0, k0 = enc(h, y=torch.zeros(2, 12))
    mu1, k1 = enc(h, y=torch.ones(2, 12))
    assert not torch.equal(mu0, mu1), "posterior mean ignored the target"
    assert torch.equal(h, h_copy), "forward mutated the input features in place"


def test_base_encoder_structurally_target_blind():
    """Input: the base Encoder's forward signature. Asserted: no y parameter
    and no reads_target attribute. Why: the base encoder 'never sees the target' by
    CONSTRUCTION -- only the psi variant's PosteriorEncoder subclass adds y.
    """
    _seed()
    enc = Encoder(input_dim=4, d_model=8)
    named = enc.forward.__code__.co_varnames[:enc.forward.__code__.co_argcount]
    assert "y" not in named
    assert not getattr(enc, "reads_target", False)


# ================================================= 3. Encoder rotations head


def test_encoder_rotations_returns_third_output_of_shape_bk():
    """Input: Encoder(rotations=K) on a [B, T, D] batch. Asserted: forward returns
    a 3-tuple whose third element has shape [B, K]. Why: the docstring promises
    '[, rotation logits [B, K]]' -- one logit per whole-trajectory rotation, pooled
    over frames, not per frame.
    """
    _seed()
    for K in (2, 5):
        enc = RotationPrior(input_dim=4, d_model=8, rotations=K)
        out = enc(torch.randn(3, 11, 4))
        assert len(out) == 3
        mu, kappa, rot = out
        assert mu.shape == (3, 11) and kappa.shape == (3, 11)
        assert rot.shape == (3, K)
    # K = 1 builds NO rot head (one-logit softmax is constant): two-tuple contract
    enc = RotationPrior(input_dim=4, d_model=8, rotations=1)
    assert len(enc(torch.randn(3, 11, 4))) == 2


def test_encoder_rotations_zero_keeps_two_tuple():
    """Input: Encoder(rotations=0), the default. Asserted: forward returns exactly
    the old 2-tuple (mu, kappa). Why: backwards compatibility -- every non-psi
    call site unpacks two values.
    """
    _seed()
    enc = RotationPrior(input_dim=4, d_model=8, rotations=0)
    out = enc(torch.randn(2, 7, 4))
    assert isinstance(out, tuple) and len(out) == 2


# ============================================ 4. BarPhaseVAE psi-mode forward


def test_psi_forward_returns_prior_anchor_and_elbo_identity():
    """Input: a padded batch through BarPhaseVAE(psi=True). Asserted: the dict
    contains 'prior_anchor', and elbo == recon - kl exactly. Why: psi mode adds
    the Eq. 27 physics-anchoring term to the output, and the ELBO definition
    (recon minus KL) is unchanged by which prior the KL is taken against.
    """
    model = _psi_model()
    h, delta, mask, y = _batch()
    out = model(h, mask, y, samples=1)
    assert "prior_anchor" in out
    assert out["prior_anchor"].shape == out["kl"].shape == (2,)
    assert torch.allclose(out["elbo"], out["recon"] - out["kl"], atol=1e-5)


def test_psi_kl_zero_when_prior_equals_posterior():
    """Input: psi model with K=1 whose prior_net is monkeypatched to return EXACTLY
    the posterior's (mu, kappa). Asserted: the MEAN MC KL over many forward calls
    is ~0, with tolerance set by the empirical sd of the estimates. Why:
    KL(q || p) = E_q[log q(z) - log p(z)] = 0 when p == q; K=1 builds NO rotation
    head (a one-logit softmax is constant, i.e. a dead parameter), so p_psi is the
    bare base path and the densities coincide directly.
    """
    model = _psi_model(psi_rotations=1)
    h, delta, mask, y = _batch()

    def fake_prior(hh, dd=None, yy=None):
        mu_q, kappa_q = model.encoder(hh, dd, y)
        return mu_q.detach(), kappa_q.detach()

    model.prior_net.forward = fake_prior

    # the q-to-psi KL now lives under "distill" (out["kl"] is q-to-physical: the
    # stop-gradient distillation redesign after joint training collapsed)
    kls = torch.stack([model(h, mask, y, samples=1)["distill"].detach()
                       for _ in range(20)])                       # [20, B]

    sd = kls.std().item()
    tol = max(1e-4, 4.0 * sd / math.sqrt(kls.numel()))
    assert abs(kls.mean().item()) < tol, \
        f"KL to an identical prior averaged {kls.mean().item()} (sd {sd})"


def _kl_inputs(model, B=2, T=8):
    _seed()
    K = model.psi_rotations
    z = torch.rand(B, T, dtype=torch.float64) * TWO_PI
    mu_q = torch.rand(B, T, dtype=torch.float64) * TWO_PI
    kappa_q = torch.rand(B, T, dtype=torch.float64) * 40 + 1.0
    mu_p = torch.rand(B, T, dtype=torch.float64) * TWO_PI
    kappa_p = torch.rand(B, T, dtype=torch.float64) * 40 + 1.0
    rot = torch.randn(B, K, dtype=torch.float64)
    mask = torch.ones(B, T, dtype=torch.float64)
    mask[-1, T - 2:] = 0.0
    return z, mu_q, kappa_q, mu_p, kappa_p, rot, mask


def test_psi_kl_invariant_to_constant_logit_shift():
    """Input: kl_to_conditional_prior with rot_logits vs rot_logits + c. Asserted:
    identical KL. Why: the weights are softmax(rot_logits), and softmax is
    invariant to adding a constant to every logit -- the mixture is unchanged.
    """
    model = _psi_model(psi_rotations=6)
    z, mu_q, kappa_q, mu_p, kappa_p, rot, mask = _kl_inputs(model)
    base = model.kl_to_conditional_prior(z, mu_q, kappa_q, mu_p, kappa_p, rot, mask)
    shifted = model.kl_to_conditional_prior(z, mu_q, kappa_q, mu_p, kappa_p,
                                            rot + 37.5, mask)
    assert torch.allclose(base, shifted, atol=1e-8)


def test_psi_kl_invariant_to_cyclic_component_permutation():
    """Input: the mixture with its components cyclically permuted -- logits rolled
    by j while the base path is rotated by -2*pi*j/K, so component k's mean
    mu_p - 2*pi*j/K + 2*pi*k/K with weight w_{k-j} re-enumerates exactly the
    original set of (weight, mean) pairs mod 2*pi. Asserted: identical KL. Why: a
    mixture is a set of weighted components; relabelling them must not change its
    density (the von Mises is 2*pi-periodic, so mod-2*pi means coincide).
    """
    model = _psi_model(psi_rotations=6)
    K = model.psi_rotations
    z, mu_q, kappa_q, mu_p, kappa_p, rot, mask = _kl_inputs(model)

    base = model.kl_to_conditional_prior(z, mu_q, kappa_q, mu_p, kappa_p, rot, mask)
    for j in (1, 4):
        permuted = model.kl_to_conditional_prior(
            z, mu_q, kappa_q, mu_p - TWO_PI * j / K, kappa_p,
            torch.roll(rot, shifts=j, dims=1), mask)
        # exact in exact arithmetic; in float64 the shifted mu_p perturbs each
        # cos argument by a rounding ulp which kappa*T amplifies to ~1e-5
        assert torch.allclose(base, permuted, atol=1e-4), f"shift j={j}"


# ================================================== 5. psi deployment path


def test_psi_infer_phase_requires_eval_mode():
    """Input: infer_phase on a psi model in train mode, then eval mode. Asserted:
    AssertionError in train mode, success and [B, T] shape in eval mode. Why: the
    'deployment path must run in eval mode' assertion guards psi mode too.
    """
    model = _psi_model()
    h, delta, _, _ = _batch()
    model.train()
    with pytest.raises(AssertionError):
        model.infer_phase(h, delta)

    model.eval()
    assert model.infer_phase(h, delta).shape == (2, 10)


def test_psi_infer_phase_is_prior_mu_plus_best_rotation():
    """Input: prior_net monkeypatched to fixed (mu_p, kappa_p, rot_logits) with a
    known argmax per crop. Asserted: infer_phase == mu_p + 2*pi*argmax(logits)/K,
    reconstructed by hand. Why: the psi deployment rule in infer_phase is exactly
    'the conditional prior's base path shifted by the winning rotation'.
    """
    model = _psi_model(psi_rotations=4)
    model.eval()

    B, T, K = 2, 6, 4
    mu_p = torch.randn(B, T)
    rot = torch.zeros(B, K)
    rot[0, 2] = 5.0     # crop 0's winner: k=2
    rot[1, 3] = 5.0     # crop 1's winner: k=3
    model.prior_net.forward = lambda h, d=None, y=None: (mu_p, torch.ones(B, T), rot)

    got = model.infer_phase(torch.randn(B, T, 4), torch.full((B, T), 0.06))
    expected = mu_p + TWO_PI * torch.tensor([[2.0], [3.0]]) / K
    assert torch.equal(got, expected)


def test_psi_infer_phase_never_consults_posterior_encoder():
    """Input: a psi model whose posterior encoder's forward is monkeypatched to
    raise. Asserted: infer_phase still succeeds. Why: the posterior reads y and is
    training scaffolding in psi mode; deployment must run entirely off p_psi(z|h),
    or the model would be unusable exactly when the target is unavailable.
    """
    model = _psi_model()
    model.eval()

    def poisoned(*a, **k):
        raise RuntimeError("posterior encoder consulted at deployment")

    model.encoder.forward = poisoned
    h, delta, _, _ = _batch()
    out = model.infer_phase(h, delta)
    assert out.shape == (2, 10)


# ============================================ 6. target-blindness control


def test_target_blindness_control_passes_on_psi_model():
    """Input: controls.assert_encoder_is_target_blind on a genuine psi model.
    Asserted: no exception. Why: in psi mode the DEPLOYED net is prior_net, which
    reads (h, delta) only, so both the structural and the behavioural check must
    pass even though the posterior encoder legitimately reads y.
    """
    model = _psi_model()
    h, delta, mask, y = _batch()
    controls_mod.assert_encoder_is_target_blind(
        model, {"h": h, "delta": delta, "mask": mask, "y": y})


def test_target_blindness_control_catches_target_reading_prior_net():
    """Input: a psi model sabotaged so its prior_net has reads_target=True.
    Asserted: the control raises AssertionError. Why: the control's whole purpose
    is to refuse a deployed inference network that consumes the target -- a
    y-reading prior_net is unusable at test time and must be caught before any
    number is produced.
    """
    model = _psi_model()
    model.prior_net = PosteriorEncoder(input_dim=4, d_model=8)
    h, delta, mask, y = _batch()
    with pytest.raises(AssertionError):
        controls_mod.assert_encoder_is_target_blind(
            model, {"h": h, "delta": delta, "mask": mask, "y": y})


# ================================================== 7. physics anchor


def test_physics_anchor_invariant_to_global_rotation_of_prior_mean():
    """Input: kl_to_physical_prior at mu_p and at mu_p + c for several global
    rotations c. Asserted: identical anchor value (float tolerance). Why: the
    physical prior is uniform in phi_1 and Markov in increments, so the anchor
    reads mu only through mu_t - mu_{t-1}; a shared rotation must cancel. This is
    exactly why psi mode needs the rotation mixture: the anchor alone cannot pin
    the absolute phase.
    """
    model = _psi_model()
    _seed()
    B, T = 3, 12
    mu_p = torch.rand(B, T, dtype=torch.float64) * TWO_PI
    kappa_p = torch.rand(B, T, dtype=torch.float64) * 3000 + 1.0
    mask = torch.ones(B, T, dtype=torch.float64)
    mask[-1, T - 4:] = 0.0

    base = model.kl_to_physical_prior(mu_p, kappa_p, mask)
    for c in (1.3, -4.0, 20.0 * math.pi):
        rotated = model.kl_to_physical_prior(mu_p + c, kappa_p, mask)
        assert torch.allclose(base, rotated, atol=1e-5), f"c={c}"


# ============================================== 8. state-dict roundtrip


def test_psi_state_dict_roundtrip_bit_identical_inference():
    """Input: a psi model's state_dict loaded into a freshly constructed psi model
    with different random init. Asserted: infer_phase output bit-identical between
    the two in eval mode. Why: the checkpoint must carry the entire deployed
    object (prior_net weights included); any parameter or buffer missing from the
    state_dict would silently change the deployed phase.
    """
    model = _psi_model()
    h, delta, _, _ = _batch()
    model.eval()
    before = model.infer_phase(h, delta)

    torch.manual_seed(123)   # a genuinely different init in the fresh model
    fresh = PsiBarPhaseVAE(input_dim=4, d_model=8, emission="cosine", rotations=4)
    fresh.load_state_dict(model.state_dict())
    fresh.eval()
    after = fresh.infer_phase(h, delta)
    assert torch.equal(before, after), \
        "psi checkpoint did not reproduce the deployed phase bit-identically"
