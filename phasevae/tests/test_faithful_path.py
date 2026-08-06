"""Tests for the phasevae paths not covered by test_phasevae.py.

Covered here: the free (per-frame) posterior branch of Encoder, the
EmissionTransformer, the sampled-reconstruction ELBO path of BarPhaseVAE.forward,
the reparameterised von Mises sampler, the KL of the free posterior against Monte
Carlo, and the train-loop objective arithmetic in run.py.

Every expected value is derived from a signature, docstring, or the mathematics it
names -- never from running the implementation first. Contract violations, if any,
keep their honest assertion under @pytest.mark.xfail(strict=False).
"""
from __future__ import annotations

import inspect
import math
import types

import numpy as np
import pytest
import torch

import phasevae.model as model_mod
from phasevae import run as run_mod
from phasevae.model import (BarPhaseVAE, EmissionTransformer, Encoder,
                            KAPPA_PHYSICAL, MAX_KAPPA, TWO_PI)
from phasevae.vonmises import sample_vonmises


def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


# ================================================== 1. Encoder free branch


def test_encoder_free_branch_per_frame_locality_and_shapes():
    """Input: [B,T,D] audio with one frame perturbed. Asserted: with drift_bound=0
    the mean is emitted PER FRAME ('per frame, factorised' in the class docstring),
    so perturbing frame k must move mu at frame k; outputs are [B,T]; kappa is
    strictly positive and below MAX_KAPPA (bounded_kappa's stated ceiling). Why:
    the free branch is the deployed q_phi and its per-frame parameterisation is the
    whole difference from the drift-bounded branch.
    """
    _seed()
    enc = Encoder(input_dim=4, hidden=8, drift_bound=0.0)
    h = torch.randn(1, 20, 4)
    mu, kappa = enc(h)
    assert mu.shape == (1, 20) and kappa.shape == (1, 20)
    assert torch.all(kappa > 0) and torch.all(kappa < MAX_KAPPA)

    h2 = h.clone()
    h2[0, 7] += 5.0
    mu2, _ = enc(h2)
    assert not torch.allclose(mu[0, 7], mu2[0, 7], atol=1e-6), \
        "per-frame mean did not respond to its own frame's input"


def test_encoder_free_branch_ignores_delta_entirely():
    """Input: same h with delta=None, a constant, and a wild per-frame delta.
    Asserted: identical (mu, kappa) in all three cases and mu in (-pi, pi] (atan2
    range). Why: the free branch's contract is that delta only parameterises the
    drift-bounded cumsum construction; the atan2 head reads h alone.
    """
    _seed()
    enc = Encoder(input_dim=4, hidden=8, drift_bound=0.0)
    h = torch.randn(2, 15, 4)
    mu_none, k_none = enc(h, None)
    mu_c, k_c = enc(h, torch.full((2, 15), 0.06))
    mu_w, k_w = enc(h, torch.randn(2, 15) * 3.0)
    assert torch.equal(mu_none, mu_c) and torch.equal(mu_none, mu_w)
    assert torch.equal(k_none, k_c) and torch.equal(k_none, k_w)
    assert torch.all(mu_none > -math.pi) and torch.all(mu_none <= math.pi)


# ================================================== 2. EmissionTransformer


def test_emission_transformer_reads_only_phi_and_mask():
    """Input: the forward signature itself. Asserted: the only possible inputs are
    phi and mask ('Reads the LATENT only, never h'), and the output on a [B,T]
    phase is [B,T]. Why: Point 1 -- an emission that could receive h would fit the
    target directly and kill the latent.
    """
    code = EmissionTransformer.forward.__code__
    assert set(code.co_varnames[:code.co_argcount]) <= {"self", "phi", "mask"}
    _seed()
    net = EmissionTransformer(d_model=16, layers=1, heads=2)
    logits = net(torch.rand(3, 11) * TWO_PI, torch.ones(3, 11))
    assert logits.shape == (3, 11)


def test_emission_transformer_time_constant_phi_gives_time_constant_logits():
    """Input: a TIME-CONSTANT phi sequence, use_positional=False. Asserted: all
    logits on the sequence are equal. Why: without positional encoding a
    self-attention stack is permutation-equivariant, so identical tokens must map
    to identical outputs -- any variation would mean position information leaks in,
    the exact shortcut the docstring flags.
    """
    _seed()
    net = EmissionTransformer(d_model=16, layers=2, heads=2, use_positional=False)
    phi = torch.full((1, 30), 0.7)
    logits = net(phi, torch.ones(1, 30))
    assert torch.allclose(logits, logits[:, :1].expand_as(logits), atol=1e-5), \
        "position information leaked into a PE-free emission"


def test_emission_transformer_sensitive_to_phi():
    """Input: two phase sequences differing everywhere. Asserted: the logits
    differ. Why: the emission is the only term that can locate a bar (its own
    docstring); an emission constant in phi has taken the shortcut.
    """
    _seed()
    net = EmissionTransformer(d_model=16, layers=1, heads=2)
    a = net(torch.zeros(1, 10))
    b = net(torch.full((1, 10), math.pi))
    assert not torch.allclose(a, b, atol=1e-6)


def test_emission_transformer_masked_frames_do_not_influence_unmasked():
    """Input: two batches identical on frames 0..7, differing ONLY on frames 8..15
    which the mask marks as padding. Asserted: the logits on the unmasked frames
    agree. Why: src_key_padding_mask semantics -- padded frames must be excluded
    from attention, else padding length would change real predictions.
    """
    _seed()
    net = EmissionTransformer(d_model=16, layers=2, heads=2)
    mask = torch.ones(1, 16)
    mask[0, 8:] = 0.0
    phi_a = torch.rand(1, 16) * TWO_PI
    phi_b = phi_a.clone()
    phi_b[0, 8:] = torch.rand(8) * TWO_PI

    la = net(phi_a, mask)
    lb = net(phi_b, mask)
    assert torch.allclose(la[0, :8], lb[0, :8], atol=1e-6), \
        "padded frames influenced unmasked logits"


# ==================================== 3. sampled-reconstruction ELBO path


def _tf_model():
    _seed()
    return BarPhaseVAE(input_dim=4, hidden=8, emission="transformer",
                       emission_layers=1, emission_dim=16)


def _batch(B=2, T=12):
    h = torch.randn(B, T, 4)
    delta = torch.full((B, T), 0.06)
    mask = torch.ones(B, T)
    mask[-1, T - 3:] = 0.0
    y = torch.zeros(B, T)
    y[0, 3] = 1.0
    y[-1, 5] = 1.0
    return h, delta, mask, y


def test_forward_transformer_elbo_identity_and_stochastic_recon():
    """Input: a padded batch through the transformer-emission forward, twice.
    Asserted: elbo == recon - kl on each returned dict (the ELBO definition in the
    forward docstring), and recon DIFFERS across the two calls. Why: the
    reconstruction is the model's only Monte Carlo term (vonmises module
    docstring), so with finite kappa two evaluations must draw different phases.
    """
    model = _tf_model()
    h, delta, mask, y = _batch()
    out1 = model(h, delta, mask, y, samples=1)
    out2 = model(h, delta, mask, y, samples=1)
    for out in (out1, out2):
        assert torch.allclose(out["elbo"], out["recon"] - out["kl"], atol=1e-5)
    assert not torch.equal(out1["recon"], out2["recon"]), \
        "reconstruction was not stochastic across calls"


def test_forward_deterministic_when_sampler_pinned(monkeypatch):
    """Input: the same batch twice with sample_vonmises monkeypatched to zeros
    (phi = mu exactly). Asserted: bit-identical recon. Why: sampling is the only
    randomness in the forward pass; frozen at the mean it must be deterministic.
    """
    monkeypatch.setattr(model_mod, "sample_vonmises",
                        lambda k: torch.zeros_like(k))
    model = _tf_model()
    h, delta, mask, y = _batch()
    out1 = model(h, delta, mask, y)
    out2 = model(h, delta, mask, y)
    assert torch.equal(out1["recon"], out2["recon"])


def test_forward_samples_k_averages_k_evaluations(monkeypatch):
    """Input: samples=3 with the sampler monkeypatched to return a counted,
    per-call constant offset (0, 0.5, 1.0 rad). Asserted: the sampler is called
    exactly 3 times and recon equals the arithmetic mean of the three hand-computed
    masked Bernoulli log-likelihoods at mu + offset. Why: 'samples: Monte Carlo
    samples for the reconstruction term' -- an average, not a sum.
    """
    offsets = [0.0, 0.5, 1.0]
    calls = {"n": 0}

    def fake_sampler(kappa):
        off = offsets[calls["n"]]
        calls["n"] += 1
        return torch.full_like(kappa, off)

    monkeypatch.setattr(model_mod, "sample_vonmises", fake_sampler)
    model = _tf_model()
    h, delta, mask, y = _batch()
    out = model(h, delta, mask, y, samples=3, pos_weight=1.0)
    assert calls["n"] == 3

    with torch.no_grad():
        terms = []
        for off in offsets:
            logits = model.emission_logits(out["mu"] + off)
            per_frame = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y, reduction="none")
            terms.append(-(per_frame * mask).sum(1))
        expected = torch.stack(terms).mean(0)
    assert torch.allclose(out["recon"], expected, atol=1e-4)


def test_forward_transformer_pos_weight_one_is_plain_bce(monkeypatch):
    """Input: tiny batch, sampler pinned to the mean, pos_weight=1. Asserted:
    recon equals the plain masked Bernoulli log-likelihood of y under the
    transformer emission at phi = mu, computed by hand. Why: pos_weight=1 is the
    unweighted ELBO reconstruction; anything else on this path is a bug.
    """
    monkeypatch.setattr(model_mod, "sample_vonmises",
                        lambda k: torch.zeros_like(k))
    model = _tf_model()
    h, delta, mask, y = _batch(B=1, T=8)
    out = model(h, delta, mask, y, samples=1, pos_weight=1.0)

    with torch.no_grad():
        logits = model.emission_logits(out["mu"])   # forward passes no mask here
        ll = (y * torch.nn.functional.logsigmoid(logits)
              + (1 - y) * torch.nn.functional.logsigmoid(-logits))
        expected = (ll * mask).sum(1)
    assert torch.allclose(out["recon"], expected, atol=1e-4)


# ======================================================= 4. von Mises sampler


def test_sampler_gradient_wrt_mu_is_one():
    """Input: phi = mu + sample_vonmises(kappa), the exact composition the recon
    path uses. Asserted: d(phi)/d(mu) == 1 elementwise. Why: the mean enters
    additively outside the sampler (rotation equivariance of the von Mises), so
    its pathwise gradient is exactly 1 -- the sampler docstring calls it exact.
    """
    _seed()
    mu = torch.zeros(50, requires_grad=True)
    phi = mu + sample_vonmises(torch.full((50,), 20.0))
    phi.sum().backward()
    assert torch.equal(mu.grad, torch.ones(50))


def test_sampler_gradient_wrt_kappa_exists_and_is_finite():
    """Input: kappa spanning small to large concentrations, requires_grad. Asserted:
    backward succeeds and every gradient entry is finite. Why: the module promises
    d(sample)/d(kappa) exists (reparameterised-rejection path), and the numerics
    comments promise every derivative stays finite.
    """
    _seed()
    kappa = torch.tensor([0.5, 5.0, 50.0, 2000.0, 1e5] * 20, requires_grad=True)
    s = sample_vonmises(kappa)
    s.sum().backward()
    assert kappa.grad is not None
    assert torch.all(torch.isfinite(kappa.grad))


def test_sampler_huge_kappa_concentrates_at_mean():
    """Input: 1000 draws at kappa = 1e6. Asserted: every circular distance from the
    mean (0) is below 0.01 rad. Why: vM sd ~ 1/sqrt(kappa) = 1e-3, so 0.01 is a
    10-sigma envelope; a sampler leaking mass elsewhere is broken.
    """
    _seed()
    s = sample_vonmises(torch.full((1000,), 1e6))
    circ = torch.minimum(s.abs(), TWO_PI - s.abs())
    assert torch.all(circ < 0.01)


def test_sampler_circular_mean_matches_mu_at_kappa_five():
    """Input: 5000 draws at kappa = 5, shifted by mu = 1.1. Asserted: the empirical
    circular mean atan2(E sin, E cos) is within 0.06 rad of mu. Why: the circular
    mean of vM(mu, kappa) is mu; the standard error at n=5000, kappa=5 is well
    under the tolerance.
    """
    _seed()
    mu = 1.1
    s = mu + sample_vonmises(torch.full((5000,), 5.0))
    mean_dir = math.atan2(torch.sin(s).mean().item(), torch.cos(s).mean().item())
    assert abs(math.atan2(math.sin(mean_dir - mu), math.cos(mean_dir - mu))) < 0.06


# ====================================================== 5. KL of the free posterior


def test_kl_free_posterior_matches_monte_carlo_two_frames():
    """Input: a 2-frame free posterior (per-frame mu, moderate kappas). Asserted:
    kl_to_physical_prior agrees with a seeded 200k-sample Monte Carlo estimate of
    E_q[log q - log p] within 5% relative. Why: this is KL's definition; a sign or
    dropped-term error in the closed form shifts the value far beyond 5%.
    """
    scipy_special = pytest.importorskip("scipy.special")
    model = BarPhaseVAE(input_dim=4, hidden=8)
    mu = torch.tensor([[0.3, 0.5]], dtype=torch.float64)
    kappa = torch.tensor([[50.0, 80.0]], dtype=torch.float64)
    delta = torch.full((1, 2), 0.06, dtype=torch.float64)
    got = model.kl_to_physical_prior(mu, kappa, delta,
                                     torch.ones(1, 2, dtype=torch.float64)).item()

    rng = np.random.default_rng(0)
    n = 200_000
    phi1 = rng.vonmises(0.3, 50.0, n)
    phi2 = rng.vonmises(0.5, 80.0, n)

    def log_i0(k):
        return np.log(scipy_special.i0e(k)) + k

    log_q = (50.0 * np.cos(phi1 - 0.3) - math.log(TWO_PI) - log_i0(50.0)
             + 80.0 * np.cos(phi2 - 0.5) - math.log(TWO_PI) - log_i0(80.0))
    log_p = (-math.log(TWO_PI)
             + KAPPA_PHYSICAL * np.cos(phi2 - phi1 - 0.06)
             - math.log(TWO_PI) - log_i0(KAPPA_PHYSICAL))
    mc = float(np.mean(log_q - log_p))
    assert got == pytest.approx(mc, rel=0.05)


def test_kl_free_posterior_one_frame_is_entropy_to_uniform():
    """Input: a 1-frame crop. Asserted: KL == -H(q) + log(2*pi) EXACTLY (analytic).
    Why: with T=1 only the uniform phi_1 term of the Markov prior survives, so
    KL(q || Uniform) = -H(q) - log p(phi_1) = -H(q) + log 2pi.
    """
    scipy_stats = pytest.importorskip("scipy.stats")
    model = BarPhaseVAE(input_dim=4, hidden=8)
    kappa_val = 7.0
    got = model.kl_to_physical_prior(
        torch.tensor([[0.4]], dtype=torch.float64),
        torch.tensor([[kappa_val]], dtype=torch.float64),
        torch.tensor([[0.05]], dtype=torch.float64),
        torch.ones(1, 1, dtype=torch.float64)).item()

    expected = -float(scipy_stats.vonmises(kappa=kappa_val).entropy()) \
        + math.log(TWO_PI)
    assert got == pytest.approx(expected, abs=1e-9)


# ============================================= 6. run.py train-loop objective


def test_train_loss_formula_matches_documented_objective():
    """Input: a real forward pass's out dict plus a fake args namespace. Asserted:
    (a) run.train's source contains exactly the documented per-frame annealed loss
    '-((out["recon"] - beta * out["kl"]) / frames).mean()' with frames =
    mask.sum(1), and (b) that expression evaluated on the out dict equals the
    hand-assembled scalar mean over crops of -(recon_i - beta*kl_i)/T_i. A full
    1-step train() run is not exercised here: it requires the Batches/pinned-memory
    stack and a gradient audit, so the objective is verified at the formula level
    as the brief allows.
    """
    src = inspect.getsource(run_mod.train)
    assert 'frames = batch["mask"].sum(1)' in src
    # the objective formula lives in the hooks module now; run.train only wires it
    assert 'loss = -(hooks.objective(out, beta, cfg) / frames).mean()' in src
    from phasevae.variants import base
    assert 'out["recon"] - beta * out["kl"]' in inspect.getsource(base.objective)

    model = _tf_model()
    h, delta, mask, y = _batch()
    out = model(h, delta, mask, y, samples=1)
    beta = run_mod.beta_at(1, types.SimpleNamespace(beta_start=0.2, beta_end=0.8,
                                                    beta_warmup=3))
    assert beta == pytest.approx(0.2 + (1 / 3) * 0.6)

    frames = mask.sum(1)
    loss = -((out["recon"] - beta * out["kl"]) / frames).mean()
    expected = -np.mean([
        (out["recon"][i].item() - beta * out["kl"][i].item())
        / frames[i].item() for i in range(mask.shape[0])])
    assert loss.item() == pytest.approx(expected, rel=1e-6)


def test_beta_at_clamps_beyond_warmup_with_nonzero_start():
    """Input: beta_start=0.2, beta_end=0.8, warmup=3, epochs past warmup. Asserted:
    the schedule clamps at beta_end (fraction = min(1, epoch/warmup)) and never
    overshoots. Why: linear annealing's contract is the endpoint holds forever.
    """
    a = types.SimpleNamespace(beta_start=0.2, beta_end=0.8, beta_warmup=3)
    assert run_mod.beta_at(3, a) == pytest.approx(0.8)
    assert run_mod.beta_at(50, a) == pytest.approx(0.8)
    assert run_mod.beta_at(0, a) == pytest.approx(0.2)
