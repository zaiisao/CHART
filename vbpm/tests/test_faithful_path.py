"""Tests for the vbpm paths not covered by test_vbpm.py."""
from __future__ import annotations

import inspect
import math
import types

import numpy as np
import pytest
import torch

import vbpm.model as model_mod
from vbpm import run as run_mod
from vbpm.model import (VBPM, EmissionTransformer, Encoder,
                            MAX_KAPPA, TWO_PI)
from vbpm.vonmises import sample_vonmises


def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


# ================================================== 1. Encoder free branch


def test_encoder_shapes_and_global_response():
    """Shapes, kappa's range, and that perturbing ONE frame moves the whole trajectory."""
    _seed()
    enc = Encoder(input_dim=4)
    h = torch.randn(1, 200, 4)
    mask = torch.ones(1, 200)
    post = enc(h, mask)
    mu, kappa, anchor = post["phase"]["mu"], post["phase"]["kappa"], post
    assert mu.shape == (1, 200) and kappa.shape == (1, 200)
    assert torch.all(kappa > 0) and torch.all(kappa < MAX_KAPPA)
    assert anchor["tempo"]["log_prior"].shape == (1,) and anchor["tempo"]["entropy"].shape == (1,)

    h2 = h.clone()
    h2[0, 7] += 5.0
    post2 = enc(h2, mask)
    mu2, anchor2 = post2["phase"]["mu"], post2
    assert not torch.allclose(mu, mu2, atol=1e-6), \
        "the trajectory did not respond to the input at all"
    assert not torch.allclose(anchor["tempo"]["log_prior"], anchor2["tempo"]["log_prior"], atol=1e-9), \
        "the tempo prior did not respond to the input at all"


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
    return VBPM(input_dim=4, d_model=8, emission="transformer",
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
    out1 = model(h, mask, y, samples=1)
    out2 = model(h, mask, y, samples=1)
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
    torch.manual_seed(7)
    out1 = model(h, mask, y)
    torch.manual_seed(7)
    out2 = model(h, mask, y)
    assert torch.equal(out1["recon"], out2["recon"])


def test_forward_samples_k_averages_k_evaluations(monkeypatch):
    """Input: samples=3 with the sampler monkeypatched to return a counted,
        per-call constant offset (0, 0.5, 1.0 rad). Asserted: the sampler is called
        exactly 3 times and recon equals the arithmetic mean of the three hand-computed
        masked Bernoulli log-likelihoods at mu + offset. Why: 'samples: Monte Carlo
        samples for the reconstruction term' -- an average, not a sum.
    """
    offsets = [0.0, 0.5, 1.0, 0.25, 0.75, 0.125]
    calls = {"n": 0}

    def fake_sampler(kappa):
        off = offsets[calls["n"]]
        calls["n"] += 1
        return torch.full_like(kappa, off)

    monkeypatch.setattr(model_mod, "sample_vonmises", fake_sampler)
    model = _tf_model()
    h, delta, mask, y = _batch()
    out = model(h, mask, y, samples=3)
    assert calls["n"] == 3

    with torch.no_grad():
        terms = []
        for i in range(3):
            phi = out["mu"] + offsets[i]
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                model.emission_logits(phi, mask), y, reduction="none")
            terms.append(-(bce * mask).sum(1))
        expected = torch.stack(terms).mean(0)
    assert torch.allclose(out["recon"], expected, atol=1e-4)


def test_forward_transformer_pos_weight_one_is_plain_bce(monkeypatch):
    """Sampler pinned to the mean: recon equals the masked BCE at phi = mu, by hand."""
    monkeypatch.setattr(model_mod, "sample_vonmises",
                        lambda k: torch.zeros_like(k))
    model = _tf_model()
    h, delta, mask, y = _batch(B=1, T=8)
    out = model(h, mask, y, samples=1)

    with torch.no_grad():
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            model.emission_logits(out["mu"], mask), y, reduction="none")
        expected = -(bce * mask).sum(1)
    assert torch.allclose(out["recon"], expected, atol=1e-4)

    # and the emission is on the training path
    model.zero_grad()
    model(h, mask, y, samples=1)["recon"].sum().backward()
    assert model.emission_net is not None
    grads = [p.grad for p in model.emission_net.parameters() if p.grad is not None]
    assert any(float(g.abs().sum()) > 0.0 for g in grads), \
        "the emission receives no reconstruction gradient"


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


def test_kl_free_posterior_matches_monte_carlo_three_frames():
    """kl_jitter agrees with a seeded 400k-sample Monte Carlo estimate of
    E_q[log q - log p] with p = vM(mu, kappa_physical) per frame -- KL's definition,
    checked without ever forming the closed-form resultant terms.
    """
    scipy_special = pytest.importorskip("scipy.special")
    model = VBPM(input_dim=4, d_model=8, kappa_physical=40.0)
    mus, kappas = (0.30, 0.36, 0.43), (50.0, 80.0, 65.0)
    got = model.kl_jitter(torch.tensor([list(mus)], dtype=torch.float64),
                          torch.tensor([list(kappas)], dtype=torch.float64),
                          torch.ones(1, 3, dtype=torch.float64)).item()

    rng = np.random.default_rng(0)
    n = 400_000
    phi = [rng.vonmises(m, k, n) for m, k in zip(mus, kappas)]

    def log_i0(k):
        return np.log(scipy_special.i0e(k)) + k

    log_q = sum(k * np.cos(p - m) - math.log(TWO_PI) - log_i0(k)
                for p, m, k in zip(phi, mus, kappas))
    log_p = sum(40.0 * np.cos(p - m) - math.log(TWO_PI) - log_i0(40.0)
                for p, m in zip(phi, mus))
    mc = float(np.mean(log_q - log_p))
    assert got == pytest.approx(mc, rel=0.02)


def test_kl_free_posterior_uniform_prior_is_entropy_to_uniform():
    """With kappa_physical = 0 the jitter prior is Uniform, so the KL must equal
    -H(q_t) + log 2pi summed over frames, exactly.
    """
    scipy_stats = pytest.importorskip("scipy.stats")
    model = VBPM(input_dim=4, d_model=8)
    model.kappa_physical = 0.0
    kappas = (7.0, 11.0)
    got = model.kl_jitter(
        torch.tensor([[0.4, 0.9]], dtype=torch.float64),
        torch.tensor([list(kappas)], dtype=torch.float64),
        torch.ones(1, 2, dtype=torch.float64)).item()

    expected = -sum(float(scipy_stats.vonmises(kappa=k).entropy()) for k in kappas) \
        + 2 * math.log(TWO_PI)
    assert got == pytest.approx(expected, abs=1e-9)


# ============================================= 6. run.py train-loop objective


def test_train_loss_formula_matches_documented_objective():
    """Input: a real forward pass's out dict plus a fake args namespace. Asserted:
        (a) run.train's source contains exactly the documented per-frame annealed loss
        '-((out["recon"] - beta * out["kl"]) / frames).mean()' with frames =
        mask.sum(1), and (b) that expression evaluated on the out dict equals the
        hand-assembled scalar mean over crops of -(recon_i - beta*kl_i)/T_i. A full
    """
    src = inspect.getsource(run_mod.train)
    assert "frames = mask.sum(1)" in src
    # the objective formula lives in the hooks module now; run.train only wires it
    assert 'loss = -(hooks.objective(out, beta, cfg) / frames).mean()' in src
    from vbpm.variants import base
    assert 'out["recon"] - beta * out["kl"]' in inspect.getsource(base.objective)

    model = _tf_model()
    h, delta, mask, y = _batch()
    out = model(h, mask, y, samples=1)
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
