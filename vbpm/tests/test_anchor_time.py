"""Behavioural tests for the time-anchored variant, from its contracts and the math only."""

from __future__ import annotations

import pytest as _pytest
_pytest.skip("variant not ported to the alignment objective (see 2191ea9)",
             allow_module_level=True)
# ---------------------------------------------------------------------------------

import math

import pytest
import torch

from vbpm.config import load_config
from vbpm.model import VBPM
from vbpm.variants import anchor_time as at

TWO_PI = 2.0 * math.pi
STRIDE = 4


# ---------------------------------------------------------------------------- helpers

def constant_cum(batch=2, frames=200, tempo=0.0628):
    """A monotone cumulative rotation with cum[:, 0] = 0, as AnchorEncoder.heads returns."""
    inc = torch.full((batch, frames), tempo)
    return torch.cumsum(inc, dim=1) - inc[:, :1]


def model(input_dim=8, frames=200, **kw):
    torch.manual_seed(0)
    return at.AnchorTimeVAE(input_dim, stride=STRIDE, harmonics=4, d_model=32,
                            emission="triangle", **kw)


def batch(input_dim=8, batch=2, frames=200, valid=None):
    torch.manual_seed(1)
    h = torch.randn(batch, frames, input_dim)
    mask = torch.ones(batch, frames)
    if valid is not None:
        mask[:, valid:] = 0.0
    y = torch.zeros(batch, frames)
    y[:, ::50] = 1.0
    return h, mask, y


# ----------------------------------------------------- 1. the anchor is where it claims

@pytest.mark.parametrize("k", [0, 1, 7, 23])
def test_candidate_k_puts_phase_zero_at_frame_k_times_stride(k):
    """candidate_anchors' contract: mu^k = cum - c_k is exactly 0 at frame k * stride."""
    cum = constant_cum()
    c, _ok = at.candidate_anchors(cum, torch.ones_like(cum), STRIDE)
    mu_k = cum - c[:, k : k + 1]
    assert torch.allclose(mu_k[:, k * STRIDE], torch.zeros(cum.shape[0]), atol=1e-6)


def test_candidate_count_follows_the_stride():
    """C = ceil(T / stride): the grid is a fixed spacing, not a fixed count."""
    cum = constant_cum(frames=201)
    c, _ok = at.candidate_anchors(cum, torch.ones_like(cum), STRIDE)
    assert c.shape[1] == math.ceil(201 / STRIDE)


# ----------------------------------------------- 2. the KL is shared across candidates

@pytest.mark.parametrize("k", [1, 5, 17])
def test_trajectory_kl_is_identical_for_every_candidate(k):
    """The whole design rests on this: a constant shift leaves the KL untouched."""
    vae = VBPM(8, emission="triangle")
    cum = constant_cum(frames=120)
    kappa = torch.full_like(cum, 300.0)
    mask = torch.ones_like(cum)

    c, _ok = at.candidate_anchors(cum, mask, STRIDE)
    reference = vae.kl_jitter(cum, kappa, mask)
    shifted = vae.kl_jitter(cum - c[:, k : k + 1], kappa, mask)
    assert torch.allclose(reference, shifted, atol=1e-4)


# --------------------------------------------------- 3. no posterior mass on the padding

def test_q_puts_no_mass_on_candidates_past_the_valid_span():
    """Candidates anchored in the padded tail read a cum that does not exist."""
    valid = 96
    vae, (h, mask, y) = model(), batch(valid=valid)
    out = vae(h, mask, y)

    q = out["logq"].exp()
    past = ~out["ok"]
    assert past.any(), "test needs some invalid candidates to be meaningful"
    assert torch.allclose(q[past], torch.zeros_like(q[past]))
    assert torch.allclose(q.sum(-1), torch.ones(q.shape[0]), atol=1e-5)


def test_n_i_counts_the_valid_candidates_not_the_allocated_ones():
    """log C_i must use the per-item count; the corpus mixes 30 s excerpts with full songs."""
    valid = 96
    vae, (h, mask, y) = model(), batch(valid=valid)
    out = vae(h, mask, y)
    assert int(out["n_i"][0]) == math.ceil(valid / STRIDE)
    assert out["logq"].shape[1] > int(out["n_i"][0]), "allocation should exceed the valid count"


def test_fully_masked_window_costs_zero_rather_than_nan():
    """run.py's backstop contract. With no valid candidate every logit would be -inf."""
    vae = model()
    h, mask, y = batch()
    mask[0] = 0.0
    out = vae(h, mask, y)
    assert torch.isfinite(out["elbo"]).all()
    assert abs(float(out["elbo"][0])) < 1e-4


def test_anchor_kl_is_zero_when_uniform_and_log_c_when_concentrated():
    """The anchor term is KL(q(k) || Uniform(C_i)), i.e. the PRICE of concentrating."""
    vae, (h, mask, y) = model(), batch()
    out = vae(h, mask, y)
    kl_traj = vae.kl_jitter(*vae.encoder(h, mask)[:2], mask)
    anchor = out["kl"] - kl_traj
    n_i = out["n_i"]

    # at init q is NEAR uniform, so the price is small but NOT zero. It must not be exactly
    # zero: Hq = 1.000000 was the signature of a head that could not escape its own init.
    assert (anchor >= 0).all() and float(anchor.max()) < 0.5

    # a point mass costs exactly log C_i; check the arithmetic the objective relies on
    logq = torch.full_like(out["logq"], float("-inf"))
    logq[:, 0] = 0.0
    q = logq.exp()
    # guarded on q, not on ok: a VALID candidate whose logit underflows also has q = 0,
    # and 0 * -inf would nan the batch. This assertion is what caught that.
    neg_entropy = torch.where(q > 0, q * logq, torch.zeros_like(q)).sum(-1)
    price = torch.log(n_i.to(q.dtype)) + neg_entropy
    assert torch.allclose(price, torch.log(n_i.to(q.dtype)), atol=1e-5)


def test_posterior_is_not_born_dead():
    """THE regression test for the first smoke run's failure: Hq was 1.000000 at every epoch."""
    vae, (h, mask, y) = model(), batch()
    out = vae(h, mask, y)
    q = out["logq"].exp()
    C = q.shape[1]
    hq = (-(q * out["logq"].nan_to_num(0.0)).sum(-1) / math.log(C)).mean()

    assert float(hq) < 0.9999, f"posterior is uniform to 4dp (Hq={float(hq):.6f}): head is dead"
    assert float(q.max()) > 1.2 / C, "no candidate stands measurably above uniform"


# ------------------------------------------- 4. the closed-form harmonic identity holds

def test_first_harmonic_matches_the_brute_force_matched_filter():
    """candidate_features' stated identity, checked against the sum it claims to compute."""
    vae = at.AnchorTimeVAE(8, stride=STRIDE, harmonics=1, d_model=32, emission="triangle")
    h, mask, _y = batch()
    cum = constant_cum(frames=h.shape[1])
    c, _ok = at.candidate_anchors(cum, mask, STRIDE)

    feat = vae.candidate_features(h, cum, c, mask)
    assert feat.shape == (h.shape[0], c.shape[1], 2)

    a = torch.sigmoid(vae.downbeat_head(h).squeeze(-1)) * mask
    # normalised by the EVIDENCE MASS, not the frame count: dividing by T put every case at
    # ~1e-2 and left the softmax uniform to six decimals. See candidate_features.
    evidence_mass = a.sum(1)
    for k in (0, 3, 11):
        delta = cum - c[:, k : k + 1]
        expect_re = (a * torch.cos(delta)).sum(1) / evidence_mass
        expect_im = (a * torch.sin(delta)).sum(1) / evidence_mass
        assert torch.allclose(feat[:, k, 0], expect_re, atol=1e-5)
        assert torch.allclose(feat[:, k, 1], expect_im, atol=1e-5)


def test_matched_filter_evidence_is_sinusoidal_in_the_anchor():
    """The identity's corollary, and the reason the head reads M > 1 harmonics."""
    vae = at.AnchorTimeVAE(8, stride=1, harmonics=1, d_model=32, emission="triangle")
    h, mask, _y = batch(frames=400)
    cum = constant_cum(batch=h.shape[0], frames=400)
    c, _ok = at.candidate_anchors(cum, mask, 1)

    re = vae.candidate_features(h, cum, c, mask)[0, :, 0]
    # a pure sinusoid in c is spanned by {1, cos c, sin c}: least squares must fit exactly
    design = torch.stack([torch.ones_like(c[0]), torch.cos(c[0]), torch.sin(c[0])], dim=1)
    residual = re - design @ torch.linalg.lstsq(design, re.unsqueeze(1)).solution.squeeze(1)
    assert float(residual.abs().max()) < 1e-4


# ---------------------------------------------------------------- gradients and wiring

def test_gradient_reaches_the_evidence_head_and_the_k_head():
    """Both heads are on the loss path, or the mechanism cannot learn at all."""
    vae, (h, mask, y) = model(), batch()
    out = vae(h, mask, y)
    (-(out["elbo"]).mean()).backward()

    for name in ("downbeat_source", "k_head"):
        grads = [p.grad for p in getattr(vae, name).parameters() if p.grad is not None]
        assert grads, f"{name} received no gradient at all"
        assert max(float(g.abs().max()) for g in grads) > 0.0, f"{name} gradient is all zero"


def test_encoder_has_no_offset_head_left_to_go_dead():
    """The anchor is enumerated, so channels 0-1 of the base encoder must be gone."""
    vae = model()
    assert set(vae.encoder.out.keys()) == {"log_phi_kappa", "log_dotphi", "residual"}


def test_residual_is_bounded_to_half_a_bin_of_phase():
    """Unbounded, the residual could shift by whole bins and undo the enumeration."""
    vae = model()
    cum = constant_cum(frames=200)
    mask = torch.ones_like(cum)
    tempo = 0.0628
    for extreme in (-50.0, 50.0):
        residual = vae.residual(torch.full_like(cum, extreme), cum, mask)
        assert float(residual.abs().max()) <= tempo * STRIDE / 2 + 1e-6


def test_deployed_path_is_target_blind_and_selects_by_q():
    """infer_phase reads audio only, and its argmax must be over q rather than R."""
    vae = model()
    vae.eval()
    h, mask, _y = batch()
    mu = vae.infer_phase(h, mask)
    assert mu.shape == (h.shape[0], h.shape[1])
    assert torch.isfinite(mu).all()


def test_config_loads_and_names_this_variant():
    """The config must parse, and its variant-only keys must be name-checked, not ignored."""
    cfg, hooks = load_config("vbpm/configs/anchor_time.yaml")
    assert cfg.variant == "anchor_time"
    assert hooks is at
    assert cfg.anchor_stride_frames == STRIDE
    assert cfg.emission in ("triangle", "cosine")


def test_build_model_refuses_a_transformer_emission():
    """B x C sequences per step; the assert exists so the run refuses instead of thrashing."""
    cfg, _hooks = load_config("vbpm/configs/anchor_time.yaml",
                              ["emission=transformer"])
    with pytest.raises(AssertionError, match="elementwise emission"):
        at.build_model(cfg, 8)


def test_mixed_length_batch_has_finite_gradients():
    """The bug that voided the first full-scale run, and the reason it hid until then."""
    for lengths in ([200, 200], [200, 120], [200, 40], [200, 0]):
        vae = model()
        h, mask, y = batch(batch=len(lengths))
        mask = torch.zeros_like(mask)
        for i, n in enumerate(lengths):
            mask[i, :n] = 1.0
        out = vae(h, mask, y)
        assert torch.isfinite(out["elbo"]).all(), f"elbo non-finite at {lengths}"
        (-out["elbo"].mean()).backward()
        nans = [n for n, p in vae.named_parameters()
                if p.grad is not None and torch.isnan(p.grad).any()]
        assert not nans, f"nan gradients at lengths {lengths}: {nans[:4]}"


def test_standardisation_cannot_manufacture_a_one_hot_posterior():
    """The mirror failure: eps must sit INSIDE the sqrt."""
    vae, (h, mask, y) = model(), batch()
    out = vae(h, mask, y)
    q = out["logq"].exp()
    C = q.shape[1]
    hq = float((-(q * out["logq"].masked_fill(~out["ok"], 0.0)).sum(-1)
                / torch.log(out["n_i"].to(q.dtype))).mean())
    assert 0.5 < hq < 0.9999, f"Hq {hq:.6f} is degenerate at init (dead or one-hot)"
    assert float(q.max()) < 0.5, "a single candidate holds half the mass before any training"


def test_epoch_note_survives_a_fully_masked_item():
    """One backstop window must not corrupt the reported Hq."""
    vae, (h, mask, y) = model(), batch()
    mask[0] = 0.0
    note = at.epoch_note(vae, {"h": h, "mask": mask, "y": y})
    assert "nan" not in note.lower() and "inf" not in note.lower(), note
    assert "Hq" in note
