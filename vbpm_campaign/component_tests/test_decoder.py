"""Characterization / oracle tests for the DECODER (emission p_theta(b | z, h)).

Component under test: faithful.model.BarPointerVAE.decode  (and z_features that
builds z_t from phi/log_tempo/meter). Per docs/ELBO_for_DBN.md 5.4 the decoder is
Bernoulli: b_hat = sigma(NN_theta(z_t, h_{1:T})); decode returns the pre-sigmoid
LOGIT of shape [B]. latent_only=True is a documented deviation that drops h.

Every check is self-checking against an ORACLE (analytic invariant, finite-difference
gradient, or exact-equality invariant) -- NOT against the code's own output.
"""
import sys
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM")

import math
import torch

from faithful.model import BarPointerVAE
from faithful.distributions import TWO_PI

torch.manual_seed(0)
DT = torch.float64  # double precision so finite-difference oracles are tight

results = []  # (property, oracle, measured, PASS/FAIL)


def record(prop, oracle, measured, ok):
    results.append((prop, str(oracle), str(measured), "PASS" if ok else "FAIL"))
    print("[%s] %s\n      oracle   = %s\n      measured = %s" % ("PASS" if ok else "FAIL", prop, oracle, measured))


def make_model(latent_only=False, h_dim=12, hidden=16, K=4):
    m = BarPointerVAE(h_dim=h_dim, hidden=hidden, num_meters=K, latent_only=latent_only)
    m = m.to(DT)
    m.eval()
    return m


def rand_inputs(m, B, seed=1):
    g = torch.Generator().manual_seed(seed)
    phi = torch.rand(B, generator=g, dtype=DT) * TWO_PI
    log_tempo = torch.randn(B, generator=g, dtype=DT)
    meter = torch.softmax(torch.randn(B, m.K, generator=g, dtype=DT), dim=-1)
    ctx = torch.randn(B, m.hidden, generator=g, dtype=DT)
    return phi, log_tempo, meter, ctx


# P1: output shape [B] and finiteness (incl. extreme / edge z values)
def test_shape_finite():
    m = make_model()
    for B in (1, 5, 8):
        phi, log_tempo, meter, ctx = rand_inputs(m, B, seed=B)
        z = m.z_features(meter, phi, log_tempo)
        out = m.decode(z, ctx)
        shape_ok = tuple(out.shape) == (B,)
        finite_ok = bool(torch.isfinite(out).all())
        record("P1 output shape/finite (B=%d)" % B,
               "shape==(%d,) & all finite" % B,
               "shape=%s, finite=%s" % (tuple(out.shape), finite_ok),
               shape_ok and finite_ok)

    B = 4
    phi = torch.tensor([0.0, math.pi, TWO_PI - 1e-9, 1e-9], dtype=DT)
    log_tempo = torch.tensor([50.0, -50.0, 0.0, 1e3], dtype=DT)
    meter = torch.tensor([[1., 0, 0, 0], [0, 0, 0, 1.], [.25, .25, .25, .25], [1e9, 0, 0, 0]], dtype=DT)
    ctx = torch.zeros(B, m.hidden, dtype=DT)
    z = m.z_features(meter, phi, log_tempo)
    out = m.decode(z, ctx)
    record("P1b extreme-z finiteness",
           "all logits finite",
           "finite=%s, vals=%s" % (bool(torch.isfinite(out).all()), out.tolist()),
           bool(torch.isfinite(out).all()))


# P2: determinism -- same (z, h) -> same logit
def test_determinism():
    m = make_model()
    phi, log_tempo, meter, ctx = rand_inputs(m, 6, seed=7)
    z = m.z_features(meter, phi, log_tempo)
    o1 = m.decode(z.clone(), ctx.clone())
    o2 = m.decode(z.clone(), ctx.clone())
    diff = (o1 - o2).abs().max().item()
    record("P2 determinism (identical inputs -> identical logit)",
           "max|o1-o2| == 0", "%.3e" % diff, diff == 0.0)


# P3: gradient flows to phi, log_tempo, meter -- autograd vs finite diff
def test_grad_flow():
    m = make_model()
    B = 4
    phi0, log_tempo0, meter0, ctx = rand_inputs(m, B, seed=3)

    phi = phi0.clone().requires_grad_(True)
    log_tempo = log_tempo0.clone().requires_grad_(True)
    meter = meter0.clone().requires_grad_(True)
    z = m.z_features(meter, phi, log_tempo)
    out = m.decode(z, ctx)
    out.sum().backward()  # outputs independent across batch: grad[i]=d out[i]/d param[i]

    g_phi = phi.grad.clone()
    g_lt = log_tempo.grad.clone()
    g_meter = meter.grad.clone()

    record("P3 grad reaches phi (nonzero)",
           "max|d logit/d phi| > 0", "%.3e" % g_phi.abs().max().item(), g_phi.abs().max().item() > 0)
    record("P3 grad reaches log_tempo (nonzero)",
           "max|d logit/d log_tempo| > 0", "%.3e" % g_lt.abs().max().item(), g_lt.abs().max().item() > 0)
    record("P3 grad reaches meter (nonzero)",
           "max|d logit/d meter| > 0", "%.3e" % g_meter.abs().max().item(), g_meter.abs().max().item() > 0)

    eps = 1e-6

    def out_i(phi_v, lt_v, i):
        z_ = m.z_features(meter0, phi_v, lt_v)
        return m.decode(z_, ctx)[i]

    fd_phi = torch.zeros(B, dtype=DT)
    fd_lt = torch.zeros(B, dtype=DT)
    for i in range(B):
        pp = phi0.clone(); pp[i] += eps
        pm = phi0.clone(); pm[i] -= eps
        fd_phi[i] = (out_i(pp, log_tempo0, i) - out_i(pm, log_tempo0, i)) / (2 * eps)
        lp = log_tempo0.clone(); lp[i] += eps
        lm = log_tempo0.clone(); lm[i] -= eps
        fd_lt[i] = (out_i(phi0, lp, i) - out_i(phi0, lm, i)) / (2 * eps)

    err_phi = (g_phi - fd_phi).abs().max().item()
    err_lt = (g_lt - fd_lt).abs().max().item()
    record("P3 autograd(phi) == finite-diff",
           "max|autograd - FD| < 1e-5", "%.3e" % err_phi, err_phi < 1e-5)
    record("P3 autograd(log_tempo) == finite-diff",
           "max|autograd - FD| < 1e-5", "%.3e" % err_lt, err_lt < 1e-5)

    fd_m0 = torch.zeros(B, dtype=DT)
    for i in range(B):
        mp = meter0.clone(); mp[i, 0] += eps
        mm = meter0.clone(); mm[i, 0] -= eps
        zp = m.z_features(mp, phi0, log_tempo0); zm = m.z_features(mm, phi0, log_tempo0)
        fd_m0[i] = (m.decode(zp, ctx)[i] - m.decode(zm, ctx)[i]) / (2 * eps)
    err_m0 = (g_meter[:, 0] - fd_m0).abs().max().item()
    record("P3 autograd(meter[:,0]) == finite-diff",
           "max|autograd - FD| < 1e-5", "%.3e" % err_m0, err_m0 < 1e-5)


# P4: latent_only truly drops h; latent_only=False truly reads h
def test_latent_only():
    m_lat = make_model(latent_only=True)
    dec_in = m_lat.decoder[0].in_features
    record("P4a latent_only decoder in_features == z_feat_dim (h absent)",
           "%d" % m_lat.z_feat_dim, "%d" % dec_in, dec_in == m_lat.z_feat_dim)

    phi, log_tempo, meter, ctx = rand_inputs(m_lat, 5, seed=11)
    z = m_lat.z_features(meter, phi, log_tempo)
    ctx2 = ctx + 100.0 * torch.randn_like(ctx)
    o1 = m_lat.decode(z, ctx)
    o2 = m_lat.decode(z, ctx2)
    diff = (o1 - o2).abs().max().item()
    record("P4a latent_only=True: logit invariant to h",
           "max|o(ctx) - o(ctx')| == 0", "%.3e" % diff, diff == 0.0)

    m_full = make_model(latent_only=False)
    dec_in_f = m_full.decoder[0].in_features
    record("P4b full decoder in_features == z_feat_dim + hidden (h present)",
           "%d" % (m_full.z_feat_dim + m_full.hidden), "%d" % dec_in_f,
           dec_in_f == m_full.z_feat_dim + m_full.hidden)

    phi, log_tempo, meter, ctx = rand_inputs(m_full, 5, seed=12)
    z = m_full.z_features(meter, phi, log_tempo)
    ctx2 = ctx + torch.randn_like(ctx)
    o1 = m_full.decode(z, ctx)
    o2 = m_full.decode(z, ctx2)
    diff = (o1 - o2).abs().max().item()
    record("P4b latent_only=False: logit responds to h",
           "max|o(ctx) - o(ctx')| > 1e-6", "%.3e" % diff, diff > 1e-6)

    ctxg = ctx.clone().requires_grad_(True)
    phiL, ltL, mtL, _ = rand_inputs(m_lat, 5, seed=13)
    zL = m_lat.z_features(mtL, phiL, ltL)
    m_lat.decode(zL, ctxg).sum().backward()
    gnorm_lat = 0.0 if ctxg.grad is None else ctxg.grad.abs().max().item()
    record("P4a latent_only=True: d logit/d h == 0",
           "grad wrt h == 0 (or None)", "%.3e" % gnorm_lat, gnorm_lat == 0.0)

    ctxg2 = ctx.clone().requires_grad_(True)
    zF = m_full.z_features(meter, phi, log_tempo)
    m_full.decode(zF, ctxg2).sum().backward()
    gnorm_full = ctxg2.grad.abs().max().item()
    record("P4b latent_only=False: d logit/d h != 0",
           "grad wrt h > 0", "%.3e" % gnorm_full, gnorm_full > 0.0)


# P5: phase response -- 2pi periodicity invariant + characterization
def test_phase_response():
    m = make_model()
    B = 1
    log_tempo = torch.zeros(B, dtype=DT)
    meter = torch.tensor([[1., 0, 0, 0]], dtype=DT)
    ctx = torch.zeros(B, m.hidden, dtype=DT)

    phi_a = torch.tensor([0.37], dtype=DT)
    phi_b = phi_a + TWO_PI
    za = m.z_features(meter, phi_a, log_tempo)
    zb = m.z_features(meter, phi_b, log_tempo)
    oa = m.decode(za, ctx); ob = m.decode(zb, ctx)
    per_err = (oa - ob).abs().max().item()
    record("P5 phase 2pi-periodicity of decoder",
           "|logit(phi) - logit(phi+2pi)| ~ 0 (<1e-9)", "%.3e" % per_err, per_err < 1e-9)

    ph0 = torch.tensor([1e-10], dtype=DT)
    phw = torch.tensor([TWO_PI - 1e-10], dtype=DT)
    o0 = m.decode(m.z_features(meter, ph0, log_tempo), ctx)
    ow = m.decode(m.z_features(meter, phw, log_tempo), ctx)
    wrap_err = (o0 - ow).abs().item()
    record("P5 phase wrap near 0/2pi continuity",
           "|logit(1e-10) - logit(2pi-1e-10)| < 1e-6", "%.3e" % wrap_err, wrap_err < 1e-6)

    phis = torch.linspace(0, TWO_PI, 25, dtype=DT)
    logits = torch.stack([m.decode(m.z_features(meter, p.view(1), log_tempo), ctx)[0] for p in phis])
    rng = (logits.max() - logits.min()).item()
    argmax_phi = phis[logits.argmax()].item()
    print("[CHAR] P5 phase sweep: logit range=%.3f, argmax at phi=%.3f rad (%.2f of bar); min=%.3f max=%.3f (random-init decoder, shape not asserted)"
          % (rng, argmax_phi, argmax_phi / TWO_PI, logits.min(), logits.max()))
    record("P5 decoder is phase-sensitive (non-degenerate)",
           "logit range over phi > 0", "%.3e" % rng, rng > 0)


if __name__ == "__main__":
    test_shape_finite()
    test_determinism()
    test_grad_flow()
    test_latent_only()
    test_phase_response()

    print("\n================ SUMMARY ================")
    n_pass = sum(1 for r in results if r[3] == "PASS")
    for prop, oracle, measured, verdict in results:
        print("[%s] %s" % (verdict, prop))
    print("\n%d/%d properties PASS" % (n_pass, len(results)))
    if n_pass != len(results):
        print("SOME PROPERTIES FAILED")
