"""Characterization / verification tests for the ELBO objective.

Component under test (imported, never modified):
  * /home/sogang/jaehoon/VBPM_reintegration/models/loss.py   -> compute_elbo_loss
  * /home/sogang/jaehoon/VBPM_reintegration/faithful/elbo.py  -> KL primitives + aggregation

Every check asserts the code against an INDEPENDENT oracle:
  * closed-form KL via scipy.special (Bessel) / numpy (Gaussian, softmax)
  * numpy BCE
  * mathematical invariants (KL>=0, KL(p||p)=0)
  * autograd finite behaviour (free-bits clamp zeroes prior gradient)
never against the code's own output.
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM")

import math
import numpy as np
import torch
from scipy.special import ive  # exponentially-scaled modified Bessel I_v

from models.loss import compute_elbo_loss
from models.distributions import categorical_kl, von_mises_kl, lognormal_kl
from faithful.distributions import (
    kl_categorical as f_kl_categorical,
    kl_von_mises as f_kl_von_mises,
    kl_log_normal as f_kl_log_normal,
)

torch.manual_seed(0)
np.random.seed(0)
DT = torch.float64  # double precision so tolerances reflect the MATH, not fp32 noise

RESULTS = []  # (property, oracle, measured, PASS/FAIL/ERROR)


def record(prop, oracle, measured, ok, tol=None):
    status = "PASS" if ok else "FAIL"
    RESULTS.append((prop, str(oracle), str(measured), status))
    tols = f"  (tol={tol})" if tol is not None else ""
    print(f"[{status}] {prop}\n        oracle   = {oracle}\n        measured = {measured}{tols}")


# ---------------------------------------------------------------------------
# Independent numpy / scipy oracles
# ---------------------------------------------------------------------------
def oracle_bce_mean(logits, targets):
    """Mean BCE-with-logits over all elements (numpy, stable)."""
    x = logits.astype(np.float64)
    y = targets.astype(np.float64)
    logsig = lambda z: -np.logaddexp(0.0, -z)
    per = -(y * logsig(x) + (1.0 - y) * logsig(-x))
    return per.mean()


def oracle_categorical_kl(lq, lp):
    """KL(softmax(lq)||softmax(lp)) per row, numpy."""
    def logsoftmax(a):
        a = a - a.max(-1, keepdims=True)
        return a - np.log(np.exp(a).sum(-1, keepdims=True))
    log_q, log_p = logsoftmax(lq), logsoftmax(lp)
    q = np.exp(log_q)
    return (q * (log_q - log_p)).sum(-1)


def oracle_vm_kl(muq, kq, mup, kp):
    """KL(vM(muq,kq)||vM(mup,kp)), scipy Bessel, overflow-safe via ive."""
    logI0 = lambda k: np.log(ive(0, k)) + k
    A = lambda k: ive(1, k) / ive(0, k)
    return logI0(kp) - logI0(kq) + A(kq) * (kq - kp * np.cos(muq - mup))


def oracle_gauss_kl(muq, sq, mup, sp):
    """KL(N(muq,sq^2)||N(mup,sp^2)) = lognormal KL in log-space."""
    return np.log(sp / sq) + (sq**2 + (muq - mup)**2) / (2.0 * sp**2) - 0.5


# ---------------------------------------------------------------------------
# Random posterior / prior parameter dicts of the right shape
# ---------------------------------------------------------------------------
def make_batch(B, T, K=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=DT)
    u = lambda *s: torch.rand(*s, generator=g, dtype=DT)

    beat_logits = r(B, T, 2)
    beat_targets = (u(B, T) > 0.5).to(DT)

    posterior = {
        "meter_logits":    r(B, T, K),
        "phase_mu":        u(B, T) * (2 * math.pi),
        "phase_log_kappa": r(B, T) * 0.5,
        "tempo_mu":        r(B, T) * 0.3,
        "tempo_log_sigma": r(B, T) * 0.3 - 0.5,
    }
    prior = {
        "meter_logits": r(B, T, K),
        "phase_mu":     u(B, T) * (2 * math.pi),
        "phase_kappa":  torch.nn.functional.softplus(r(B, T)) + 0.05,
        "tempo_mu":     r(B, T) * 0.3,
        "tempo_sigma":  torch.nn.functional.softplus(r(B, T)) + 0.05,
    }
    return beat_logits, beat_targets, posterior, prior


# ===========================================================================
# PROPERTY 1 - default weights: total == bce + (kl_m+kl_phi+kl_tempo); aux == 0
# ===========================================================================
def test_p1_default_is_strict_elbo():
    B, T, K = 4, 5, 3
    bl, bt, post, prior = make_batch(B, T, K, seed=1)
    total, comp = compute_elbo_loss(bl, bt, post, prior)  # ALL defaults

    lhs = float(total)
    rhs = float(comp["bce"] + comp["kl_meter"] + comp["kl_phase"] + comp["kl_tempo"])
    ok_a = abs(lhs - rhs) < 1e-9
    record("P1a total == bce+kl_m+kl_phi+kl_tempo (default weights)", rhs, lhs, ok_a, 1e-9)

    aux_keys = ["kl_taubar", "kl_barphase", "taubar_sup", "meter_sup",
                "phase_sup", "barphase_sup", "tempo_density"]
    aux_vals = {k: float(comp[k]) for k in aux_keys}
    ok_b = all(v == 0.0 for v in aux_vals.values())
    record("P1b all aux terms == 0 at default weights", 0.0, aux_vals, ok_b)

    o_bce = oracle_bce_mean(bl[:, :, 0].numpy(), bt.numpy())
    o_klm = oracle_categorical_kl(post["meter_logits"].numpy(), prior["meter_logits"].numpy()).mean()
    o_klp = oracle_vm_kl(post["phase_mu"].numpy(), post["phase_log_kappa"].exp().numpy(),
                         prior["phase_mu"].numpy(), prior["phase_kappa"].numpy()).mean()
    o_klt = oracle_gauss_kl(post["tempo_mu"].numpy(), post["tempo_log_sigma"].exp().numpy(),
                            prior["tempo_mu"].numpy(), prior["tempo_sigma"].numpy()).mean()
    o_total = o_bce + o_klm + o_klp + o_klt
    ok_c = abs(lhs - o_total) < 1e-6
    record("P1c total matches independent numpy/scipy oracle", o_total, lhs, ok_c, 1e-6)


# ===========================================================================
# PROPERTY 2 - each KL term >= 0 (incl. extreme-kappa / kappa->0 edge cases)
# ===========================================================================
def test_p2_kl_nonnegative():
    B, T, K = 4, 6, 4
    _, _, post, prior = make_batch(B, T, K, seed=2)
    klm = categorical_kl(post["meter_logits"], prior["meter_logits"])
    klp = von_mises_kl(post["phase_mu"], post["phase_log_kappa"].exp(),
                       prior["phase_mu"], prior["phase_kappa"])
    klt = lognormal_kl(post["tempo_mu"], post["tempo_log_sigma"].exp(),
                       prior["tempo_mu"], prior["tempo_sigma"])
    m = min(float(klm.min()), float(klp.min()), float(klt.min()))
    ok = m >= -1e-9
    record("P2a KL>=0 for meter/phase/tempo (random params)", ">= 0", m, ok, 1e-9)

    muq = torch.tensor([0.0, 2 * math.pi - 1e-6, math.pi, 0.3], dtype=DT)
    mup = torch.tensor([1e-6, 1e-6, 0.0, 0.3], dtype=DT)
    kq = torch.tensor([1e-6, 2000.0, 1e-6, 700.0], dtype=DT)
    kp = torch.tensor([2000.0, 1e-6, 1e-6, 700.0], dtype=DT)
    klp_ext = von_mises_kl(muq, kq, mup, kp)
    o_ext = oracle_vm_kl(muq.numpy(), kq.numpy(), mup.numpy(), kp.numpy())
    ok_ext_nn = float(klp_ext.min()) >= -1e-6
    ok_ext_or = np.allclose(klp_ext.numpy(), o_ext, atol=1e-4, rtol=1e-4)
    record("P2b vM KL>=0 at extreme/near-0/large kappa + wrapped phase",
           ">= 0", float(klp_ext.min()), ok_ext_nn, 1e-6)
    record("P2c vM KL matches scipy Bessel oracle at extreme kappa",
           list(np.round(o_ext, 5)), list(np.round(klp_ext.numpy(), 5)), ok_ext_or, 1e-4)

    lg = torch.randn(5, K, dtype=DT)
    z_cat = float(categorical_kl(lg, lg).abs().max())
    mu = torch.rand(5, dtype=DT) * 2 * math.pi
    ka = torch.rand(5, dtype=DT) * 5 + 0.1
    z_vm = float(von_mises_kl(mu, ka, mu, ka).abs().max())
    m2 = torch.randn(5, dtype=DT)
    s2 = torch.rand(5, dtype=DT) + 0.1
    z_ln = float(lognormal_kl(m2, s2, m2, s2).abs().max())
    ok_zero = max(z_cat, z_vm, z_ln) < 1e-9
    record("P2d KL(p||p)==0 (cat/vM/logN identity)", 0.0,
           {"cat": z_cat, "vM": z_vm, "logN": z_ln}, ok_zero, 1e-9)


# ===========================================================================
# PROPERTY 3 - reconstruction reduction over T  (CHARACTERIZATION)
# ===========================================================================
def test_p3_recon_reduction_over_T():
    B, T, K = 2, 5, 3
    bl, bt, post, prior = make_batch(B, T, K, seed=3)
    _, comp1 = compute_elbo_loss(bl, bt, post, prior)

    def tile(d):
        return {k: torch.cat([v, v], dim=1) for k, v in d.items()}
    bl2 = torch.cat([bl, bl], dim=1)
    bt2 = torch.cat([bt, bt], dim=1)
    _, comp2 = compute_elbo_loss(bl2, bt2, tile(post), tile(prior))

    r1, r2 = float(comp1["bce"]), float(comp2["bce"])
    o_mean = oracle_bce_mean(bl[:, :, 0].numpy(), bt.numpy())
    ok_mean = abs(r2 - r1) < 1e-9 and abs(r1 - o_mean) < 1e-9
    record("P3a recon is MEAN over frames: double-T recon == 1x-T recon",
           f"{o_mean} (unchanged)", {"T": r1, "2T": r2}, ok_mean, 1e-9)

    ratio = r2 / r1
    record("P3b [FINDING] models/loss.py uses reduction='mean', NOT sum_t",
           "ratio 2.0 if summed", f"ratio={ratio:.4f} (mean=>1.0)", abs(ratio - 1.0) < 1e-9)


# ===========================================================================
# PROPERTY 4 - free-bits prior starvation (SUSPECT-A)
# ===========================================================================
def test_p4_free_bits_prior_starvation():
    B, T, K = 1, 4, 3
    bl, bt, post, prior = make_batch(B, T, K, seed=4)
    post["tempo_mu"] = prior["tempo_mu"].clone() + 1e-3
    post["tempo_log_sigma"] = torch.log(prior["tempo_sigma"]).clone()

    with torch.no_grad():
        klt = lognormal_kl(post["tempo_mu"], post["tempo_log_sigma"].exp(),
                           prior["tempo_mu"], prior["tempo_sigma"]).mean(-1)
    kl_val = float(klt.mean())

    def prior_tempo_grad(fb):
        p = {k: (v.clone().requires_grad_(True) if k == "tempo_mu" else v.clone())
             for k, v in prior.items()}
        total, _ = compute_elbo_loss(bl, bt,
                                     {k: v.clone() for k, v in post.items()},
                                     p, free_bits_tempo=fb)
        total.backward()
        g = p["tempo_mu"].grad
        return 0.0 if g is None else float(g.abs().sum())

    fb_high = kl_val * 4 + 1.0
    fb_zero = 0.0
    g_clamped = prior_tempo_grad(fb_high)
    g_free = prior_tempo_grad(fb_zero)

    ok_clamp = g_clamped == 0.0
    ok_free = g_free > 1e-8
    record("P4a below free-bits floor => PRIOR tempo grad == 0 (starvation)",
           "0.0", g_clamped, ok_clamp)
    record("P4b at free_bits=0 (above floor) => PRIOR tempo grad != 0",
           "> 0", g_free, ok_free)
    record("P4  [FINDING] free-bits clamp zeroes prior-side gradient when KL<floor",
           f"kl={kl_val:.4g}, floor={fb_high:.4g}",
           {"grad_clamped": g_clamped, "grad_free": g_free}, ok_clamp and ok_free)


# ===========================================================================
# PROPERTY 5 - beta scales ONLY the KL block
# ===========================================================================
def test_p5_beta_scales_kl_only():
    B, T, K = 3, 5, 3
    bl, bt, post, prior = make_batch(B, T, K, seed=5)
    t1, c1 = compute_elbo_loss(bl, bt, post, prior, beta=1.0)
    t2, c2 = compute_elbo_loss(bl, bt, post, prior, beta=2.0)

    kl_block = float(c1["kl_meter"] + c1["kl_phase"] + c1["kl_tempo"])
    delta = float(t2) - float(t1)
    ok_delta = abs(delta - kl_block) < 1e-9
    record("P5a total(beta=2)-total(beta=1) == KL block", kl_block, delta, ok_delta, 1e-9)

    ok_bce = abs(float(c2["bce"]) - float(c1["bce"])) < 1e-12
    record("P5b bce is invariant to beta", float(c1["bce"]), float(c2["bce"]), ok_bce, 1e-12)


# ===========================================================================
# PROPERTY 6 - faithful KL == models KL; + aggregation residual (sum-T vs mean-T)
# ===========================================================================
def test_p6_faithful_vs_models_parity():
    B, T, K = 3, 4, 3
    _, _, post, prior = make_batch(B, T, K, seed=6)

    m_cat = categorical_kl(post["meter_logits"], prior["meter_logits"])
    f_cat = f_kl_categorical(torch.log_softmax(post["meter_logits"], -1),
                             torch.log_softmax(prior["meter_logits"], -1))
    ok_cat = torch.allclose(m_cat, f_cat, atol=1e-10)

    m_vm = von_mises_kl(post["phase_mu"], post["phase_log_kappa"].exp(),
                        prior["phase_mu"], prior["phase_kappa"])
    f_vm = f_kl_von_mises(post["phase_mu"], post["phase_log_kappa"].exp(),
                          prior["phase_mu"], prior["phase_kappa"])
    ok_vm = torch.allclose(m_vm, f_vm, atol=1e-9)

    m_ln = lognormal_kl(post["tempo_mu"], post["tempo_log_sigma"].exp(),
                        prior["tempo_mu"], prior["tempo_sigma"])
    f_ln = f_kl_log_normal(post["tempo_mu"], post["tempo_log_sigma"].exp(),
                           prior["tempo_mu"], prior["tempo_sigma"])
    ok_ln = torch.allclose(m_ln, f_ln, atol=1e-10)

    record("P6a categorical KL: models == faithful", 0.0,
           float((m_cat - f_cat).abs().max()), ok_cat, 1e-10)
    record("P6b von Mises KL: models == faithful", 0.0,
           float((m_vm - f_vm).abs().max()), ok_vm, 1e-9)
    record("P6c log-Normal KL: models == faithful", 0.0,
           float((m_ln - f_ln).abs().max()), ok_ln, 1e-10)

    bl, bt, post2, prior2 = make_batch(B, T, K, seed=6)
    m_total, mc = compute_elbo_loss(bl, bt, post2, prior2)
    recon_sumT = torch.nn.functional.binary_cross_entropy_with_logits(
        bl[:, :, 0], bt, reduction="none").sum(1)
    kl_sumT = (m_cat.sum(1) + m_vm.sum(1) + m_ln.sum(1))
    faithful_total = (recon_sumT + kl_sumT).mean()
    ok_resid = abs(float(faithful_total) - float(m_total) * T) < 1e-6
    record("P6d [FINDING] faithful = T x models (sum-over-T vs mean-over-T)",
           f"{float(m_total)*T:.6f}", f"{float(faithful_total):.6f}", ok_resid, 1e-6)


def main():
    tests = [
        test_p1_default_is_strict_elbo,
        test_p2_kl_nonnegative,
        test_p3_recon_reduction_over_T,
        test_p4_free_bits_prior_starvation,
        test_p5_beta_scales_kl_only,
        test_p6_faithful_vs_models_parity,
    ]
    for t in tests:
        print(f"\n===== {t.__name__} =====")
        try:
            t()
        except Exception as e:
            import traceback
            traceback.print_exc()
            RESULTS.append((t.__name__, "no exception", f"ERROR: {e}", "ERROR"))

    print("\n\n==================== SUMMARY ====================")
    n_pass = sum(1 for r in RESULTS if r[3] == "PASS")
    n_fail = sum(1 for r in RESULTS if r[3] == "FAIL")
    n_err = sum(1 for r in RESULTS if r[3] == "ERROR")
    for prop, oracle, measured, status in RESULTS:
        print(f"[{status}] {prop}")
    print(f"\n{n_pass} PASS / {n_fail} FAIL / {n_err} ERROR  (of {len(RESULTS)})")


if __name__ == "__main__":
    main()
