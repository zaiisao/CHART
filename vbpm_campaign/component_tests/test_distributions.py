"""Characterization / oracle tests for the DISTRIBUTIONS component.

Files under test (imported, NEVER modified):
  - faithful/distributions.py
  - models/distributions.py

Every check compares the real code's output against an INDEPENDENT oracle:
analytic closed forms, brute-force numerical integrals, scipy Bessel functions,
torch.distributions KLs, or a mathematical invariant (sum-to-1, A'(k) identity).
"""
import sys
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM")

import math
import numpy as np
import scipy.special as sp
import scipy.integrate as si
import torch

import faithful.distributions as F
import models.distributions as M

torch.manual_seed(0)
np.random.seed(0)
DT = torch.float64

RESULTS = []


def record(prop, oracle, measured, ok, extra=""):
    res = "PASS" if ok else "FAIL"
    RESULTS.append(dict(property=prop, oracle=str(oracle), measured=str(measured), result=res))
    print(f"[{res}] {prop}")
    print(f"        oracle   = {oracle}")
    print(f"        measured = {measured}")
    if extra:
        print(f"        {extra}")


def A_scipy(k):
    return sp.ive(1, k) / sp.ive(0, k)


TWO_PI = 2 * math.pi


def test_bessel_helpers():
    ks = torch.tensor([0.01, 0.2, 1.0, 5.0, 20.0, 200.0, 700.0], dtype=DT)
    o_logi0 = np.log(sp.ive(0, ks.numpy())) + ks.numpy()
    err_f = float(np.max(np.abs(F.log_i0(ks).numpy() - o_logi0)))
    err_m = float(np.max(np.abs(M._log_i0(ks).numpy() - o_logi0)))
    record("log_i0 faithful vs scipy log I0", "max|err|<1e-9", f"max|err|={err_f:.2e}", err_f < 1e-9)
    record("log_i0 models vs scipy log I0", "max|err|<1e-9", f"max|err|={err_m:.2e}", err_m < 1e-9)
    o_A = A_scipy(ks.numpy())
    ef = float(np.max(np.abs(F.A_kappa(ks).numpy() - o_A)))
    em = float(np.max(np.abs(M._A(ks).numpy() - o_A)))
    record("A_kappa faithful vs scipy I1e/I0e", "max|err|<1e-12", f"max|err|={ef:.2e}", ef < 1e-12)
    record("A models vs scipy I1e/I0e", "max|err|<1e-12", f"max|err|={em:.2e}", em < 1e-12)


def _sampler_stats(sampler, kappa, mu, N):
    k = torch.full((N,), kappa, dtype=DT)
    m = torch.full((N,), mu, dtype=DT)
    z = sampler(m, k)
    d = z - m
    return float(torch.cos(d).mean()), float(torch.sin(d).mean())


def test_vm_sampler_moments():
    N = 300_000
    for kappa in [0.01, 0.2, 1.0, 5.0, 20.0, 200.0]:
        A = float(A_scipy(kappa))
        mu = 0.7
        tol = 5.0 / math.sqrt(N) + 3e-3
        Ec_f, Es_f = _sampler_stats(F.sample_von_mises, kappa, mu, N)
        record(f"faithful sampler E[cos(z-mu)]->A(k) k={kappa}", f"A={A:.5f} (|d|<{tol:.4f})",
               f"E[cos]={Ec_f:.5f} E[sin]={Es_f:.5f}", abs(Ec_f - A) < tol and abs(Es_f) < tol)
        Ec_m, Es_m = _sampler_stats(M.von_mises_sample, kappa, mu, N)
        record(f"models sampler E[cos(z-mu)]->A(k) k={kappa}", f"A={A:.5f} (|d|<{tol:.4f})",
               f"E[cos]={Ec_m:.5f} E[sin]={Es_m:.5f}", abs(Ec_m - A) < tol and abs(Es_m) < tol)
        cv_f = 1 - Ec_f
        cv_o = 1 - A
        record(f"faithful circular-variance k={kappa}", f"1-A={cv_o:.5f} (|d|<{tol:.4f})",
               f"{cv_f:.5f}", abs(cv_f - cv_o) < tol)


def _vm_density(theta, mu, kappa):
    return np.exp(kappa * np.cos(theta - mu)) / (2 * np.pi * sp.iv(0, kappa))


def _vm_kl_brute(mu_q, kappa_q, mu_p, kappa_p):
    def integrand(t):
        q = _vm_density(t, mu_q, kappa_q)
        p = _vm_density(t, mu_p, kappa_p)
        return q * (np.log(q) - np.log(p))
    val, _ = si.quad(integrand, -np.pi, np.pi, limit=400)
    return val


def test_vm_kl():
    pairs = [
        (0.0, 1.0, 0.0, 0.5),
        (0.3, 5.0, -0.5, 2.0),
        (0.0, 20.0, 0.0, 15.0),
        (0.0, 50.0, 0.0, 1.0),
        (0.1, 3.0, 0.1, 3.0),
        (1.2, 8.0, -1.0, 8.0),
    ]
    for (mq, kq, mp, kp) in pairs:
        o = _vm_kl_brute(mq, kq, mp, kp)
        t = lambda x: torch.tensor(x, dtype=DT)
        klf = float(F.kl_von_mises(t(mq), t(kq), t(mp), t(kp)))
        klm = float(M.von_mises_kl(t(mq), t(kq), t(mp), t(kp)))
        tol = 1e-6 + 1e-4 * abs(o)
        record(f"faithful kl_von_mises {(mq,kq,mp,kp)}", f"brute={o:.6f}", f"{klf:.6f}", abs(klf - o) < tol)
        record(f"models von_mises_kl {(mq,kq,mp,kp)}", f"brute={o:.6f}", f"{klm:.6f}", abs(klm - o) < tol)


def test_lognormal_kl():
    cases = [
        (0.0, 1.0, 0.0, 1.0),
        (0.5, 0.3, -0.2, 1.5),
        (2.0, 0.1, 2.0, 2.0),
        (-1.0, 2.0, 1.0, 0.5),
    ]
    from scipy.stats import lognorm
    for (mq, sq, mp, sp_) in cases:
        # GENUINELY INDEPENDENT oracle: brute-force integral of the ACTUAL LogNormal
        # densities over x in (0, inf).  This tests the code's "reduces to Gaussian KL
        # in log-space" CLAIM, not merely re-derives its own formula.
        # scipy lognorm(s=sigma, scale=exp(mu)).
        def _integrand(x, mq=mq, sq=sq, mp=mp, sp_=sp_):
            q = lognorm.pdf(x, sq, scale=math.exp(mq))
            pp = lognorm.pdf(x, sp_, scale=math.exp(mp))
            if q <= 0.0:
                return 0.0
            return q * (math.log(q) - math.log(pp))
        o, _ = si.quad(_integrand, 0.0, np.inf, limit=400)
        t = lambda x: torch.tensor(x, dtype=DT)
        klf = float(F.kl_log_normal(t(mq), t(sq), t(mp), t(sp_)))
        klm = float(M.lognormal_kl(t(mq), t(sq), t(mp), t(sp_)))
        tol = 1e-6 + 1e-5 * abs(o)
        record(f"faithful kl_log_normal {(mq,sq,mp,sp_)}", f"gaussKL={o:.9f}", f"{klf:.9f}", abs(klf - o) < tol)
        record(f"models lognormal_kl {(mq,sq,mp,sp_)}", f"gaussKL={o:.9f}", f"{klm:.9f}", abs(klm - o) < tol)


def test_categorical_kl():
    logits_q = torch.tensor([[2.0, 0.5, -1.0, 0.0], [0.1, 0.1, 0.1, 3.0]], dtype=DT)
    logits_p = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 0.5, 0.2]], dtype=DT)
    q = torch.softmax(logits_q, -1)
    p = torch.softmax(logits_p, -1)
    o = torch.distributions.kl_divergence(
        torch.distributions.Categorical(probs=q),
        torch.distributions.Categorical(probs=p)).numpy()
    klf = F.kl_categorical(torch.log_softmax(logits_q, -1), torch.log_softmax(logits_p, -1)).numpy()
    klm = M.categorical_kl(logits_q, logits_p).numpy()
    ef = float(np.max(np.abs(klf - o)))
    em = float(np.max(np.abs(klm - o)))
    record("faithful kl_categorical vs torch.dist", f"{o}", f"{klf}", ef < 1e-12)
    record("models categorical_kl vs torch.dist", f"{o}", f"{klm}", em < 1e-12)
    z_f = float(F.kl_categorical(torch.log_softmax(logits_q, -1), torch.log_softmax(logits_q, -1)).abs().max())
    z_m = float(M.categorical_kl(logits_q, logits_q).abs().max())
    record("categorical KL(q||q)=0 (both impls)", "0", f"faithful={z_f:.2e} models={z_m:.2e}",
           z_f < 1e-12 and z_m < 1e-12)


def _Aprime(k):
    A = float(A_scipy(k))
    return 1.0 - A * A - A / k


def _grad_Ecos(sample_fn, kappa, N):
    k = torch.tensor(kappa, dtype=DT, requires_grad=True)
    mu = torch.zeros(N, dtype=DT)
    kk = k.expand(N)
    z = sample_fn(mu, kk)
    loss = torch.cos(z - mu).mean()
    (g,) = torch.autograd.grad(loss, k)
    return float(g)


def test_implicit_reparam_grad():
    N = 400_000
    for kappa in [0.5, 2.0, 5.0, 10.0]:
        o = _Aprime(kappa)
        h = 1e-4
        fd = float((A_scipy(kappa + h) - A_scipy(kappa - h)) / (2 * h))
        record(f"A'(k) identity (analytic vs FD) k={kappa}", f"analytic={o:.5f}", f"FD={fd:.5f}", abs(o - fd) < 1e-5)
        gf = _grad_Ecos(F.sample_von_mises, kappa, N)
        gm = _grad_Ecos(M.von_mises_sample, kappa, N)
        tol = 0.02 + 0.05 * abs(o)
        record(f"faithful d/dk E[cos] vs A'(k) k={kappa}", f"A'={o:.5f} (|d|<{tol:.4f})", f"{gf:.5f}", abs(gf - o) < tol)
        record(f"models d/dk E[cos] vs A'(k) k={kappa}", f"A'={o:.5f} (|d|<{tol:.4f})", f"{gm:.5f}", abs(gm - o) < tol)


def test_vm_normalization():
    grid = torch.linspace(-math.pi, math.pi, 400_001, dtype=DT)
    for kappa in [0.2, 1.0, 5.0, 20.0]:
        k = torch.full_like(grid, kappa)
        integ_f = float(torch.trapz(F.von_mises_pdf(grid, k), grid))
        record(f"faithful von_mises_pdf integrates to 1 k={kappa}", "1.0", f"{integ_f:.8f}", abs(integ_f - 1.0) < 1e-5)
        integ_m = float(torch.trapz(1.0 / M._inv_prob(grid, k), grid))
        record(f"models 1/_inv_prob integrates to 1 k={kappa}", "1.0", f"{integ_m:.8f}", abs(integ_m - 1.0) < 1e-5)


def test_vm_cdf():
    from scipy.stats import vonmises
    zs = torch.linspace(-math.pi, math.pi, 50, dtype=DT)
    for kappa in [0.5, 3.0, 15.0]:
        k = torch.full_like(zs, kappa)
        cdf = F.von_mises_cdf(zs, k)
        mono = bool((cdf[1:] - cdf[:-1] >= -1e-9).all())
        end = abs(float(cdf[0])) < 1e-3 and abs(float(cdf[-1]) - 1.0) < 1e-3
        o_mid = float(vonmises.cdf(0.0, kappa))
        m_mid = float(F.von_mises_cdf(torch.tensor(0.0, dtype=DT), torch.tensor(kappa, dtype=DT)))
        record(f"faithful CDF monotone+endpoints k={kappa}", "mono & F(-pi)=0 & F(pi)=1",
               f"mono={mono} F0={float(cdf[0]):.3e} F1={float(cdf[-1]):.5f}", mono and end)
        record(f"faithful CDF(0) vs scipy k={kappa}", f"scipy={o_mid:.5f}", f"{m_mid:.5f}", abs(m_mid - o_mid) < 2e-3)
    zt = torch.linspace(-3.0, 3.0, 13, dtype=DT)
    for kappa in [1.0, 8.0, 30.0]:
        k = torch.full_like(zt, kappa)
        if kappa < 10.5:
            cdf_m, _ = M._von_mises_cdf_series(zt, k)
        else:
            cdf_m = M._von_mises_cdf_normal(zt, k)
        o = vonmises.cdf(zt.numpy(), kappa)
        err = float(np.max(np.abs(cdf_m.numpy() - o)))
        tol = 1e-4 if kappa < 10.5 else 5e-3
        record(f"models CDF vs scipy k={kappa}", f"max|err|<{tol}", f"max|err|={err:.2e}", err < tol)


def test_cross_impl_agreement():
    t = lambda x: torch.tensor(x, dtype=DT)
    args = (0.3, 8.0, -0.4, 3.0)
    df = float(F.kl_von_mises(t(args[0]), t(args[1]), t(args[2]), t(args[3])))
    dm = float(M.von_mises_kl(t(args[0]), t(args[1]), t(args[2]), t(args[3])))
    record("cross-impl vM KL agree", f"faithful={df:.8f}", f"models={dm:.8f}", abs(df - dm) < 1e-9)
    big = (0.0, 2000.0, 0.0, 1.0)
    df2 = float(F.kl_von_mises(t(big[0]), t(big[1]), t(big[2]), t(big[3])))
    dm2 = float(M.von_mises_kl(t(big[0]), t(big[1]), t(big[2]), t(big[3])))
    record("cross-impl vM KL agree @k=2000 (BUG-2 fix)", f"faithful={df2:.4f}", f"models={dm2:.4f}",
           abs(df2 - dm2) / abs(df2) < 1e-9)
    lf = float(F.kl_log_normal(t(0.4), t(0.7), t(-0.3), t(1.1)))
    lm = float(M.lognormal_kl(t(0.4), t(0.7), t(-0.3), t(1.1)))
    record("cross-impl logN KL agree", f"faithful={lf:.9f}", f"models={lm:.9f}", abs(lf - lm) < 1e-12)
    lg_q = torch.tensor([1.0, -0.5, 2.0], dtype=DT)
    lg_p = torch.tensor([0.2, 0.3, -1.0], dtype=DT)
    cf = float(F.kl_categorical(torch.log_softmax(lg_q, -1), torch.log_softmax(lg_p, -1)))
    cm = float(M.categorical_kl(lg_q, lg_p))
    record("cross-impl categorical KL agree", f"faithful={cf:.9f}", f"models={cm:.9f}", abs(cf - cm) < 1e-12)


def test_edge_cases():
    t = lambda x: torch.tensor(x, dtype=DT)
    small = 1e-3
    A_small = float(A_scipy(small))
    record("A(k->0) ~ k/2", f"{small/2:.6f}", f"A={A_small:.6f}", abs(A_small - small / 2) < 1e-5)
    k1 = float(F.kl_von_mises(t(0.0), t(4.0), t(0.0), t(2.0)))
    k2 = float(F.kl_von_mises(t(2 * math.pi), t(4.0), t(0.0), t(2.0)))
    record("vM KL 2pi-periodic in mu_q", f"{k1:.9f}", f"{k2:.9f}", abs(k1 - k2) < 1e-9)
    z = F.sample_von_mises(t(0.0), t(5.0))
    zm = M.von_mises_sample(t(0.0), t(5.0))
    record("scalar (T=1) sampler finite", "finite", f"faithful={float(z):.4f} models={float(zm):.4f}",
           bool(torch.isfinite(z).all()) and bool(torch.isfinite(zm).all()))
    khuge = float(M.von_mises_kl(t(0.01), t(5000.0), t(0.0), t(10.0)))
    record("vM KL @ saturating k=5000 finite & >=0", ">=0 finite", f"{khuge:.3f}",
           math.isfinite(khuge) and khuge >= 0)
    self_vm = float(M.von_mises_kl(t(0.7), t(12.0), t(0.7), t(12.0)))
    self_ln = float(M.lognormal_kl(t(1.0), t(0.4), t(1.0), t(0.4)))
    record("vM & logN self-KL == 0", "0", f"vM={self_vm:.2e} logN={self_ln:.2e}",
           abs(self_vm) < 1e-9 and abs(self_ln) < 1e-12)


if __name__ == "__main__":
    print("=" * 78)
    print("DISTRIBUTIONS component - oracle tests")
    print("=" * 78)
    for fn in [test_bessel_helpers, test_vm_sampler_moments, test_vm_kl,
               test_lognormal_kl, test_categorical_kl, test_implicit_reparam_grad,
               test_vm_normalization, test_vm_cdf, test_cross_impl_agreement, test_edge_cases]:
        print("\n--- " + fn.__name__ + " ---")
        try:
            fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            record(fn.__name__ + " (EXCEPTION)", "no exception", repr(e), False)
    n_pass = sum(1 for r in RESULTS if r["result"] == "PASS")
    n_fail = len(RESULTS) - n_pass
    print("\n" + "=" * 78)
    print(f"SUMMARY: {n_pass} PASS / {n_fail} FAIL  (total {len(RESULTS)})")
    print("=" * 78)
    if n_fail:
        print("FAILURES:")
        for r in RESULTS:
            if r["result"] == "FAIL":
                print(f"  - {r['property']}: measured {r['measured']} vs oracle {r['oracle']}")
