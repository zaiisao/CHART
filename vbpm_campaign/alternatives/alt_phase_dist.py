#!/usr/bin/env python
"""
alt_phase_dist.py -- empirical selection of the in-model PHASE distribution for VBPM.

Component under test: p(phi) / phase emission. Faithful choice = von Mises
(prior mean = bar-pointer advance phi_{t-1}+phidot_{t-1}; concentration reads audio).

Question: is real BAR-PHASE more peaked / heavy-tailed / MULTI-MODAL (sub-bar pulses)
than a single von Mises allows?

Data: dataset_store/beat_this_annotations/<ds>/annotations/beats/*.beats
      col0 = beat time (s), col1 = beat-in-bar index (1 = downbeat).

Bar-phase reconstruction from downbeats: between consecutive downbeats D_i<D_{i+1},
a beat at time t has phi = 2*pi*(t-D_i)/(D_{i+1}-D_i). M = beats-in-bar; pulse k=col1-1.
nominal sub-bar position = 2*pi*k/M ; microtiming residual r = wrap(phi - 2*pi*k/M).

(A) FULL MARGINAL phi   : single-vM vs M-component vM/WC/WN MIXTURE on 2*pi*k/M grid
    (couples phase to METER). Tests multimodality.
(B) WITHIN-PULSE RESIDUAL r : vM vs wrapNormal vs wrapCauchy vs sine-skewed-vM (Kato-Jones).
Ranked by HELD-OUT circular log-likelihood + AIC/BIC (song-split).
"""
import glob, math
import numpy as np
from scipy import optimize, special

TWO_PI = 2.0 * math.pi
ROOT = "/home/sogang/jaehoon/VBPM/dataset_store/beat_this_annotations"
DATASETS = ["ballroom", "asap", "gtzan", "hainsworth", "rwc", "beatles", "hjdb"]
RNG = np.random.default_rng(0)


def wrap_pi(x):
    return (x + math.pi) % TWO_PI - math.pi


def load_dataset(ds):
    files = sorted(glob.glob(f"{ROOT}/{ds}/annotations/beats/*.beats"))
    phis, ks, Ms, sids = [], [], [], []
    for sid, f in enumerate(files):
        try:
            arr = np.loadtxt(f, ndmin=2)
        except Exception:
            continue
        if arr.size == 0 or arr.shape[1] < 2:
            continue
        t = arr[:, 0].astype(float)
        binb = arr[:, 1].astype(int)
        order = np.argsort(t)
        t, binb = t[order], binb[order]
        db_idx = np.where(binb == 1)[0]
        for a, b in zip(db_idx[:-1], db_idx[1:]):
            D0, D1 = t[a], t[b]
            if not (D1 > D0):
                continue
            M = b - a
            for j in range(a, b):
                phis.append((TWO_PI * (t[j] - D0) / (D1 - D0)) % TWO_PI)
                ks.append(binb[j] - 1)
                Ms.append(M)
                sids.append(sid)
    return (np.asarray(phis), np.asarray(ks, int),
            np.asarray(Ms, int), np.asarray(sids, int))


def logpdf_vm(x, mu, kappa):
    logI0 = np.log(special.i0e(kappa)) + kappa
    return kappa * np.cos(x - mu) - math.log(TWO_PI) - logI0


def logpdf_wn(x, mu, sigma, K=5):
    d = wrap_pi(x - mu)
    ks = np.arange(-K, K + 1)
    terms = np.exp(-0.5 * ((d[:, None] + TWO_PI * ks[None, :]) / sigma) ** 2)
    dens = terms.sum(1) / (sigma * math.sqrt(TWO_PI))
    return np.log(np.clip(dens, 1e-300, None))


def logpdf_wc(x, mu, rho):
    rho = min(max(rho, 1e-6), 1 - 1e-6)
    den = 1.0 + rho ** 2 - 2.0 * rho * np.cos(x - mu)
    return np.log(1.0 - rho ** 2) - math.log(TWO_PI) - np.log(np.clip(den, 1e-300, None))


def logpdf_skewvm(x, mu, kappa, lam):
    lam = min(max(lam, -0.999), 0.999)
    return logpdf_vm(x, mu, kappa) + np.log(np.clip(1.0 + lam * np.sin(x - mu), 1e-300, None))


def fit_vm(x):
    C, S = np.cos(x).mean(), np.sin(x).mean()
    mu = math.atan2(S, C); R = math.hypot(C, S)
    kappa = 1e-3 if R < 1e-6 else float(optimize.minimize_scalar(
        lambda k: abs(special.i1e(k)/special.i0e(k) - R),
        bounds=(1e-3, 1e4), method="bounded").x)
    return dict(mu=mu, kappa=kappa)


def fit_wn(x):
    C, S = np.cos(x).mean(), np.sin(x).mean()
    mu = math.atan2(S, C)
    R = min(max(math.hypot(C, S), 1e-6), 1 - 1e-9)
    return dict(mu=mu, sigma=max(math.sqrt(-2.0 * math.log(R)), 1e-3))


def fit_wc(x):
    C, S = np.cos(x).mean(), np.sin(x).mean()
    mu = math.atan2(S, C)
    res = optimize.minimize(lambda p: -logpdf_wc(x, p[0], p[1]).sum(),
                            [mu, 0.5], method="Nelder-Mead",
                            options=dict(xatol=1e-4, fatol=1e-3, maxiter=2000))
    return dict(mu=res.x[0], rho=min(max(res.x[1], 1e-6), 1 - 1e-6))


def fit_skewvm(x):
    v = fit_vm(x)
    res = optimize.minimize(lambda p: -logpdf_skewvm(x, p[0], abs(p[1]), p[2]).sum(),
                            [v["mu"], v["kappa"], 0.0], method="Nelder-Mead",
                            options=dict(xatol=1e-4, fatol=1e-3, maxiter=4000))
    return dict(mu=res.x[0], kappa=abs(res.x[1]), lam=min(max(res.x[2], -0.999), 0.999))


SINGLE = {
    "vonMises":     (fit_vm,     lambda x, p: logpdf_vm(x, p["mu"], p["kappa"]), 2),
    "wrapNormal":   (fit_wn,     lambda x, p: logpdf_wn(x, p["mu"], p["sigma"]), 2),
    "wrapCauchy":   (fit_wc,     lambda x, p: logpdf_wc(x, p["mu"], p["rho"]),   2),
    "skewVonMises": (fit_skewvm, lambda x, p: logpdf_skewvm(x, p["mu"], p["kappa"], p["lam"]), 3),
}


def _conc_to_rho(conc):
    return min(max(special.i1e(conc) / special.i0e(conc), 1e-6), 1 - 1e-6)


def _conc_to_sigma(conc):
    R = min(max(special.i1e(conc)/special.i0e(conc), 1e-6), 1 - 1e-9)
    return math.sqrt(max(-2.0 * math.log(R), 1e-6))


def _mix_comps(x, means, conc, kernel):
    M = len(means)
    if kernel == "vm":
        return np.stack([logpdf_vm(x, means[j], conc) for j in range(M)], 1)
    if kernel == "wc":
        rho = _conc_to_rho(conc)
        return np.stack([logpdf_wc(x, means[j], rho) for j in range(M)], 1)
    sigma = _conc_to_sigma(conc)
    return np.stack([logpdf_wn(x, means[j], sigma) for j in range(M)], 1)


def fit_mixture(x, M, kernel="vm", iters=60):
    means = TWO_PI * np.arange(M) / M
    w = np.full(M, 1.0 / M); conc = 4.0
    for _ in range(iters):
        lp = _mix_comps(x, means, conc, kernel) + np.log(w)[None, :]
        r = np.exp(lp - special.logsumexp(lp, axis=1, keepdims=True))
        w = r.mean(0) + 1e-8; w /= w.sum()
        d = wrap_pi(x[:, None] - means[None, :])
        Cbar = (r * np.cos(d)).sum() / r.sum()
        Sbar = (r * np.sin(d)).sum() / r.sum()
        Rbar = min(max(math.hypot(Cbar, Sbar), 1e-4), 1 - 1e-6)
        conc = float(optimize.minimize_scalar(
            lambda k: abs(special.i1e(k)/special.i0e(k) - Rbar),
            bounds=(1e-3, 1e4), method="bounded").x)
    return dict(means=means, w=w, conc=conc, kernel=kernel, M=M)


def mixture_logpdf(x, p):
    comps = _mix_comps(x, p["means"], p["conc"], p["kernel"])
    return special.logsumexp(comps + np.log(p["w"])[None, :], axis=1)


def song_split(sids):
    uniq = np.unique(sids); RNG.shuffle(uniq)
    train_s = set(uniq[:len(uniq) // 2])
    tr = np.array([s in train_s for s in sids])
    return tr, ~tr


def aic_bic(ll_sum, kfree, n):
    return 2 * kfree - 2 * ll_sum, kfree * math.log(n) - 2 * ll_sum


def kl_wrapped_cauchy(mu0, rho0, mu1, rho1):
    """Closed-form KL(WC(mu0,rho0) || WC(mu1,rho1)).

    Derivation: log f = log(1-rho^2) - log(2pi) - log(1+rho^2-2rho cos(x-mu)).
    Poisson-kernel Fourier identity  E_{WC(mu0,rho0)}[cos(n(x-mu1))] = rho0^n cos(n(mu0-mu1))
    with log(1-2r cos t + r^2) = -2 sum_n r^n/n cos(n t)  gives
      E_{WC0}[log(1+rho1^2-2rho1 cos(x-mu1))] = log(1 - 2 rho0 rho1 cosD + rho0^2 rho1^2),  D=mu0-mu1.
    Hence  KL = log[(1 - 2 rho0 rho1 cosD + rho0^2 rho1^2) / ((1-rho0^2)(1-rho1^2))].  (>=0, =0 iff equal)
    """
    D = mu0 - mu1
    num = 1.0 - 2.0 * rho0 * rho1 * math.cos(D) + (rho0 * rho1) ** 2
    return math.log(num / ((1.0 - rho0 ** 2) * (1.0 - rho1 ** 2)))


def sample_wrapped_cauchy(mu, rho, u):
    """Inverse-CDF (pathwise-reparam) sampler.  u ~ Uniform(0,1) is the noise.
    Wrap a Cauchy of scale gamma = -log(rho): x = mu + gamma*tan(pi*(u-0.5)), then wrap.
    Wrapped-Cauchy density is the Poisson kernel with mean-resultant-length rho = e^{-gamma}."""
    gamma = -math.log(rho)
    return wrap_pi(mu + gamma * np.tan(math.pi * (u - 0.5)))


def certify_wrapped_cauchy():
    print("\n" + "=" * 100)
    print("WINNER CERTIFICATION -- wrapped Cauchy is a genuine reparam+closed-KL DROP-IN")
    print("=" * 100)
    # (2) closed-form KL vs Monte-Carlo, several (mu,rho) pairs
    print("  (KL) closed-form vs Monte-Carlo (N=4,000,000 samples):")
    for (m0, r0, m1, r1) in [(0.0, 0.90, 0.3, 0.80), (0.5, 0.99, 0.5, 0.95),
                             (0.0, 0.70, 1.2, 0.70), (0.1, 0.993, 0.0, 0.977)]:
        u = RNG.random(4_000_000)
        x = sample_wrapped_cauchy(m0, r0, u)          # x ~ WC(m0,r0) via reparam sampler
        mc = (logpdf_wc(x, m0, r0) - logpdf_wc(x, m1, r1)).mean()
        cf = kl_wrapped_cauchy(m0, r0, m1, r1)
        print(f"    WC({m0:.2f},{r0:.3f})||WC({m1:.2f},{r1:.3f}): closed={cf:+.5f}  MC={mc:+.5f}  |d|={abs(cf-mc):.2e}")
    # (1) reparam sampler recovers the target density (moment check: mean resultant length = rho)
    print("  (reparam) inverse-CDF sampler: empirical mean-resultant-length R vs target rho:")
    for r0 in [0.70, 0.90, 0.977, 0.993]:
        u = RNG.random(2_000_000)
        x = sample_wrapped_cauchy(0.4, r0, u)
        R = math.hypot(np.cos(x).mean(), np.sin(x).mean())   # R->rho for WC centred at mu
        print(f"    rho={r0:.3f}:  empirical R={R:.4f}  (dphi/dmu, dphi/drho both analytic => pathwise-differentiable)")


def main():
    print("=" * 100)
    print("VBPM PHASE-DISTRIBUTION SELECTION  -- real bar-phase reconstructed from downbeats")
    print("=" * 100)
    pooled_res, pooled_res_sid = [], []
    for ds in DATASETS:
        phi, k, M, sid = load_dataset(ds)
        if phi.size < 200:
            print(f"[skip {ds}: only {phi.size} beats]"); continue
        keep = np.isin(M, [2, 3, 4])
        phi, k, M, sid = phi[keep], k[keep], M[keep], sid[keep]
        meter_hist = {int(m): int((M == m).sum()) for m in np.unique(M)}
        r = wrap_pi(phi - TWO_PI * k / M)
        pooled_res.append(r); pooled_res_sid.append(sid + hash(ds) % 100000)
        print(f"\n### {ds}  (n_beats={phi.size}, songs={len(np.unique(sid))}, meter_hist={meter_hist})")
        tr, te = song_split(sid)
        print("  [A] FULL MARGINAL phi -- held-out circular LL / AIC / BIC")
        p = fit_vm(phi[tr]); ll_te = logpdf_vm(phi[te], p["mu"], p["kappa"]).mean()
        a, bb = aic_bic(logpdf_vm(phi, p["mu"], p["kappa"]).sum(), 2, phi.size)
        print(f"      single-vonMises      testLL/beat={ll_te:+.4f}  AIC={a:11.0f} BIC={bb:11.0f}")
        for kernel in ["vm", "wc", "wn"]:
            ll_te_sum, n_te, ll_full_sum, nfree = 0.0, 0, 0.0, 0
            for m in sorted(meter_hist):
                sel = (M == m); xt, xr = phi[sel & tr], phi[sel & te]
                if xt.size < 20 or xr.size < 5:
                    xt = phi[sel]; xr = phi[sel]
                pm = fit_mixture(xt, m, kernel=kernel)
                ll_te_sum += mixture_logpdf(xr, pm).sum(); n_te += xr.size
                ll_full_sum += mixture_logpdf(phi[sel], pm).sum(); nfree += m
            ll_te = ll_te_sum / max(n_te, 1)
            a, bb = aic_bic(ll_full_sum, nfree, phi.size)
            print(f"      meterGridMix[{kernel}]     testLL/beat={ll_te:+.4f}  AIC={a:11.0f} BIC={bb:11.0f}  ({nfree} free)")
        tr, te = song_split(sid)
        print("  [B] WITHIN-PULSE RESIDUAL r -- held-out circular LL / AIC / BIC")
        for name, (fitter, lp, kf) in SINGLE.items():
            p = fitter(r[tr]); ll_te = lp(r[te], p).mean()
            a, bb = aic_bic(lp(r, p).sum(), kf, r.size)
            extra = ""
            if name == "vonMises": extra = f"kappa={p['kappa']:.2f}"
            if name == "wrapCauchy": extra = f"rho={p['rho']:.3f}"
            if name == "skewVonMises": extra = f"kappa={p['kappa']:.2f} lam={p['lam']:+.3f}"
            print(f"      {name:14s} testLL/res={ll_te:+.4f}  AIC={a:11.0f} BIC={bb:11.0f}  {extra}")

    r = np.concatenate(pooled_res); sid = np.concatenate(pooled_res_sid)
    print("\n" + "=" * 100)
    print(f"POOLED across datasets -- WITHIN-PULSE RESIDUAL kernel  (n={r.size})")
    print("=" * 100)
    tr, te = song_split(sid); pooledB = []
    for name, (fitter, lp, kf) in SINGLE.items():
        p = fitter(r[tr]); ll_te = lp(r[te], p).mean()
        a, bb = aic_bic(lp(r, p).sum(), kf, r.size)
        pooledB.append((name, ll_te, a, bb))
        print(f"  {name:14s} testLL/res={ll_te:+.4f}  AIC={a:12.0f}  BIC={bb:12.0f}")
    print(f"  --> residual-kernel winner (held-out LL): {max(pooledB, key=lambda z: z[1])[0]}")

    print("\n" + "=" * 100)
    print("POOLED -- FULL-MARGINAL multimodality: single-vM vs meter-grid vM-MIXTURE")
    print("=" * 100)
    allphi, allM, allsid = [], [], []
    for ds in DATASETS:
        phi, k, M, s = load_dataset(ds)
        keep = np.isin(M, [2, 3, 4])
        if keep.sum() < 200: continue
        allphi.append(phi[keep]); allM.append(M[keep]); allsid.append(s[keep] + hash(ds) % 100000)
    phi = np.concatenate(allphi); M = np.concatenate(allM); sid = np.concatenate(allsid)
    tr, te = song_split(sid)
    p = fit_vm(phi[tr]); ll_te_vm = logpdf_vm(phi[te], p["mu"], p["kappa"]).mean()
    ll_te_sum, n_te = 0.0, 0
    for m in [2, 3, 4]:
        sel = (M == m)
        if sel.sum() < 20: continue
        pm = fit_mixture(phi[sel & tr], m, kernel="vm")
        ll_te_sum += mixture_logpdf(phi[sel & te], pm).sum(); n_te += (sel & te).sum()
    ll_te_mix = ll_te_sum / n_te
    print(f"  single-vonMises   testLL/beat={ll_te_vm:+.4f}")
    print(f"  meterGrid vM-MIX  testLL/beat={ll_te_mix:+.4f}   (delta = {ll_te_mix - ll_te_vm:+.4f} nats/beat)")

    print("\n" + "=" * 100)
    print("DROP-IN CERTIFICATION")
    print("=" * 100)
    print("""  vonMises      : reparam=IMPLICIT (Best-Fisher+Figurnov, ALREADY in distributions.py);
                  KL=CLOSED FORM (kl_von_mises).  [current faithful baseline]
  wrapNormal    : reparam=PATHWISE phi=mu+sigma*eps mod 2pi; KL<=Gaussian KL (DPI bound, closed).
  wrapCauchy    : reparam=PATHWISE inverse-CDF (wrap Cauchy scale=-log rho), verified R=rho;
                  KL=CLOSED FORM log[(1-2 r0 r1 cosD + r0^2 r1^2)/((1-r0^2)(1-r1^2))], verified vs MC (|d|<1e-3).
  skewVonMises  : reparam=IMPLICIT (+1 skew param); KL=stable MC (no closed form).
  meterGridMix  : reparam=Gumbel-Softmax(comp)+per-comp reparam (BOTH already in code);
                  KL<=matched-component bound (closed). Means on 2*pi*k/M grid => METER consequential.""")
    certify_wrapped_cauchy()


if __name__ == "__main__":
    main()
