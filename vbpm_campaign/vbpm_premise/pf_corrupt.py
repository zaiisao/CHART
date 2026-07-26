"""PREMISE-4 driver: bootstrap PF over a bar-pointer transition that can be CONTINUOUSLY
CORRUPTED from the hand-specified physical law toward the learned VAE prior's measured
behaviour.  Verbatim copy of vbpm_final/pf.py with corruption knobs added; with all knobs
at their defaults it is bit-identical to the original (asserted by a regression test).

Corruption modes (one axis at a time):
  mode='none'    : original physical transition.
  mode='rev'     : the pointer increment delta = exp(lt) + sigma_phi*eta keeps its magnitude
                   but its SIGN is flipped with probability p_rev.  p_rev=0.5 -> zero drift,
                   frac_neg -> 0.5 (the learned prior's measured monotonicity failure), while
                   the STEP SIZE stays physical.  Isolates loss-of-drift.
  mode='cauchy'  : phi_t ~ WrappedCauchy(mu = phi_{t-1} + exp(lt), rho).  This is EXACTLY the
                   learned prior's own family (vbpm.distributions.sample_wrapped_cauchy in
                   e3_pf_learned.py).  rho=1 -> deterministic physical advance; rho->0 ->
                   uniform.  Isolates loss-of-concentration.
  mode='gauss'   : phi_t = phi_{t-1} + exp(lt) + sigma_phi*eta with sigma_phi swept up.
                   Gaussian analogue of 'cauchy' (light tails).
  mode='blend'   : phi_t = phi_{t-1} + exp(lt) + u,  u ~ g_w ∝ q_sloppy^(1-w) p_physical^w,
                   the log-linear (geometric) bridge between the sloppy learned-style kernel
                   and the physical kernel.  w = lambda/(1+lambda) is what a physical-prior
                   ANCHORING term of strength lambda buys.  w=0 -> pure sloppy, w=1 -> physical.
  mode='ltnoise' : original phase law, but the log-bar-advance random walk sigma_lt swept up.
                   Isolates tempo-drift sloppiness.
"""
from __future__ import annotations

import math
import numpy as np

TWO_PI = 2.0 * math.pi
METERS = (2, 3, 4)


def lt_band(m, fps=50.0, bpm_lo=55.0, bpm_hi=215.0):
    lo = math.log(TWO_PI * bpm_lo / (60.0 * fps * m))
    hi = math.log(TWO_PI * bpm_hi / (60.0 * fps * m))
    return lo, hi


def _systematic_resample(w, rng):
    K = len(w)
    pos = (np.arange(K) + rng.random()) / K
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    return np.searchsorted(cdf, pos).clip(max=K - 1)


def _wrapped_cauchy(mu, rho, rng):
    """rho in [0,1); rho=0 -> uniform.  Inverse-cdf sampler."""
    u = rng.random(len(mu))
    scale = (1.0 - rho) / (1.0 + rho)
    return (mu + 2.0 * np.arctan(scale * np.tan(math.pi * (u - 0.5)))) % TWO_PI


def particle_filter(LL, nb, K=600, alpha=1.0, sigma_lt=0.005, sigma_phi=0.0, p_switch=0.005,
                    meter_prior=None, fps=50.0, seed=0, noise="gauss", tempo_prior=None,
                    tp_mode="init", tp_rho=0.999, keep_path=True, ess_frac=0.5,
                    # ---- corruption knobs (defaults = original physical transition) ----
                    mode="none", p_rev=0.0, rho_phase=1.0, blend_sampler=None):
    rng = np.random.default_rng(seed)
    T = LL.shape[0]
    nbv = np.zeros(5, np.int64)
    for m in METERS:
        nbv[m] = nb[m]
    lo = np.zeros(5); hi = np.zeros(5)
    for m in METERS:
        lo[m], hi[m] = lt_band(m, fps)

    pm = np.asarray(meter_prior if meter_prior is not None else [0, 0, 1 / 3, 1 / 3, 1 / 3],
                    float)[list(METERS)]
    pm = pm / pm.sum()
    mid = rng.choice(METERS, size=K, p=pm)
    phi = rng.random(K) * TWO_PI
    if tempo_prior is None:
        lt = lo[mid] + (hi[mid] - lo[mid]) * rng.random(K)
    else:
        tmu = np.zeros(5); tsd = np.ones(5)
        for m in METERS:
            tmu[m], tsd[m] = tempo_prior[m]
        lt = np.clip(tmu[mid] + tsd[mid] * rng.standard_normal(K), lo[mid], hi[mid])

    idx = np.arange(K)
    b = (phi / TWO_PI * nbv[mid]).astype(np.int64)
    logw = alpha * LL[0, mid - 1, b]
    _l0 = alpha * LL[0, mid - 1, b]
    _mx = _l0.max()
    logZ = float(_mx + math.log(np.exp(_l0 - _mx).mean()))   # log (1/K) sum_k exp(ll_0)
    logw -= logw.max()
    w = np.exp(logw); w /= w.sum()

    ph_mean = np.empty(T); ph_map = np.empty(T); mt_map = np.empty(T, np.int64)
    map_idx = np.empty(T, np.int64)
    lt_map = np.empty(T)
    ess_h = np.empty(T)
    phi_hist = np.empty((T, K), np.float32) if keep_path else None
    m_hist = np.empty((T, K), np.int8) if keep_path else None
    anc = np.empty((T, K), np.int32) if keep_path else None

    def record(t):
        ph_mean[t] = math.atan2(float((w * np.sin(phi)).sum()),
                                float((w * np.cos(phi)).sum())) % TWO_PI
        j = int(w.argmax())
        map_idx[t] = j
        ph_map[t] = phi[j]; mt_map[t] = mid[j]; lt_map[t] = lt[j]
        ess_h[t] = 1.0 / float((w ** 2).sum())
        if keep_path:
            phi_hist[t] = phi; m_hist[t] = mid

    record(0)
    if keep_path:
        anc[0] = idx
    n_resample = 0

    for t in range(1, T):
        # ------------------------------ transition ------------------------------
        step = np.exp(lt)                      # physical pointer advance, rad/frame
        if mode == "blend":
            # increment offset u ~ g_w ∝ q_sloppy^(1-w) * p_physical^w  (log-linear bridge)
            ug, cdf = blend_sampler
            u = np.interp(rng.random(K), cdf, ug)
            adv_det = phi + step
            cross = adv_det >= TWO_PI
            phi = (adv_det + u) % TWO_PI
        elif mode == "cauchy":
            adv_det = phi + step
            cross = adv_det >= TWO_PI
            phi = _wrapped_cauchy(adv_det % TWO_PI, rho_phase, rng)
        else:
            adv = phi + step                    # identical order/arithmetic to vbpm_final/pf.py
            if sigma_phi > 0:
                adv = adv + sigma_phi * rng.standard_normal(K)
            if mode == "rev" and p_rev > 0:     # flip the SIGN of the increment w.p. p_rev
                flip = rng.random(K) < p_rev
                adv = np.where(flip, 2.0 * phi - adv, adv)
            cross = adv >= TWO_PI
            phi = adv % TWO_PI
        # ------------------------------ tempo RW --------------------------------
        if noise == "laplace":
            e = rng.laplace(0.0, 1.0 / math.sqrt(2.0), K)
        else:
            e = rng.standard_normal(K)
        if tempo_prior is not None and tp_mode == "ou":
            lt = tmu[mid] + tp_rho * (lt - tmu[mid]) + sigma_lt * e
        else:
            lt = lt + sigma_lt * e
        if p_switch > 0:
            sw = cross & (rng.random(K) < p_switch)
            if sw.any():
                new = rng.choice(METERS, size=int(sw.sum()))
                old = mid[sw]
                lt[sw] = lt[sw] + np.log(new / old)
                mid[sw] = new
        np.clip(lt, lo[mid], hi[mid], out=lt)

        # ------------------------------ weight ----------------------------------
        b = (phi / TWO_PI * nbv[mid]).astype(np.int64)
        _ll = alpha * LL[t, mid - 1, b]
        _mx = _ll.max()
        # unbiased bootstrap-PF evidence increment: log sum_k W_{t-1,k} exp(ll_t,k)
        logZ += float(_mx + math.log(float((w * np.exp(_ll - _mx)).sum())))
        logw = logw + _ll
        logw -= logw.max()
        w = np.exp(logw); w /= w.sum()
        record(t)

        if keep_path:
            anc[t] = idx
        if ess_h[t] < ess_frac * K:
            a = _systematic_resample(w, rng)
            phi, lt, mid = phi[a], lt[a], mid[a]
            if keep_path:
                anc[t] = a
            logw = np.zeros(K)
            w = np.full(K, 1.0 / K)
            n_resample += 1

    out = dict(phase_mean=ph_mean, phase_map=ph_map, meter_map=mt_map, lt_map=lt_map,
               ess=float(ess_h.mean()), n_resample=n_resample,
               logZ=logZ, logZ_per_frame=logZ / T, T=T)
    if keep_path:
        j = int(map_idx[T - 1])
        pp = np.empty(T); pmt = np.empty(T, np.int64)
        for t in range(T - 1, -1, -1):
            pp[t] = phi_hist[t, j]; pmt[t] = m_hist[t, j]
            if t > 0:
                j = int(anc[t - 1][j])
        out["phase_path"] = pp
        out["meter_path"] = pmt
    return out
