"""Bootstrap particle filter over a bar-pointer transition, weighted by a SUPERVISED
p(activation | bar phase, meter).  Pure numpy (the emission is a lookup table).

Transition (stated explicitly, this is the "simple bar-pointer transition" option):
    phi_t   = (phi_{t-1} + exp(lt_{t-1}) + sigma_phi*eta_t) mod 2pi   # pointer advance
    lt_t    = lt_{t-1} + sigma_lt * eps_t                  # log bar-advance random walk
              (eps ~ N(0,1) or Laplace(0,1)/sqrt(2)), reflected into the 55-215 BPM band
    m_t     = m_{t-1}, except at a bar crossing where it switches w.p. p_switch
              (beat tempo preserved: lt shifts by log(m_new/m_old))
Weights   logw += alpha * log p(o_t | phi_t, m_t);  systematic resample at ESS < K/2.
"""
from __future__ import annotations

import math

import numpy as np

TWO_PI = 2.0 * math.pi
METERS = (2, 3, 4)


def lt_band(m, fps=50.0, bpm_lo=55.0, bpm_hi=215.0):
    """log bar-advance (rad/frame) band for beat-BPM in [bpm_lo, bpm_hi] at meter m."""
    lo = math.log(TWO_PI * bpm_lo / (60.0 * fps * m))
    hi = math.log(TWO_PI * bpm_hi / (60.0 * fps * m))
    return lo, hi


def _systematic_resample(w, rng):
    K = len(w)
    pos = (np.arange(K) + rng.random()) / K
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    return np.searchsorted(cdf, pos).clip(max=K - 1)


def particle_filter(LL, nb, K=600, alpha=1.0, sigma_lt=0.005, sigma_phi=0.0, p_switch=0.005,
                    meter_prior=None, fps=50.0, seed=0, noise="gauss", tempo_prior=None,
                    tp_mode="init", tp_rho=0.999,
                    keep_path=True, ess_frac=0.5, n_hist_bins=72):
    """LL: [T, 4, Bmax] padded log p(o_t|bin, meter-1).  nb: dict meter -> n_bins.
    Returns dict with phase_mean / phase_map / phase_path / meter tracks and diagnostics."""
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
    mid = rng.choice(METERS, size=K, p=pm)                    # meter value per particle
    phi = rng.random(K) * TWO_PI
    if tempo_prior is None:
        lt = lo[mid] + (hi[mid] - lo[mid]) * rng.random(K)     # madmom-style uniform band
    else:                       # TRAIN-FITTED log-tempo prior N(mu_m, sd_m) per meter
        tmu = np.zeros(5); tsd = np.ones(5)
        for m in METERS:
            tmu[m], tsd[m] = tempo_prior[m]
        lt = np.clip(tmu[mid] + tsd[mid] * rng.standard_normal(K), lo[mid], hi[mid])

    idx = np.arange(K)
    b = (phi / TWO_PI * nbv[mid]).astype(np.int64)
    logw = alpha * LL[0, mid - 1, b]
    logw -= logw.max()
    w = np.exp(logw); w /= w.sum()

    ph_mean = np.empty(T); ph_map = np.empty(T); mt_map = np.empty(T, np.int64)
    map_idx = np.empty(T, np.int64)
    lt_map = np.empty(T)
    ess_h = np.empty(T)
    phi_hist = np.empty((T, K), np.float32) if keep_path else None
    m_hist = np.empty((T, K), np.int8) if keep_path else None
    anc = np.empty((T, K), np.int32) if keep_path else None
    NH = int(n_hist_bins)
    hist_psi = np.zeros((T, NH), np.float32)   # posterior over beat phase psi=(m*phi) mod 2pi
    hist_phi = np.zeros((T, NH), np.float32)   # posterior over bar phase phi

    def record(t):
        ph_mean[t] = math.atan2(float((w * np.sin(phi)).sum()),
                                float((w * np.cos(phi)).sum())) % TWO_PI
        j = int(w.argmax())
        map_idx[t] = j
        ph_map[t] = phi[j]; mt_map[t] = mid[j]; lt_map[t] = lt[j]
        ess_h[t] = 1.0 / float((w ** 2).sum())
        psi = (mid * phi) % TWO_PI
        hist_psi[t] = np.bincount((psi / TWO_PI * NH).astype(np.int64) % NH,
                                  weights=w, minlength=NH)
        hist_phi[t] = np.bincount((phi / TWO_PI * NH).astype(np.int64) % NH,
                                  weights=w, minlength=NH)
        if keep_path:
            phi_hist[t] = phi; m_hist[t] = mid

    record(0)
    if keep_path:
        anc[0] = idx
    n_resample = 0

    for t in range(1, T):
        # --- transition ---
        adv = phi + np.exp(lt)
        if sigma_phi > 0:
            adv = adv + sigma_phi * rng.standard_normal(K)
        cross = adv >= TWO_PI
        phi = adv % TWO_PI
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
                lt[sw] = lt[sw] + np.log(new / old)     # preserve beat tempo
                mid[sw] = new
        np.clip(lt, lo[mid], hi[mid], out=lt)

        # --- weight ---
        b = (phi / TWO_PI * nbv[mid]).astype(np.int64)
        logw = logw + alpha * LL[t, mid - 1, b]
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
               hist_psi=hist_psi, hist_phi=hist_phi)
    if keep_path:
        # ancestral backtrace of the final MAP particle.  anc[t][p] = index (in the
        # PRE-resample set at t-1) of the parent of pre-resample particle p at time t.
        j = int(map_idx[T - 1])
        pp = np.empty(T); pmt = np.empty(T, np.int64)
        for t in range(T - 1, -1, -1):
            pp[t] = phi_hist[t, j]; pmt[t] = m_hist[t, j]
            if t > 0:
                j = int(anc[t - 1][j])
        out["phase_path"] = pp
        out["meter_path"] = pmt
    return out
