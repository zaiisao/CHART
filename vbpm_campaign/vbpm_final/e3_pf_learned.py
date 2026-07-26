"""Bootstrap particle filter over the LEARNED VAE transition (deploy path of E3).

Copy of vbpm_fix/variant_b.py::particle_filter with two additions and no other change:
  * an ancestral-path backtrace, so the reported trajectory is a genuine sample path of
    the filter rather than the per-frame argmax (per-frame argmax alone makes frac_neg
    look ~0.5 even for a well-behaved filter);
  * all history kept on-device and moved to host once, so the path option is cheap.

vbpm_fix/ is NOT modified.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from vbpm.distributions import TWO_PI, sample_wrapped_cauchy, sample_student_t


def _systematic_resample(w):
    K = w.shape[0]
    pos = (torch.arange(K, device=w.device, dtype=w.dtype)
           + torch.rand(1, device=w.device)) / K
    cdf = torch.cumsum(w, 0)
    cdf = cdf / cdf[-1].clamp(min=1e-30)
    return torch.searchsorted(cdf.contiguous(), pos.contiguous()).clamp(max=K - 1)


def _circ_wmean(phi, w):
    return torch.atan2((w * torch.sin(phi)).sum(), (w * torch.cos(phi)).sum()) % TWO_PI


@torch.no_grad()
def particle_filter_learned(model, h, obs, K=600, alpha=1.0, diffuse_init=True,
                            lt_lo=-3.55, lt_hi=-2.18, ess_frac=0.5, keep_path=True):
    assert h.shape[0] == 1
    T = h.shape[1]
    dv = h.device
    prior_ctx = model.encode_prior(h)
    ctx = prior_ctx[0]
    dof = model.tempo_dof()
    a_lv = model.level_ar()
    Km = model.K

    p_m, p_ph_mu, p_ph_rho, p_lv_mu, p_lv_s, _a, _b = model.unpack(
        model.prior_init_head(prior_ctx.mean(1)))
    if diffuse_init:
        phi = torch.rand(K, device=dv) * TWO_PI
        level = lt_lo + (lt_hi - lt_lo) * torch.rand(K, device=dv)
    else:
        phi = sample_wrapped_cauchy(p_ph_mu.expand(K), p_ph_rho.expand(K))
        level = sample_student_t(dof, p_lv_mu.expand(K), p_lv_s.expand(K))
    dev = torch.zeros(K, device=dv)
    log_tempo = level + dev
    m_idx = torch.multinomial(F.softmax(p_m, -1).expand(K, Km), 1).squeeze(1)
    meter = F.one_hot(m_idx, Km).to(h.dtype)
    anchor = level.clone()

    zf = model.z_features(meter, phi, log_tempo)
    logw = alpha * model.obs_logp(zf, obs[0, 0].unsqueeze(0).expand(K, -1))
    logw = logw - logw.max()
    w = F.softmax(logw, 0)

    idx0 = torch.arange(K, device=dv, dtype=torch.int32)
    ph_mean = torch.empty(T, device=dv); ph_map = torch.empty(T, device=dv)
    lt_map = torch.empty(T, device=dv); ess_h = torch.empty(T, device=dv)
    mt_map = torch.empty(T, device=dv, dtype=torch.int64)
    map_idx = torch.empty(T, device=dv, dtype=torch.int64)
    phi_hist = torch.empty((T, K), device=dv) if keep_path else None
    m_hist = torch.empty((T, K), device=dv, dtype=torch.int8) if keep_path else None
    anc = torch.empty((T, K), device=dv, dtype=torch.int32) if keep_path else None

    def record(t):
        ph_mean[t] = _circ_wmean(phi, w)
        j = w.argmax()
        map_idx[t] = j
        ph_map[t] = phi[j]
        lt_map[t] = log_tempo[j]
        mt_map[t] = meter[j].argmax()
        ess_h[t] = 1.0 / (w ** 2).sum()
        if keep_path:
            phi_hist[t] = phi
            m_hist[t] = meter.argmax(-1).to(torch.int8)

    record(0)
    if keep_path:
        anc[0] = idx0
    n_resample = 0

    for t in range(1, T):
        ctx_t = ctx[t].unsqueeze(0).expand(K, -1)
        advance = phi + torch.exp(log_tempo.clamp(-12.0, 6.0))
        cross = advance >= TWO_PI
        p_ph_mu_t = advance % TWO_PI
        rho = model.prior_phase_conc(ctx_t)
        a = model.prior_dev_coef(ctx_t)
        s_lv = model.prior_level_scale(ctx_t)
        s_dv = model.prior_dev_scale(ctx_t)

        phi_new = sample_wrapped_cauchy(p_ph_mu_t, rho)
        level_new = sample_student_t(dof, anchor + a_lv * (level - anchor), s_lv)
        dev_new = a * dev + s_dv * torch.randn(K, device=dv)
        lt_new = level_new + dev_new

        log_pi = model.meter_prior_logp(meter, phi_new, phi, ctx_t)
        draw = torch.multinomial(log_pi.exp().clamp(min=1e-12), 1).squeeze(1)
        meter_new = torch.where(cross.unsqueeze(-1), F.one_hot(draw, Km).to(h.dtype), meter)

        phi, level, dev, log_tempo, meter = phi_new, level_new, dev_new, lt_new, meter_new

        zf = model.z_features(meter, phi, log_tempo)
        logw = logw + alpha * model.obs_logp(zf, obs[0, t].unsqueeze(0).expand(K, -1))
        logw = logw - logw.max()
        w = F.softmax(logw, 0)
        record(t)

        if keep_path:
            anc[t] = idx0
        if float(ess_h[t]) < ess_frac * K:
            a_i = _systematic_resample(w)
            phi, level, dev, log_tempo = phi[a_i], level[a_i], dev[a_i], log_tempo[a_i]
            meter, anchor = meter[a_i], anchor[a_i]
            if keep_path:
                anc[t] = a_i.to(torch.int32)
            logw = torch.zeros(K, device=dv)
            w = torch.full((K,), 1.0 / K, device=dv)
            n_resample += 1

    out = {"phase_mean": ph_mean.cpu().numpy().astype(float),
           "phase_map": ph_map.cpu().numpy().astype(float),
           "log_tempo": lt_map.cpu().numpy().astype(float),
           "meter_map": mt_map.cpu().numpy() + getattr(model, "meter_offset", 1),
           "ess": float(ess_h.mean()), "n_resample": n_resample}
    if keep_path:
        PH = phi_hist.cpu().numpy(); MH = m_hist.cpu().numpy()
        AN = anc.cpu().numpy(); MI = map_idx.cpu().numpy()
        j = int(MI[T - 1])
        pp = np.empty(T); pmt = np.empty(T, np.int64)
        for t in range(T - 1, -1, -1):
            pp[t] = PH[t, j]; pmt[t] = MH[t, j]
            if t > 0:
                j = int(AN[t - 1][j])
        out["phase_path"] = pp
        out["meter_path"] = pmt + getattr(model, "meter_offset", 1)
    return out
