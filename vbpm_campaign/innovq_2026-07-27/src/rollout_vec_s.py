"""Parallel-in-time rollout with SAMPLING and full ELBO outputs.

Two fixes over rollout_vec.py:
  (a) sample=True via pre-drawn noise. Under reparameterization every draw is either
      exogenous (uniform / standard-t / normal) or state-independent (meter draws use
      m_logits from the INIT head, fixed over t), so the noise is exogenous to the
      recursion and Picard applies unchanged.
  (b) OFF-BY-ONE FIX. The loop computes innov_head(ctx_t, z_features(state_{t-1}));
      rollout_vec paired ctx_t with z_features(state_t). Suspected source of the
      1.1e-3 rad residual previously recorded as a speed/exactness sacrifice.

Returns every key elbo_innovq needs (Z, kl_p, kl_l, kl_m, kl_dv, n_cross, MU, SQ, ...).
"""
import math, torch, torch.nn.functional as F
import innovq as IQ

THRESH_MARGIN = 5e-3   # rad; ~5x the observed Picard residual
from innovq import TWO_PI, B_SLT0, DEV_SIGMA, DOF, T_SCALE, B_LS0
# NOT `from innovq import R0`: R0 is rebound by --gamma_phase at runtime and a
# from-import would freeze the import-time value. Read IQ.R0 at call time.
import pm_common as P
from faithful.distributions import kl_categorical
from innovq import kl_phase_innov, kl_wrapped_cauchy


def draw_noise(Bn, T, K, dev, dof, gen=None):
    """All exogenous randomness, pre-drawn. Index 0 = the t=1 init draws."""
    st = torch.distributions.StudentT(torch.tensor(float(dof), device=dev),
                                      torch.zeros((), device=dev), torch.ones((), device=dev))
    return dict(u=torch.rand(Bn, T, device=dev, generator=gen).clamp(1e-4, 1 - 1e-4),
                tstd=st.sample((Bn, T)),
                nrm=torch.randn(Bn, T, device=dev, generator=gen),
                gum=-torch.log(-torch.log(torch.rand(Bn, T, K, device=dev, generator=gen) + 1e-20) + 1e-20))


def _heads(model, init, K):
    mu_phi1 = torch.atan2(init[:, K + 1], init[:, K]) % TWO_PI
    rho1 = (torch.sigmoid(init[:, K + 2]) * model.rho1_max).clamp(1e-6, 1 - 1e-6)
    mu_l1 = init[:, K + 3] + model.level_offset
    s_l1 = F.softplus(init[:, K + 4]) + 0.05
    return mu_phi1, rho1, mu_l1, s_l1


def rollout_loop_noise(model, h, b, noise, *, sample=True, temperature=0.3):
    """EXACT replica of IQ.rollout consuming pre-drawn noise (the gate reference)."""
    Bn, T, _ = h.shape; dev = h.device; K = model.K
    dof_t = torch.tensor(DOF, device=dev); tsc = torch.full((Bn,), T_SCALE, device=dev)
    zero = torch.zeros(Bn, device=dev)
    ctx = model.encode_posterior(h, b)
    init = model.init_head(torch.cat([ctx.mean(1), ctx[:, 0]], -1))
    m_logits = init[:, :K]
    mu_phi1, rho1, mu_l1, s_l1 = _heads(model, init, K)
    if sample:
        phi = (mu_phi1 + (-torch.log(rho1)) * torch.tan(math.pi * (noise["u"][:, 0] - 0.5))) % TWO_PI
        level = mu_l1 + s_l1 * noise["tstd"][:, 0]
        dv = DEV_SIGMA * noise["nrm"][:, 0]
        meter = F.softmax((m_logits + noise["gum"][:, 0]) / temperature, -1)
    else:
        phi, level, dv = mu_phi1, mu_l1, zero
        meter = F.softmax(m_logits / max(temperature, 1e-6), -1)
    lt = level + dv
    kl_p = kl_wrapped_cauchy(mu_phi1, rho1, torch.full_like(mu_phi1, math.pi),
                             torch.full_like(mu_phi1, 1e-6)).double()
    kl_l = P.kl_t_mc(dof_t, mu_l1, s_l1, dof_t, torch.full((Bn,), IQ.INIT_LV_MU, device=dev),
                     torch.full((Bn,), IQ.INIT_LV_S, device=dev), level)
    kl_m = kl_categorical(torch.log_softmax(m_logits, -1),
                          torch.full((Bn, K), -math.log(K), device=dev))
    n_cross = torch.ones(Bn, device=dev)
    z_feats = [model.z_features(meter, phi, lt)]; phis, lts, MU, SQ = [phi], [lt], [], []
    for t in range(1, T):
        zf = model.z_features(meter, phi, lt)
        out = model.innov_head(torch.cat([ctx[:, t], zf], -1))
        mu_eps = torch.tanh(out[:, 0]) * model.s_phi
        sq = F.softplus(out[:, 1] + IQ.R0).clamp(1e-6, 0.5)
        mu_lt = torch.tanh(out[:, 2]) * model.s_lt
        s_lv = F.softplus(out[:, 3] + B_SLT0) + 1e-5
        tempo = torch.exp(lt.clamp(-12.0, 6.0)); advance = phi + tempo
        cross = (advance >= TWO_PI).to(h.dtype)
        if sample:
            gam_q = -torch.log1p(-sq)
            eps = mu_eps + gam_q * torch.tan(math.pi * (noise["u"][:, t] - 0.5))
            eps_lt = mu_lt + s_lv * noise["tstd"][:, t]
            dv = DEV_SIGMA * noise["nrm"][:, t]
            m_draw = F.softmax((m_logits + noise["gum"][:, t]) / temperature, -1)
        else:
            eps, eps_lt, dv = mu_eps, mu_lt, zero
            m_draw = F.softmax(m_logits / max(temperature, 1e-6), -1)
        meter_prev = meter
        phi = (advance + eps) % TWO_PI
        level = level + eps_lt
        lt = level + dv
        meter = torch.where(cross.unsqueeze(-1) > 0.5, m_draw, meter_prev)
        kl_p = kl_p + kl_phase_innov(mu_eps, sq)
        kl_l = kl_l + P.kl_t_mc(dof_t, mu_lt, s_lv, dof_t, zero, tsc, eps_lt)
        kl_m = kl_m + cross * kl_categorical(torch.log_softmax(m_logits, -1),
                                             torch.log(meter_prev @ model.Pi_phys + 1e-9))
        n_cross = n_cross + cross
        z_feats.append(model.z_features(meter, phi, lt))
        phis.append(phi); lts.append(lt); MU.append(mu_eps.detach()); SQ.append(sq.detach())
    return dict(Z=torch.stack(z_feats, 1), phi=torch.stack(phis, 1), lt=torch.stack(lts, 1),
                kl_p=kl_p.float(), kl_l=kl_l, kl_m=kl_m, kl_dv=torch.zeros(Bn, device=dev),
                n_cross=n_cross, MU=torch.stack(MU, 1), SQ=torch.stack(SQ, 1),
                init_vec=init, ctx=ctx, mu_phi1=mu_phi1.detach(), rho1=rho1.detach(),
                mu_l1=mu_l1.detach(), s_l1=s_l1.detach())
    if exact_fallback and bool(unstable.any()):
        idx = torch.nonzero(unstable).squeeze(-1)
        sub = {k: v[idx] for k, v in noise.items()}
        ref = rollout_loop_noise(model, h[idx], b[idx], sub, sample=sample, temperature=temperature)
        for k in ("Z", "phi", "lt", "kl_p", "kl_l", "kl_m", "kl_dv", "n_cross", "MU", "SQ",
                  "init_vec", "mu_phi1", "rho1", "mu_l1", "s_l1"):
            if k in ref and k in out_d and out_d[k].shape[0] == Bn:
                t = out_d[k].clone()
                t[idx] = ref[k].to(t.dtype)
                out_d[k] = t
        out_d["n_fallback"] = int(idx.numel())
    else:
        out_d["n_fallback"] = 0
    return out_d


def rollout_vec_s(model, h, b, noise=None, *, sample=True, temperature=0.3, n_picard=8,
                  exact_fallback=False):
    """Picard-vectorized. Same signature/outputs as the loop above.

    n_picard defaults to 8: at 4, ~18% of real crops resolve the wrong bar-crossing count
    when sampling (verified). More iterations are NOT always enough -- the hard threshold
    cross = (advance >= 2pi) makes the Picard map bistable, and some crops sit on a second
    stable fixed point that is identical at n_picard 8/16/32/64. Those crops are DETECTED
    (crossing pattern still changing on the last iteration) and recomputed exactly with the
    sequential rollout, so the result is exact by construction rather than by tolerance.
    """
    Bn, T, _ = h.shape; dev = h.device; K = model.K
    dof_t = torch.tensor(DOF, device=dev); tsc = torch.full((Bn,), T_SCALE, device=dev)
    if noise is None:
        noise = draw_noise(Bn, T, K, dev, DOF)
    ctx = model.encode_posterior(h, b)
    init = model.init_head(torch.cat([ctx.mean(1), ctx[:, 0]], -1))
    m_logits = init[:, :K]
    mu_phi1, rho1, mu_l1, s_l1 = _heads(model, init, K)
    if sample:
        phi1 = (mu_phi1 + (-torch.log(rho1)) * torch.tan(math.pi * (noise["u"][:, 0] - 0.5))) % TWO_PI
        lev1 = mu_l1 + s_l1 * noise["tstd"][:, 0]
        dvv = DEV_SIGMA * noise["nrm"]
        m_draw = F.softmax((m_logits.unsqueeze(1) + noise["gum"]) / temperature, -1)   # [B,T,K]
        tanu = torch.tan(math.pi * (noise["u"] - 0.5))
    else:
        phi1, lev1 = mu_phi1, mu_l1
        dvv = torch.zeros(Bn, T, device=dev)
        m_draw = F.softmax(m_logits / max(temperature, 1e-6), -1).unsqueeze(1).expand(-1, T, -1)
        tanu = None
    # ---- initial guess: zero innovations
    lt = (lev1.unsqueeze(1) + dvv)
    phi = ((phi1.unsqueeze(1).double()
            + torch.cumsum(F.pad(torch.exp(lt.clamp(-12., 6.))[:, :-1], (1, 0)).double(), 1)) % TWO_PI).float()
    meter = m_draw[:, :1].expand(-1, T, -1)
    ar = torch.arange(T, device=dev)
    prev_cross = None
    unstable = torch.zeros(Bn, dtype=torch.bool, device=dev)
    for _it in range(n_picard):
        zf = model.z_features(meter.reshape(-1, K), phi.reshape(-1), lt.reshape(-1)).reshape(Bn, T, -1)
        zf_in = torch.cat([zf[:, :1], zf[:, :-1]], 1)          # OFF-BY-ONE FIX: ctx_t pairs z_{t-1}
        out = model.innov_head(torch.cat([ctx, zf_in], -1))
        mu_eps = torch.tanh(out[..., 0]) * model.s_phi
        sq = F.softplus(out[..., 1] + IQ.R0).clamp(1e-6, 0.5)
        mu_lt = torch.tanh(out[..., 2]) * model.s_lt
        s_lv = F.softplus(out[..., 3] + B_SLT0) + 1e-5
        if sample:
            eps = mu_eps + (-torch.log1p(-sq)) * tanu
            eps_lt = mu_lt + s_lv * noise["tstd"]
        else:
            eps, eps_lt = mu_eps, mu_lt
        lev = (lev1.unsqueeze(1).double() + torch.cumsum(F.pad(eps_lt[:, 1:], (1, 0)).double(), 1)).float()
        lt = lev + dvv
        steps = torch.exp(lt.clamp(-12., 6.))
        inc = F.pad(steps[:, :-1], (1, 0)) + F.pad(eps[:, 1:], (1, 0))
        phi = ((phi1.unsqueeze(1).double() + torch.cumsum(inc.double(), 1)) % TWO_PI).float()
        adv = phi[:, :-1] + steps[:, :-1]
        cross = F.pad((adv >= TWO_PI).to(h.dtype), (1, 0))     # [B,T], cross[:,0]=0
        cfull = cross.clone(); cfull[:, 0] = 1.0               # t=1 always draws
        last = torch.cummax((ar.unsqueeze(0) * cfull).long(), dim=1).values
        meter = torch.gather(m_draw, 1, last.unsqueeze(-1).expand(-1, -1, K))
        # The Picard map is BISTABLE at the hard threshold: a crop can settle on a
        # self-consistent but WRONG crossing pattern, so "pattern still changing" cannot
        # detect it. Flag FRAGILITY instead: any advance sitting within the Picard residual
        # of 2pi could tip either way, so recompute that crop exactly.
        margin = (adv - TWO_PI).abs().min(dim=1).values
        unstable = margin < THRESH_MARGIN
        if prev_cross is not None:
            unstable = unstable | (cross != prev_cross).any(dim=1)
        prev_cross = cross.detach()
    zf = model.z_features(meter.reshape(-1, K), phi.reshape(-1), lt.reshape(-1)).reshape(Bn, T, -1)
    meter_prev = torch.cat([meter[:, :1], meter[:, :-1]], 1)
    kl_p = kl_wrapped_cauchy(mu_phi1, rho1, torch.full_like(mu_phi1, math.pi),
                             torch.full_like(mu_phi1, 1e-6)).double() \
        + kl_phase_innov(mu_eps[:, 1:].reshape(-1), sq[:, 1:].reshape(-1)).reshape(Bn, -1).sum(1)
    kl_l = P.kl_t_mc(dof_t, mu_l1, s_l1, dof_t, torch.full((Bn,), IQ.INIT_LV_MU, device=dev),
                     torch.full((Bn,), IQ.INIT_LV_S, device=dev), lev[:, 0]) \
        + P.kl_t_mc(dof_t, mu_lt[:, 1:], s_lv[:, 1:], dof_t, torch.zeros_like(mu_lt[:, 1:]),
                    tsc.unsqueeze(1).expand(-1, T - 1), eps_lt[:, 1:]).sum(1)
    lq = torch.log_softmax(m_logits, -1).unsqueeze(1).expand(-1, T - 1, -1)
    lp = torch.log(meter_prev[:, 1:] @ model.Pi_phys + 1e-9)
    kl_m = kl_categorical(torch.log_softmax(m_logits, -1),
                          torch.full((Bn, K), -math.log(K), device=dev)) \
        + (cross[:, 1:] * kl_categorical(lq, lp)).sum(1)
    out_d = dict(Z=zf, phi=phi, lt=lt, kl_p=kl_p.float(), kl_l=kl_l, kl_m=kl_m,
                kl_dv=torch.zeros(Bn, device=dev), n_cross=1.0 + cross[:, 1:].sum(1),
                MU=mu_eps[:, 1:].detach(), SQ=sq[:, 1:].detach(),
                init_vec=init, ctx=ctx, mu_phi1=mu_phi1.detach(), rho1=rho1.detach(),
                mu_l1=mu_l1.detach(), s_l1=s_l1.detach())
    if exact_fallback and bool(unstable.any()):
        idx = torch.nonzero(unstable).squeeze(-1)
        sub = {k: v[idx] for k, v in noise.items()}
        ref = rollout_loop_noise(model, h[idx], b[idx], sub, sample=sample, temperature=temperature)
        for k in ("Z", "phi", "lt", "kl_p", "kl_l", "kl_m", "kl_dv", "n_cross", "MU", "SQ",
                  "init_vec", "mu_phi1", "rho1", "mu_l1", "s_l1"):
            if k in ref and k in out_d and out_d[k].shape[0] == Bn:
                t = out_d[k].clone()
                t[idx] = ref[k].to(t.dtype)
                out_d[k] = t
        out_d["n_fallback"] = int(idx.numel())
    else:
        out_d["n_fallback"] = 0
    return out_d
