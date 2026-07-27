"""S1 postmortem common: data, oracle z, a UNIFIED elbo runner that can swap
  q  in {amortized (the trained encoder), pinned (oracle z, calibrated widths), free (SVI)}
  p  in {trained (the ckpt's own prior heads), physical (the fitted physical law)}
  decoders in {trained, refit}
  recon in {bce (exactly as elbo_b), stmask (v17b/Beat-This +-3 zero-weight), stpool (official
            ShiftTolerantBCELoss: max-pool spread + mask)}
Everything else is a line-for-line copy of vbpm_fix/variant_b.py::elbo_b so the numbers are
directly comparable to the trained arm_ii log.
"""
import sys, math, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
import variant_b as VB                                                    # noqa: E402
from vbpm.distributions import (TWO_PI, gumbel_softmax, sample_wrapped_cauchy,   # noqa: E402
                                sample_student_t, kl_categorical,
                                kl_wrapped_cauchy, kl_log_normal)

FPS = 50.0
CROP = 256
ARMS = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms"
CKPT = f"{ARMS}/arm_i_ii_bern.pt"
OUT = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem"

# ---- physical law (established fits) ------------------------------------------------
PHYS = dict(gamma_phase=5.5e-4,          # wrapped Cauchy scale  -> rho = exp(-gamma)
            t_dof=2.0, t_scale=0.00125,  # Student-t log-tempo LEVEL increments
            dev_ar=0.0, dev_sigma=1e-3,  # "no fast deviation": dev pinned at the param floor
            meter_self=0.995,            # sticky meter, per BAR (bar-gated)
            init_level_mu=-2.66, init_level_scale=0.3)


# ------------------------------------------------------------------ data
def load_songs(split="train"):
    d = np.load(f"{ARMS}/act_{split}.npz", allow_pickle=True)
    stems = [s for s in sorted(set(k.rsplit('|', 1)[0] for k in d.files))
             if s + '|act' in d.files]
    return [dict(stem=s, act=np.asarray(d[s + '|act'], np.float32),
                 beats=np.asarray(d[s + '|beats'], float),
                 downs=np.asarray(d[s + '|downs'], float),
                 dataset=str(d[s + '|dataset'])) for s in stems]


def oracle_z(beats, downs, T):
    """GT bar phase (0 at downbeat -> 2pi), per-bar log bar-advance, meter. None if <2 downbeats."""
    d = downs[(downs >= 0) & (downs < T / FPS)]
    if len(d) < 2:
        return None
    t = (np.arange(T) + 0.5) / FPS
    ph = np.zeros(T); lt = np.full(T, np.nan)
    for i in range(len(d) - 1):
        a, b = d[i], d[i + 1]; m = (t >= a) & (t < b)
        ph[m] = TWO_PI * (t[m] - a) / max(b - a, 1e-6)
        lt[m] = math.log(TWO_PI / (max(b - a, 1e-6) * FPS))
    lt[np.isnan(lt)] = np.nanmedian(lt)
    ph[t < d[0]] = (TWO_PI * (t[t < d[0]] - d[0]) / max(d[1] - d[0], 1e-6)) % TWO_PI
    ph[t >= d[-1]] = (TWO_PI * (t[t >= d[-1]] - d[-1]) / max(d[-1] - d[-2], 1e-6)) % TWO_PI
    bpb = np.median([np.sum((beats >= d[i]) & (beats < d[i + 1])) for i in range(len(d) - 1)])
    return ph, lt, int(max(2, min(round(bpb), 4)))


def targets(beats, downs, T):
    b = np.zeros(T, np.float32); db = np.zeros(T, np.float32)
    for x in beats:
        i = int(round(x * FPS))
        if 0 <= i < T: b[i] = 1
    for x in downs:
        i = int(round(x * FPS))
        if 0 <= i < T: db[i] = 1
    return b, db


def st_mask(y, tol=3):
    """v17b / Beat-This pattern: zero LOSS WEIGHT within +-tol frames of a positive,
    except the annotated frame itself (weight 1)."""
    m = torch.ones_like(y)
    for dshift in list(range(-tol, 0)) + list(range(1, tol + 1)):
        m = m * (1.0 - torch.roll(y, dshift, dims=-1))
    m = torch.clamp(m, 0.0, 1.0)
    return torch.maximum(m, y)


def build_crops(songs, n_per_song=4, seed=0, crop=CROP, dev="cuda:0"):
    rng = np.random.default_rng(seed)
    H, B, DB, PHI, LT, MI, SID = [], [], [], [], [], [], []
    for si, s in enumerate(songs):
        act = np.clip(s["act"], 1e-4, 1.0 - 1e-4).astype(np.float32)
        T = len(act)
        oz = oracle_z(s["beats"], s["downs"], T)
        if oz is None or T < crop + 1:
            continue
        ph, lt, m = oz
        b, db = targets(s["beats"], s["downs"], T)
        for _ in range(n_per_song):
            st = int(rng.integers(0, T - crop))
            sl = slice(st, st + crop)
            H.append(act[sl]); B.append(b[sl]); DB.append(db[sl])
            PHI.append(ph[sl]); LT.append(lt[sl]); MI.append(m - 1); SID.append(si)
    t = lambda x, dt=torch.float32: torch.tensor(np.array(x), dtype=dt, device=dev)
    return dict(h=t(H), b=t(B), db=t(DB), obs=t(H),
                phi=t(PHI), lt=t(LT),
                m=torch.tensor(np.array(MI), dtype=torch.long, device=dev),
                song=np.array(SID))


def load_model(dev="cuda:0"):
    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    m = VB.BarPointerVAE_B(h_dim=2, hidden=128, num_meters=4, obs_dim=2, obs_type="bern").to(dev)
    missing, unexpected = m.load_state_dict(ck["model"], strict=False)
    assert not [k for k in missing], missing
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def new_decoders(dev="cuda:0", hidden=128):
    dec = nn.Sequential(nn.Linear(7, hidden), nn.Tanh(), nn.Linear(hidden, 2)).to(dev)
    hdec = nn.Sequential(nn.Linear(7, hidden), nn.Tanh(), nn.Linear(hidden, 2)).to(dev)
    return dec, hdec


# ------------------------------------------------------------------ recon

# ---------------------------------------------------- displacement likelihood
# BCE is pointwise: no information about HOW FAR a predicted beat is from a true one,
# so its tempo landscape is a flat plateau with a zero-width pit at the answer
# (measured: pit half-width ~ w/T -> 0% at T=1500). The Cramer / L1-on-CDF distance
# compares CUMULATIVE beat counts, so a k-frame displacement costs ~k and a count
# error accumulates linearly. Used as a log-density: log p(b|z) = -lam*Cramer + const,
# i.e. lam is the observation precision. Gated in vbpm_innovq/cramer_gate2.py:
# min at mult=1.00, far-field slope 89250 vs BCE 63.7, gradient points home 0.7x-1.3x.
CRAMER_LAM = 0.03          # scales the term to BCE magnitude so beta*KL stays live

def _smooth_dirac(y, sig=1.44):
    """Gaussian-smooth a Dirac beat train, mass preserved (matches the pulse width)."""
    k = int(4 * sig) | 1
    t = torch.arange(k, device=y.device, dtype=y.dtype) - k // 2
    g = torch.exp(-0.5 * (t / sig) ** 2)
    g = g / g.sum()
    return F.conv1d(y.unsqueeze(1), g.view(1, 1, -1), padding=k // 2).squeeze(1)

def _cramer(p, tgt, scales=(50, 200, 800)):
    """sum_t |cumsum(p)_t - cumsum(tgt)_t| at several horizons -> [B]."""
    tot = 0.0
    for L in tuple(scales) + (p.shape[-1],):
        pad = (-p.shape[-1]) % L
        pp = F.pad(p, (0, pad)).unflatten(-1, (-1, L))
        tt = F.pad(tgt, (0, pad)).unflatten(-1, (-1, L))
        tot = tot + (pp.cumsum(-1) - tt.cumsum(-1)).abs().sum((-1, -2))
    return tot


def recon_terms(dec, hdec, Z, b, db, obs, recon="bce", tol=3):
    """Returns (recon_b, recon_db, recon_obs) each [B]  (summed over time)."""
    lg = dec(Z)
    ob = hdec(Z)
    if recon == "bce":
        rb = F.binary_cross_entropy_with_logits(lg[..., 0], b, reduction="none").sum(-1)
        rd = F.binary_cross_entropy_with_logits(lg[..., 1], db, reduction="none").sum(-1)
    elif recon == "stmask":
        mb, md = st_mask(b, tol), st_mask(db, tol)
        rb = (F.binary_cross_entropy_with_logits(lg[..., 0], b, reduction="none") * mb).sum(-1)
        rd = (F.binary_cross_entropy_with_logits(lg[..., 1], db, reduction="none") * md).sum(-1)
    elif recon == "stpool":
        # official ShiftTolerantBCELoss: max-pool the LOGITS over 1+2*tol, crop, mask +-2*tol
        def stp(p, y):
            sp = F.max_pool1d(p.unsqueeze(1), 1 + 2 * tol, 1).squeeze(1)[..., tol:-tol or None]
            ct = y[..., 2 * tol:-2 * tol or None]
            spread2 = F.max_pool1d(y.unsqueeze(1), 1 + 4 * tol, 1).squeeze(1)
            look = ct + (1.0 - spread2)
            return (F.binary_cross_entropy_with_logits(sp, ct, reduction="none") * look).sum(-1)
        rb, rd = stp(lg[..., 0], b), stp(lg[..., 1], db)
    elif recon == "hybrid":
        # BCE keeps the decoder phase-sensitive (measured +38.4% vs Cramer's +1.3%);
        # Cramer supplies the far-field tempo gradient BCE lacks. CRAMER_LAM is annealed
        # to 0 by the caller so the CONVERGED objective is a strict Bernoulli ELBO.
        rb = (F.binary_cross_entropy_with_logits(lg[..., 0], b, reduction="none").sum(-1)
              + CRAMER_LAM * _cramer(torch.sigmoid(lg[..., 0]), _smooth_dirac(b)))
        rd = (F.binary_cross_entropy_with_logits(lg[..., 1], db, reduction="none").sum(-1)
              + CRAMER_LAM * _cramer(torch.sigmoid(lg[..., 1]), _smooth_dirac(db)))
    elif recon == "cramer":
        rb = CRAMER_LAM * _cramer(torch.sigmoid(lg[..., 0]), _smooth_dirac(b))
        rd = CRAMER_LAM * _cramer(torch.sigmoid(lg[..., 1]), _smooth_dirac(db))
    else:
        raise ValueError(recon)
    ro = F.binary_cross_entropy_with_logits(ob, obs, reduction="none").sum(-1).sum(-1)
    return rb, rd, ro


def kl_t_mc(dq, lq, sq, dp, lp, sp, z):
    return (torch.distributions.StudentT(dq, lq, sq).log_prob(z)
            - torch.distributions.StudentT(dp, lp, sp).log_prob(z))


def _stat_dev_sigma(sigma, a):
    return sigma / torch.sqrt((1.0 - a ** 2).clamp(min=1e-3))


# ------------------------------------------------------------------ unified ELBO
def elbo_run(model, D, dec, hdec, *, q_mode="amortized", prior_mode="trained",
             widths=None, free=None, temperature=0.3, beta=1.0, obs_w=1.0,
             sample=True, recon="bce", tol=3, idx=None, ctx_cache=None,
             ret_traj=False, enc_grad=False, tf_z=False):
    """One pass of the arm_ii objective with swappable q / p / decoders / recon."""
    h = D["h"]; b = D["b"]; db = D["db"]; obs = D["obs"]
    if idx is not None:
        h, b, db, obs = h[idx], b[idx], db[idx], obs[idx]
    Bn, T, _ = h.shape
    dev = h.device
    K = model.K

    if ctx_cache is not None:
        prior_ctx = ctx_cache["prior"] if idx is None else ctx_cache["prior"][idx]
        post_ctx = ctx_cache["post"] if idx is None else ctx_cache["post"][idx]
    elif enc_grad:
        # TRAINING mode: the amortized encoder trunk must receive gradient. The default
        # (enc_grad=False) keeps the original post-hoc-analysis behaviour bit-for-bit.
        prior_ctx = model.encode_prior(h)
        post_ctx = model.encode_posterior(h, b)
    else:
        with torch.no_grad():
            prior_ctx = model.encode_prior(h)
            post_ctx = model.encode_posterior(h, b)

    o_phi = D["phi"] if idx is None else D["phi"][idx]
    o_lt = D["lt"] if idx is None else D["lt"][idx]
    o_m = D["m"] if idx is None else D["m"][idx]

    def _oracle_zfeat(t):
        """TRUE z_t features (teacher forcing of the posterior recurrence)."""
        return model.z_features(F.one_hot(o_m, K).float(), o_phi[:, t], o_lt[:, t])

    dof_q = model.tempo_dof()
    a_lv = model.level_ar()
    rho_phys = math.exp(-PHYS["gamma_phase"])
    Pi_phys = torch.full((K, K), (1.0 - PHYS["meter_self"]) / (K - 1), device=dev)
    Pi_phys.fill_diagonal_(PHYS["meter_self"])
    logPi_phys = torch.log(Pi_phys)

    W = widths or {}
    if q_mode == "free" and idx is not None:
        free = {k: v[idx] for k, v in free.items()}   # all free params have leading dim = n_crops

    def wget(name, t, default):
        v = W.get(name, default)
        if torch.is_tensor(v) and v.dim() >= 2:
            return v[idx][:, t] if idx is not None and v.shape[0] != Bn else v[:, t]
        if torch.is_tensor(v) and v.dim() == 1 and v.shape[0] == Bn:
            return v
        return torch.as_tensor(v, dtype=torch.float32, device=dev).expand(Bn)

    def q_params(t, z_prev_feat):
        if q_mode == "amortized":
            return model.unpack(model.post_head(torch.cat([post_ctx[:, t], z_prev_feat], -1)))
        if q_mode == "pinned":
            gap = float(W.get("meter_gap", 10.0))
            qm = F.one_hot(o_m, K).float() * gap
            return (qm, o_phi[:, t] % TWO_PI, wget("rho", t, 0.999),
                    o_lt[:, t], wget("s_lv", t, 0.00125),
                    torch.zeros(Bn, device=dev), wget("s_dv", t, 1e-3))
        if q_mode == "free":
            return (free["mlog"], free["phi"][:, t] % TWO_PI,
                    torch.sigmoid(free["rho_raw"][:, t]) * (1.0 - 1e-4),
                    free["lv"][:, t], F.softplus(free["slv_raw"][:, t]) + 1e-3,
                    free["dv"][:, t], F.softplus(free["sdv_raw"][:, t]) + 1e-3)
        raise ValueError(q_mode)

    kl_m = h.new_zeros(Bn); kl_p = h.new_zeros(Bn)
    kl_lv = h.new_zeros(Bn); kl_dv = h.new_zeros(Bn)
    n_cross = h.new_zeros(Bn)
    z_feats = []
    q_rhos = []

    z0 = model.z0.unsqueeze(0).expand(Bn, -1)
    q_m, q_ph_mu, q_ph_rho, q_lv_mu, q_lv_s, q_dv_mu, q_dv_s = q_params(0, z0)
    p_m, p_ph_mu, p_ph_rho, p_lv_mu, p_lv_s, _a, _c = model.unpack(
        model.prior_init_head(prior_ctx.mean(1)))
    a0 = model.prior_dev_coef(prior_ctx[:, 0])
    p_dv_mu = torch.zeros_like(q_dv_mu)
    p_dv_s = _stat_dev_sigma(model.prior_dev_scale(prior_ctx[:, 0]), a0)
    dof_p = dof_q
    if prior_mode == "physical":
        p_m = torch.zeros(Bn, K, device=dev)
        p_ph_mu = torch.full((Bn,), math.pi, device=dev)
        p_ph_rho = torch.full((Bn,), 1e-6, device=dev)
        p_lv_mu = torch.full((Bn,), PHYS["init_level_mu"], device=dev)
        p_lv_s = torch.full((Bn,), PHYS["init_level_scale"], device=dev)
        p_dv_mu = torch.zeros(Bn, device=dev)
        p_dv_s = torch.full((Bn,), PHYS["dev_sigma"], device=dev)
        dof_p = torch.tensor(PHYS["t_dof"], device=dev)

    if sample:
        meter = gumbel_softmax(q_m, temperature)
        phi = sample_wrapped_cauchy(q_ph_mu, q_ph_rho)
        level = sample_student_t(dof_q, q_lv_mu, q_lv_s)
        dev_ = q_dv_mu + q_dv_s * torch.randn_like(q_dv_mu)
    else:
        meter = F.softmax(q_m / max(temperature, 1e-6), -1)
        phi, level, dev_ = q_ph_mu, q_lv_mu, q_dv_mu
    log_tempo = level + dev_

    kl_m = kl_m + kl_categorical(torch.log_softmax(q_m, -1), torch.log_softmax(p_m, -1))
    kl_p = kl_p + kl_wrapped_cauchy(q_ph_mu, q_ph_rho, p_ph_mu, p_ph_rho)
    kl_lv = kl_lv + kl_t_mc(dof_q, q_lv_mu, q_lv_s, dof_p, p_lv_mu, p_lv_s, level)
    kl_dv = kl_dv + kl_log_normal(q_dv_mu, q_dv_s, p_dv_mu, p_dv_s)
    n_cross = n_cross + 1.0

    z_feats.append(model.z_features(meter, phi, log_tempo))
    q_rhos.append(q_ph_rho)
    level_anchor = level
    meter_prev, phi_prev = meter, phi
    level_prev, dev_prev, log_tempo_prev = level, dev_, log_tempo

    for t in range(1, T):
        z_prev_feat = (_oracle_zfeat(t - 1) if tf_z
                       else model.z_features(meter_prev, phi_prev, log_tempo_prev))
        q_m, q_ph_mu, q_ph_rho, q_lv_mu, q_lv_s, q_dv_mu, q_dv_s = q_params(t, z_prev_feat)

        tempo_prev = torch.exp(log_tempo_prev.clamp(-12.0, 6.0))
        advance = phi_prev + tempo_prev
        cross = (advance >= TWO_PI).to(h.dtype)
        p_ph_mu = advance % TWO_PI

        if prior_mode == "trained":
            p_ph_rho = model.prior_phase_conc(prior_ctx[:, t])
            a = model.prior_dev_coef(prior_ctx[:, t])
            p_lv_mu = level_anchor + a_lv * (level_prev - level_anchor)
            p_lv_s = model.prior_level_scale(prior_ctx[:, t])
            p_dv_mu = a * dev_prev
            p_dv_s = model.prior_dev_scale(prior_ctx[:, t])
            dof_p = dof_q
        else:
            p_ph_rho = torch.full((Bn,), rho_phys, device=dev)
            p_lv_mu = level_prev
            p_lv_s = torch.full((Bn,), PHYS["t_scale"], device=dev)
            p_dv_mu = PHYS["dev_ar"] * dev_prev
            p_dv_s = torch.full((Bn,), PHYS["dev_sigma"], device=dev)
            dof_p = torch.tensor(PHYS["t_dof"], device=dev)

        if sample:
            phi = sample_wrapped_cauchy(q_ph_mu, q_ph_rho)
            level = sample_student_t(dof_q, q_lv_mu, q_lv_s)
            dev_ = q_dv_mu + q_dv_s * torch.randn_like(q_dv_mu)
            q_meter_draw = gumbel_softmax(q_m, temperature)
        else:
            phi, level, dev_ = q_ph_mu, q_lv_mu, q_dv_mu
            q_meter_draw = F.softmax(q_m / max(temperature, 1e-6), -1)
        log_tempo = level + dev_
        meter = torch.where(cross.unsqueeze(-1) > 0.5, q_meter_draw, meter_prev)

        if prior_mode == "trained":
            log_pi_p = model.meter_prior_logp(meter_prev, phi, phi_prev, prior_ctx[:, t])
        else:
            log_pi_p = torch.log(meter_prev @ Pi_phys + 1e-9)

        kl_p = kl_p + kl_wrapped_cauchy(q_ph_mu, q_ph_rho, p_ph_mu, p_ph_rho)
        kl_lv = kl_lv + kl_t_mc(dof_q, q_lv_mu, q_lv_s, dof_p, p_lv_mu, p_lv_s, level)
        kl_dv = kl_dv + kl_log_normal(q_dv_mu, q_dv_s, p_dv_mu, p_dv_s)
        kl_m = kl_m + cross * kl_categorical(torch.log_softmax(q_m, -1), log_pi_p)
        n_cross = n_cross + cross

        z_feats.append(model.z_features(meter, phi, log_tempo))
        q_rhos.append(q_ph_rho)
        meter_prev, phi_prev = meter, phi
        level_prev, dev_prev, log_tempo_prev = level, dev_, log_tempo

    Z = torch.stack(z_feats, 1)
    rb, rd, ro = recon_terms(dec, hdec, Z, b, db, obs, recon=recon, tol=tol)
    L_kl = kl_m + kl_p + kl_lv + kl_dv
    loss = (rb + rd + obs_w * ro + beta * L_kl).mean()
    info = dict(loss=float(loss), recon_beat=float(rb.mean()), recon_db=float(rd.mean()),
                recon_obs=float(ro.mean()), kl=float(L_kl.mean()),
                kl_phase=float(kl_p.mean()), kl_level=float(kl_lv.mean()),
                kl_dev=float(kl_dv.mean()), kl_meter=float(kl_m.mean()),
                n_cross=float(n_cross.mean()),
                q_rho=float(torch.stack(q_rhos, 1).mean()))
    info["q_rho_t"] = torch.stack(q_rhos, 1)     # [B,T] WITH grad (additive; float keys unchanged)
    if ret_traj:
        return loss, info, Z
    return loss, info


# ===========================================================================================
# VECTORISED ELBO for a PINNED / FREE q.
# Key structural fact: with a non-amortised q the posterior means are free parameters, so
# z_t ~ q_t depends on NOTHING earlier -- the only sequential couplings are (a) the PRIOR mean
# phi_{t-1}+exp(lt_{t-1}) and (b) the meter carry between bar crossings. (a) is a shift of the
# already-drawn samples; (b) is a "value at the most recent crossing" gather (cummax of crossing
# indices). So the entire 256-step loop collapses to batched ops -- identical maths, ~100x faster.
# Verified against elbo_run (loop version) in verify_vec.py.
# ===========================================================================================
def elbo_vec(model, D, dec, hdec, *, widths=None, free=None, prior_mode="trained",
             temperature=0.3, beta=1.0, obs_w=1.0, recon="bce", tol=3, idx=None,
             prior_ctx=None, ret_traj=False):
    h = D["h"]; b = D["b"]; db = D["db"]; obs = D["obs"]
    o_phi, o_lt, o_m = D["phi"], D["lt"], D["m"]
    if idx is not None:
        h, b, db, obs, o_phi, o_lt, o_m = (h[idx], b[idx], db[idx], obs[idx],
                                           o_phi[idx], o_lt[idx], o_m[idx])
        if free is not None:
            free = {k: v[idx] for k, v in free.items()}
    B, T, _ = h.shape
    dv = h.device
    K = model.K
    ctx = model.encode_prior(h) if prior_ctx is None else (
        prior_ctx if idx is None else prior_ctx[idx])

    # ---- q parameters, all [B,T] (meter logits [B,K]) ----
    if free is None:
        W = widths or {}
        def ex(v):
            v = torch.as_tensor(v, dtype=torch.float32, device=dv) if not torch.is_tensor(v) else v
            return v.expand(B, T) if v.dim() == 0 else (v[idx] if (idx is not None and v.shape[0] != B) else v)
        q_ph_mu = o_phi % TWO_PI
        q_ph_rho = ex(W.get("rho", 0.999)).clamp(1e-6, 1 - 1e-4)
        q_lv_mu = o_lt
        q_lv_s = ex(W.get("s_lv", 0.00125))
        q_dv_mu = torch.zeros(B, T, device=dv)
        q_dv_s = ex(W.get("s_dv", 1e-3))
        q_m = F.one_hot(o_m, K).float() * float(W.get("meter_gap", 10.0))
    else:
        q_ph_mu = free["phi"] % TWO_PI
        q_ph_rho = (torch.sigmoid(free["rho_raw"]) * (1 - 1e-4)).clamp(1e-6, 1 - 1e-4)
        q_lv_mu = free["lv"]; q_lv_s = F.softplus(free["slv_raw"]) + 1e-3
        q_dv_mu = free["dv"]; q_dv_s = F.softplus(free["sdv_raw"]) + 1e-3
        q_m = free["mlog"]

    # ---- draw the whole trajectory at once ----
    dof = model.tempo_dof()
    gam = -torch.log(q_ph_rho)
    u = torch.rand(B, T, device=dv)
    phi = (q_ph_mu + gam * torch.tan(math.pi * (u - 0.5))) % TWO_PI
    level = torch.distributions.StudentT(dof, q_lv_mu, q_lv_s).rsample()
    dev_ = q_dv_mu + q_dv_s * torch.randn(B, T, device=dv)
    lt = level + dev_

    # ---- meter: fresh gumbel draw per frame, carried between bar crossings ----
    draws = gumbel_softmax(q_m.unsqueeze(1).expand(B, T, K), temperature)      # [B,T,K]
    adv = phi[:, :-1] + torch.exp(lt[:, :-1].clamp(-12.0, 6.0))
    cross = torch.zeros(B, T, device=dv)
    cross[:, 1:] = (adv >= TWO_PI).float()
    cross[:, 0] = 1.0
    ar = torch.arange(T, device=dv).unsqueeze(0).expand(B, T)
    last = torch.cummax(torch.where(cross > 0.5, ar, torch.zeros_like(ar)), dim=1).values
    meter = torch.gather(draws, 1, last.unsqueeze(-1).expand(B, T, K))
    meter_prev = torch.cat([meter[:, :1], meter[:, :-1]], 1)

    # ---- prior parameters, all [B,T] ----
    p_ph_mu = torch.empty(B, T, device=dv); p_lv_mu = torch.empty(B, T, device=dv)
    p_dv_mu = torch.empty(B, T, device=dv)
    p_ph_mu[:, 1:] = adv % TWO_PI
    anchor = level[:, :1]
    pinit = model.unpack(model.prior_init_head(ctx.mean(1)))
    a0 = model.prior_dev_coef(ctx[:, 0])
    if prior_mode == "trained":
        p_ph_rho = model.prior_phase_conc(ctx)
        a = model.prior_dev_coef(ctx)
        p_lv_s = model.prior_level_scale(ctx)
        p_dv_s = model.prior_dev_scale(ctx)
        a_lv = model.level_ar()
        p_lv_mu[:, 1:] = anchor + a_lv * (level[:, :-1] - anchor)
        p_dv_mu[:, 1:] = a[:, 1:] * dev_[:, :-1]
        p_m_log = torch.log_softmax(pinit[0], -1)
        dof_p = dof
        p_ph_rho = p_ph_rho.clone(); p_lv_s = p_lv_s.clone(); p_dv_s = p_dv_s.clone()
        p_ph_rho[:, 0] = pinit[2]; p_lv_mu[:, 0] = pinit[3]; p_lv_s[:, 0] = pinit[4]
        p_ph_mu[:, 0] = pinit[1]
        p_dv_mu[:, 0] = 0.0
        p_dv_s = p_dv_s.clone(); p_dv_s[:, 0] = _stat_dev_sigma(model.prior_dev_scale(ctx[:, 0]), a0)
    else:
        p_ph_rho = torch.full((B, T), math.exp(-PHYS["gamma_phase"]), device=dv)
        p_lv_s = torch.full((B, T), PHYS["t_scale"], device=dv)
        p_dv_s = torch.full((B, T), PHYS["dev_sigma"], device=dv)
        p_lv_mu[:, 1:] = level[:, :-1]
        p_dv_mu[:, 1:] = PHYS["dev_ar"] * dev_[:, :-1]
        p_ph_mu[:, 0] = math.pi; p_ph_rho = p_ph_rho.clone(); p_ph_rho[:, 0] = 1e-6
        p_lv_mu[:, 0] = PHYS["init_level_mu"]
        p_lv_s = p_lv_s.clone(); p_lv_s[:, 0] = PHYS["init_level_scale"]
        p_dv_mu[:, 0] = 0.0
        p_m_log = torch.full((B, K), -math.log(K), device=dv)
        dof_p = torch.tensor(PHYS["t_dof"], device=dv)

    kl_p = kl_wrapped_cauchy(q_ph_mu, q_ph_rho, p_ph_mu, p_ph_rho).sum(1)
    kl_lv = kl_t_mc(dof, q_lv_mu, q_lv_s, dof_p, p_lv_mu, p_lv_s, level).sum(1)
    kl_dv = kl_log_normal(q_dv_mu, q_dv_s, p_dv_mu, p_dv_s).sum(1)
    lq_m = torch.log_softmax(q_m, -1)
    if prior_mode == "trained":
        lp_t = model.meter_prior_logp(
            meter_prev[:, 1:].reshape(-1, K), phi[:, 1:].reshape(-1),
            phi[:, :-1].reshape(-1), ctx[:, 1:].reshape(-1, ctx.shape[-1])).reshape(B, T - 1, K)
    else:
        Pi = torch.full((K, K), (1 - PHYS["meter_self"]) / (K - 1), device=dv)
        Pi.fill_diagonal_(PHYS["meter_self"])
        lp_t = torch.log(meter_prev[:, 1:] @ Pi + 1e-9)
    kl_m = (kl_categorical(lq_m, p_m_log)
            + (cross[:, 1:] * kl_categorical(lq_m.unsqueeze(1).expand(B, T - 1, K), lp_t)).sum(1))

    Z = torch.cat([torch.cos(phi).unsqueeze(-1), torch.sin(phi).unsqueeze(-1),
                   lt.clamp(-12.0, 6.0).unsqueeze(-1), meter], -1)
    rb, rd, ro = recon_terms(dec, hdec, Z, b, db, obs, recon=recon, tol=tol)
    L_kl = kl_m + kl_p + kl_lv + kl_dv
    loss = (rb + rd + obs_w * ro + beta * L_kl).mean()
    info = dict(loss=float(loss), recon_beat=float(rb.mean()), recon_db=float(rd.mean()),
                recon_obs=float(ro.mean()), kl=float(L_kl.mean()), kl_phase=float(kl_p.mean()),
                kl_level=float(kl_lv.mean()), kl_dev=float(kl_dv.mean()),
                kl_meter=float(kl_m.mean()), n_cross=float(cross.sum(1).mean()))
    return (loss, info, Z) if ret_traj else (loss, info)
