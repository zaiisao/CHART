"""P4 (decisive): is the ELBO OPTIMUM in the right place?

Clamp the posterior to the ORACLE latents (true bar phase, true log-phidot, true meter),
train ONLY the decoder + prior heads on the SAME objective, and compare the term budget
and free-run F against the freely-trained model. If the oracle posterior gives a HIGHER
(worse) ELBO than the collapsed one, the objective is misspecified, not under-optimised.
"""
import sys, glob, math, time
import numpy as np, torch
import torch.nn.functional as Fn
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
from vbpm.model import BarPointerVAE
from vbpm.elbo import strict_elbo, free_run, _stationary_dev_sigma
from vbpm.distributions import (TWO_PI, gumbel_softmax, sample_wrapped_cauchy, sample_student_t,
                                kl_categorical, kl_wrapped_cauchy, kl_log_normal, kl_student_t_mc)
from vbpm.evaluate import beats_from_barphase, f_measure, metronome, _estimate_meter

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:2"
CACHE = "/disk1/jaehoon/vbpm_mert_cache"; fps = 50.0; H_DIM = 8
FR, BS, STEPS = 256, 16, 600
torch.manual_seed(0); rng = np.random.default_rng(0)

def load(split, cap=None):
    out = []
    for f in sorted(glob.glob(f"{CACHE}/{split}__*.npz")):
        d = np.load(f, allow_pickle=True)
        out.append(dict(T=int(d["feats"].shape[1]), beats=np.asarray(d["beats"], float),
                        downs=np.asarray(d["downs"], float)))
    return out[:cap] if cap else out
train = load("train"); ev = load("eval", 30)

def dirac_h(beats, downs, start, n):
    h = np.random.randn(n, H_DIM).astype(np.float32) * 0.01
    for t in beats:
        i = int(round(t * fps)) - start
        if 0 <= i < n: h[i, 0] += 1.0
    for t in downs:
        i = int(round(t * fps)) - start
        if 0 <= i < n: h[i, 1] += 1.0
    return h

def sample_oracle(s, st, n):
    """returns h, b, db, true bar phase, true log phidot, meter idx -- or None."""
    dd = s["downs"]; bb = s["beats"]
    if len(dd) < 3: return None
    tt = (np.arange(n) + st) / fps
    ph = np.interp(tt, dd, np.arange(len(dd)) * TWO_PI, left=np.nan, right=np.nan)
    if not np.isfinite(ph).all(): return None
    m = _estimate_meter(bb, dd)
    phid = np.gradient(ph) * 1.0                      # rad per frame (bar phase)
    b = np.zeros(n, np.float32); db = np.zeros(n, np.float32)
    for t in bb:
        i = int(round(t * fps)) - st
        if 0 <= i < n: b[i] = 1.0
    for t in dd:
        i = int(round(t * fps)) - st
        if 0 <= i < n: db[i] = 1.0
    return (dirac_h(bb, dd, st, n), b, db, (ph % TWO_PI).astype(np.float32),
            np.log(np.clip(phid, 1e-4, None)).astype(np.float32), m)

def batch(nb=BS):
    H, B, D, PH, LT, M = [], [], [], [], [], []
    tries = 0
    while len(H) < nb and tries < 300:
        tries += 1
        s = train[rng.integers(len(train))]
        if s["T"] <= FR + 2: continue
        st = int(rng.integers(0, s["T"] - FR))
        o = sample_oracle(s, st, FR)
        if o is None: continue
        H.append(o[0]); B.append(o[1]); D.append(o[2]); PH.append(o[3]); LT.append(o[4]); M.append(o[5])
    f = lambda x: torch.from_numpy(np.stack(x)).to(DEV)
    return f(H), f(B), f(D), f(PH), f(LT), torch.tensor(M, device=DEV)

# ---------------- ELBO with the posterior CLAMPED to the oracle ----------------
def elbo_oracle(model, h, b, d, ph_true, lt_true, m_true, rho_q, temperature=0.5, beta=1.0):
    B, T, _ = h.shape
    prior_ctx = model.encode_prior(h)
    dof = model.tempo_dof()
    K = model.K
    q_m_fixed = Fn.one_hot(torch.clamp(m_true - 1, 0, K - 1), K).float() * 8.0
    sig = torch.full((B,), 0.02, device=h.device)
    kl_p = h.new_zeros(B); kl_lv = h.new_zeros(B); kl_dv = h.new_zeros(B); kl_m = h.new_zeros(B)
    z_feats = []
    p_m, p_ph_mu, p_ph_rho, p_lv_mu0, p_lv_s, _a, _b2 = model.unpack(model.prior_init_head(prior_ctx.mean(1)))
    a0 = model.prior_dev_coef(prior_ctx[:, 0]); sd0 = model.prior_dev_scale(prior_ctx[:, 0])
    p_dv_s0 = _stationary_dev_sigma(sd0, a0)
    meter = torch.softmax(q_m_fixed, -1)
    phi = ph_true[:, 0]; level = lt_true[:, 0]; devv = torch.zeros_like(level); log_tempo = level
    kl_m = kl_m + kl_categorical(torch.log_softmax(q_m_fixed, -1), torch.log_softmax(p_m, -1))
    kl_p = kl_p + kl_wrapped_cauchy(phi, rho_q.expand(B), p_ph_mu, p_ph_rho)
    kl_lv = kl_lv + kl_student_t_mc(dof, level, sig, p_lv_mu0, p_lv_s, level)
    kl_dv = kl_dv + kl_log_normal(torch.zeros_like(level), sig, torch.zeros_like(level), p_dv_s0)
    z_feats.append(model.z_features(meter, phi, log_tempo))
    level_anchor = level; a_lv = model.level_ar()
    meter_prev, phi_prev, level_prev, dev_prev, lt_prev = meter, phi, level, devv, log_tempo
    ncross = h.new_zeros(B)
    for t in range(1, T):
        tempo_prev = torch.exp(lt_prev.clamp(-12., 6.)); advance = phi_prev + tempo_prev
        cross = (advance >= TWO_PI).to(h.dtype); p_ph_mu = advance % TWO_PI
        p_ph_rho = model.prior_phase_conc(prior_ctx[:, t]); a = model.prior_dev_coef(prior_ctx[:, t])
        p_lv_mu = level_anchor + a_lv * (level_prev - level_anchor); p_lv_s = model.prior_level_scale(prior_ctx[:, t])
        p_dv_mu = a * dev_prev; p_dv_s = model.prior_dev_scale(prior_ctx[:, t])
        phi = ph_true[:, t]; level = lt_true[:, t]; devv = torch.zeros_like(level); log_tempo = level
        meter = meter_prev
        lpp = model.meter_prior_logp(meter_prev, phi, phi_prev, prior_ctx[:, t])
        kl_p = kl_p + kl_wrapped_cauchy(phi, rho_q.expand(B), p_ph_mu, p_ph_rho)
        kl_lv = kl_lv + kl_student_t_mc(dof, level, sig, p_lv_mu, p_lv_s, level)
        kl_dv = kl_dv + kl_log_normal(torch.zeros_like(level), sig, p_dv_mu, p_dv_s)
        kl_m = kl_m + cross * kl_categorical(torch.log_softmax(q_m_fixed, -1), lpp)
        ncross = ncross + cross
        z_feats.append(model.z_features(meter, phi, log_tempo))
        meter_prev, phi_prev, level_prev, dev_prev, lt_prev = meter, phi, level, devv, log_tempo
    logits = torch.stack([model.decode(z_feats[t], prior_ctx[:, t]) for t in range(T)], 1)
    rb = Fn.binary_cross_entropy_with_logits(logits[..., 0], b, reduction="none").sum(1)
    rd = Fn.binary_cross_entropy_with_logits(logits[..., 1], d, reduction="none").sum(1)
    terms = {"recon_beat": rb.mean(), "recon_db": rd.mean(), "kl_phase": kl_p.mean(),
             "kl_level": kl_lv.mean(), "kl_dev": kl_dv.mean(), "kl_meter": kl_m.mean()}
    return terms, {"n_cross": float(ncross.mean()), "p_lv_mu0": p_lv_mu0.detach()}

model = BarPointerVAE(h_dim=H_DIM, hidden=128, num_meters=4).to(DEV)
rho_q = torch.tensor([0.99], device=DEV)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
print(f"=== P4 oracle-posterior ELBO (rho_q={float(rho_q):.3f}) ===", flush=True)
for step in range(1, STEPS + 1):
    h, b, d, ph, lt, m = batch()
    terms, dg = elbo_oracle(model, h, b, d, ph, lt, m, rho_q)
    loss = sum(terms.values())
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    if step % 100 == 0 or step == 1:
        print(f"[ORACLE] s{step:4d} loss {float(loss):8.2f} | rec_b {float(terms['recon_beat']):7.2f} "
              f"rec_db {float(terms['recon_db']):7.2f} | KL phi {float(terms['kl_phase']):8.2f} "
              f"lv {float(terms['kl_level']):8.2f} dv {float(terms['kl_dev']):7.2f} m {float(terms['kl_meter']):5.2f} "
              f"| ncross {dg['n_cross']:.1f} p_lv_mu0 {float(dg['p_lv_mu0'].mean()):+.3f} "
              f"(true lt {float(lt.mean()):+.3f})", flush=True)

# free-run with the oracle-trained priors
@torch.no_grad()
def fr(model):
    model.eval(); acc = {"F": [], "ratio": []}
    for s in ev:
        T = min(s["T"], 1600)
        hh = torch.from_numpy(dirac_h(s["beats"], s["downs"], 0, T)).unsqueeze(0).to(DEV)
        out = free_run(model, hh)
        pm = out["phase_mu"][0, :T].cpu().numpy()
        ref = s["beats"][s["beats"] < T / fps]; dref = s["downs"][s["downs"] < T / fps]
        if len(ref) < 2: continue
        mm = _estimate_meter(ref, dref)
        dp = np.diff(pm); dp = np.where(dp < -math.pi, dp + TWO_PI, dp)
        acc["F"].append(f_measure(ref, beats_from_barphase(pm, mm, fps)))
        acc["ratio"].append(float(np.mean(dp)) / (TWO_PI / (np.median(np.diff(s["beats"])) * mm * fps)))
    model.train(); return {k: float(np.mean(v)) for k, v in acc.items()}
print("[ORACLE] free-run after oracle-posterior training:", fr(model), flush=True)

# ---- how much recon can the decoder REALLY buy with perfect latents? (fit decoder alone) ----
print("\n=== decoder head-room on ORACLE z_features (no KL, decoder only) ===")
dec = torch.nn.Sequential(torch.nn.Linear(model.z_feat_dim, 128), torch.nn.Tanh(),
                          torch.nn.Linear(128, 2)).to(DEV)
o2 = torch.optim.AdamW(dec.parameters(), lr=3e-3)
for step in range(1, 1501):
    h, b, d, ph, lt, m = batch()
    mm = Fn.one_hot(torch.clamp(m - 1, 0, model.K - 1), model.K).float().unsqueeze(1).expand(-1, FR, -1)
    z = torch.cat([torch.cos(ph).unsqueeze(-1), torch.sin(ph).unsqueeze(-1), lt.unsqueeze(-1), mm], -1)
    lo = dec(z)
    rb = Fn.binary_cross_entropy_with_logits(lo[..., 0], b, reduction="none").sum(1).mean()
    rd = Fn.binary_cross_entropy_with_logits(lo[..., 1], d, reduction="none").sum(1).mean()
    l = rb + rd
    o2.zero_grad(); l.backward(); o2.step()
    if step % 300 == 0:
        print(f"  s{step} rec_b {float(rb):6.2f} (base 41.4)  rec_db {float(rd):6.2f} (base 13.5) "
              f"-> REALISED recon head-room {41.4-float(rb)+13.5-float(rd):6.2f} nats/seq", flush=True)
