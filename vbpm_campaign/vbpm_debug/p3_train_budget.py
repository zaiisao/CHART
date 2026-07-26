"""P3: Dirac training with a term-by-term ELBO budget + PER-TERM GRADIENT NORMS,
posterior-vs-truth traces, and a deploy attribution (tempo error vs phase-offset error).

usage: p3_train_budget.py <tag> <device> <beta_mode>   beta_mode in {warm, zero, one}
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

TAG, DEV, BETA_MODE = sys.argv[1], sys.argv[2], sys.argv[3]
CACHE = "/disk1/jaehoon/vbpm_mert_cache"; fps = 50.0; H_DIM = 8
STEPS, WARM, BS, FR = 600, 300, 16, 256
torch.manual_seed(0); rng = np.random.default_rng(0)

def load(split, cap=None):
    out = []
    for f in sorted(glob.glob(f"{CACHE}/{split}__*.npz")):
        d = np.load(f, allow_pickle=True)
        out.append(dict(T=int(d["feats"].shape[1]), beats=np.asarray(d["beats"], float),
                        downs=np.asarray(d["downs"], float)))
    return out[:cap] if cap else out

def dirac_h(beats, downs, start, n):
    h = np.random.randn(n, H_DIM).astype(np.float32) * 0.01
    for t in beats:
        i = int(round(t * fps)) - start
        if 0 <= i < n: h[i, 0] += 1.0
    for t in downs:
        i = int(round(t * fps)) - start
        if 0 <= i < n: h[i, 1] += 1.0
    return h

def targets(beats, downs, start, n):
    b = np.zeros(n, np.float32); db = np.zeros(n, np.float32)
    for t in beats:
        i = int(round(t * fps)) - start
        if 0 <= i < n: b[i] = 1.0
    for t in downs:
        i = int(round(t * fps)) - start
        if 0 <= i < n: db[i] = 1.0
    return b, db

def true_barphase(beats, downs, start, n):
    """GT bar phase [0,2pi) at each frame (piecewise linear between downbeats)."""
    tt = (np.arange(n) + start) / fps
    if len(downs) < 2: return None
    ph = np.interp(tt, downs, np.arange(len(downs)) * TWO_PI, left=np.nan, right=np.nan)
    return ph % TWO_PI, np.isfinite(ph)

train = load("train"); ev = load("eval", 30)

# ---------- ELBO re-implementation returning SEPARATE term tensors (identical math) ----------
def elbo_terms(model, h, b, d, temperature):
    B, T, _ = h.shape
    post_ctx = model.encode_posterior(h, b); prior_ctx = model.encode_prior(h)
    dof = model.tempo_dof()
    kl_m = h.new_zeros(B); kl_p = h.new_zeros(B); kl_lv = h.new_zeros(B); kl_dv = h.new_zeros(B)
    z_feats = []; qph = []; qlt = []; kl_lv_terms = []
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    q_m, q_ph_mu, q_ph_rho, q_lv_mu, q_lv_s, q_dv_mu, q_dv_s = model.unpack(model.post_head(torch.cat([post_ctx[:, 0], z0], -1)))
    p_m, p_ph_mu0, p_ph_rho, p_lv_mu0, p_lv_s, _a, _b2 = model.unpack(model.prior_init_head(prior_ctx.mean(1)))
    a0 = model.prior_dev_coef(prior_ctx[:, 0]); sd0 = model.prior_dev_scale(prior_ctx[:, 0])
    p_dv_s = _stationary_dev_sigma(sd0, a0)
    meter = gumbel_softmax(q_m, temperature); phi = sample_wrapped_cauchy(q_ph_mu, q_ph_rho)
    level = sample_student_t(dof, q_lv_mu, q_lv_s); devv = q_dv_mu + q_dv_s * torch.randn_like(q_dv_mu)
    log_tempo = level + devv
    kl_m = kl_m + kl_categorical(torch.log_softmax(q_m, -1), torch.log_softmax(p_m, -1))
    kl_p = kl_p + kl_wrapped_cauchy(q_ph_mu, q_ph_rho, p_ph_mu0, p_ph_rho)
    kl_lv1 = kl_student_t_mc(dof, q_lv_mu, q_lv_s, p_lv_mu0, p_lv_s, level)
    kl_lv = kl_lv + kl_lv1
    kl_dv = kl_dv + kl_log_normal(q_dv_mu, q_dv_s, torch.zeros_like(q_dv_mu), p_dv_s)
    z_feats.append(model.z_features(meter, phi, log_tempo)); qph.append(q_ph_mu); qlt.append(log_tempo)
    level_anchor = level; a_lv = model.level_ar()
    meter_prev, phi_prev, level_prev, dev_prev, lt_prev = meter, phi, level, devv, log_tempo
    ncross = h.new_zeros(B); rho_q_acc = []; rho_p_acc = []
    for t in range(1, T):
        zpf = model.z_features(meter_prev, phi_prev, lt_prev)
        q_m, q_ph_mu, q_ph_rho, q_lv_mu, q_lv_s, q_dv_mu, q_dv_s = model.unpack(model.post_head(torch.cat([post_ctx[:, t], zpf], -1)))
        tempo_prev = torch.exp(lt_prev.clamp(-12., 6.)); advance = phi_prev + tempo_prev
        cross = (advance >= TWO_PI).to(h.dtype); p_ph_mu = advance % TWO_PI
        p_ph_rho = model.prior_phase_conc(prior_ctx[:, t]); a = model.prior_dev_coef(prior_ctx[:, t])
        p_lv_mu = level_anchor + a_lv * (level_prev - level_anchor); p_lv_s = model.prior_level_scale(prior_ctx[:, t])
        p_dv_mu = a * dev_prev; p_dv_s = model.prior_dev_scale(prior_ctx[:, t])
        phi = sample_wrapped_cauchy(q_ph_mu, q_ph_rho); level = sample_student_t(dof, q_lv_mu, q_lv_s)
        devv = q_dv_mu + q_dv_s * torch.randn_like(q_dv_mu); log_tempo = level + devv
        qd = gumbel_softmax(q_m, temperature); meter = torch.where(cross.unsqueeze(-1) > .5, qd, meter_prev)
        lpp = model.meter_prior_logp(meter_prev, phi, phi_prev, prior_ctx[:, t])
        kl_p = kl_p + kl_wrapped_cauchy(q_ph_mu, q_ph_rho, p_ph_mu, p_ph_rho)
        klv_t = kl_student_t_mc(dof, q_lv_mu, q_lv_s, p_lv_mu, p_lv_s, level)
        kl_lv = kl_lv + klv_t; kl_lv_terms.append(klv_t.detach())
        kl_dv = kl_dv + kl_log_normal(q_dv_mu, q_dv_s, p_dv_mu, p_dv_s)
        kl_m = kl_m + cross * kl_categorical(torch.log_softmax(q_m, -1), lpp)
        ncross = ncross + cross
        z_feats.append(model.z_features(meter, phi, log_tempo)); qph.append(q_ph_mu); qlt.append(log_tempo)
        rho_q_acc.append(q_ph_rho.detach()); rho_p_acc.append(p_ph_rho.detach())
        meter_prev, phi_prev, level_prev, dev_prev, lt_prev = meter, phi, level, devv, log_tempo
    logits = torch.stack([model.decode(z_feats[t], prior_ctx[:, t]) for t in range(T)], 1)
    rb = Fn.binary_cross_entropy_with_logits(logits[..., 0], b, reduction="none").sum(1)
    rd = Fn.binary_cross_entropy_with_logits(logits[..., 1], d, reduction="none").sum(1)
    terms = {"recon_beat": rb.mean(), "recon_db": rd.mean(), "kl_phase": kl_p.mean(),
             "kl_level": kl_lv.mean(), "kl_dev": kl_dv.mean(), "kl_meter": kl_m.mean()}
    diag = {"kl_level_t1": kl_lv1.mean().detach(), "n_cross": ncross.mean().detach(),
            "q_ph_mu": torch.stack(qph, 1).detach(), "log_tempo": torch.stack(qlt, 1).detach(),
            "rho_q": torch.stack(rho_q_acc, 1).detach(), "rho_p": torch.stack(rho_p_acc, 1).detach(),
            "p_lv_mu0": p_lv_mu0.detach(), "p_ph_mu0": p_ph_mu0.detach(),
            "kl_lv_terms": torch.stack(kl_lv_terms, 1) if kl_lv_terms else None,
            "beat_logit": logits[..., 0].detach()}
    return terms, diag

model = BarPointerVAE(h_dim=H_DIM, hidden=128, num_meters=4).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
params = [p for p in model.parameters() if p.requires_grad]
pnames = [n for n, p in model.named_parameters() if p.requires_grad]

def gnorm(g, filt=None):
    return math.sqrt(sum(float((gi ** 2).sum()) for n, gi in zip(pnames, g)
                         if gi is not None and (filt is None or filt in n)))

@torch.no_grad()
def deploy_attrib(model, songs, max_frames=1600):
    """Free-run and ATTRIBUTE the failure: tempo error vs phase-offset error."""
    model.eval()
    acc = {"free_F": [], "ratio": [], "F_modeltempo_bestoff": [], "F_truetempo_modeloff": [],
           "F_truetempo_bestoff": [], "metro": []}
    offs = np.linspace(0, TWO_PI, 64, endpoint=False)
    for s in songs:
        T = min(s["T"], max_frames)
        h = torch.from_numpy(dirac_h(s["beats"], s["downs"], 0, T)).unsqueeze(0).to(DEV)
        out = free_run(model, h)
        pm = out["phase_mu"][0, :T].cpu().numpy()
        ref = s["beats"][s["beats"] < T / fps]; dref = s["downs"][s["downs"] < T / fps]
        if len(ref) < 2: continue
        m = _estimate_meter(ref, dref)
        dphi = np.diff(pm); dphi = np.where(dphi < -math.pi, dphi + TWO_PI, dphi)
        mtempo = float(np.mean(dphi)); moff = float(pm[0])
        ttempo = TWO_PI / (np.median(np.diff(s["beats"])) * m * fps)
        sc = lambda pd, o: f_measure(ref, beats_from_barphase((np.arange(T) * pd + o) % TWO_PI, m, fps))
        acc["free_F"].append(f_measure(ref, beats_from_barphase(pm, m, fps)))
        acc["ratio"].append(mtempo / ttempo)
        acc["F_modeltempo_bestoff"].append(max(sc(mtempo, o) for o in offs))
        acc["F_truetempo_modeloff"].append(sc(ttempo, moff))
        acc["F_truetempo_bestoff"].append(max(sc(ttempo, o) for o in offs))
        acc["metro"].append(f_measure(ref, metronome(T, fps)))
    model.train()
    return {k: float(np.mean(v)) for k, v in acc.items()}

print(f"=== {TAG} beta_mode={BETA_MODE} dev={DEV} ===", flush=True)
t0 = time.time()
for step in range(1, STEPS + 1):
    beta = {"warm": min(1.0, step / WARM), "zero": 0.0, "one": 1.0}[BETA_MODE]
    temp = 1.0 + (0.3 - 1.0) * min(step / STEPS, 1.0)
    hs, bs_, ds, gts = [], [], [], []
    for _ in range(BS):
        s = train[rng.integers(len(train))]
        st = int(rng.integers(0, s["T"] - FR))
        hs.append(torch.from_numpy(dirac_h(s["beats"], s["downs"], st, FR)))
        bb, dd = targets(s["beats"], s["downs"], st, FR); bs_.append(torch.from_numpy(bb)); ds.append(torch.from_numpy(dd))
        tb = true_barphase(s["beats"], s["downs"], st, FR)
        gts.append(tb[0] if tb is not None else np.full(FR, np.nan))
    h = torch.stack(hs).to(DEV); b = torch.stack(bs_).to(DEV); d = torch.stack(ds).to(DEV)
    gtph = np.stack(gts)

    terms, diag = elbo_terms(model, h, b, d, temp)
    loss = terms["recon_beat"] + terms["recon_db"] + beta * (terms["kl_phase"] + terms["kl_level"] + terms["kl_dev"] + terms["kl_meter"])
    if step % 50 == 0 or step == 1:
        # --- per-term gradient norms (separate backprops) ---
        gl = {}
        for k, v in terms.items():
            g = torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)
            gl[k] = (gnorm(g), gnorm(g, "prior_init_head"), gnorm(g, "decoder"), gnorm(g, "post_head"))
        # --- posterior vs truth ---
        qph = diag["q_ph_mu"].cpu().numpy(); ok = np.isfinite(gtph)
        if ok.sum() > 100:
            dphase = (qph - gtph + math.pi) % TWO_PI - math.pi
            circR = float(np.abs(np.mean(np.exp(1j * dphase[ok]))))   # 1 = perfectly locked (up to const offset)
        else:
            circR = float("nan")
        lt = diag["log_tempo"].cpu().numpy()
        print(f"[{TAG}] s{step:4d} b={beta:.2f} | recon_b {float(terms['recon_beat']):7.2f} (base 41.4) "
              f"recon_db {float(terms['recon_db']):7.2f} (base 13.5) | KL phi {float(terms['kl_phase']):7.2f} "
              f"lv {float(terms['kl_level']):7.2f} dv {float(terms['kl_dev']):6.2f} m {float(terms['kl_meter']):5.2f} "
              f"| ncross {float(diag['n_cross']):5.1f} lt {lt.mean():+.2f}+-{lt.std():.2f} "
              f"rho_q {float(diag['rho_q'].mean()):.3f} rho_p {float(diag['rho_p'].mean()):.3f} "
              f"p_lv_mu0 {float(diag['p_lv_mu0'].mean()):+.3f} circR_post {circR:.3f}", flush=True)
        print(f"        GRAD |all|/|prior_init|/|decoder|/|post_head|: " +
              "  ".join(f"{k}={gl[k][0]:.1f}/{gl[k][1]:.2f}/{gl[k][2]:.1f}/{gl[k][3]:.1f}" for k in gl), flush=True)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
    if step % 200 == 0 or step == STEPS:
        a = deploy_attrib(model, ev)
        print(f"   >>> [{TAG} s{step}] FREE-RUN beat_F={a['free_F']:.3f} | metro={a['metro']:.3f} | "
              f"tempo ratio model/true={a['ratio']:.2f} | modelTempo+bestOff={a['F_modeltempo_bestoff']:.3f} "
              f"trueTempo+modelOff={a['F_truetempo_modeloff']:.3f} trueTempo+bestOff={a['F_truetempo_bestoff']:.3f}", flush=True)
print(f"[{TAG}] done in {time.time()-t0:.0f}s", flush=True)
