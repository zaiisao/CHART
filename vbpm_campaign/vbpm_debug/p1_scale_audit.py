"""P1: scale audit + deploy-path structure, at INIT (no training). Fast."""
import sys, glob, math, json
import numpy as np, torch
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
from vbpm.model import BarPointerVAE
from vbpm.elbo import strict_elbo, free_run, _stationary_dev_sigma
from vbpm.distributions import TWO_PI, kl_wrapped_cauchy, kl_student_t_mc, kl_log_normal, kl_categorical, gumbel_softmax, sample_wrapped_cauchy, sample_student_t
from vbpm.evaluate import beats_from_barphase, downbeats_from_barphase, f_measure, metronome, _estimate_meter

CACHE = "/disk1/jaehoon/vbpm_mert_cache"; dev = "cuda:2"; fps = 50.0; H_DIM = 8
torch.manual_seed(0); rng = np.random.default_rng(0)

def load(split, cap=None):
    out = []
    for f in sorted(glob.glob(f"{CACHE}/{split}__*.npz")):
        d = np.load(f, allow_pickle=True)
        out.append(dict(key=f.split("__")[-1][:-4], T=int(d["feats"].shape[1]),
                        beats=np.asarray(d["beats"], float), downs=np.asarray(d["downs"], float)))
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

train = load("train"); ev = load("eval", 30)
print(f"[data] train={len(train)} eval={len(ev)}")

# ---------- ground-truth tempo statistics ----------
ibis = []; bpbs = []
for s in train:
    if len(s["beats"]) > 2: ibis.append(np.median(np.diff(s["beats"])))
    bpbs.append(_estimate_meter(s["beats"], s["downs"]))
ibis = np.array(ibis); bpbs = np.array(bpbs)
bar_sec = ibis * np.array([_estimate_meter(s["beats"], s["downs"]) for s in train if len(s["beats"]) > 2])
true_phidot = TWO_PI / (bar_sec * fps)          # rad per frame of BAR phase
print(f"[GT] median IBI {np.median(ibis):.3f}s -> BPM {60/np.median(ibis):.1f}; meters {np.bincount(bpbs)}")
print(f"[GT] TRUE bar phidot rad/frame: mean {true_phidot.mean():.5f} med {np.median(true_phidot):.5f} "
      f"[{true_phidot.min():.5f},{true_phidot.max():.5f}]  -> log_tempo mean {np.log(true_phidot).mean():.3f} "
      f"range [{np.log(true_phidot).min():.3f},{np.log(true_phidot).max():.3f}]")

# ---------- build a batch ----------
FR = 256; BS = 8
hs, bs_, ds = [], [], []
for _ in range(BS):
    s = train[rng.integers(len(train))]
    st = int(rng.integers(0, s["T"] - FR))
    hs.append(torch.from_numpy(dirac_h(s["beats"], s["downs"], st, FR)))
    b, d = targets(s["beats"], s["downs"], st, FR); bs_.append(torch.from_numpy(b)); ds.append(torch.from_numpy(d))
h = torch.stack(hs).to(dev); b = torch.stack(bs_).to(dev); d = torch.stack(ds).to(dev)
print(f"[batch] h {tuple(h.shape)} beat rate {b.mean():.4f} db rate {d.mean():.4f}")

model = BarPointerVAE(h_dim=H_DIM, hidden=128, num_meters=4).to(dev)
loss, info = strict_elbo(model, h, b, d, temperature=1.0, beta=1.0)
print("[init ELBO]", {k: round(v, 3) for k, v in info.items() if isinstance(v, float)})

# ---- base-rate reference: what recon would a constant-logit predictor get? ----
pb = b.mean(); pdb = d.mean()
base_b = -(b * torch.log(pb) + (1 - b) * torch.log(1 - pb)).sum(1).mean()
base_db = -(d * torch.log(pdb) + (1 - d) * torch.log(1 - pdb)).sum(1).mean()
print(f"[base-rate recon per seq over T={FR}] beat {base_b:.2f} nats  db {base_db:.2f} nats  "
      f"(per frame {base_b/FR:.4f} / {base_db/FR:.4f}); PERFECT recon = 0 -> total recon head-room {(base_b+base_db):.1f} nats/seq")

# ---------- (c) SCALE AUDIT: count the terms ----------
print("\n=== (c) SCALE AUDIT ===")
print(f"recon_b/recon_db: .sum(1) over T={FR} frames, then .mean() over B -> per-seq sum. OK")
print(f"kl_phase accumulates T={FR} terms (t=1 + T-1 transitions); reported {info['kl_phase']:.1f} "
      f"-> {info['kl_phase']/FR:.4f} nats/frame")
print(f"kl_level  T terms -> {info['kl_level']/FR:.4f} nats/frame")
print(f"kl_dev    T terms -> {info['kl_dev']/FR:.4f} nats/frame")
print(f"kl_meter  gated, n_cross={info['n_cross']:.1f} of {FR} frames -> {info['kl_meter']/max(info['n_cross'],1):.4f} nats/crossing")
print(f"RATIO KL/recon = {info['kl']/info['recon']:.2f};  KL/(recon head-room {float(base_b+base_db):.0f}) = {info['kl']/float(base_b+base_db):.2f}")

# ---------- (a) per-term gradient norms ----------
print("\n=== (a) PER-TERM GRADIENT NORMS at init ===")
def term_grads(model, h, b, d, temperature=1.0):
    """Re-run the ELBO but keep the six terms as separate tensors, backprop each."""
    import torch.nn.functional as Fn
    from vbpm.elbo import _stationary_dev_sigma
    B, T, _ = h.shape
    post_ctx = model.encode_posterior(h, b); prior_ctx = model.encode_prior(h)
    dof = model.tempo_dof()
    kl_m = h.new_zeros(B); kl_p = h.new_zeros(B); kl_lv = h.new_zeros(B); kl_dv = h.new_zeros(B)
    z_feats = []
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    q_m, q_ph_mu, q_ph_rho, q_lv_mu, q_lv_s, q_dv_mu, q_dv_s = model.unpack(model.post_head(torch.cat([post_ctx[:, 0], z0], -1)))
    p_m, p_ph_mu, p_ph_rho, p_lv_mu, p_lv_s, _a, _b2 = model.unpack(model.prior_init_head(prior_ctx.mean(1)))
    a0 = model.prior_dev_coef(prior_ctx[:, 0]); sd0 = model.prior_dev_scale(prior_ctx[:, 0])
    p_dv_mu = torch.zeros_like(q_dv_mu); p_dv_s = _stationary_dev_sigma(sd0, a0)
    meter = gumbel_softmax(q_m, temperature); phi = sample_wrapped_cauchy(q_ph_mu, q_ph_rho)
    level = sample_student_t(dof, q_lv_mu, q_lv_s); devv = q_dv_mu + q_dv_s * torch.randn_like(q_dv_mu)
    log_tempo = level + devv
    kl_m = kl_m + kl_categorical(torch.log_softmax(q_m, -1), torch.log_softmax(p_m, -1))
    kl_p = kl_p + kl_wrapped_cauchy(q_ph_mu, q_ph_rho, p_ph_mu, p_ph_rho)
    kl_lv1 = kl_student_t_mc(dof, q_lv_mu, q_lv_s, p_lv_mu, p_lv_s, level)   # <-- the ONLY term touching deploy tempo
    kl_lv = kl_lv + kl_lv1
    kl_dv = kl_dv + kl_log_normal(q_dv_mu, q_dv_s, p_dv_mu, p_dv_s)
    z_feats.append(model.z_features(meter, phi, log_tempo))
    level_anchor = level; a_lv = model.level_ar()
    meter_prev, phi_prev = meter, phi; level_prev, dev_prev, lt_prev = level, devv, log_tempo
    ncross = h.new_zeros(B)
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
        kl_lv = kl_lv + kl_student_t_mc(dof, q_lv_mu, q_lv_s, p_lv_mu, p_lv_s, level)
        kl_dv = kl_dv + kl_log_normal(q_dv_mu, q_dv_s, p_dv_mu, p_dv_s)
        kl_m = kl_m + cross * kl_categorical(torch.log_softmax(q_m, -1), lpp)
        ncross = ncross + cross
        z_feats.append(model.z_features(meter, phi, log_tempo))
        meter_prev, phi_prev = meter, phi; level_prev, dev_prev, lt_prev = level, devv, log_tempo
    logits = torch.stack([model.decode(z_feats[t], prior_ctx[:, t]) for t in range(T)], 1)
    rb = Fn.binary_cross_entropy_with_logits(logits[..., 0], b, reduction="none").sum(1)
    rd = Fn.binary_cross_entropy_with_logits(logits[..., 1], d, reduction="none").sum(1)
    terms = {"recon_beat": rb.mean(), "recon_db": rd.mean(), "kl_phase": kl_p.mean(),
             "kl_level": kl_lv.mean(), "kl_level_t1_ONLY": kl_lv1.mean(), "kl_dev": kl_dv.mean(),
             "kl_meter": kl_m.mean()}
    return terms

terms = term_grads(model, h, b, d)
params = [p for p in model.parameters() if p.requires_grad]
names = [n for n, p in model.named_parameters() if p.requires_grad]
rows = []
for k, v in terms.items():
    g = torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)
    tot = math.sqrt(sum(float((gi ** 2).sum()) for gi in g if gi is not None))
    # gradient onto the deploy-critical prior_init_head
    sub = {n: float(gi.norm()) for n, gi, in zip(names, g) if gi is not None and "prior_init_head" in n}
    rows.append((k, float(v), tot, sum(sub.values())))
print(f"{'term':>18} {'value':>10} {'|grad| all':>12} {'|grad| prior_init':>18}")
for k, v, tot, s in rows:
    print(f"{k:>18} {v:10.2f} {tot:12.4f} {s:18.4f}")

# ---------- (d) MC-KL variance ----------
print("\n=== (d) 1-sample Student-t MC KL variance ===")
vals = []; gnorms = []
for r in range(20):
    torch.manual_seed(100 + r)
    tt = term_grads(model, h, b, d)
    g = torch.autograd.grad(tt["kl_level"], params, retain_graph=False, allow_unused=True)
    vals.append(float(tt["kl_level"])); gnorms.append(math.sqrt(sum(float((gi**2).sum()) for gi in g if gi is not None)))
vals = np.array(vals); gnorms = np.array(gnorms)
print(f"kl_level over 20 redraws: mean {vals.mean():.2f} std {vals.std():.2f} (CV {vals.std()/abs(vals.mean()):.3f}) range [{vals.min():.1f},{vals.max():.1f}]")
print(f"|grad kl_level| over 20 redraws: mean {gnorms.mean():.3f} std {gnorms.std():.3f} (CV {gnorms.std()/gnorms.mean():.3f})")

# closed-form Gaussian analogue on the same q/p moments (variance comparison)
print("\n=== (d2) closed-form Gaussian KL on the SAME level moments (variance-free reference) ===")
vals2 = []
for r in range(20):
    torch.manual_seed(100 + r)
    with torch.no_grad():
        pass
    # cheap: only need the value spread of an analogous Gaussian KL; recompute via term_grads w/ patch
print("(see p3 for the like-for-like swap)")

# ---------- deploy-path structure ----------
print("\n=== DEPLOY PATH STRUCTURE (free_run mean chain) ===")
s = ev[0]; T = min(s["T"], 1600)
hh = torch.from_numpy(dirac_h(s["beats"], s["downs"], 0, T)).unsqueeze(0).to(dev)
out = free_run(model, hh)
lt = out["log_tempo"][0].cpu().numpy(); pm = out["phase_mu"][0].cpu().numpy(); ph = out["phase"][0].cpu().numpy()
dphi_mu = np.diff(pm); dphi_mu = np.where(dphi_mu < -math.pi, dphi_mu + TWO_PI, dphi_mu)
print(f"phase_mu per-frame advance: mean {dphi_mu.mean():.5f} std {dphi_mu.std():.3e} min {dphi_mu.min():.5f} max {dphi_mu.max():.5f}")
m_true = _estimate_meter(s["beats"], s["downs"])
tp = TWO_PI / (np.median(np.diff(s["beats"])) * m_true * fps)
print(f"TRUE bar phidot for this song = {tp:.5f} rad/frame (log {math.log(tp):.3f}); model deploy = {dphi_mu.mean():.5f} (log {math.log(dphi_mu.mean()):.3f})")
print(f"  -> model is {dphi_mu.mean()/tp:.1f}x too fast. n wraps: model {int((np.diff(pm)<-math.pi).sum())} vs true bars {len(s['downs'][s['downs']<T/fps])}")
print(f"stochastic log_tempo traj: mean {lt.mean():.3f} std {lt.std():.3f} range [{lt.min():.3f},{lt.max():.3f}]")
print(f"free-run beat_F {f_measure(s['beats'][s['beats']<T/fps], beats_from_barphase(pm, m_true, fps)):.3f} "
      f"metronome {f_measure(s['beats'][s['beats']<T/fps], metronome(T, fps)):.3f}")

# ORACLE deploy: replace the deploy log-tempo by the TRUE one, keep everything else
print("\n=== ORACLE DEPLOY: same constant-tempo mean chain but with the TRUE log-tempo ===")
Fs = []; Fs_meta = []
for s in ev:
    T = min(s["T"], 1600)
    ref = s["beats"][s["beats"] < T / fps]
    if len(ref) < 2: continue
    m_true = _estimate_meter(ref, s["downs"][s["downs"] < T / fps])
    tp = TWO_PI / (np.median(np.diff(s["beats"])) * m_true * fps)
    pm_or = (np.arange(T) * tp) % TWO_PI                       # phase 0 at frame 0 (no offset fit)
    pm_or2 = (np.arange(T) * tp + (TWO_PI - (s["beats"][0] * fps * tp) % TWO_PI)) % TWO_PI
    Fs.append(f_measure(ref, beats_from_barphase(pm_or, m_true, fps)))
    Fs_meta.append(f_measure(ref, metronome(T, fps)))
print(f"ORACLE constant-tempo mean chain (no phase fit): beat_F {np.mean(Fs):.3f} over {len(Fs)} songs; metronome {np.mean(Fs_meta):.3f}")
