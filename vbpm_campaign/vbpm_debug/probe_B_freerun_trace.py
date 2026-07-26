"""PROBE B: complete input/output trace of the DEPLOY path elbo.free_run on a TRAINED
Dirac model. Answers (a)-(f) of the brief. Nothing in vbpm/ is modified: the oracle-tempo
variant is a local *copy* of free_run with two scalars overridden.
"""
import sys, glob, math, json
import numpy as np, torch
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
from vbpm.model import BarPointerVAE
from vbpm.elbo import free_run, strict_elbo
from vbpm.distributions import (TWO_PI, gumbel_softmax, sample_wrapped_cauchy, sample_student_t)
from vbpm.evaluate import (beats_from_barphase, downbeats_from_barphase, beats_from_activation,
                           metronome, f_measure, _estimate_meter)

CACHE = "/disk1/jaehoon/vbpm_mert_cache"; dev = "cuda:2"; fps = 50.0; H_DIM = 8
CKPT = sys.argv[1] if len(sys.argv) > 1 else "/home/sogang/jaehoon/VBPM_reintegration/vbpm_debug/dirac_step600.pt"
MAXF = 1600


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


def true_barphase(beats, downs, T):
    t = (np.arange(T) + 0.5) / fps
    a = downs if len(downs) >= 2 else beats[::4]
    if len(a) < 2: return None
    ph = np.zeros(T)
    for i in range(len(a) - 1):
        msk = (t >= a[i]) & (t < a[i + 1])
        ph[msk] = TWO_PI * (t[msk] - a[i]) / max(a[i + 1] - a[i], 1e-6)
    return ph


# --------------------------------------------------------------------------
# local COPY of elbo.free_run with optional oracle overrides (vbpm/ untouched)
# --------------------------------------------------------------------------
def _stationary_dev_sigma(sigma, a):
    return sigma / torch.sqrt((1.0 - a ** 2).clamp(min=1e-3))


@torch.no_grad()
def free_run_copy(model, h, temperature=0.3, force_lv_mu=None, force_ph_mu=None):
    B, T, _ = h.shape
    prior_ctx = model.encode_prior(h)
    dof = model.tempo_dof()
    p_m, p_ph_mu, p_ph_rho, p_lv_mu, p_lv_s, _a, _b = model.unpack(model.prior_init_head(prior_ctx.mean(1)))
    if force_lv_mu is not None: p_lv_mu = torch.as_tensor(force_lv_mu, device=h.device, dtype=h.dtype).expand_as(p_lv_mu)
    if force_ph_mu is not None: p_ph_mu = torch.as_tensor(force_ph_mu, device=h.device, dtype=h.dtype).expand_as(p_ph_mu)
    a0 = model.prior_dev_coef(prior_ctx[:, 0]); sd0 = model.prior_dev_scale(prior_ctx[:, 0])
    p_dv_s = _stationary_dev_sigma(sd0, a0)
    meter = gumbel_softmax(p_m, temperature)
    phi = sample_wrapped_cauchy(p_ph_mu, p_ph_rho)
    level = sample_student_t(dof, p_lv_mu, p_lv_s)
    dev = p_dv_s * torch.randn_like(p_dv_s)
    log_tempo = level + dev
    phi_mu = p_ph_mu % TWO_PI; level_mu = p_lv_mu; dev_mu = torch.zeros_like(p_lv_mu)
    level_anchor = level; level_mu_anchor = level_mu; a_lv = model.level_ar()
    z_feats = [model.z_features(meter, phi, log_tempo)]
    phase_traj, phase_mu_traj = [phi], [phi_mu]
    log_tempo_traj, meter_traj, ltmu_traj, rho_traj = [log_tempo], [meter.argmax(-1)], [level_mu + dev_mu], [p_ph_rho]
    meter_prev, phi_prev, log_tempo_prev = meter, phi, log_tempo
    level_prev, dev_prev = level, dev
    for t in range(1, T):
        tempo_prev = torch.exp(log_tempo_prev.clamp(-12.0, 6.0))
        advance = phi_prev + tempo_prev
        cross = (advance >= TWO_PI)
        p_ph_mu = advance % TWO_PI
        p_ph_rho = model.prior_phase_conc(prior_ctx[:, t])
        a = model.prior_dev_coef(prior_ctx[:, t])
        p_lv_s = model.prior_level_scale(prior_ctx[:, t]); p_dv_s = model.prior_dev_scale(prior_ctx[:, t])
        phi = sample_wrapped_cauchy(p_ph_mu, p_ph_rho)
        level = sample_student_t(dof, level_anchor + a_lv * (level_prev - level_anchor), p_lv_s)
        dev = a * dev_prev + p_dv_s * torch.randn_like(p_dv_s)
        log_tempo = level + dev
        q_meter_draw = gumbel_softmax(model.meter_prior_logp(meter_prev, phi, phi_prev, prior_ctx[:, t]), temperature)
        meter = torch.where(cross.unsqueeze(-1), q_meter_draw, meter_prev)
        level_mu = level_mu_anchor + a_lv * (level_mu - level_mu_anchor)
        dev_mu = a * dev_mu
        log_tempo_mu = level_mu + dev_mu
        phi_mu = (phi_mu + torch.exp(log_tempo_mu)) % TWO_PI
        z_feats.append(model.z_features(meter, phi, log_tempo))
        phase_traj.append(phi); phase_mu_traj.append(phi_mu); ltmu_traj.append(log_tempo_mu)
        log_tempo_traj.append(log_tempo); meter_traj.append(meter.argmax(-1)); rho_traj.append(p_ph_rho)
        meter_prev, phi_prev, log_tempo_prev = meter, phi, log_tempo
        level_prev, dev_prev = level, dev
    logits = torch.stack([model.decode(z_feats[t], prior_ctx[:, t]) for t in range(T)], dim=1)
    return {"phase": torch.stack(phase_traj, 1), "phase_mu": torch.stack(phase_mu_traj, 1),
            "log_tempo": torch.stack(log_tempo_traj, 1), "log_tempo_mu": torch.stack(ltmu_traj, 1),
            "meter": torch.stack(meter_traj, 1), "prior_rho": torch.stack(rho_traj, 1),
            "decoder_prob": torch.sigmoid(logits[..., 0]), "downbeat_prob": torch.sigmoid(logits[..., 1]),
            "p_lv_mu": p_lv_mu, "p_lv_s": p_lv_s, "p_ph_mu0": p_ph_mu}


ck = torch.load(CKPT, map_location=dev)
model = BarPointerVAE(h_dim=H_DIM, hidden=128, num_meters=4).to(dev)
model.load_state_dict(ck["sd"]); model.eval()
print(f"loaded {CKPT} (step {ck['step']})")
ev = load("eval", 30)
torch.manual_seed(0); np.random.seed(0)

print()
print("=" * 100)
print("(a)+(b)  FREE-RUN TRAJECTORY DUMP vs GROUND TRUTH   [phase_mu = the SCORED read-out]")
print("=" * 100)
hdr = (f"{'song':28s} {'m':>2s} {'trueBPM':>8s} {'modelBPM':>9s} {'ratio':>6s} "
       f"{'trueBar_f':>9s} {'phaseMu_f':>9s} {'nEst':>6s} {'nRef':>5s} {'est/ref':>7s} {'beatF':>6s} {'dbF':>6s} {'decF':>6s} {'metro':>6s}")
print(hdr); print("-" * len(hdr))
rows = []
for s in ev:
    T = min(s["T"], MAXF)
    h = torch.from_numpy(dirac_h(s["beats"], s["downs"], 0, T)).unsqueeze(0).to(dev)
    out = free_run_copy(model, h)
    pm = out["phase_mu"][0].cpu().numpy(); dec = out["decoder_prob"][0].cpu().numpy()
    ref = s["beats"][s["beats"] < T / fps]; dref = s["downs"][s["downs"] < T / fps]
    if len(ref) < 2: continue
    m = _estimate_meter(ref, dref)
    ibi = float(np.median(np.diff(ref))); trueBPM = 60.0 / ibi
    true_bar_frames = ibi * m * fps
    r = float(math.exp(out["p_lv_mu"][0].item()))
    modelBPM = 60.0 * fps * m * r / TWO_PI
    est = beats_from_barphase(pm, m, fps)
    # measured period of the phase_mu sawtooth
    w = np.where(np.diff(pm) < -math.pi)[0]
    meas_bar = float(np.median(np.diff(w))) if len(w) > 2 else float("nan")
    bF = f_measure(ref, est); dF = f_measure(dref, downbeats_from_barphase(pm, fps)) if len(dref) >= 2 else float("nan")
    dcF = f_measure(ref, beats_from_activation(dec, fps)); mF = f_measure(ref, metronome(T, fps))
    rows.append(dict(key=s["key"], m=m, trueBPM=trueBPM, modelBPM=modelBPM, rate=r,
                     true_bar=true_bar_frames, meas_bar=meas_bar, nest=len(est), nref=len(ref),
                     bF=bF, dF=dF, dcF=dcF, mF=mF,
                     lv_mu=float(out["p_lv_mu"][0]), lv_s=float(out["p_lv_s"][0]),
                     lt_mean=float(out["log_tempo"][0].mean()), lt_std=float(out["log_tempo"][0].std()),
                     ltmu_std=float(out["log_tempo_mu"][0].std()),
                     rho=float(out["prior_rho"][0].mean()),
                     dec_min=float(dec.min()), dec_max=float(dec.max()), dec_mean=float(dec.mean()),
                     dec_std=float(dec.std()), meter_mode=int(np.bincount(out["meter"][0].cpu().numpy()).argmax()),
                     meter_switches=int((np.diff(out["meter"][0].cpu().numpy()) != 0).sum()), T=T))
    print(f"{s['key'][:28]:28s} {m:2d} {trueBPM:8.1f} {modelBPM:9.1f} {modelBPM/trueBPM:6.2f} "
          f"{true_bar_frames:9.1f} {meas_bar:9.1f} {len(est):6d} {len(ref):5d} {len(est)/len(ref):7.2f} "
          f"{bF:6.3f} {dF:6.3f} {dcF:6.3f} {mF:6.3f}")

A = lambda k: float(np.nanmean([r[k] for r in rows]))
print("-" * len(hdr))
print(f"MEAN  beat_F={A('bF'):.4f}  db_F={A('dF'):.4f}  decoder_F={A('dcF'):.4f}  metronome={A('mF'):.4f}")
print(f"      model/true BPM ratio  mean={np.nanmean([r['modelBPM']/r['trueBPM'] for r in rows]):.3f}  "
      f"median={np.nanmedian([r['modelBPM']/r['trueBPM'] for r in rows]):.3f}")
print(f"      est/ref beat-count ratio mean={np.nanmean([r['nest']/r['nref'] for r in rows]):.3f}")
print(f"      p_lv_mu across songs: mean={A('lv_mu'):.4f} std={np.std([r['lv_mu'] for r in rows]):.4f} "
      f"min={min(r['lv_mu'] for r in rows):.4f} max={max(r['lv_mu'] for r in rows):.4f}")
print(f"      TRUE log-tempo across songs: mean="
      f"{np.mean([math.log(TWO_PI/r['true_bar']) for r in rows]):.4f} "
      f"std={np.std([math.log(TWO_PI/r['true_bar']) for r in rows]):.4f}")
print(f"      log_tempo_mu (mean chain) per-song std over t: {A('ltmu_std'):.3e}  <- 0 => CONSTANT rate")
print(f"      stochastic log_tempo: mean={A('lt_mean'):.3f} std_over_t={A('lt_std'):.3f}")
print(f"      prior phase rho: {A('rho'):.4f} -> wrapped-Cauchy gamma={-math.log(A('rho')):.4f} rad")
print(f"      meter: mode={[r['meter_mode'] for r in rows][:12]} switches/song mean={A('meter_switches'):.1f}")

print()
print("=" * 100)
print("(e)  DECODER PROBABILITY DISTRIBUTION at deploy")
print("=" * 100)
print(f"  min={A('dec_min'):.4f} max={A('dec_max'):.4f} mean={A('dec_mean'):.4f} std={A('dec_std'):.4f}")
s = ev[0]; T = min(s["T"], MAXF)
h = torch.from_numpy(dirac_h(s["beats"], s["downs"], 0, T)).unsqueeze(0).to(dev)
o = free_run_copy(model, h); dec = o["decoder_prob"][0].cpu().numpy()
b_t, db_t = targets(s["beats"], s["downs"], 0, T)
print(f"  song {s['key']}: base rate p(beat)={b_t.mean():.4f}")
print(f"    decoder_prob AT true beat frames : mean={dec[b_t > 0.5].mean():.4f} std={dec[b_t > 0.5].std():.4f}")
print(f"    decoder_prob at non-beat frames  : mean={dec[b_t < 0.5].mean():.4f} std={dec[b_t < 0.5].std():.4f}")
print(f"    quantiles of decoder_prob: {np.percentile(dec, [0, 5, 25, 50, 75, 95, 100]).round(4).tolist()}")
print(f"    #frames with prob>0.5: {(dec > 0.5).sum()} / {T}")
# training-time decoder for the SAME song, teacher-forced
b_ten = torch.from_numpy(b_t).unsqueeze(0).to(dev); d_ten = torch.from_numpy(db_t).unsqueeze(0).to(dev)
with torch.no_grad():
    _, info = strict_elbo(model, h, b_ten, d_ten, temperature=0.3, beta=1.0)
tp = info["beat_prob"][0].cpu().numpy()
print(f"  TRAINING (teacher-forced) decoder on same song:")
print(f"    at true beats: mean={tp[b_t > 0.5].mean():.4f} | elsewhere: mean={tp[b_t < 0.5].mean():.4f} | max={tp.max():.4f}")
print(f"    train-rollout beat_F (peak-pick) = {f_measure(s['beats'][s['beats'] < T / fps], beats_from_activation(tp, fps)):.4f}")
qpm = info["post_phase_mu"][0].cpu().numpy()
mm = _estimate_meter(s["beats"][s["beats"] < T / fps], s["downs"][s["downs"] < T / fps])
print(f"    train-rollout POSTERIOR phase read-out beat_F = "
      f"{f_measure(s['beats'][s['beats'] < T / fps], beats_from_barphase(qpm, mm, fps)):.4f}")
tb = true_barphase(s["beats"], s["downs"], T)
if tb is not None:
    circ = np.abs(((qpm - tb + math.pi) % TWO_PI) - math.pi)
    print(f"    posterior phase_mu vs TRUE bar phase: mean circ err={circ.mean():.3f} rad ({circ.mean()/TWO_PI*100:.1f}% of a bar)")
    fpm = o["phase_mu"][0].cpu().numpy()
    circ2 = np.abs(((fpm - tb + math.pi) % TWO_PI) - math.pi)
    print(f"    FREE-RUN phase_mu vs TRUE bar phase : mean circ err={circ2.mean():.3f} rad "
          f"(chance = {math.pi/2:.3f})")

print()
print("=" * 100)
print("(f)  ORACLE-TEMPO free-run: force p_lv_mu = log(true bar phidot)")
print("=" * 100)
res = {"as_is": [], "oracle_tempo": [], "oracle_tempo_phase": [], "metro": []}
for s in ev:
    T = min(s["T"], MAXF)
    h = torch.from_numpy(dirac_h(s["beats"], s["downs"], 0, T)).unsqueeze(0).to(dev)
    ref = s["beats"][s["beats"] < T / fps]; dref = s["downs"][s["downs"] < T / fps]
    if len(ref) < 2: continue
    m = _estimate_meter(ref, dref)
    ibi = float(np.median(np.diff(ref)))
    true_lv = math.log(TWO_PI / (ibi * m * fps))
    # oracle initial bar phase: where the first downbeat sits
    d0 = dref[0] if len(dref) else ref[0]
    ph0 = float((-d0 * fps * math.exp(true_lv)) % TWO_PI)
    o1 = free_run_copy(model, h)
    o2 = free_run_copy(model, h, force_lv_mu=true_lv)
    o3 = free_run_copy(model, h, force_lv_mu=true_lv, force_ph_mu=ph0)
    for k, oo in [("as_is", o1), ("oracle_tempo", o2), ("oracle_tempo_phase", o3)]:
        res[k].append(f_measure(ref, beats_from_barphase(oo["phase_mu"][0].cpu().numpy(), m, fps)))
    res["metro"].append(f_measure(ref, metronome(T, fps)))
for k in res:
    print(f"  {k:20s} beat_F = {np.nanmean(res[k]):.4f}")

json.dump(rows, open("/home/sogang/jaehoon/VBPM_reintegration/vbpm_debug/probe_B_rows.json", "w"), indent=1)
