"""P3: does the POSTERIOR phase mean track the TRUE bar phase under Dirac input?
Trains 600 steps on cuda:2 (same recipe as probe_dirac.py) and measures, at init and after:
  * circular corr / mean abs circular error  q_ph_mu  vs  true bar phase
  * phase_rho (posterior AND prior) -> wrapped-Cauchy gamma in radians
  * |(u,v)| radius feeding atan2 (gradient conditioning of the phase mean)
  * recon headroom vs the KL price of a sharp phase
  * per-module grad norms (who actually gets trained)
  * free-run beat_F vs metronome
NO source file is modified; the posterior params are read with a forward hook on post_head.
"""
import sys, math, glob, time, json
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
from vbpm.model import BarPointerVAE
from vbpm.elbo import strict_elbo, free_run
from vbpm.evaluate import beats_from_barphase, downbeats_from_barphase, beats_from_activation, metronome, f_measure, _estimate_meter
from vbpm.distributions import TWO_PI

CACHE = "/disk1/jaehoon/vbpm_mert_cache"; dev = "cuda:2"; fps = 50.0; H_DIM = 8
FR, BS, STEPS, WARM = 256, 16, 600, 600

def load(split, cap=None):
    out = []
    for f in sorted(glob.glob(f"{CACHE}/{split}__*.npz")):
        d = np.load(f, allow_pickle=True)
        out.append(dict(key=f.split("__")[1][:-4], T=int(d["feats"].shape[1]),
                        beats=np.asarray(d["beats"], float), downs=np.asarray(d["downs"], float)))
        if cap and len(out) >= cap: break
    return out
train, ev = load("train"), load("eval", 30)

def dirac_h(beats, downs, start, n):
    h = np.random.randn(n, H_DIM).astype(np.float32) * 0.01
    for t in beats:
        i = int(round(t*fps)) - start
        if 0 <= i < n: h[i, 0] += 1.0
    for t in downs:
        i = int(round(t*fps)) - start
        if 0 <= i < n: h[i, 1] += 1.0
    return h

def targets(beats, downs, start, n):
    b = np.zeros(n, np.float32); db = np.zeros(n, np.float32)
    for t in beats:
        i = int(round(t*fps)) - start
        if 0 <= i < n: b[i] = 1.0
    for t in downs:
        i = int(round(t*fps)) - start
        if 0 <= i < n: db[i] = 1.0
    return b, db

def oracle_barphase(beats, downs, start, n):
    """0 at each downbeat -> 2pi at the next; NaN outside the annotated span."""
    t = (np.arange(n) + start + 0.5) / fps
    a = downs
    if len(a) < 2: return None
    ph = np.full(n, np.nan)
    for i in range(len(a)-1):
        m = (t >= a[i]) & (t < a[i+1])
        ph[m] = TWO_PI * (t[m]-a[i]) / max(a[i+1]-a[i], 1e-6)
    return ph

def circ_corr(a, b):
    m = ~(np.isnan(a) | np.isnan(b)); a, b = a[m], b[m]
    if len(a) < 10: return float("nan")
    ab = math.atan2(np.sin(a).mean(), np.cos(a).mean()); bb = math.atan2(np.sin(b).mean(), np.cos(b).mean())
    sa, sb = np.sin(a-ab), np.sin(b-bb)
    return float((sa*sb).sum() / math.sqrt((sa**2).sum()*(sb**2).sum()))

def circ_err(a, b, best_shift=True):
    m = ~(np.isnan(a) | np.isnan(b)); a, b = a[m], b[m]
    if len(a) < 10: return float("nan")
    d = a - b
    if best_shift:
        s = math.atan2(np.sin(d).mean(), np.cos(d).mean()); d = d - s
    return float(np.abs((d + math.pi) % TWO_PI - math.pi).mean())

# ---- fixed diagnostic batch (same crops before/after) ----
rngd = np.random.default_rng(7)
dh, db_, dd, dph = [], [], [], []
for s in ev:
    if s["T"] <= FR or len(s["downs"]) < 4: continue
    st = int(round(s["downs"][1]*fps))
    if st + FR > s["T"]: continue
    p = oracle_barphase(s["beats"], s["downs"], st, FR)
    if p is None or np.isnan(p).mean() > 0.2: continue
    dh.append(dirac_h(s["beats"], s["downs"], st, FR))
    bb, ddd = targets(s["beats"], s["downs"], st, FR); db_.append(bb); dd.append(ddd); dph.append(p)
    if len(dh) >= 16: break
DH = torch.from_numpy(np.stack(dh)).to(dev); DB = torch.from_numpy(np.stack(db_)).to(dev)
DD = torch.from_numpy(np.stack(dd)).to(dev); DPH = np.stack(dph)
print(f"diagnostic batch: {DH.shape}, mean beats/crop={DB.sum(1).mean():.1f}, downbeats/crop={DD.sum(1).mean():.1f}", flush=True)

torch.manual_seed(0); rng = np.random.default_rng(0)
model = BarPointerVAE(h_dim=H_DIM, hidden=128, num_meters=4).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

_cap = []
def hook(mod, inp, out): _cap.append(out.detach())
hh = model.post_head.register_forward_hook(hook)

@torch.no_grad()
def diagnose(tag):
    model.eval(); _cap.clear()
    loss, info = strict_elbo(model, DH, DB, DD, temperature=0.5, beta=1.0)
    P = torch.stack(_cap, 1)                              # [B,T,param_dim]
    B, T, _ = P.shape; K = model.K
    u, v = P[..., K], P[..., K+1]
    rq = torch.sigmoid(P[..., K+2])
    qmu = info["post_phase_mu"].cpu().numpy()             # [B,T]
    with torch.no_grad():
        pctx = model.encode_prior(DH)
        rp = torch.stack([model.prior_phase_conc(pctx[:, t]) for t in range(T)], 1)
    rad = torch.sqrt(u**2 + v**2)
    print(f"\n### {tag} ###")
    print(f"  loss={info['loss']:.1f} recon_beat={info['recon_beat']:.2f} recon_db={info['recon_db']:.2f} "
          f"kl_phase={info['kl_phase']:.2f} kl_level={info['kl_level']:.2f} kl_dev={info['kl_dev']:.2f} kl_meter={info['kl_meter']:.2f}")
    pbase = float(DB.mean()); Hb = -(pbase*math.log(pbase+1e-9)+(1-pbase)*math.log(1-pbase+1e-9))*T
    print(f"  RECON HEADROOM: base-rate beat BCE for this crop = {Hb:.1f} nats (perfect = 0) "
          f"-> total beat+db headroom ~{Hb*2:.0f} nats/crop")
    cc = [circ_corr(qmu[i], DPH[i]) for i in range(B)]
    e_raw = [circ_err(qmu[i], DPH[i], False) for i in range(B)]
    e_sh = [circ_err(qmu[i], DPH[i], True) for i in range(B)]
    print(f"  q_ph_mu vs TRUE bar phase:  circ_corr = {np.nanmean(cc):+.4f} (per-song "
          f"min {np.nanmin(cc):+.3f} max {np.nanmax(cc):+.3f})")
    print(f"                              mean|circ err| raw = {np.nanmean(e_raw):.4f} rad, "
          f"best-shift = {np.nanmean(e_sh):.4f} rad   (chance = {math.pi/2:.4f})")
    dq = np.diff(qmu, axis=1); w_q = (dq < -math.pi).sum(1); w_t = np.nanmax(np.nan_to_num(DPH), axis=1)*0
    true_bars = [(np.diff(DPH[i]) < -math.pi).sum() for i in range(B)]
    print(f"  q_ph_mu 2pi-wraps/crop = {w_q.mean():.1f} (TRUE bars/crop = {np.mean(true_bars):.1f}); "
          f"forward jumps>pi = {(dq > math.pi).sum(1).mean():.1f}")
    print(f"  POSTERIOR rho_q: mean={rq.mean():.4f} min={rq.min():.4f} max={rq.max():.4f} "
          f"-> gamma mean={-math.log(float(rq.mean())):.4f} rad "
          f"(need <~0.05 rad to place a beat within 1 frame)")
    print(f"  PRIOR     rho_p: mean={rp.mean():.4f} min={rp.min():.4f} max={rp.max():.4f} "
          f"-> gamma mean={-math.log(float(rp.mean())):.4f} rad")
    print(f"  atan2 radius |(u,v)|: mean={rad.mean():.4f} min={rad.min():.4f} "
          f"(d mu/d(u,v) ~ 1/r = {1.0/float(rad.mean()):.2f})")
    # sampled phi spread around its own mean (use the real sampler at the real rho_q)
    from vbpm.distributions import sample_wrapped_cauchy as _swc
    mu_f = info["post_phase_mu"].to(dev).reshape(-1); rq_f = rq.reshape(-1)
    devs = []
    for _ in range(32):
        s_ = _swc(mu_f, rq_f)
        devs.append(((s_ - mu_f + math.pi) % TWO_PI - math.pi).cpu().numpy())
    dd_ = np.concatenate(devs)
    print(f"  SAMPLED phi spread around q mean: median|dev|={np.median(np.abs(dd_)):.4f} rad, "
          f"P(|dev|>0.1)={np.mean(np.abs(dd_)>0.1):.3f}, P(|dev|>0.5)={np.mean(np.abs(dd_)>0.5):.3f}"
          f"  [true inter-beat phase gap = 2pi/m = {TWO_PI/4:.3f} rad]")
    qm = info["post_phase_mu"].cpu().numpy()
    print(f"  q_ph_mu temporal shape: per-song std over t = {qm.std(1).mean():.4f} rad, "
          f"range = {(qm.max(1)-qm.min(1)).mean():.4f} rad, mean = {qm.mean():.4f}")
    model.train()
    return dict(cc=float(np.nanmean(cc)), err=float(np.nanmean(e_sh)), rq=float(rq.mean()), rp=float(rp.mean()))

@torch.no_grad()
def ev_freerun(tag, cap=20):
    model.eval(); acc = {"beat_phase": [], "downbeat_phase": [], "decoder": [], "metronome": []}
    bpms = []
    for s in ev[:cap]:
        T = min(s["T"], 1600)
        h = torch.from_numpy(dirac_h(s["beats"], s["downs"], 0, T)).unsqueeze(0).to(dev)
        out = free_run(model, h)
        pm = out["phase_mu"][0, :T].cpu().numpy(); dec = out["decoder_prob"][0, :T].cpu().numpy()
        ref = s["beats"][s["beats"] < T/fps]; dref = s["downs"][s["downs"] < T/fps]
        if len(ref) < 2: continue
        m = _estimate_meter(ref, dref)
        acc["beat_phase"].append(f_measure(ref, beats_from_barphase(pm, m, fps)))
        if len(dref) >= 2: acc["downbeat_phase"].append(f_measure(dref, downbeats_from_barphase(pm, fps)))
        acc["decoder"].append(f_measure(ref, beats_from_activation(dec, fps)))
        acc["metronome"].append(f_measure(ref, metronome(T, fps)))
        adv = float(np.median(np.diff(pm) % TWO_PI))
        bpms.append((60.0*fps*m*adv/TWO_PI, 60.0/np.median(np.diff(ref))))
    model.train()
    r = {k: float(np.mean(v)) for k, v in acc.items()}
    print(f"  [FREE-RUN {tag}] beat_F={r['beat_phase']:.3f} db_F={r['downbeat_phase']:.3f} "
          f"decoder_F={r['decoder']:.3f} metronome={r['metronome']:.3f}")
    print(f"    deploy BPM (est,true) first 6: {[(round(a),round(b)) for a,b in bpms[:6]]}  "
          f"median est={np.median([a for a,_ in bpms]):.0f} vs true={np.median([b for _,b in bpms]):.0f}")
    return r

d0 = diagnose("INIT (step 0)")
ev_freerun("init")

t0 = time.time()
for step in range(1, STEPS+1):
    beta = min(1.0, step/WARM); temp = 1.0 + (0.3-1.0)*min(step/STEPS, 1.0)
    hs, bs_, ds = [], [], []
    for _ in range(BS):
        s = train[rng.integers(len(train))]
        if s["T"] <= FR: continue
        st = int(rng.integers(0, s["T"]-FR))
        hs.append(torch.from_numpy(dirac_h(s["beats"], s["downs"], st, FR)))
        b, d = targets(s["beats"], s["downs"], st, FR); bs_.append(torch.from_numpy(b)); ds.append(torch.from_numpy(d))
    h = torch.stack(hs).to(dev); b = torch.stack(bs_).to(dev); d = torch.stack(ds).to(dev)
    _cap.clear()
    opt.zero_grad(); loss, info = strict_elbo(model, h, b, d, temperature=temp, beta=beta)
    loss.backward()
    if step in (1, 100, 300, 600):
        gn = {}
        for name, mod in [("post_gru", model.post_gru), ("post_head", model.post_head),
                          ("prior_gru", model.prior_gru), ("prior_init_head", model.prior_init_head),
                          ("prior_phase_rho", model.prior_phase_rho), ("prior_level_sigma", model.prior_level_sigma),
                          ("decoder", model.decoder), ("meter_prior", model.meter_prior)]:
            g = [p.grad.norm().item() for p in mod.parameters() if p.grad is not None]
            gn[name] = float(np.sqrt(np.sum(np.square(g)))) if g else 0.0
        gn["z0"] = float(model.z0.grad.norm()) if model.z0.grad is not None else 0.0
        gn["level_ar_logit"] = float(model.level_ar_logit.grad.abs()) if model.level_ar_logit.grad is not None else 0.0
        print(f"  [grad s{step}] " + " ".join(f"{k}={v:.3e}" for k, v in gn.items()), flush=True)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    if step % 100 == 0:
        print(f"s{step:4d} b={beta:.2f} rec_b={info['recon_beat']:6.2f} rec_db={info['recon_db']:6.2f} "
              f"kl(phi={info['kl_phase']:.2f} lv={info['kl_level']:.2f} dv={info['kl_dev']:.2f} m={info['kl_meter']:.2f}) "
              f"| {step/(time.time()-t0):.2f} it/s", flush=True)

d1 = diagnose(f"AFTER {STEPS} STEPS")
ev_freerun("after")
hh.remove()

print("\n" + "="*92)
print("SUMMARY  init -> trained")
print("="*92)
print(f"  circ_corr(q_ph_mu, true bar phase): {d0['cc']:+.4f} -> {d1['cc']:+.4f}")
print(f"  mean|circ err| (best shift):        {d0['err']:.4f} -> {d1['err']:.4f} rad (chance {math.pi/2:.3f})")
print(f"  posterior rho_q:                    {d0['rq']:.4f} -> {d1['rq']:.4f}  (gamma {-math.log(d0['rq']):.3f} -> {-math.log(d1['rq']):.3f} rad)")
print(f"  prior     rho_p:                    {d0['rp']:.4f} -> {d1['rp']:.4f}  (gamma {-math.log(d0['rp']):.3f} -> {-math.log(d1['rp']):.3f} rad)")
