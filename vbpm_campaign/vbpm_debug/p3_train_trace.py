"""P3: TRAIN the Dirac model and trace the TEMPO PATH.

(b) does log_tempo converge to the true per-song log bar-phidot?  print pred vs true BPM.
(c) is KL_level fighting the move?  measure KL magnitude vs distance to travel.
(e) FIX: bias-init the level_mu output of post_head / prior_init_head at
    log(2pi/(median_IBI*m*fps)).  Before/after free-run beat_F.

usage: p3_train_trace.py [--fix] [--steps N] [--dev cuda:1] [--tag NAME]
NOTE: the fix is applied by writing into the model's OWN parameter tensors (bias init),
which is what a real bias-init would do; vbpm/ source is never touched.
"""
import sys, glob, math, time, argparse, json
import numpy as np, torch
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
from vbpm.model import BarPointerVAE
from vbpm.elbo import strict_elbo, free_run
from vbpm.distributions import kl_student_t_mc, sample_student_t
from vbpm.evaluate import (beats_from_barphase, downbeats_from_barphase, beats_from_activation,
                           metronome, f_measure, _estimate_meter, bpm_from_logtempo)

ap = argparse.ArgumentParser()
ap.add_argument("--fix", action="store_true")
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--dev", default="cuda:1")
ap.add_argument("--tag", default="base")
A = ap.parse_args()

CACHE = "/disk1/jaehoon/vbpm_mert_cache"; fps = 50.0; TWO_PI = 2 * math.pi
dev = A.dev; H_DIM = 8


def load(split, cap=None):
    out = []
    for f in sorted(glob.glob(f"{CACHE}/{split}__*.npz")):
        d = np.load(f, allow_pickle=True)
        b = np.asarray(d["beats"], float); dn = np.asarray(d["downs"], float)
        if len(b) < 4: continue
        m = _estimate_meter(b, dn)
        ibi = float(np.median(np.diff(b)))
        out.append(dict(key=f.split("__")[-1][:-4], T=int(d["feats"].shape[1]), beats=b, downs=dn,
                        m=m, ibi=ibi, log_phidot=math.log(TWO_PI / (ibi * m * fps)), bpm=60.0 / ibi))
    return out[:cap] if cap else out


train = load("train"); ev = load("eval", 30)
TARGET_LOG = float(np.median([s["log_phidot"] for s in train]))
print(f"[{A.tag}] train={len(train)} eval={len(ev)}  dataset-median log_phidot={TARGET_LOG:.4f}"
      f" (={math.exp(TARGET_LOG):.5f} rad/frame, {bpm_from_logtempo(TARGET_LOG,4,fps):.1f} BPM @m=4)", flush=True)


def dirac_h(beats, downs, start, n, rng):
    h = rng.standard_normal((n, H_DIM)).astype(np.float32) * 0.01
    for t in beats:
        i = int(round(t * fps)) - start
        if 0 <= i < n: h[i, 0] += 1.0
    for t in downs:
        i = int(round(t * fps)) - start
        if 0 <= i < n: h[i, 1] += 1.0
    return h


def targets(beats, downs, start, n):
    b = np.zeros(n, np.float32); d = np.zeros(n, np.float32)
    for t in beats:
        i = int(round(t * fps)) - start
        if 0 <= i < n: b[i] = 1.0
    for t in downs:
        i = int(round(t * fps)) - start
        if 0 <= i < n: d[i] = 1.0
    return b, d


torch.manual_seed(0); rng = np.random.default_rng(0)
model = BarPointerVAE(h_dim=H_DIM, hidden=128, num_meters=4).to(dev)
K = model.K
LEVEL_IDX = K + 3

if A.fix:
    with torch.no_grad():
        model.post_head[-1].bias[LEVEL_IDX] += TARGET_LOG
        model.prior_init_head[-1].bias[LEVEL_IDX] += TARGET_LOG
    print(f"[{A.tag}] FIX APPLIED: level_mu bias += {TARGET_LOG:.4f} on post_head + prior_init_head", flush=True)


@torch.no_grad()
def tempo_report(step):
    """(b) predicted vs true per-song tempo, and the free-run scores."""
    model.eval()
    preds, trues, keys, ms = [], [], [], []
    fb, fd, fdec, fmet = [], [], [], []
    for s in ev:
        T = min(s["T"], 1600)
        h = torch.from_numpy(dirac_h(s["beats"], s["downs"], 0, T, np.random.default_rng(1))).unsqueeze(0).to(dev)
        pc = model.encode_prior(h)
        _m, _ph, _rho, plv_mu, _ls, _dm, _ds = model.unpack(model.prior_init_head(pc.mean(1)))
        preds.append(float(plv_mu[0])); trues.append(s["log_phidot"]); keys.append(s["key"]); ms.append(s["m"])
        out = free_run(model, h)
        pm = out["phase_mu"][0, :T].cpu().numpy(); dec = out["decoder_prob"][0, :T].cpu().numpy()
        ref = s["beats"][s["beats"] < T / fps]; dref = s["downs"][s["downs"] < T / fps]
        if len(ref) < 2: continue
        fb.append(f_measure(ref, beats_from_barphase(pm, s["m"], fps)))
        if len(dref) >= 2: fd.append(f_measure(dref, downbeats_from_barphase(pm, fps)))
        fdec.append(f_measure(ref, beats_from_activation(dec, fps)))
        fmet.append(f_measure(ref, metronome(T, fps)))
    model.train()
    p = np.array(preds); t = np.array(trues)
    r = float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-9 else float("nan")
    return dict(step=step, pred=p, true=t, keys=keys, ms=ms, corr=r,
                beat_F=float(np.mean(fb)), db_F=float(np.mean(fd)) if fd else float("nan"),
                dec_F=float(np.mean(fdec)), met_F=float(np.mean(fmet)))


def show(rep, full=False):
    p, t = rep["pred"], rep["true"]
    pb = np.array([bpm_from_logtempo(pp, m, fps) for pp, m in zip(p, rep["ms"])])
    tb = np.array([bpm_from_logtempo(tt, m, fps) for tt, m in zip(t, rep["ms"])])
    print(f"  [s{rep['step']:4d}] log_tempo pred: mean={p.mean():7.3f} std={p.std():.4f} | true mean={t.mean():.3f} std={t.std():.3f}"
          f" | bias={p.mean()-t.mean():+7.3f} | MAE={np.abs(p-t).mean():.3f} | corr={rep['corr']:+.3f}", flush=True)
    print(f"          BPM pred: med={np.median(pb):9.1f} [{pb.min():.1f},{pb.max():.1f}] vs TRUE med={np.median(tb):.1f} [{tb.min():.1f},{tb.max():.1f}]"
          f" | median ratio={np.median(pb/tb):.3f}x", flush=True)
    print(f"          FREE-RUN beat_F={rep['beat_F']:.3f} db_F={rep['db_F']:.3f} decoder_F={rep['dec_F']:.3f} metronome={rep['met_F']:.3f}", flush=True)
    if full:
        print("          per-song (first 12):  key  true_BPM  pred_BPM  ratio")
        for i in range(min(12, len(p))):
            print(f"            {rep['keys'][i][:34]:34s} m={rep['ms'][i]}  {tb[i]:7.1f}  {pb[i]:10.2f}  {pb[i]/tb[i]:7.3f}x", flush=True)


rep0 = tempo_report(0); print("INIT:"); show(rep0, full=True)

# ---------------- (c) KL_level pressure measurement, at INIT geometry -------------
print(f"\n[{A.tag}] (c) KL_level PRESSURE: how big is the level KL vs the distance to travel?", flush=True)
with torch.no_grad():
    dof = model.tempo_dof()
    d_travel = TARGET_LOG - float(rep0["pred"].mean())
    # t=1 KL: q at true value, p at init value, using the model's own scales
    s0 = ev[0]; T = 1600
    h = torch.from_numpy(dirac_h(s0["beats"], s0["downs"], 0, T, np.random.default_rng(1))).unsqueeze(0).to(dev)
    pc = model.encode_prior(h)
    _m, _ph, _rho, plv_mu, plv_s, _dm, _ds = model.unpack(model.prior_init_head(pc.mean(1)))
    sig = plv_s
    for qmu in [float(plv_mu[0]), float(plv_mu[0]) - 1.0, TARGET_LOG]:
        q = torch.tensor([qmu], device=dev); z = q.clone()
        kl = kl_student_t_mc(dof, q, sig, plv_mu, sig, z)
        print(f"    q_level_mu={qmu:+7.3f} (p={float(plv_mu[0]):+.3f}, sigma={float(sig):.3f}) -> KL_level(t=1) ~ {float(kl):8.3f} nats", flush=True)
    print(f"    distance to travel = {d_travel:.3f} nats-of-log-tempo; per-frame OU/RW sigma={float(model.prior_level_scale(pc[:,1])):.3f}", flush=True)
    print(f"    (t>=2 the prior mean is the OU anchor = the model's OWN t=1 sample, so only the t=1 KL", flush=True)
    print(f"     term ties log_tempo to prior_init_head; a whole-song shift costs ~ (d/sigma)^2/2 = "
          f"{(d_travel/float(sig))**2/2:.2f} nats ONCE, vs {T} recon frames.)", flush=True)

# --------------------------------- train ------------------------------------
STEPS, WARM, BS, FR = A.steps, max(1, A.steps // 2), 16, 256
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
t0 = time.time()
hist = []
for step in range(1, STEPS + 1):
    beta = min(1.0, step / WARM); temp = 1.0 + (0.3 - 1.0) * min(step / STEPS, 1.0)
    hs, bs_, ds = [], [], []
    for _ in range(BS):
        s = train[rng.integers(len(train))]
        if s["T"] <= FR: continue
        st = int(rng.integers(0, s["T"] - FR))
        hs.append(torch.from_numpy(dirac_h(s["beats"], s["downs"], st, FR, rng)))
        b, d = targets(s["beats"], s["downs"], st, FR)
        bs_.append(torch.from_numpy(b)); ds.append(torch.from_numpy(d))
    h = torch.stack(hs).to(dev); b = torch.stack(bs_).to(dev); d = torch.stack(ds).to(dev)
    opt.zero_grad()
    loss, info = strict_elbo(model, h, b, d, temperature=temp, beta=beta)
    if not torch.isfinite(loss):
        print("NaN@", step, flush=True); break
    loss.backward()
    gl = float(model.prior_init_head[-1].bias.grad[LEVEL_IDX])
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()
    if step % 100 == 0:
        print(f"  s{step:4d} b={beta:.2f} rec_b={info['recon_beat']:7.2f} rec_db={info['recon_db']:7.2f} "
              f"kl(phi={info['kl_phase']:8.2f} lv={info['kl_level']:8.2f} dv={info['kl_dev']:7.2f} m={info['kl_meter']:6.2f}) "
              f"ncross={info['n_cross']:6.1f} nu={info['tempo_dof']:.2f} dL/db_lv={gl:+.4f} | {step/(time.time()-t0):.2f} it/s", flush=True)
    if step % 200 == 0 or step == STEPS:
        rep = tempo_report(step); show(rep, full=(step == STEPS)); hist.append(rep)

print(f"\n[{A.tag}] SUMMARY  init beat_F={rep0['beat_F']:.3f} -> final beat_F={hist[-1]['beat_F']:.3f}"
      f"  (metronome {hist[-1]['met_F']:.3f})", flush=True)
np.save(f"/home/sogang/jaehoon/VBPM_reintegration/vbpm_debug/hist_{A.tag}.npy",
        np.array([{k: v for k, v in r.items()} for r in [rep0] + hist], dtype=object), allow_pickle=True)
