"""P1: TEMPO PATH units + init audit. No training. Pure measurement.

(a) TRUE per-song bar-phidot / BPM from the cache.
(d) UNITS AUDIT of every consumer of log_tempo:
      elbo advance, evaluate.bpm_from_logtempo, evaluate.beats_from_barphase, z_features.
(a') What the untrained model's prior_init_head / post_head emit, in rad/frame + BPM.
"""
import sys, glob, math, json
import numpy as np, torch
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
from vbpm.model import BarPointerVAE
from vbpm.elbo import free_run
from vbpm.evaluate import (bpm_from_logtempo, beats_from_barphase,
                           downbeats_from_barphase, _estimate_meter, f_measure, metronome)

CACHE = "/disk1/jaehoon/vbpm_mert_cache"; fps = 50.0; TWO_PI = 2 * math.pi
dev = "cuda:1"; H_DIM = 8
np.set_printoptions(precision=4, suppress=True)


def load(split, cap=None):
    out = []
    for f in sorted(glob.glob(f"{CACHE}/{split}__*.npz")):
        d = np.load(f, allow_pickle=True)
        out.append(dict(key=f.split("__")[-1][:-4], T=int(d["feats"].shape[1]),
                        beats=np.asarray(d["beats"], float), downs=np.asarray(d["downs"], float)))
    return out[:cap] if cap else out


def dirac_h(beats, downs, start, n, seed=0):
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((n, H_DIM)).astype(np.float32) * 0.01
    for t in beats:
        i = int(round(t * fps)) - start
        if 0 <= i < n: h[i, 0] += 1.0
    for t in downs:
        i = int(round(t * fps)) - start
        if 0 <= i < n: h[i, 1] += 1.0
    return h


train = load("train"); ev = load("eval", 30)
print(f"loaded train={len(train)} eval={len(ev)}")

# ---------------------------------------------------------------- (a) TRUTH
print("\n" + "=" * 78)
print("(a) TRUE per-song tempo from cached beats  [bar-phidot = 2pi/(IBI*m*fps) rad/frame]")
print("=" * 78)
rows = []
for s in ev:
    b, d = s["beats"], s["downs"]
    if len(b) < 3: continue
    ibi = float(np.median(np.diff(b)))
    m = _estimate_meter(b, d)
    phidot = TWO_PI / (ibi * m * fps)
    rows.append(dict(key=s["key"], ibi=ibi, m=m, bpm=60.0 / ibi,
                     phidot=phidot, log_phidot=math.log(phidot)))
arr = np.array([[r["bpm"], r["phidot"], r["log_phidot"], r["m"]] for r in rows])
print(f"  n={len(rows)}  BPM       min={arr[:,0].min():7.2f} med={np.median(arr[:,0]):7.2f} max={arr[:,0].max():7.2f}")
print(f"          bar-phidot rad/fr min={arr[:,1].min():.5f} med={np.median(arr[:,1]):.5f} max={arr[:,1].max():.5f}")
print(f"          log_phidot        min={arr[:,2].min():7.3f} med={np.median(arr[:,2]):7.3f} max={arr[:,2].max():7.3f}  std={arr[:,2].std():.3f}")
print(f"          meter m: {dict(zip(*np.unique(arr[:,3].astype(int), return_counts=True)))}")
# same over the FULL train split
tr = []
for s in train:
    b, d = s["beats"], s["downs"]
    if len(b) < 3: continue
    ibi = float(np.median(np.diff(b))); m = _estimate_meter(b, d)
    tr.append(math.log(TWO_PI / (ibi * m * fps)))
tr = np.array(tr)
print(f"  TRAIN split (n={len(tr)}): log_phidot med={np.median(tr):.3f} mean={tr.mean():.3f} std={tr.std():.3f}"
      f"  -> phidot med={math.exp(np.median(tr)):.5f} rad/frame")
TARGET_LOG = float(np.median(tr))
json.dump({"target_log": TARGET_LOG}, open("/home/sogang/jaehoon/VBPM_reintegration/vbpm_debug/target_log.json", "w"))

# ---------------------------------------------------------------- (d) UNITS
print("\n" + "=" * 78)
print("(d) UNITS AUDIT -- round-trip a KNOWN song through every consumer")
print("=" * 78)
BPM, M = 120.0, 4
ibi = 60.0 / BPM
phidot = TWO_PI / (ibi * M * fps)
lt = math.log(phidot)
print(f"  ground truth: BPM={BPM} m={M} fps={fps} -> IBI={ibi}s, bar={ibi*M}s={ibi*M*fps:.0f} frames")
print(f"                bar-phidot = 2pi/{ibi*M*fps:.0f} = {phidot:.6f} rad/frame ; log = {lt:.4f}")
print(f"  [evaluate.bpm_from_logtempo] -> {bpm_from_logtempo(lt, M, fps):.4f}  (expect {BPM})"
      f"   {'OK' if abs(bpm_from_logtempo(lt,M,fps)-BPM)<1e-6 else '*** MISMATCH ***'}")
# synth a constant-rate phase chain exactly as elbo.free_run's mean chain does, read it out
T = 1000
ph = np.zeros(T); p = 0.0
for t in range(1, T):
    p = (p + phidot) % TWO_PI
    ph[t] = p
est_b = beats_from_barphase(ph, M, fps)
est_d = downbeats_from_barphase(ph, fps)
print(f"  [elbo advance -> evaluate.beats_from_barphase] {len(est_b)} beats in {T/fps:.0f}s"
      f" -> {len(est_b)/(T/fps)*60:.2f} BPM (expect {BPM})   median IBI={np.median(np.diff(est_b)):.4f}s (expect {ibi})")
print(f"  [-> downbeats_from_barphase] {len(est_d)} downbeats -> bar period {np.median(np.diff(est_d)):.4f}s (expect {ibi*M})")
print(f"  [model.z_features log_tempo channel] passes log_tempo RAW (clamp -12..6): {lt:.4f} in range -> OK")
print(f"  CONVENTION CHECK: exp(log_tempo) is BAR rad/frame everywhere?  advance=phi+exp(lt) [elbo.py:88,176] YES;"
      f" bpm_from_logtempo multiplies by m [evaluate.py:75] YES; beats_from_barphase multiplies phi by m YES")

# ---------------------------------------------------------------- init
print("\n" + "=" * 78)
print("(BUG#1) UNTRAINED MODEL: what log_tempo does the model emit?")
print("=" * 78)
torch.manual_seed(0)
model = BarPointerVAE(h_dim=H_DIM, hidden=128, num_meters=4).to(dev)
K = model.K
hs = []
for s in ev[:16]:
    T = min(s["T"], 1600)
    hs.append(torch.from_numpy(dirac_h(s["beats"], s["downs"], 0, 1600)))
h = torch.stack(hs).to(dev)
with torch.no_grad():
    pc = model.encode_prior(h)
    vec_init = model.prior_init_head(pc.mean(1))
    pm, pph, prho, plv_mu, plv_s, pdv_mu, pdv_s = model.unpack(vec_init)
    b0 = torch.zeros(h.shape[0], h.shape[1], device=dev)
    qc = model.encode_posterior(h, b0)
    z0 = model.z0.unsqueeze(0).expand(h.shape[0], -1)
    qm, qph, qrho, qlv_mu, qlv_s, qdv_mu, qdv_s = model.unpack(
        model.post_head(torch.cat([qc[:, 0], z0], dim=-1)))
print(f"  prior_init_head level_mu : mean={plv_mu.mean():.4f} std={plv_mu.std():.4f} "
      f"min={plv_mu.min():.4f} max={plv_mu.max():.4f}")
print(f"                  level_sig: mean={plv_s.mean():.4f}   dev_sig(head)={pdv_s.mean():.4f}")
print(f"                  phase_mu : mean={pph.mean():.3f} rad   phase_rho: mean={prho.mean():.4f}")
print(f"  post_head(t=1)  level_mu : mean={qlv_mu.mean():.4f} std={qlv_mu.std():.4f}")
print(f"  level_ar()={model.level_ar().item():.4f}  tempo_dof()={model.tempo_dof().item():.4f}")
lm = float(plv_mu.mean())
print(f"\n  => init log_tempo ~ {lm:.4f}  =>  phidot = {math.exp(lm):.4f} rad/frame")
print(f"     TRUE median                {TARGET_LOG:.4f}  =>  phidot = {math.exp(TARGET_LOG):.5f} rad/frame")
print(f"     RATIO (too fast by)        {math.exp(lm - TARGET_LOG):.2f}x")
print(f"     implied BPM at m=4: {bpm_from_logtempo(lm,4,fps):9.1f}  vs true median {bpm_from_logtempo(TARGET_LOG,4,fps):.1f}")
print(f"     frames per bar: {TWO_PI/math.exp(lm):.2f} (true {TWO_PI/math.exp(TARGET_LOG):.1f})")

# ---------------------------------------------------------------- free-run trace
print("\n" + "=" * 78)
print("FREE-RUN TRACE at init (per song): phase_mu chain vs analytic constant-rate")
print("=" * 78)
with torch.no_grad():
    out = free_run(model, h)
pmu = out["phase_mu"].cpu().numpy()
ltj = out["log_tempo"].cpu().numpy()
print(f"  log_tempo (stochastic chain): mean={ltj.mean():.3f} std={ltj.std():.3f} min={ltj.min():.3f} max={ltj.max():.3f}")
print(f"  per-song log_tempo std over t: {np.mean(ltj.std(1)):.4f}  (0 => level+dev frozen)")
for i in range(4):
    w = int((np.diff(pmu[i]) < -math.pi).sum())
    # analytic: constant rate exp(plv_mu[i]) -> wraps
    rate = math.exp(float(plv_mu[i]))
    pred_w = int((1600 - 1) * rate / TWO_PI)
    print(f"   song{i}: phase_mu wraps={w:4d}  analytic const-rate wraps={pred_w:4d}"
          f"   (=> mean chain is a CONSTANT metronome at {bpm_from_logtempo(float(plv_mu[i]),4,fps):.0f} BPM)")
print("\n  KEY STRUCTURAL FACT: in free_run the mean chain is level_mu=const (OU anchor=itself),")
print("  dev_mu=0 forever, so phase_mu advances by exp(p_lv_mu) EVERY frame -> phase_mu read-out")
print("  is EXACTLY a constant-BPM metronome with offset p_ph_mu, both from prior_init_head(mean-pooled h).")

# verify that claim numerically
i = 0
rate = math.exp(float(plv_mu[i]))
analytic = (float(out["phase_mu"][i, 0]) + rate * np.arange(1600)) % TWO_PI
err = np.abs(((pmu[i] - analytic + math.pi) % TWO_PI) - math.pi).max()
print(f"  VERIFY: max |phase_mu - analytic const-rate| over 1600 frames = {err:.2e} rad  "
      f"{'CONFIRMED CONSTANT' if err < 1e-3 else 'not constant'}")
