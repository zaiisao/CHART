"""PROBE C: (d) is the TRAINING rollout the same recursion as free_run? and (e) is the
emission collapsed already at training time? Quantifies every asymmetry numerically.
"""
import sys, glob, math
import numpy as np, torch
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
from vbpm.model import BarPointerVAE
from vbpm.elbo import strict_elbo, free_run
from vbpm.distributions import TWO_PI
from vbpm.evaluate import beats_from_barphase, beats_from_activation, f_measure, _estimate_meter

CACHE = "/disk1/jaehoon/vbpm_mert_cache"; dev = "cuda:2"; fps = 50.0; H_DIM = 8
CKPT = sys.argv[1] if len(sys.argv) > 1 else "/home/sogang/jaehoon/VBPM_reintegration/vbpm_debug/dirac_step600.pt"
FR = 256


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


ck = torch.load(CKPT, map_location=dev)
model = BarPointerVAE(h_dim=H_DIM, hidden=128, num_meters=4).to(dev)
model.load_state_dict(ck["sd"]); model.eval()
print(f"loaded {CKPT} (step {ck['step']})")
ev = load("eval", 30)
torch.manual_seed(0); np.random.seed(0); rng = np.random.default_rng(0)

# ---- build a batch of 16 crops exactly like training ----
hs, bs_, ds = [], [], []
for i in range(16):
    s = ev[i]
    st = int(rng.integers(0, max(1, s["T"] - FR)))
    hs.append(torch.from_numpy(dirac_h(s["beats"], s["downs"], st, FR)))
    b, d = targets(s["beats"], s["downs"], st, FR)
    bs_.append(torch.from_numpy(b)); ds.append(torch.from_numpy(d))
h = torch.stack(hs).to(dev); b = torch.stack(bs_).to(dev); d = torch.stack(ds).to(dev)

with torch.no_grad():
    loss, info = strict_elbo(model, h, b, d, temperature=0.3, beta=1.0)

pb = b.mean().item(); pd = d.mean().item()
base_b = -(pb * math.log(pb) + (1 - pb) * math.log(1 - pb)) * FR
base_d = -(pd * math.log(pd) + (1 - pd) * math.log(1 - pd)) * FR
print()
print("=" * 90); print("(e-1) EMISSION COLLAPSE AT TRAINING TIME (teacher-forced, best case)"); print("=" * 90)
print(f"  p(beat)={pb:.4f}  p(downbeat)={pd:.4f}")
print(f"  recon_beat = {info['recon_beat']:.2f}   base-rate (constant-prob) NLL = {base_b:.2f}"
      f"   -> {'*** AT BASE RATE = COLLAPSED ***' if info['recon_beat'] > 0.95*base_b else 'informative'}")
print(f"  recon_db   = {info['recon_db']:.2f}   base-rate NLL = {base_d:.2f}"
      f"   -> {'*** AT BASE RATE = COLLAPSED ***' if info['recon_db'] > 0.95*base_d else 'informative'}")
bp = info["beat_prob"].cpu().numpy(); dbp = info["db_prob"].cpu().numpy()
bt = b.cpu().numpy()
print(f"  train beat_prob: min={bp.min():.5f} max={bp.max():.5f} mean={bp.mean():.5f} std={bp.std():.5f}")
print(f"    at true beats  : {bp[bt>0.5].mean():.5f}   elsewhere: {bp[bt<0.5].mean():.5f}   "
      f"separation ratio={bp[bt>0.5].mean()/max(bp[bt<0.5].mean(),1e-9):.3f}")
print(f"  KL terms: phase={info['kl_phase']:.2f} level={info['kl_level']:.2f} dev={info['kl_dev']:.2f} "
      f"meter={info['kl_meter']:.2f}  n_cross={info['n_cross']:.1f}/{FR}")

# ---- posterior phase quality ----
qpm = info["post_phase_mu"].cpu().numpy()
print()
print("=" * 90); print("(e-2) POSTERIOR (training) phase: is it a clean pointer at the true rate?"); print("=" * 90)
dq = (np.diff(qpm, axis=1) + math.pi) % TWO_PI - math.pi
print(f"  posterior phase circular increment: mean={dq.mean():+.4f} rad/frame  std={dq.std():.4f}  "
      f"frac_neg={float((dq<0).mean()):.3f}")
print(f"  -> implied bar period {TWO_PI/max(dq.mean(),1e-9):.1f} frames; TRUE bar period ~"
      f"{np.mean([np.median(np.diff(ev[i]['beats']))*_estimate_meter(ev[i]['beats'],ev[i]['downs'])*fps for i in range(16)]):.1f} frames")

# ---- deploy vs train, same crops ----
with torch.no_grad():
    fo = free_run(model, h, temperature=0.3)
fpm = fo["phase_mu"].cpu().numpy(); fdec = fo["decoder_prob"].cpu().numpy()
fph = fo["phase"].cpu().numpy(); flt = fo["log_tempo"].cpu().numpy()
dfp = (np.diff(fpm, axis=1) + math.pi) % TWO_PI - math.pi
dfs = (np.diff(fph, axis=1) + math.pi) % TWO_PI - math.pi
print()
print("=" * 90); print("(d) TRAIN ROLLOUT vs FREE-RUN on the SAME 16 crops"); print("=" * 90)
print(f"  {'quantity':38s} {'TRAIN (posterior)':>22s} {'DEPLOY (free_run)':>22s}")
print(f"  {'phase increment mean (rad/frame)':38s} {dq.mean():22.5f} {dfs.mean():22.5f}")
print(f"  {'phase increment std':38s} {dq.std():22.5f} {dfs.std():22.5f}")
print(f"  {'MEAN-chain phase increment':38s} {'n/a (no mean chain)':>22s} {dfp.mean():22.5f}")
print(f"  {'log_tempo mean':38s} {'-':>22s} {flt.mean():22.5f}")
print(f"  {'beat_prob mean':38s} {bp.mean():22.5f} {fdec.mean():22.5f}")
print(f"  {'beat_prob std':38s} {bp.std():22.5f} {fdec.std():22.5f}")
print(f"  {'beat_prob max':38s} {bp.max():22.5f} {fdec.max():22.5f}")
print(f"  {'beat_prob at true beats':38s} {bp[bt>0.5].mean():22.5f} {fdec[bt>0.5].mean():22.5f}")
print(f"  {'beat_prob elsewhere':38s} {bp[bt<0.5].mean():22.5f} {fdec[bt<0.5].mean():22.5f}")

# ---- z-feature distribution shift: what the decoder SEES at train vs deploy ----
print()
print("=" * 90); print("(d-2) DECODER INPUT DISTRIBUTION SHIFT (z_features)"); print("=" * 90)
print(f"  log_tempo fed to decoder:  TRAIN not exposed directly; DEPLOY mean={flt.mean():.3f} std={flt.std():.3f} "
      f"min={flt.min():.3f} max={flt.max():.3f}")
print(f"  (physically correct log bar-phidot ~ {math.log(TWO_PI/(0.5*4*fps)):.3f})")

print()
print("=" * 90); print("(d-3) EXPLICIT recursion diff train vs deploy"); print("=" * 90)
diffs = [
 ("t=1 phase source",      "q_ph_mu,q_ph_rho (posterior, reads b)", "p_ph_mu,p_ph_rho from prior_init_head(prior_ctx.MEAN(1))"),
 ("t=1 level source",      "StudentT(q_lv_mu,q_lv_s)",              "StudentT(p_lv_mu,p_lv_s)"),
 ("t=1 dev source",        "q_dv_mu + q_dv_s*eps  (nonzero mean)",  "0 + p_dv_s*eps       (mean 0)"),
 ("t>1 phase",             "q sample; prior mean only via KL",      "sample from prior mean = dead reckoning"),
 ("t>1 meter at crossing", "gumbel(q_m) (posterior logits)",        "gumbel(meter_prior_logp)"),
 ("OU level anchor",       "level_anchor = t=1 POSTERIOR level",    "level_anchor = t=1 PRIOR StudentT draw (nu~3 lottery)"),
 ("MEAN chain phi_mu",     "*** DOES NOT EXIST -> never trained ***","exists and IS the scored read-out"),
 ("exp() clamp on advance","clamp(-12,6) on log_tempo_prev",        "clamp on stochastic chain; NO clamp on exp(log_tempo_mu)"),
]
for a, t, dd in diffs:
    print(f"  {a:24s}\n      train : {t}\n      deploy: {dd}")
