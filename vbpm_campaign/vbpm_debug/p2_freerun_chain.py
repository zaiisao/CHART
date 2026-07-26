"""P2: what does the DEPLOY read-out chain (phase_mu) actually do?

Claim under test (d): free_run's phase_mu advances by exp(log_tempo_mu) where
level_mu = anchor + a_lv*(level_mu - anchor)  with level_mu INITIALISED to anchor
        => level_mu == anchor for all t (fixed point)
dev_mu  = a * dev_mu with dev_mu init 0 => 0 for all t
=> log_tempo_mu is EXACTLY CONSTANT over the whole song, and equals p_lv_mu from
   prior_init_head(prior_ctx.mean(1)).  i.e. the deploy read-out is a METRONOME.
Verified numerically from the returned phase_mu (no source mutation).
"""
import sys, math, glob
import numpy as np, torch
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
from vbpm.model import BarPointerVAE
from vbpm.elbo import free_run
from vbpm.distributions import TWO_PI

dev = "cuda:2"; fps = 50.0
torch.manual_seed(0)
model = BarPointerVAE(h_dim=8, hidden=128, num_meters=4).to(dev)

# --- Dirac h from real songs ---
CACHE = "/disk1/jaehoon/vbpm_mert_cache"
def load(split, cap=None):
    out = []
    for f in sorted(glob.glob(f"{CACHE}/{split}__*.npz")):
        d = np.load(f, allow_pickle=True)
        out.append(dict(key=f.split("__")[1][:-4], T=int(d["feats"].shape[1]),
                        beats=np.asarray(d["beats"], float), downs=np.asarray(d["downs"], float)))
        if cap and len(out) >= cap: break
    return out
ev = load("eval", 8)

def dirac_h(beats, downs, n, H=8):
    h = np.random.randn(n, H).astype(np.float32) * 0.01
    for t in beats:
        i = int(round(t * fps))
        if 0 <= i < n: h[i, 0] += 1.0
    for t in downs:
        i = int(round(t * fps))
        if 0 <= i < n: h[i, 1] += 1.0
    return h

T = 600
hs = np.stack([dirac_h(s["beats"], s["downs"], T) for s in ev])
h = torch.from_numpy(hs).to(dev)
out = free_run(model, h)
pm = out["phase_mu"].cpu().numpy()          # [B,T]
ph = out["phase"].cpu().numpy()
lt = out["log_tempo"].cpu().numpy()

print("="*92)
print("P2a  IS phase_mu A CONSTANT-RATE METRONOME?  (unwrapped per-frame advance)")
print("="*92)
d = np.diff(pm, axis=1) % TWO_PI            # advance per frame, unwrapped
print(f"{'song':<34} {'adv[0]':>9} {'adv[-1]':>9} {'std(adv)':>11} {'max|adv-adv0|':>14} {'implied BPM(m=4)':>17}")
for i, s in enumerate(ev):
    a = d[i]
    bpm = 60.0 * fps * 4 * a[0] / TWO_PI
    print(f"{s['key'][:34]:<34} {a[0]:9.5f} {a[-1]:9.5f} {a.std():11.3e} {np.abs(a-a[0]).max():14.3e} {bpm:17.1f}")
tru = [60.0/np.median(np.diff(s["beats"])) for s in ev]
print(f"  TRUE beat BPMs of these songs: {['%.1f'%x for x in tru]}")
print("  -> if std(adv) == 0 the deploy read-out is EXACTLY a metronome: it cannot track any tempo change.")

print()
print("="*92)
print("P2b  DEPLOY RATE = ONE SCALAR from prior_init_head.  Where does it come from?")
print("="*92)
with torch.no_grad():
    pctx = model.encode_prior(h)
    pk = model.unpack(model.prior_init_head(pctx.mean(1)))
    p_lv_mu = pk[3]
    print(f"  p_lv_mu (init-head level mean)  = {p_lv_mu.cpu().numpy().round(5)}")
    print(f"  exp(p_lv_mu)                    = {p_lv_mu.exp().cpu().numpy().round(5)}  rad/frame")
    print(f"  measured phase_mu advance       = {d[:,0].round(5)}  rad/frame")
    print(f"  identical? max|diff| = {np.abs(d[:,0]-p_lv_mu.exp().cpu().numpy()).max():.3e}")
    print(f"  physically correct for these songs (rad/frame) = "
          f"{[round(TWO_PI/(60.0/b*4*fps),5) for b in tru]}")
print("  NOTE: prior_init_head is used in strict_elbo ONLY at t=1 (p_m,p_ph_mu,p_lv_mu,p_lv_s),")
print("        so this scalar receives gradient from ONE frame per 256-frame crop.")

print()
print("="*92)
print("P2c  STOCHASTIC phase chain (the one z_features/decoder actually sees) coherence")
print("="*92)
dph = np.diff(ph, axis=1)
print(f"{'song':<28} {'down-wraps':>11} {'up-jumps':>9} {'true bars':>10} {'med|adv|':>9} {'frac |adv|>0.5':>15}")
for i, s in enumerate(ev):
    dn = int((dph[i] < -math.pi).sum()); up = int((dph[i] > math.pi).sum())
    truebars = int(((s["downs"] < T/fps)).sum())
    adv = (dph[i] + math.pi) % TWO_PI - math.pi
    print(f"{s['key'][:28]:<28} {dn:11d} {up:9d} {truebars:10d} {np.median(np.abs(adv)):9.4f} {np.mean(np.abs(adv)>0.5):15.3f}")

print()
print("="*92)
print("P2d  prior phase concentration rho_p actually used in free-run (per frame)")
print("="*92)
with torch.no_grad():
    rp = torch.stack([model.prior_phase_conc(pctx[:, t]) for t in range(T)], 1)
r = rp.cpu().numpy()
print(f"  rho_p: min={r.min():.4f} mean={r.mean():.4f} max={r.max():.4f}  -> gamma=-log rho: "
      f"min={-math.log(r.max()):.4f} mean={-math.log(r.mean()):.4f} max={-math.log(r.min()):.4f} rad")
print(f"  true per-frame phase advance is ~{TWO_PI/(0.5*4*fps):.4f} rad; the prior's own Cauchy scale is "
      f"{-math.log(r.mean())/(TWO_PI/(0.5*4*fps)):.1f}x the whole advance")
