"""ENCODER PROBE A (static / init):
(a) what does the encoder actually SEE?  stats of cat([h, b])
(d) post_head input geometry: shapes + how much of the pre-activation comes from z_prev
(e) atan2(u,v) phase resolution: radius r=|(u,v)| at init, achievable phase increments
(c-init) per-latent KL at init
"""
import sys, math
import numpy as np, torch
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_debug")
from enc_common import *  # noqa
from vbpm.model import BarPointerVAE
from vbpm.elbo import strict_elbo

dev = "cuda:3"
torch.manual_seed(0)
rng = np.random.default_rng(0)
ev = load("eval", 30)
FR = 256

# ---------- build one batch exactly like probe_dirac ----------
hs, bs_, ds, phs, oks, pds = [], [], [], [], [], []
for s in ev[:8]:
    st = 200
    hs.append(dirac_h(s["beats"], s["downs"], st, FR, rng))
    b, d = targets(s["beats"], s["downs"], st, FR)
    bs_.append(b); ds.append(d)
    ph, ok = oracle_barphase(s["beats"], s["downs"], st, FR)
    phs.append(ph); oks.append(ok); pds.append(true_phidot(s))
h = torch.from_numpy(np.stack(hs)).to(dev)
b = torch.from_numpy(np.stack(bs_)).to(dev)
d = torch.from_numpy(np.stack(ds)).to(dev)
phs = np.stack(phs); oks = np.stack(oks); pds = np.array(pds)

print("=" * 78)
print("(a) ENCODER INPUT  x = cat([h, b])  -- is the Dirac informative and is b visible?")
print("=" * 78)
x = torch.cat([h, b.unsqueeze(-1)], dim=-1)
print(f"  h        shape {tuple(h.shape)}  min {h.min():.3f} max {h.max():.3f} mean {h.mean():.5f} std {h.std():.4f}")
print(f"  x        shape {tuple(x.shape)}  (h_dim+1 = {x.shape[-1]})")
for c in range(x.shape[-1]):
    col = x[..., c]
    on = (col > 0.5).float().mean().item()
    print(f"    ch{c}: min {col.min():+.3f} max {col.max():+.3f} std {col.std():.4f}  frac>0.5 = {on:.4f}"
          + ("   <- BEAT impulses" if c == 0 else "   <- DOWNBEAT impulses" if c == 1
             else "   <- b TARGET (beats)" if c == x.shape[-1] - 1 else "   (noise only)"))
print(f"  beat frames per 256: {(b>0.5).sum(1).tolist()}   downbeat frames: {(d>0.5).sum(1).tolist()}")
print(f"  agreement h[:,:,0]>0.5 vs b>0.5 : {((h[...,0]>0.5)==(b>0.5)).float().mean():.4f}  (1.0 = identical)")
print(f"  true phidot per song (rad/frame): {np.round(pds,4).tolist()}")
print(f"    -> log_tempo TRUE mean = {np.log(pds).mean():+.3f}")

model = BarPointerVAE(h_dim=H_DIM, hidden=128, num_meters=4).to(dev)
print()
print("=" * 78)
print("(d) post_head INPUT GEOMETRY: cat([context_t, z_prev_feat])")
print("=" * 78)
post_ctx = model.encode_posterior(h, b)
print(f"  post_ctx  {tuple(post_ctx.shape)}   min {post_ctx.min():+.3f} max {post_ctx.max():+.3f} std {post_ctx.std():.4f}")
z0 = model.z0.unsqueeze(0).expand(h.shape[0], -1)
print(f"  z_feat_dim = {model.z_feat_dim}  (cos, sin, log_tempo, onehot(K={model.K}))")
cat = torch.cat([post_ctx[:, 5], z0], -1)
print(f"  post_head input {tuple(cat.shape)} = hidden {model.hidden} + z_feat {model.z_feat_dim}")

# how much of the first-layer pre-activation is DRIVEN by z_prev vs by context?
W = model.post_head[0].weight            # [hidden, hidden+zdim]
Wc, Wz = W[:, :model.hidden], W[:, model.hidden:]
# realistic z_prev feature scale: cos/sin in [-1,1], log_tempo ~ true (-3), meter onehot
zp = torch.zeros(1, model.z_feat_dim, device=dev)
zp[0, 0] = 1.0; zp[0, 1] = 0.0; zp[0, 2] = -3.0; zp[0, 3 + 2] = 1.0
ctx_contrib = (post_ctx[:, 5] @ Wc.T)
z_contrib = (zp @ Wz.T)
print(f"  |context contribution| std {ctx_contrib.std():.4f}   |z_prev contribution| std {z_contrib.std():.4f}"
      f"   ratio z/ctx = {(z_contrib.std()/ctx_contrib.std()):.3f}")
# the phase part of z_prev alone (cos, sin only)
zp_ph = torch.zeros(1, model.z_feat_dim, device=dev); zp_ph[0, 0] = 1.0
print(f"  ONLY (cos,sin) part: std {(zp_ph @ Wz.T).std():.4f}"
      f"   -> phase-carry / context ratio = {((zp_ph @ Wz.T).std()/ctx_contrib.std()):.4f}")
# sensitivity of head OUTPUT to a 1-frame phase advance in z_prev (0.063 rad)
def head_out(zvec, ctx):
    return model.post_head(torch.cat([ctx, zvec], -1))
ctx1 = post_ctx[:1, 5]
outs = []
for dphi in [0.0, 0.063, 0.63, 3.14]:
    zz = torch.zeros(1, model.z_feat_dim, device=dev)
    zz[0, 0] = math.cos(dphi); zz[0, 1] = math.sin(dphi); zz[0, 2] = -3.0; zz[0, 5] = 1.0
    outs.append(head_out(zz, ctx1))
base = outs[0]
for k, dphi in enumerate([0.063, 0.63, 3.14]):
    print(f"    d(head output) for z_prev phase shift {dphi:.3f} rad: L2 {(outs[k+1]-base).norm():.5f}"
          f"  (vs output scale {base.norm():.4f})")

print()
print("=" * 78)
print("(e) atan2(u,v) PHASE RESOLUTION at init")
print("=" * 78)
with torch.no_grad():
    vecs = torch.stack([model.post_head(torch.cat([post_ctx[:, t], z0], -1)) for t in range(0, 256)], 1)
u, v = vecs[..., model.K], vecs[..., model.K + 1]
r = torch.sqrt(u ** 2 + v ** 2)
print(f"  u range [{u.min():+.4f},{u.max():+.4f}]  v range [{v.min():+.4f},{v.max():+.4f}]")
print(f"  radius r = |(u,v)| : mean {r.mean():.5f} min {r.min():.6f} max {r.max():.5f}")
print(f"  -> d(phase)/d(u,v) ~ 1/r = {1.0/r.mean():.1f} rad per unit; a 0.063-rad step needs"
      f" |d(u,v)| = {0.063*r.mean():.6f}")
ph_init = torch.atan2(v, u) % TWO_PI
dph = circ_diff(ph_init[:, 1:].cpu().numpy(), ph_init[:, :-1].cpu().numpy())
print(f"  q phase_mu(t) at init: per-frame increment mean {dph.mean():+.4f} std {dph.std():.4f}"
      f" (TRUE should be ~{pds.mean():+.4f})")
print(f"  |increment| median {np.median(np.abs(dph)):.4f} vs true {pds.mean():.4f}"
      f"  -> init phase is {np.median(np.abs(dph))/pds.mean():.1f}x too JUMPY")
# float resolution sanity: can atan2 express 0.063 rad steps at this radius?
uu = torch.tensor([1.0], dtype=torch.float32); vv = torch.tensor([0.0])
steps = [float(torch.atan2(torch.tensor([math.sin(k*0.063)]), torch.tensor([math.cos(k*0.063)]))) for k in range(3)]
print(f"  float32 atan2 sanity: {steps} (exact 0.063 steps -> representable: "
      f"{abs((steps[1]-steps[0])-0.063)<1e-6})")

print()
print("=" * 78)
print("(c-init) PER-LATENT KL AT INIT (one strict_elbo forward, T=256)")
print("=" * 78)
torch.manual_seed(0)
loss, info = strict_elbo(model, h, b, d, temperature=1.0, beta=1.0)
for k in ["loss", "recon", "recon_beat", "recon_db", "kl", "kl_meter", "kl_phase", "kl_level",
          "kl_dev", "n_cross", "tempo_dof"]:
    print(f"  {k:12s} {info[k]:12.4f}")
pm = info["post_phase_mu"].cpu().numpy()
print(f"  post_phase_mu {pm.shape} min {pm.min():.3f} max {pm.max():.3f}")
# correlation with the ORACLE bar phase
cd = circ_diff(pm, phs)
print(f"  circular error vs ORACLE bar phase: mean|err| {np.abs(cd[oks]).mean():.3f} rad"
      f"  (chance = {math.pi/2:.3f})")
R = np.abs(np.exp(1j * cd[oks]).mean())
print(f"  circular concentration R of (q - oracle) = {R:.4f}   (0 = no relation, 1 = locked)")
