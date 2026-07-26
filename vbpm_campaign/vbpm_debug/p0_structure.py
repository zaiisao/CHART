"""P0: STRUCTURAL claim -- what does the deployed beat_phase read-out actually depend on?

Claim to test: in vbpm.elbo.free_run, the `phase_mu` chain (the ONLY thing
beats_from_barphase / downbeats_from_barphase consume) is a *deterministic constant-tempo
ramp* whose two free parameters are BOTH produced by prior_init_head(prior_ctx.mean(1)):
    phi_mu[0]   = p_ph_mu
    log_tempo_mu = p_lv_mu   (constant for all t, because level_mu_anchor == level_mu)
    phi_mu[t]   = (phi_mu[t-1] + exp(p_lv_mu)) mod 2pi
If true: meter, the wrapped-Cauchy sampler, the AR(1) deviation, the decoder and every
per-frame prior head are IRRELEVANT to the reported free-run beat_F.
"""
import sys, math
import numpy as np, torch
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
from vbpm.model import BarPointerVAE
from vbpm.elbo import free_run
from vbpm.distributions import TWO_PI

dev = "cuda:1"
torch.manual_seed(0)
m = BarPointerVAE(h_dim=8, hidden=128, num_meters=4).to(dev)
h = torch.randn(3, 400, 8, device=dev) * 0.01

torch.manual_seed(11); o1 = free_run(m, h)
torch.manual_seed(22); o2 = free_run(m, h)
pm1 = o1["phase_mu"].cpu().numpy(); pm2 = o2["phase_mu"].cpu().numpy()
print("=" * 78)
print("P0a  phase_mu reproducibility across two different sampling seeds")
print(f"   max |phase_mu(seed11) - phase_mu(seed22)| = {np.abs(pm1-pm2).max():.3e}")
print(f"   (sampled phase differs? max|d phase| = {np.abs(o1['phase'].cpu().numpy()-o2['phase'].cpu().numpy()).max():.3f})")
print(f"   VERDICT: phase_mu is {'DETERMINISTIC (noise-free, sampler-independent)' if np.abs(pm1-pm2).max()<1e-6 else 'stochastic'}")

# is it a constant-rate ramp?
d = np.diff(pm1, axis=1) % TWO_PI
print()
print("P0b  is phase_mu a CONSTANT-tempo ramp?")
for b in range(3):
    print(f"   song{b}: d(phase_mu) mean={d[b].mean():.6f} std={d[b].std():.3e} "
          f"min={d[b].min():.6f} max={d[b].max():.6f}  rad/frame")
lt = o1["log_tempo"].cpu().numpy()
print(f"   sampled log_tempo (stochastic path): mean={lt.mean():.3f} std={lt.std():.3f}")
print(f"   phase_mu implied log_tempo = {np.log(d[:,0])}")

# compare with a hand-rolled 2-scalar reconstruction from prior_init_head
with torch.no_grad():
    pc = m.encode_prior(h)
    p = m.unpack(m.prior_init_head(pc.mean(1)))
    p_ph_mu, p_lv_mu = p[1], p[3]
    rec = [(p_ph_mu % TWO_PI)]
    for t in range(1, 400):
        rec.append((rec[-1] + torch.exp(p_lv_mu)) % TWO_PI)
    rec = torch.stack(rec, 1).cpu().numpy()
print()
print("P0c  2-scalar reconstruction (init phase + init level ONLY, no per-frame heads)")
print(f"   max |free_run.phase_mu - hand_ramp| = {np.abs(pm1-rec).max():.3e}")
print(f"   p_ph_mu = {p_ph_mu.cpu().numpy()}   p_lv_mu = {p_lv_mu.cpu().numpy()}")
print(f"   exp(p_lv_mu) = {torch.exp(p_lv_mu).cpu().numpy()} rad/frame")
print(f"   physically correct 120bpm 4/4 @50fps = {0.5*TWO_PI/50:.5f} rad/frame (log={math.log(0.5*TWO_PI/50):.3f})")
print(f"   VERDICT: {'CONFIRMED -- deploy beat read-out = 2 scalars from a MEAN-POOLED head' if np.abs(pm1-rec).max()<1e-5 else 'refuted'}")

# how many bar wraps
print()
print("P0d  bar wraps in 400 frames (8 s): ", [(np.diff(pm1[b]) < -math.pi).sum() for b in range(3)],
      " (true ~4 bars at 120bpm 4/4)")
