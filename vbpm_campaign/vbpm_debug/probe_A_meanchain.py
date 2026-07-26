"""PROBE A (structural, untrained model): what IS the free_run read-out chain?

Claim under test: evaluate.py scores `phase_mu`, and in elbo.free_run the phase_mu chain is a
CLOSED-FORM CONSTANT-RATE sawtooth determined by exactly TWO scalars from
prior_init_head(prior_ctx.mean(1)) -- p_ph_mu (initial phase) and p_lv_mu (log tempo) --
with NO per-frame audio input and NO coupling to the stochastic chain.

Verified by predicting phase_mu[t] = (p_ph_mu + t*exp(p_lv_mu)) mod 2pi and comparing.
"""
import sys, math, glob
import numpy as np, torch
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
from vbpm.model import BarPointerVAE
from vbpm.elbo import free_run
from vbpm.distributions import TWO_PI

dev = "cuda:2"; fps = 50.0
torch.manual_seed(0)
m0 = BarPointerVAE(h_dim=8, hidden=128, num_meters=4).to(dev).eval()

T = 600
h = torch.randn(3, T, 8, device=dev) * 0.01
out = free_run(m0, h)

pm = out["phase_mu"].cpu().numpy()
lt = out["log_tempo"].cpu().numpy()
ph = out["phase"].cpu().numpy()

# recover the two scalars directly
with torch.no_grad():
    pc = m0.encode_prior(h)
    p_m, p_ph_mu, p_ph_rho, p_lv_mu, p_lv_s, _a, _b = m0.unpack(m0.prior_init_head(pc.mean(1)))
p_ph_mu_np = (p_ph_mu % TWO_PI).cpu().numpy(); p_lv_mu_np = p_lv_mu.cpu().numpy()

print("=" * 78)
print("A1  phase_mu is a closed-form constant-rate sawtooth (NO per-frame audio)")
print("=" * 78)
for i in range(3):
    pred = (p_ph_mu_np[i] + np.arange(T) * math.exp(p_lv_mu_np[i])) % TWO_PI
    err = np.abs(((pm[i] - pred + math.pi) % TWO_PI) - math.pi)
    print(f"  seq{i}: p_ph_mu={p_ph_mu_np[i]:.4f} p_lv_mu={p_lv_mu_np[i]:.4f} "
          f"-> rate={math.exp(p_lv_mu_np[i]):.4f} rad/frame | max|phase_mu - closed_form| = {err.max():.3e}")
print("  => the deploy read-out is an OPEN-LOOP METRONOME parameterised by 2 scalars.")

print()
print("=" * 78)
print("A2  wrap rate vs physically-correct rate")
print("=" * 78)
true_rate = 0.5 * TWO_PI / fps   # 120bpm, 4/4, 50fps -> bar advance rad/frame
for i in range(3):
    w = int((np.diff(pm[i]) < -math.pi).sum())
    print(f"  seq{i}: phase_mu bar wraps in {T} frames ({T/fps:.0f}s) = {w}  (true@120bpm ~ {T*true_rate/TWO_PI:.1f})"
          f"  ratio={w/(T*true_rate/TWO_PI):.1f}x")
print(f"  correct log rate = {math.log(true_rate):.3f}; model p_lv_mu mean = {p_lv_mu_np.mean():.3f}")

print()
print("=" * 78)
print("A3  the STOCHASTIC phase chain (what feeds the decoder) -- is it even monotone?")
print("=" * 78)
for i in range(3):
    d = np.diff(ph[i])
    dd = (d + math.pi) % TWO_PI - math.pi        # signed circular increment
    print(f"  seq{i}: circ-increment mean={dd.mean():+.4f} std={dd.std():.4f} "
          f"frac_negative={float((dd<0).mean()):.3f}  (a clean pointer would be ~all positive, tiny std)")
    print(f"          log_tempo: mean={lt[i].mean():.3f} std={lt[i].std():.3f} min={lt[i].min():.3f} max={lt[i].max():.3f}")
with torch.no_grad():
    rho = m0.prior_phase_conc(pc[:, 1:])
print(f"  prior phase rho (concentration): mean={float(rho.mean()):.4f} -> wrapped-Cauchy gamma=-log rho ="
      f" {float(-torch.log(rho).mean()):.4f} rad half-width")

print()
print("=" * 78)
print("A4  does free_run touch b at all?  (leakage audit)")
print("=" * 78)
import inspect
src = inspect.getsource(free_run)
print("  free_run signature:", str(inspect.signature(free_run)))
print("  mentions 'post_'   :", "post_" in src, " | mentions 'encode_posterior':", "encode_posterior" in src)
print("  mentions ' b '     :", any(tok in src for tok in ["(h, b", "b)", "b,"]) and "b=" in src)
print("  => only h enters. No beat leakage in free_run itself.")
