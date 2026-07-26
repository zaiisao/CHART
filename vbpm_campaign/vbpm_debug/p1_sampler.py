"""P1: wrapped-Cauchy sampler + KL sanity, and what rho actually buys you in RADIANS."""
import sys, math
import numpy as np, torch
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
from vbpm.distributions import sample_wrapped_cauchy, kl_wrapped_cauchy, TWO_PI

torch.manual_seed(0)
dev = "cuda:2"
N = 400000

def circ(x):  # wrap to (-pi, pi]
    return (x + math.pi) % TWO_PI - math.pi

print("="*88)
print("P1a  sampler correctness: WC(mu,rho) must have E[cos(phi-mu)] = rho exactly")
print("="*88)
print(f"{'rho':>6} {'gamma=-log rho':>14} {'R_emp':>8} {'R_theory':>9} {'medAbsDev':>10} {'P(|d|>0.1)':>11} {'P(|d|>0.5)':>11} {'P(|d|>pi/2)':>12}")
mu = torch.full((N,), 1.0, device=dev)
for rho_v in [0.10, 0.30, 0.50, 0.70, 0.90, 0.95, 0.99, 0.999]:
    rho = torch.full((N,), rho_v, device=dev)
    phi = sample_wrapped_cauchy(mu, rho)
    d = circ((phi - mu).cpu().numpy())
    R = float(np.cos(d).mean())
    print(f"{rho_v:6.3f} {-math.log(rho_v):14.4f} {R:8.4f} {rho_v:9.4f} "
          f"{np.median(np.abs(d)):10.4f} {np.mean(np.abs(d)>0.1):11.4f} "
          f"{np.mean(np.abs(d)>0.5):11.4f} {np.mean(np.abs(d)>math.pi/2):12.4f}")

print()
print("="*88)
print("P1b  closed-form KL vs MC (is kl_wrapped_cauchy right?)")
print("="*88)
def mc_kl(mq, rq, mp, rp, n=2000000):
    # WC density: (1/2pi)(1-r^2)/(1+r^2-2r cos(x-mu))
    mqt = torch.full((n,), mq, device=dev); rqt = torch.full((n,), rq, device=dev)
    x = sample_wrapped_cauchy(mqt, rqt)
    def lp(x, mu, r):
        return math.log(1.0/TWO_PI) + math.log(1-r*r) - torch.log(1+r*r-2*r*torch.cos(x-mu))
    return float((lp(x, mq, rq) - lp(x, mp, rp)).mean())
for (mq, rq, mp, rp) in [(1.0,0.5,1.0,0.5),(1.0,0.9,1.0,0.5),(1.0,0.5,2.0,0.5),(0.5,0.95,2.5,0.3)]:
    cf = float(kl_wrapped_cauchy(torch.tensor(mq), torch.tensor(rq), torch.tensor(mp), torch.tensor(rp)))
    print(f"  q(mu={mq},rho={rq}) || p(mu={mp},rho={rp}):  closed={cf:9.5f}  MC={mc_kl(mq,rq,mp,rp):9.5f}")

print()
print("="*88)
print("P1c  KL price of a SHARP phase posterior (mu_q == mu_p), per frame and per 256-frame crop")
print("="*88)
rp = torch.tensor(0.5)
for rq_v in [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]:
    k = float(kl_wrapped_cauchy(torch.tensor(0.0), torch.tensor(rq_v), torch.tensor(0.0), rp))
    print(f"  rho_q={rq_v:6.3f} (gamma={-math.log(rq_v):.4f} rad)  KL/frame={k:8.3f} nats  KL over T=256: {k*256:9.1f} nats")
print("  (recon headroom for a whole 256-frame crop with ~8 beats is only ~tens of nats)")

print()
print("="*88)
print("P1d  ACCUMULATION: what a Cauchy-noised phase chain does over 256 frames")
print("="*88)
# stochastic free-run chain: phi_t = WC(phi_{t-1} + dphi, rho); count spurious/missed wraps
true_dphi = 0.0628  # 120bpm 4/4 @50fps
for rho_v in [0.5, 0.9, 0.99]:
    B, T = 2000, 256
    phi = torch.zeros(B, device=dev)
    traj = []
    for t in range(T):
        phi = sample_wrapped_cauchy(phi + true_dphi, torch.full((B,), rho_v, device=dev))
        traj.append(phi)
    tr = torch.stack(traj, 1).cpu().numpy()
    d = np.diff(tr, axis=1)
    wraps = (d < -math.pi).sum(1)
    fwd = (d > math.pi).sum(1)
    print(f"  rho={rho_v:5.2f}: mean 2pi-down-wraps/256fr = {wraps.mean():7.2f} (TRUE = {256*true_dphi/TWO_PI:.2f}),"
          f" spurious UP-jumps = {fwd.mean():6.2f}")
print("  a coherent chain would give ~2.56 down-wraps and 0 up-jumps in 256 frames")
