"""KL( q_corrupt || p_physical ) per frame, in nats, for every point on every corruption axis.

p_physical = the hand-specified pointer kernel actually used at the published operating point:
   delta = phi_t - phi_{t-1} ~ N(step, sigma_phi=0.03)   [wrapped],  step = exp(lt)
The anchoring penalty lambda * KL(p_psi || p_physical) is charged PER FRAME, so this is the
price (in nats/frame) that lambda has to pay to drag a kernel back to the physical end.
"""
import math, json
import numpy as np

TWO_PI = 2*math.pi
G = np.linspace(-math.pi, math.pi, 400001)[:-1]      # increment grid, wrapped
DX = G[1]-G[0]

def wrapped_normal(mu, s, nw=6):
    d = np.zeros_like(G)
    for k in range(-nw, nw+1):
        d += np.exp(-0.5*((G-mu+k*TWO_PI)/s)**2)/(s*math.sqrt(TWO_PI))
    return d

def wrapped_cauchy(mu, rho):
    if rho <= 0: return np.full_like(G, 1.0/TWO_PI)
    if rho >= 1: return None            # point mass: KL to any density is +inf
    return (1-rho**2)/(TWO_PI*(1+rho**2-2*rho*np.cos(G-mu)))

def kl(q, p):
    if q is None: return float('inf')
    q = q/ (q.sum()*DX); p = p/(p.sum()*DX)
    m = q > 1e-300
    return float(np.sum(q[m]*np.log(q[m]/np.maximum(p[m],1e-300)))*DX)

def curves(step=0.0832, s_phys=0.03, s_lt_phys=0.05):
    P = wrapped_normal(step, s_phys)
    out = {}
    out['rev_p'] = {p: kl((1-p)*wrapped_normal(step, s_phys) + p*wrapped_normal(-step, s_phys), P)
                    for p in [0.0,0.01,0.02,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50]}
    rr = [1.0,0.9999,0.9995,0.999,0.995,0.99,0.98,0.95,0.90,0.80,0.60,0.30,0.0]
    out['cauchy_rho'] = {r: kl(wrapped_cauchy(step, r), P) for r in rr}
    fine = np.linspace(0.990, 0.99999, 400)
    kf = [kl(wrapped_cauchy(step, r), P) for r in fine]
    j = int(np.argmin(kf))
    out['_cauchy_kl_min'] = {'rho_star': float(fine[j]), 'kl_min': float(kf[j])}
    out['gauss_sphi'] = {s: kl(wrapped_normal(step, s), P)
                         for s in [0.03,0.06,0.10,0.15,0.20,0.30,0.40,0.80,1.50,3.00]}
    # tempo axis: phase kernel untouched, KL is on the log-bar-advance random walk
    out['sigma_lt'] = {s: float(math.log(s_lt_phys/s) + s**2/(2*s_lt_phys**2) - 0.5)
                       for s in [0.05,0.10,0.20,0.40,0.80,1.60]}
    return out

if __name__ == '__main__':
    c = curves()
    for ax, d in c.items():
        print(f'--- {ax} : KL(q||p_physical) nats/frame')
        for k, v in d.items():
            print(f'    {k:>8} : {v:12.4g}' + ('  (rho=1 is a point mass -> KL infinite in the limit)' if v==float('inf') else ''))
    json.dump({a:{str(k):v for k,v in d.items()} for a,d in c.items()},
              open('/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise/kl_to_physical.json','w'), indent=1)
