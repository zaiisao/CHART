"""Q5: what did the trained PRIOR transition become? (the other half of the deploy failure)"""
import math, sys
import numpy as np, torch
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms")
import variant_b as VB
from audit_common import load_split, FPS
ARMS = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms"
DEV = "cuda:0"
ev = load_split("eval", cap=20)
d = np.load(f"{ARMS}/act_eval.npz", allow_pickle=True)
ck = torch.load(f"{ARMS}/arm_i_ii_bern.pt", map_location=DEV)
m = VB.BarPointerVAE_B(h_dim=2, hidden=ck["config"]["hidden"], num_meters=4,
                       obs_dim=2, obs_type="bern").to(DEV)
m.load_state_dict(ck["model"]); m.eval()
rho, slv, sdv, aar = [], [], [], []
with torch.no_grad():
    for s in ev:
        a = np.clip(np.asarray(d[s["stem"] + "|act"], np.float32), 1e-4, 1 - 1e-4)
        h = torch.from_numpy(a).unsqueeze(0).to(DEV)
        c = m.encode_prior(h)[0]
        rho.append(m.prior_phase_conc(c).cpu().numpy())
        slv.append(m.prior_level_scale(c).cpu().numpy())
        sdv.append(m.prior_dev_scale(c).cpu().numpy())
        aar.append(m.prior_dev_coef(c).cpu().numpy())
r = np.concatenate(rho); l = np.concatenate(slv); v = np.concatenate(sdv); aa = np.concatenate(aar)
print(f"TRAINED PRIOR TRANSITION on {len(ev)} eval songs ({len(r)} frames):")
print(f"  phase concentration rho : mean {r.mean():.5f}  p50 {np.median(r):.5f}  p99 {np.percentile(r,99):.5f}  max {r.max():.5f}")
print(f"     -> wrapped-Cauchy scale gamma = -log rho : median {-math.log(np.median(r)):.2f} rad "
      f"(bar advance per frame is only ~0.070 rad)")
print(f"  level sigma  : mean {l.mean():.4f} p50 {np.median(l):.4f} max {l.max():.4f}")
print(f"  dev   sigma  : mean {v.mean():.4f} p50 {np.median(v):.4f} max {v.max():.4f}")
print(f"  dev   AR coef: mean {aa.mean():+.4f}")
print(f"  level_ar (OU): {float(m.level_ar()):.4f}   tempo dof {float(m.tempo_dof()):.2f}")
print(f"  level_offset : {float(m.level_offset):.3f}")
