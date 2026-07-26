"""Q6: contrast of the VAE-learned emission vs the SUPERVISED one, SAME code path, eval fold."""
import math, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms")
import variant_b as VB
from audit_common import load_split, ideal_barphase, FPS
from vbpm.evaluate import _estimate_meter
import q3_probe4_fit as Q3
ARMS = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms"
DEV = "cuda:0"; TWO_PI = 2 * math.pi


def contrast_generic(logp, songs, acts, n_off=12, K=4):
    out = []
    for s in songs:
        T = s["T"]; dref = s["downs"]
        if len(dref) < 3: continue
        ph = ideal_barphase(dref, T, FPS, mode="extrap")
        if ph is None: continue
        m = _estimate_meter(s["beats"], dref)
        o = torch.tensor(acts[s["stem"]], dtype=torch.float32, device=DEV)
        mt = F.one_hot(torch.full((T,), m - 1, device=DEV), K).float()
        p = torch.tensor(ph, dtype=torch.float32, device=DEV)
        t = float(logp(mt, p, o).mean())
        offs = [float(logp(mt, (p + TWO_PI * k / n_off) % TWO_PI, o).mean()) for k in range(1, n_off)]
        out.append(t - float(np.mean(offs)))
    return np.array(out)


ev = load_split("eval")
d = np.load(f"{ARMS}/act_eval.npz", allow_pickle=True)
aev = {s["stem"]: np.clip(np.asarray(d[s["stem"] + "|act"], np.float32), 1e-4, 1 - 1e-4) for s in ev}
tr = load_split("train")
dtr = np.load(f"{ARMS}/act_train.npz", allow_pickle=True)
atr = {s["stem"]: np.clip(np.asarray(dtr[s["stem"] + "|act"], np.float32), 1e-4, 1 - 1e-4) for s in tr}

rows = {}
for tag, hdim in (("ii_bern", 2), ("i_bern", 768)):
    ck = torch.load(f"{ARMS}/arm_i_{tag}.pt", map_location=DEV)
    mo = VB.BarPointerVAE_B(h_dim=hdim, hidden=ck["config"]["hidden"], num_meters=4,
                            obs_dim=2, obs_type="bern").to(DEV)
    mo.load_state_dict(ck["model"]); mo.eval()
    for lt_val in (-2.66,):
        def lp(mt, p, o, mo=mo, lt_val=lt_val):
            ltv = torch.full((p.shape[0],), lt_val, device=DEV)
            return mo.obs_logp(mo.z_features(mt, p, ltv), o)
        with torch.no_grad():
            dl = contrast_generic(lp, ev, aev)
        rows[f"VAE {tag}"] = dl
for nb in (36, 72):
    mu, _, _ = Q3.fit_table(tr, atr, nb, "bar")
    em = Q3.BinEmission(mu, dev=DEV).to(DEV)
    def lp2(mt, p, o, em=em):
        return em.obs_logp(em.z_features(mt, p, torch.zeros_like(p)), o)
    with torch.no_grad():
        rows[f"SUPERVISED nb={nb}"] = contrast_generic(lp2, ev, aev)
# an oracle von Mises for scale
for kap in (8.0,):
    def lp3(mt, p, o, kap=kap):
        return kap * torch.cos(p - p) * 0 + kap * torch.cos(torch.zeros_like(p))  # placeholder
print(f"{'emission':22s} {'nats/frame':>11s} {'geo-mean ratio':>15s} {'sd':>8s} {'n':>4s}")
for k, v in rows.items():
    print(f"{k:22s} {v.mean():11.4f} {math.exp(v.mean()):15.4f} {v.std():8.4f} {len(v):4d}")
