"""2x2 CAUSAL DESIGN: {VBPM learned transition, simple bar-pointer transition}
                    x {VAE-learned emission, supervised phase emission}
All four cells use the SAME model object (arm_ii ckpt) and the SAME read-out code; only the
transition and the obs_logp differ.  The simple transition is implemented here in torch so the
emission can be an arbitrary nn module (the VAE's h_dec).
"""
from __future__ import annotations
import argparse, json, math, sys, time
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_final")
import variant_b as VB                                                     # noqa
from emission import PhaseEmission, load_act, load_split, METERS, TWO_PI, FPS, _estimate_meter  # noqa
from run_exp2 import score_traj, summarize, pr                             # noqa
from swap_test import SupEmisModel                                         # noqa
from common import smooth_phase                                            # noqa
from pf import lt_band                                                     # noqa

ARMS = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms"


@torch.no_grad()
def simple_pf(model, obs, K=600, alpha=0.25, sigma_lt=0.05, sigma_phi=0.03,
              p_switch=0.005, meter_prior=None, seed=0, ess_frac=0.5):
    """Simple bar-pointer transition (same law as vbpm_final/pf.py), torch, pluggable emission."""
    dv = obs.device
    T = obs.shape[0]
    g = torch.Generator(device=dv).manual_seed(seed)
    lo = torch.tensor([lt_band(m)[0] for m in (1, 2, 3, 4)], device=dv)
    hi = torch.tensor([lt_band(m)[1] for m in (1, 2, 3, 4)], device=dv)
    pm = torch.tensor(meter_prior, device=dv, dtype=torch.float32)
    mi = torch.multinomial(pm, K, replacement=True, generator=g)          # meter index 0..3
    phi = torch.rand(K, device=dv, generator=g) * TWO_PI
    lt = lo[mi] + (hi[mi] - lo[mi]) * torch.rand(K, device=dv, generator=g)

    def logw_of(phi, lt, mi, t):
        zf = model.z_features(F.one_hot(mi, 4).float(), phi, lt)
        return alpha * model.obs_logp(zf, obs[t].unsqueeze(0).expand(K, -1))

    logw = logw_of(phi, lt, mi, 0); logw = logw - logw.max()
    w = torch.softmax(logw, 0)
    phi_h = torch.empty(T, K, device=dv); m_h = torch.empty(T, K, dtype=torch.long, device=dv)
    anc = torch.empty(T, K, dtype=torch.long, device=dv)
    idx = torch.arange(K, device=dv)
    ph_mean = np.empty(T); ph_map = np.empty(T); map_idx = np.empty(T, np.int64); ess = np.empty(T)

    def rec(t):
        ph_mean[t] = math.atan2(float((w * phi.sin()).sum()), float((w * phi.cos()).sum())) % TWO_PI
        j = int(w.argmax()); map_idx[t] = j; ph_map[t] = float(phi[j])
        ess[t] = 1.0 / float((w ** 2).sum())
        phi_h[t] = phi; m_h[t] = mi
    rec(0); anc[0] = idx
    for t in range(1, T):
        adv = phi + lt.exp() + sigma_phi * torch.randn(K, device=dv, generator=g)
        cross = adv >= TWO_PI
        phi = adv % TWO_PI
        lt = lt + sigma_lt * torch.randn(K, device=dv, generator=g)
        if p_switch > 0:
            sw = cross & (torch.rand(K, device=dv, generator=g) < p_switch)
            if bool(sw.any()):
                new = torch.multinomial(pm, int(sw.sum()), replacement=True, generator=g)
                lt[sw] = lt[sw] + torch.log((new + 1).float() / (mi[sw] + 1).float())
                mi[sw] = new
        lt = torch.clamp(lt, lo[mi], hi[mi])
        logw = logw + logw_of(phi, lt, mi, t)
        logw = logw - logw.max()
        w = torch.softmax(logw, 0)
        rec(t); anc[t] = idx
        if ess[t] < ess_frac * K:
            pos = (torch.arange(K, device=dv) + torch.rand(1, device=dv, generator=g)) / K
            cdf = torch.cumsum(w, 0); cdf = cdf / cdf[-1]
            a = torch.searchsorted(cdf.contiguous(), pos.contiguous()).clamp(max=K - 1)
            phi, lt, mi = phi[a], lt[a], mi[a]
            anc[t] = a
            logw = torch.zeros(K, device=dv); w = torch.full((K,), 1.0 / K, device=dv)
    phi_h = phi_h.cpu().numpy(); m_h = m_h.cpu().numpy(); anc = anc.cpu().numpy()
    j = int(map_idx[T - 1]); pp = np.empty(T); pmt = np.empty(T, np.int64)
    for t in range(T - 1, -1, -1):
        pp[t] = phi_h[t, j]; pmt[t] = m_h[t, j] + 1
        if t > 0: j = int(anc[t - 1][j])
    return dict(phase_mean=ph_mean, phase_map=ph_map, phase_path=pp, meter_path=pmt,
                ess=float(ess.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transition", default="simple", choices=["simple", "learned"])
    ap.add_argument("--emission", default="vae", choices=["vae", "sup"])
    ap.add_argument("--lik", default="gauss"); ap.add_argument("--bpb", type=int, default=24)
    ap.add_argument("--K", type=int, default=600); ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--sigma_lt", type=float, default=0.05)
    ap.add_argument("--sigma_phi", type=float, default=0.03)
    ap.add_argument("--n_eval", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--tag", default="cell")
    a = ap.parse_args()
    dev = "cuda:0"
    tr = load_split("train"); ev = load_split("eval")
    if a.n_eval: ev = ev[:a.n_eval]
    at = load_act("train"); ae = load_act("eval")
    ck = torch.load(f"{ARMS}/arm_i_ii_bern.pt", map_location="cpu")
    cls = SupEmisModel if a.emission == "sup" else VB.BarPointerVAE_B
    model = cls(h_dim=2, hidden=128, num_meters=4, obs_dim=2, obs_type="bern").to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    if a.emission == "sup":
        model.attach(PhaseEmission(bins_per_beat=a.bpb, likelihood=a.lik, smooth=0.0)
                     .fit(tr, at), dev)
    prior = np.zeros(4)
    for s in tr:
        m = _estimate_meter(s["beats"], s["downs"])
        prior[m - 1] += 1
    prior = prior / prior.sum()

    rows = {k: [] for k in ("mean", "map", "path", "pf_meter_path")}
    t0 = time.time()
    for i, s in enumerate(ev):
        act = ae[s["stem"]]
        T = min(len(act), s["T"]) if not a.max_frames else min(len(act), s["T"], a.max_frames)
        ref = s["beats"][s["beats"] < T / FPS]; dref = s["downs"][s["downs"] < T / FPS]
        if len(ref) < 3: continue
        m_gt = _estimate_meter(s["beats"], s["downs"])
        obs = torch.from_numpy(act[:T]).to(dev)
        torch.manual_seed(1234 + i)
        if a.transition == "simple":
            out = simple_pf(model, obs, K=a.K, alpha=a.alpha, sigma_lt=a.sigma_lt,
                            sigma_phi=a.sigma_phi, meter_prior=prior, seed=1234 + i)
            mp = out["meter_path"]
        else:
            o = obs.unsqueeze(0)
            r = VB.particle_filter(model, o, o, K=a.K, alpha=a.alpha)
            out = dict(phase_mean=r["phase_mean"].numpy(), phase_map=r["phase_map"].numpy(),
                       phase_path=r["phase_map"].numpy(), ess=r["ess"])
            mp = np.asarray(r["meter_map"]) + 1
        base = dict(stem=s["stem"], n_true=len(ref), n_true_db=len(dref), ess=out["ess"],
                    obs_contrast=float("nan"),
                    meter_ok=float(int(np.bincount(mp).argmax()) == m_gt))
        for k in ("mean", "map", "path"):
            rows[k].append({**base, **score_traj(out[f"phase_{k}"], m_gt, ref, dref, T)})
        rows["pf_meter_path"].append({**base, **score_traj(out["phase_path"],
                                      int(np.bincount(mp).argmax()), ref, dref, T)})
        if i % 20 == 0: print(f"  {i}/{len(ev)} {time.time()-t0:.0f}s", flush=True)
    res = {"config": vars(a)}
    for k in rows:
        d = summarize(rows[k], f"{a.transition}-trans + {a.emission}-emis {k}")
        pr(d); res[k] = d
    json.dump(res, open(f"/home/sogang/jaehoon/VBPM_reintegration/vbpm_final/{a.tag}.json", "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
