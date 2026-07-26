"""SWAP TEST: VBPM's OWN learned prior transition (arm_ii ckpt) + {VAE-learned | SUPERVISED}
emission, evaluated through vbpm_fix/variant_b.py::particle_filter (untouched).

Isolates the emission as the causal variable: same transition, same PF code, same read-out.
"""
from __future__ import annotations
import argparse, json, math, sys, time
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_final")

import variant_b as VB                                                    # noqa: E402
from emission import PhaseEmission, load_act, load_split, METERS, TWO_PI, FPS, _estimate_meter  # noqa
from run_exp2 import score_traj, summarize, pr                            # noqa: E402
from common import smooth_phase                                           # noqa: E402

ARMS = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms"


class SupEmisModel(VB.BarPointerVAE_B):
    """Same model / same learned prior transition; obs_logp REPLACED by the supervised table."""

    def attach(self, emis, device):
        self.emis = emis
        bpb = emis.bpb
        Bmax = emis.Bmax
        self.nb_t = torch.tensor([bpb, 2 * bpb, 3 * bpb, 4 * bpb], device=device)
        W = torch.zeros(4, Bmax, 2, device=device)
        W2 = torch.zeros(4, Bmax, 2, device=device)
        C0 = torch.zeros(4, Bmax, device=device)
        for m in METERS:
            if emis.likelihood == "bern":
                w, c0 = emis.C[m]
                W[m - 1, :emis.nb[m]] = torch.tensor(w, device=device, dtype=torch.float32)
            else:
                w1, w2, c0 = emis.C[m]
                W[m - 1, :emis.nb[m]] = torch.tensor(w1, device=device, dtype=torch.float32)
                W2[m - 1, :emis.nb[m]] = torch.tensor(w2, device=device, dtype=torch.float32)
            C0[m - 1, :emis.nb[m]] = torch.tensor(c0, device=device, dtype=torch.float32)
        # meter index 0 == "1 beat per bar": fold the 4/4 table over its 4 beats (no bar structure)
        for A, src in ((W, W), (W2, W2), (C0, C0)):
            pass
        W[0, :bpb] = W[3, :4 * bpb].reshape(4, bpb, 2).mean(0)
        W2[0, :bpb] = W2[3, :4 * bpb].reshape(4, bpb, 2).mean(0)
        C0[0, :bpb] = C0[3, :4 * bpb].reshape(4, bpb).mean(0)
        self.W, self.W2, self.C0 = W, W2, C0
        return self

    def obs_logp(self, z_feat, o_t):
        phi = torch.atan2(z_feat[:, 1], z_feat[:, 0]) % TWO_PI
        mi = z_feat[:, 3:].argmax(-1)
        nb = self.nb_t[mi]
        b = (phi / TWO_PI * nb).long().clamp(min=0)
        b = torch.minimum(b, nb - 1)
        v = o_t if self.emis.likelihood == "bern" else torch.log(o_t / (1 - o_t))
        ll = (self.W[mi, b] * v).sum(-1) + self.C0[mi, b]
        if self.emis.likelihood == "gauss":
            ll = ll + (self.W2[mi, b] * v ** 2).sum(-1)
        return ll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emission", default="sup", choices=["sup", "vae"])
    ap.add_argument("--lik", default="gauss")
    ap.add_argument("--bpb", type=int, default=24)
    ap.add_argument("--K", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--n_eval", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--tag", default="swap")
    a = ap.parse_args()
    dev = "cuda:0"

    tr = load_split("train"); ev = load_split("eval")
    if a.n_eval: ev = ev[:a.n_eval]
    at = load_act("train"); ae = load_act("eval")

    ck = torch.load(f"{ARMS}/arm_i_ii_bern.pt", map_location="cpu")
    cls = SupEmisModel if a.emission == "sup" else VB.BarPointerVAE_B
    model = cls(h_dim=2, hidden=ck["config"]["hidden"], num_meters=4,
                obs_dim=2, obs_type="bern").to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    if a.emission == "sup":
        emis = PhaseEmission(bins_per_beat=a.bpb, likelihood=a.lik, smooth=0.0).fit(tr, at)
        model.attach(emis, dev)
        print(f"attached supervised emission lik={a.lik} bpb={a.bpb}", flush=True)

    rows = {k: [] for k in ("mean", "map", "smooth")}
    t0 = time.time()
    for i, s in enumerate(ev):
        act = ae[s["stem"]]
        T = min(len(act), s["T"]) if not a.max_frames else min(len(act), s["T"], a.max_frames)
        ref = s["beats"][s["beats"] < T / FPS]; dref = s["downs"][s["downs"] < T / FPS]
        if len(ref) < 3: continue
        m_gt = _estimate_meter(s["beats"], s["downs"])
        obs = torch.from_numpy(act[:T]).unsqueeze(0).to(dev)
        torch.manual_seed(1234 + i)
        out = VB.particle_filter(model, obs, obs, K=a.K, alpha=a.alpha)
        base = dict(stem=s["stem"], n_true=len(ref), n_true_db=len(dref), ess=out["ess"],
                    obs_contrast=float("nan"), meter_ok=float("nan"))
        for k, ph in (("mean", out["phase_mean"].numpy()), ("map", out["phase_map"].numpy()),
                      ("smooth", smooth_phase(out["phase_mean"].numpy(), 5))):
            rows[k].append({**base, **score_traj(ph, m_gt, ref, dref, T)})
        if i % 20 == 0:
            print(f"  {i}/{len(ev)}  {time.time()-t0:.0f}s", flush=True)
    res = {"config": vars(a)}
    for k in rows:
        d = summarize(rows[k], f"arm_ii-transition + {a.emission}-emission {k}")
        pr(d); res[k] = d
    json.dump(res, open(f"/home/sogang/jaehoon/VBPM_reintegration/vbpm_final/{a.tag}.json", "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
