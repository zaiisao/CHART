"""Re-evaluate a trained E3 checkpoint with BOTH read-outs (GT meter and the PF's own
inferred meter), so a meter collapse can be separated from a phase failure."""
from __future__ import annotations

import argparse, json, sys, time
import numpy as np
import torch

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_final")
import e3_common as C                                              # noqa: E402
from e3_common import FPS, FINAL, _estimate_meter, score_traj, summarize, pr  # noqa
from e3_emission import PhaseEmission, load_act                    # noqa: E402
from e3_model import FrozenPhaseEmission, E3VAE                    # noqa: E402
from e3_pf_learned import particle_filter_learned                  # noqa: E402
from e3_vae import LayerMerge, build_obs, model_obs_contrast       # noqa: E402
from audit_common import load_split                                # noqa: E402
from common import smooth_phase                                    # noqa: E402

DEV = "cuda:0"
ARMS = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0])
    ap.add_argument("--K", type=int, default=600)
    ap.add_argument("--n_eval", type=int, default=0)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]; NM = cfg.get("num_meters", 3); MET0 = 2 if NM == 3 else 1
    train = load_split("train", with_feats=False)
    ev = load_split("eval", with_feats=True, cap=(a.n_eval or None))
    emis_t = None
    if cfg["emission"] == "frozen":
        at = load_act("train")
        emis = PhaseEmission(bins_per_beat=cfg["bpb"], likelihood=cfg["lik"],
                             smooth=0.0).fit(train, at, phase_mode="downbeat")
        emis_t = FrozenPhaseEmission(emis, meters=tuple(range(MET0, MET0 + NM))).to(DEV)
    merge = LayerMerge().to(DEV); merge.load_state_dict(ck["merge"])
    model = E3VAE(h_dim=768, emission=emis_t, hidden=cfg["hidden"], num_meters=NM,
                  meter_offset=MET0, drop_tempo_from_decoder=bool(cfg["drop_tempo"])).to(DEV)
    model.load_state_dict(ck["model"]); merge.eval(); model.eval()
    obs_ev = build_obs(ev, f"{ARMS}/act_eval.npz")

    res = {"ckpt": a.ckpt, "config": cfg}
    for alpha in a.alphas:
        rows = {k: [] for k in ("mean", "map", "path", "smooth_mean", "path_own_meter")}
        t1 = time.time()
        for i, s in enumerate(ev):
            T = s["feats"].shape[1]
            ref = s["beats"][s["beats"] < T / FPS]; dref = s["downs"][s["downs"] < T / FPS]
            if len(ref) < 3:
                continue
            m_gt = _estimate_meter(ref, dref)
            f = torch.from_numpy(s["feats"][:, :T, :].astype(np.float32)).unsqueeze(0).to(DEV)
            with torch.no_grad():
                h = merge(f)
                obs = torch.from_numpy(obs_ev[s["stem"]][:T]).unsqueeze(0).to(DEV)
                torch.manual_seed(1234 + i)
                out = particle_filter_learned(model, h, obs, K=a.K, alpha=alpha)
                oc = model_obs_contrast(model, obs[0], dref, ref, T)
            m_pf = int(np.bincount(out["meter_path"]).argmax())
            base = dict(stem=s["stem"], dataset=s["dataset"], T=T, n_true=len(ref),
                        n_true_db=len(dref), ess=out["ess"], obs_contrast=oc,
                        meter_ok=float(m_pf == m_gt))
            for k, ph in (("mean", out["phase_mean"]), ("map", out["phase_map"]),
                          ("path", out["phase_path"]),
                          ("smooth_mean", smooth_phase(out["phase_mean"], 5))):
                rows[k].append({**base, **score_traj(ph, m_gt, ref, dref, T)})
            rows["path_own_meter"].append({**base, **score_traj(out["phase_path"], m_pf,
                                                                ref, dref, T)})
            del f, h, obs
            if i % 20 == 0:
                print(f"  {i}/{len(ev)} {time.time()-t1:.0f}s", flush=True)
        for k, rr in rows.items():
            d = summarize(rr, f"{a.tag} {k} a={alpha}")
            pr(d)
            res.setdefault("pf", {})[f"a{alpha}_{k}"] = d
    json.dump(res, open(f"{FINAL}/{a.tag}.json", "w"), indent=1, default=float)
    print("WROTE", a.tag + ".json", flush=True)


if __name__ == "__main__":
    main()
