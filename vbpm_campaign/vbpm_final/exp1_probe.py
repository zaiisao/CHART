"""EXP1 stage (2) re-run from checkpoints: does the PHASE revive? -> exp1prb_<tag>.json"""
import json, sys, torch
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_final")
from audit_common import load_split, banner
from exp1_cut_tempo import CutModel, build_obs_cache, side_channel_probe, decoder_weight_report, VIEWS

DEV, OUT = "cuda:0", "/home/sogang/jaehoon/VBPM_reintegration/vbpm_final"
ARMS = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms"

train = load_split("train"); ev = load_split("eval", cap=40)
otr = build_obs_cache(train, f"{ARMS}/act_train.npz")
oev = build_obs_cache(ev, f"{ARMS}/act_eval.npz")
out = {}
for view in VIEWS:
    for seed in (0, 1):
        tag = f"{view}_s{seed}"
        ck = torch.load(f"{OUT}/exp1_{tag}.pt", map_location=DEV)
        m = CutModel(view, h_dim=2, hidden=ck["config"]["hidden"], num_meters=4,
                     obs_dim=2, obs_type="bern").to(DEV)
        m.load_state_dict(ck["model"]); m.eval()
        r = {"weights": decoder_weight_report(m)}
        for lab, songs, cache in (("train", train, otr), ("eval", ev, oev)):
            r[lab] = side_channel_probe(m, songs, cache, 7, 6, 16, 256, lab)
        out[tag] = r
        p = r["eval"]
        banner(f"{tag}")
        print(f"  eval corr(cos,b)={p['corr_cosphi_beat']:+.4f} corr(sin,b)={p['corr_sinphi_beat']:+.4f} "
              f"corr(cos m*phi,b)={p['corr_cos_m_phi_beat']:+.4f} corr(logT,b)={p['corr_logtempo_beat']:+.4f}")
        print(f"  eval d_rec_b: phase-rand {p['d_rec_b_phase_random']:+.3f} phase-const "
              f"{p['d_rec_b_phase_const']:+.3f} tempo-flat {p['d_rec_b_tempo_flat']:+.3f} "
              f"tempo-shuf {p['d_rec_b_tempo_shuf']:+.3f}")
        print(f"  eval d_obs/frame: phase {p['d_obs_phase_random']:+.5f} tempo {p['d_obs_tempo_flat']:+.5f}"
              f"  | rec_b FULL {p['ablation']['FULL z']['rec_b']:.2f} vs baserate "
              f"{p['ablation']['_baserate']['rec_b']:.2f}", flush=True)
json.dump(out, open(f"{OUT}/exp1prb_all.json", "w"), indent=1, default=float)
print("WROTE exp1prb_all.json")
