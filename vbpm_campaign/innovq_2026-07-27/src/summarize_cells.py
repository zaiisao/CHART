"""Extract the decisive curves per cell from innovq_<tag>.json artifacts."""
import json, glob, os
D = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq"
out = {}
for f in sorted(glob.glob(f"{D}/innovq_*.json")):
    tag = os.path.basename(f)[7:-5]
    if tag.startswith("smoke"):
        continue
    d = json.load(open(f))
    rows = d["hist"]
    cell = dict(teacher_corr_pooled=d.get("teacher_corr"),
                teacher_corr_percrop=d.get("teacher_corr_percrop"), curve=[])
    for r in rows:
        tr, ev = r.get("tr", {}), r.get("ev", {})
        cell["curve"].append(dict(
            stage=r["stage"], step=r["step"], beta=r.get("beta"),
            tr_corr_pooled=tr.get("corr"), tr_corr_percrop=tr.get("corr_percrop"),
            ev_corr_pooled=ev.get("corr"), ev_tf_F=ev.get("tf_F"),
            ev_tf_F_corr=ev.get("tf_F_corr"),
            kl_phase=tr.get("kl_phase"), kl_level=tr.get("kl_level"),
            kl_meter=tr.get("kl_meter"), mean_abs_innov=tr.get("mean_abs_innov"),
            sat_frac=tr.get("sat_frac"), rho_q=tr.get("rho_q"), max_sq=tr.get("max_sq")))
    hand = [c for c in cell["curve"] if c["stage"] == "handover"]
    beta1 = [c for c in cell["curve"] if c["stage"] == "elbo" and (c["beta"] or 0) >= 1.0]
    fin = [c for c in cell["curve"] if c["stage"] in ("final", "abort")]
    cell["handover"] = hand[-1] if hand else None
    cell["at_beta1_first"] = beta1[0] if beta1 else None
    cell["at_beta1_last"] = beta1[-1] if beta1 else None
    cell["final"] = fin[-1] if fin else None
    if beta1:
        cell["corr_percrop_min_at_beta1"] = min(c["tr_corr_percrop"] for c in beta1)
        cell["corr_pooled_min_at_beta1"] = min(c["tr_corr_pooled"] for c in beta1)
        cell["kl_phase_max_at_beta1"] = max(c["kl_phase"] for c in beta1)
        cell["ev_tf_F_last"] = beta1[-1]["ev_tf_F"]
    out[tag] = cell
json.dump(out, open(f"{D}/campaign_summary.json", "w"), indent=1)
for tag, c in out.items():
    print("=" * 20, tag, "=" * 20)
    for k in ("handover", "at_beta1_first", "at_beta1_last", "final"):
        print(f" {k}: {json.dumps(c[k])}")
    for k in ("corr_percrop_min_at_beta1", "kl_phase_max_at_beta1", "ev_tf_F_last"):
        if k in c:
            print(f" {k}: {c[k]}")
