"""E3 side-by-side accounting table: assembles e3_abl_eval.json + the e3_* VAE runs."""
from __future__ import annotations

import json
import sys

FINAL = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_final"
COLS = [("beat_F", "beat_F"), ("blind_best_offset", "blindB"),
        ("margin_over_blind", "MARGIN"), ("n_ratio", "n_rat"),
        ("downbeat_F", "db_F"), ("blind_db_best", "dbBlnd"),
        ("margin_db_over_blind", "MARG_db"), ("n_ratio_db", "n_rdb"),
        ("obs_contrast", "contr"), ("frac_neg", "fneg"), ("meter_acc", "m_acc")]


def row(name, d):
    out = f"{name:<44s}"
    for k, _ in COLS:
        v = d.get(k, float("nan"))
        out += f" {v:>8.4f}" if isinstance(v, float) else f" {str(v):>8s}"
    return out


def header():
    h = f"{'method':<44s}"
    for _, lbl in COLS:
        h += f" {lbl:>8s}"
    return h


def main():
    lines = [header(), "-" * len(header())]
    abl = json.load(open(f"{FINAL}/e3_abl_eval.json"))["summary"]
    order_pre = ["oracle_phase"]
    for k in order_pre:
        lines.append(row(abl[k]["name"], abl[k]))
    lines.append("-" * len(header()))
    for tag, label in sys.argv[1:] and [] or []:
        pass
    runs = []
    for tag in ("e3_A_both", "e3_B_fix2only", "e3_C_fix1only"):
        try:
            r = json.load(open(f"{FINAL}/{tag}.json"))
        except FileNotFoundError:
            continue
        for k, d in r.get("pf", {}).items():
            runs.append((f"{tag} {k}", d))
    for nm, d in runs:
        lines.append(row(nm, d))
    lines.append("-" * len(header()))
    for k in ("rigid_autocorr", "rigid_autocorr_own", "rigid_map", "rigid_map_own",
              "simple_pf_mean", "simple_pf_map", "simple_pf_path", "simple_pf_path_own",
              "act_peak", "metronome120"):
        if k in abl:
            lines.append(row(abl[k]["name"], abl[k]))
    print("\n".join(lines))
    open(f"{FINAL}/e3_TABLE.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
