"""Print the final Variant-B comparison table from the result JSONs."""
import json, sys, glob, os

D = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/varB_pf"


def row(name, d):
    if not d:
        return
    print(f"  {name:<34} beat_F={d['beat_F']:.3f}  db_F={d['db_F']:.3f}  "
          f"n_est/n_true={d['n_ratio']:.2f}  density-matched-blind-floor={d['blind_floor']:.3f}"
          f"  (N={d['N']})")


def show(path):
    if not os.path.exists(path):
        return
    R = json.load(open(path))
    print("=" * 100)
    print(path)
    print("=" * 100)
    if "base_freerun" in R:
        print(" BASELINE (unmodified VBPM, vbpm.free_run):")
        row("free_run", R["base_freerun"].get("fr"))
        print(f"   metronome floor = {R['base_freerun']['metronome']:.3f}")
    v = R.get("varB", {})
    if v:
        print(" VARIANT B (same trained weights, two deploy paths):")
        row("OPEN-LOOP vbpm.free_run", v.get("fr"))
        for k in ["pf_circ", "pf_circ_mono", "pf_map", "pf_anc", "pf_anc_mono"]:
            row("PARTICLE FILTER " + k[3:], v.get(k))
        print(f"   metronome floor = {v['metronome']:.3f}   mean_ESS = {v.get('mean_ess', float('nan')):.0f}"
              f"   PF/true BPM = {v.get('pf_bpm_ratio', float('nan')):.2f}"
              f"   alpha = {v.get('alpha', R.get('alpha_chosen', 'n/a'))}")
    if "obs_profile" in R:
        print(f" obs decoder p(beat|phi) sharpness: {R['obs_profile']}")
    if "shift" in R:
        print(" SHIFT TEST (max |circular diff| of the deploy phase trajectory):")
        for k, d in R["shift"].items():
            print(f"   {k:<10} max={d['max']:.4f} rad   mean={d['mean']:.4f} rad")
    if "alpha_chosen" in R:
        print(f" alpha tuned on held-out TRAIN songs -> {R['alpha_chosen']}")
    print()


def show_extra(path):
    if not os.path.exists(path):
        return
    R = json.load(open(path))
    print("=" * 100)
    print(path, "  [EVIDENCE-OFF CONTROL]")
    print("=" * 100)
    for al, o in R.items():
        print(f" alpha={al}  (alpha=0 => identical PF machinery, observation term OFF)")
        for k in ["pf_anc", "pf_anc_mono", "pf_circ", "pf_circ_mono"]:
            row(k, o.get(k))
        print(f"   PF/true BPM ratio = {o['pf_bpm_ratio']:.2f}")
    print()


for p in sorted(glob.glob(f"{D}/res_*.json")):
    if "extra" in p:
        show_extra(p)
    else:
        show(p)
