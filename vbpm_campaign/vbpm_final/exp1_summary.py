import glob, json, os
import numpy as np
D = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_final"
ORDER = ["full", "cut_tempo", "cut_phase"]
prb = json.load(open(f"{D}/exp1prb_all.json"))
dep = {os.path.basename(f)[8:-5]: json.load(open(f)) for f in sorted(glob.glob(f"{D}/exp1dep_*.json"))}
key = lambda t: (ORDER.index(t.rsplit("_s", 1)[0]), t)

print("=" * 122)
print("PROBE (a)  DOES THE PHASE REVIVE?   eval fold (held out), teacher-forced posterior, nats per 256-frame crop")
print("=" * 122)
print(f"{'run':14s} {'corr(cos,b)':>11s} {'corr(sin,b)':>11s} {'corr(logT,b)':>12s} | "
      f"{'d_rec_b PHASE':>13s} {'d_rec_b TEMPO':>13s} | {'d_obs/fr PHASE':>14s} {'TEMPO':>9s} | "
      f"{'rec_b':>7s} {'baserate':>8s}")
for t in sorted(prb, key=key):
    p = prb[t]["eval"]
    print(f"{t:14s} {p['corr_cosphi_beat']:+11.4f} {p['corr_sinphi_beat']:+11.4f} "
          f"{p['corr_logtempo_beat']:+12.4f} | {p['d_rec_b_phase_random']:+13.3f} "
          f"{p['d_rec_b_tempo_shuf']:+13.3f} | {p['d_obs_phase_random']:+14.5f} "
          f"{p['d_obs_tempo_flat']:+9.5f} | {p['ablation']['FULL z']['rec_b']:7.2f} "
          f"{p['ablation']['_baserate']['rec_b']:8.2f}")

print()
print("=" * 122)
print("(c) DEPLOY  bootstrap PF, eval fold 0 (40 songs, full length).  MANDATORY density-matched blind control.")
print("=" * 122)
hd = (f"{'run':14s} {'cfg':16s} {'beat_F':>7s} {'blind0':>7s} {'blindb':>7s} {'MARGIN':>8s} | "
      f"{'db_F':>6s} {'db_bl':>6s} {'M_db':>7s} | {'n_rat':>6s} {'fneg':>5s} {'c_bar':>6s} "
      f"{'c_beat':>6s} {'lockbar':>7s} {'lockbt':>6s} {'phid':>5s}")
print(hd)
agg = {}
for t in sorted(dep, key=key):
    for cfg, s in dep[t]["pf"].items():
        print(f"{t:14s} {cfg:16s} {s['beat_F']:7.4f} {s['blind_same_density']:7.4f} "
              f"{s['blind_best_offset']:7.4f} {s['margin_over_blind']:+8.4f} | "
              f"{s['downbeat_F']:6.4f} {s['blind_db_best']:6.4f} {s['margin_db_over_blind']:+7.4f} | "
              f"{s['n_ratio']:6.3f} {s['frac_neg']:5.3f} {s['obs_contrast']:6.4f} "
              f"{s['contrast_beat']:6.4f} {s['lock_bar']:7.4f} {s['lock_beat']:6.4f} "
              f"{s['phidot_ratio']:5.2f}")
        agg.setdefault((t.rsplit('_s',1)[0], cfg.rsplit('_',1)[1]), []).append(s)
    print()

print("=" * 122)
print("MEAN over both seeds x 4 PF configs, per view x read-out")
print("=" * 122)
print(f"{'view':11s} {'readout':8s} {'beat_F':>7s} {'blindb':>7s} {'MARGIN':>8s} | {'db_F':>6s} "
      f"{'M_db':>7s} | {'n_rat':>6s} {'fneg':>5s} {'c_bar':>6s} {'c_beat':>6s} {'lockbar':>7s} "
      f"{'lockbt':>6s} {'phid':>5s}")
for view in ORDER:
    for rd in ("mean", "map", "smooth"):
        L = agg.get((view, rd))
        if not L:
            continue
        f = lambda k: float(np.mean([s[k] for s in L]))
        print(f"{view:11s} {rd:8s} {f('beat_F'):7.4f} {f('blind_best_offset'):7.4f} "
              f"{f('margin_over_blind'):+8.4f} | {f('downbeat_F'):6.4f} "
              f"{f('margin_db_over_blind'):+7.4f} | {f('n_ratio'):6.3f} {f('frac_neg'):5.3f} "
              f"{f('obs_contrast'):6.4f} {f('contrast_beat'):6.4f} {f('lock_bar'):7.4f} "
              f"{f('lock_beat'):6.4f} {f('phidot_ratio'):5.2f}")
print()
print("REFERENCE: frozen activation-head peak-pick beat_F 0.811 / db 0.534 | oracle PF ~0.96 |")
print("           120BPM metronome 0.290 | arm_ii published baseline 0.383 (margin -0.046)")
