"""Cross-view comparison table for EXPERIMENT 1 (cut the tempo side-channel)."""
import glob, json, math, os
import numpy as np

D = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_final"
ORDER = ["full", "cut_tempo", "cut_phase"]

runs = {}
for f in sorted(glob.glob(f"{D}/exp1_*_s[01].json")):
    r = json.load(open(f))
    runs[os.path.basename(f)[5:-5]] = r

def line(s):
    print(s, flush=True)

line("=" * 118)
line("TRAIN FIT (last logged step, train fold)  +  HELD-OUT teacher-forced recon (eval fold, nats/256-frame crop)")
line("=" * 118)
line(f"{'run':16s} {'rec_b':>8s} {'rec_db':>8s} {'rec_obs':>9s} {'kl':>7s} | "
     f"{'EV rec_b':>9s} {'EV rec_db':>9s} {'EV rec_obs':>10s} | {'baserate_b':>10s}")
for k in sorted(runs, key=lambda x: (ORDER.index(x.rsplit('_s',1)[0]), x)):
    r = runs[k]; h = r["train_hist"][-1]; p = r["probe_eval"]
    a = p["ablation"]
    line(f"{k:16s} {h['recon_beat']:8.2f} {h['recon_db']:8.2f} {h['recon_obs']:9.2f} {h['kl']:7.2f} | "
         f"{a['FULL z']['rec_b']:9.2f} {a['FULL z']['rec_db']:9.2f} {a['FULL z']['rec_obs']:10.2f} | "
         f"{a['_baserate']['rec_b']:10.2f}")

line("")
line("=" * 118)
line("PROBE (a): PHASE REVIVAL  (eval fold, teacher-forced posterior)")
line("=" * 118)
line(f"{'run':16s} {'corr(cos,b)':>12s} {'corr(sin,b)':>12s} {'corr(cos m*phi,b)':>18s} "
     f"{'corr(logT,b)':>13s} | {'d_rec_b PHASE-rand':>19s} {'PHASE-const':>12s} {'TEMPO-flat':>11s} {'TEMPO-shuf':>11s}")
for k in sorted(runs, key=lambda x: (ORDER.index(x.rsplit('_s',1)[0]), x)):
    p = runs[k]["probe_eval"]
    line(f"{k:16s} {p['corr_cosphi_beat']:+12.4f} {p['corr_sinphi_beat']:+12.4f} "
         f"{p['corr_cos_m_phi_beat']:+18.4f} {p['corr_logtempo_beat']:+13.4f} | "
         f"{p['d_rec_b_phase_random']:+19.3f} {p['d_rec_b_phase_const']:+12.3f} "
         f"{p['d_rec_b_tempo_flat']:+11.3f} {p['d_rec_b_tempo_shuf']:+11.3f}")

line("")
line(f"{'run':16s} {'q_rho':>7s} {'p_rho':>7s} {'TF frac_neg':>12s} {'TF adv':>9s} {'TF jitter':>10s} "
     f"{'d_obs/frame PHASE':>18s} {'TEMPO':>9s}")
for k in sorted(runs, key=lambda x: (ORDER.index(x.rsplit('_s',1)[0]), x)):
    p = runs[k]["probe_eval"]
    line(f"{k:16s} {p['post_rho']:7.4f} {p['prior_rho']:7.4f} {p['tf_frac_neg']:12.3f} "
         f"{p['tf_mean_adv']:+9.4f} {p['tf_jitter']:10.4f} {p['d_obs_phase_random']:+18.5f} "
         f"{p['d_obs_tempo_flat']:+9.5f}")

line("")
line("=" * 118)
line("(c) DEPLOY: BOOTSTRAP PARTICLE FILTER, eval fold 0, full length, 40 songs")
line("    MANDATORY controls: blind0 = uniform grid with the SAME beat count; blindbest = best of 12 phase offsets")
line("=" * 118)
hdr = (f"{'run':16s} {'cfg':16s} {'beat_F':>7s} {'blind0':>7s} {'blindb':>7s} {'MARGIN':>8s} | "
       f"{'db_F':>6s} {'db_blindb':>9s} {'MARG_db':>8s} | {'n_ratio':>7s} {'n_r_db':>7s} "
       f"{'frac_neg':>8s} {'contrast':>9s} {'ESS':>6s}")
line(hdr)
best = {}
for k in sorted(runs, key=lambda x: (ORDER.index(x.rsplit('_s',1)[0]), x)):
    r = runs[k]
    for cfg, s in r["pf"].items():
        line(f"{k:16s} {cfg:16s} {s['beat_F']:7.4f} {s['blind_same_density']:7.4f} "
             f"{s['blind_best_offset']:7.4f} {s['margin_over_blind']:+8.4f} | "
             f"{s['downbeat_F']:6.4f} {s['blind_db_best']:9.4f} {s['margin_db_over_blind']:+8.4f} | "
             f"{s['n_ratio']:7.3f} {s['n_ratio_db']:7.3f} {s['frac_neg']:8.3f} "
             f"{s['obs_contrast']:9.4f} {s['ess']:6.1f}")
        best.setdefault(k, []).append((s['margin_over_blind'], cfg, s))
    line("")

line("=" * 118)
line("BEST-MARGIN configuration per run  (and the per-view mean over both seeds / all 12 cfgs)")
line("=" * 118)
by_view = {}
for k, v in best.items():
    m, cfg, s = max(v, key=lambda x: x[0])
    view = k.rsplit('_s', 1)[0]
    by_view.setdefault(view, []).append([x[2] for x in v])
    line(f"{k:16s} best={cfg:16s} beat_F={s['beat_F']:.4f} blindbest={s['blind_best_offset']:.4f} "
         f"MARGIN={m:+.4f}  db_F={s['downbeat_F']:.4f} MARG_db={s['margin_db_over_blind']:+.4f} "
         f"n_ratio={s['n_ratio']:.3f} frac_neg={s['frac_neg']:.3f} contrast={s['obs_contrast']:.4f}")
line("")
for view in ORDER:
    if view not in by_view:
        continue
    allc = [s for lst in by_view[view] for s in lst]
    f = lambda key: float(np.mean([s[key] for s in allc]))
    line(f"{view:12s} MEAN over {len(allc):2d} cfgs: beat_F={f('beat_F'):.4f} "
         f"blindbest={f('blind_best_offset'):.4f} MARGIN={f('margin_over_blind'):+.4f}  "
         f"db_F={f('downbeat_F'):.4f} MARG_db={f('margin_db_over_blind'):+.4f}  "
         f"n_ratio={f('n_ratio'):.3f} frac_neg={f('frac_neg'):.3f} "
         f"contrast={f('obs_contrast'):.4f}")
line("")
line("REFERENCE BARS: frozen activation head peak-pick beat_F 0.811 / db 0.534 | oracle-likelihood PF ~0.96 |")
line("                120BPM metronome 0.290 | arm_ii (published baseline) beat_F 0.385, MARGIN -0.032..-0.062")
