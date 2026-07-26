import json, glob, os, sys
R="/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/ab/runs"
NAME={"baseline_freerun":"UNFIXED free_run (open-loop)",
      "noA_freerun":"free_run, corr OFF (A ablated)",
      "A_freerun":"(i) A only  : free_run + audio-cond. prior mean",
      "B_filter":"(ii) B only : particle filter, corr OFF",
      "AB_filter":"(iii) A+B   : particle filter + audio-cond. prior mean",
      "B_filter|circmean":"(ii) B only, circ-weighted-mean read-out",
      "AB_filter|circmean":"(iii) A+B , circ-weighted-mean read-out"}
for f in sorted(glob.glob(f"{R}/*/result.json")):
    r=json.load(open(f)); tag=r["tag"]
    print(f"\n################ {tag}  (obs_weight={r['args']['obs_weight']}, K={r['args']['K']}, temper={r['args']['temper']})")
    print(f"{'deploy path':52s} {'beat_F':>7s} {'db_F':>7s} {'metro':>7s} {'n_est/n_true':>13s} {'n':>4s}")
    for k,v in r["final"].items():
        print(f"{NAME.get(k,k):52s} {v['beat_F']:7.3f} {v['db_F']:7.3f} {v['metro_F']:7.3f} "
              f"{v['n_est/n_true']:13.3f} {v['n_songs']:4d}")
    if "temper_sweep" in r:
        print("  temper sweep (AB_filter, 15 songs):")
        for tp,d in r["temper_sweep"].items():
            for k,v in d.items():
                print(f"    temper={tp:5s} {k:28s} beat_F={v['beat_F']:.3f} db_F={v['db_F']:.3f} n_est/n_true={v['n_est/n_true']:.2f}")
    if "shift_test" in r:
        print("  SHIFT TEST  (+25 frames = 0.5 s), max |circular diff| in rad:")
        for k,v in r["shift_test"].items():
            print(f"    {NAME.get(k,k):50s} max={v['max_over_songs']:.4f}  mean-of-per-song-max={v['mean_of_per_song_max']:.4f}")
