"""Collate verify*.json into the final table, with the artifact controls read out."""
import sys, json, math
import numpy as np

ROWS = ["A_freerun", "noA_freerun", "B_filter", "AB_filter"]


def show(path, title):
    d = json.load(open(path))
    print(f"\n########## {title}   ({path})")
    print(f"metronome-120 floor = {d['_metronome_F']:.3f}")
    hdr = (f"{'path':<14}{'beat_F':>8}{'blindFl':>8}{'marg':>7}{'dens':>6}{'barsAdv':>9}{'barsT':>7}"
           f"{'rollF':>7}{'rollMg':>7}{'shiftD':>8}{'seedD':>7}{'shOff':>7}{'F|orig':>8}{'F|shft':>8}{'dbF':>7}")
    print(hdr)
    for k in ROWS:
        if k not in d:
            continue
        v = d[k]
        print(f"{k:<14}{v['beat_F']:>8.3f}{v['blind_grid_floor']:>8.3f}{v['margin_over_floor']:>7.3f}"
              f"{v['density']:>6.2f}{v['bars_adv']:>9.2f}{v['bars_true']:>7.2f}"
              f"{v['roll_beat_F']:>7.3f}{v['roll_margin']:>7.3f}{v['shift_maxcirc']:>8.3f}"
              f"{v['seed_maxcirc']:>7.3f}{v['shift_offset_sec']:>7.3f}"
              f"{v['shift_F_vs_orig']:>8.3f}{v['shift_F_vs_shifted']:>8.3f}{v['db_F']:>7.3f}")
    # informative subset for the +25-frame shift test: songs where 0.5 s is NOT ~a whole beat
    for k in ROWS:
        if k not in d or "per_song" not in d[k]:
            continue
        ps = d[k]["per_song"]
        bp = np.asarray(ps["shift_beatphase"], float)
        sel = (bp > 0.25) & (bp < 0.75)
        if sel.sum() >= 3:
            fo = np.asarray(ps["shift_F_vs_orig"], float)[sel]
            fs = np.asarray(ps["shift_F_vs_shifted"], float)[sel]
            f0 = np.asarray(ps["beat_F"], float)[sel]
            print(f"   [{k}] shift-informative subset n={int(sel.sum())} (0.5 s is 0.25-0.75 of a beat): "
                  f"F(no shift)={np.nanmean(f0):.3f}  F(shifted est vs ORIG ref)={np.nanmean(fo):.3f}  "
                  f"F(shifted est vs SHIFTED ref)={np.nanmean(fs):.3f}")
    print(f"   median IBI = {d[ROWS[0]]['median_ibi']:.3f} s"
          if ROWS[0] in d and "median_ibi" in d[ROWS[0]] else "")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        show(p, p.split("/")[-1])
