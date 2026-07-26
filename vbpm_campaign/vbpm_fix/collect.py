"""Collect all Variant-A run results into one table."""
import json, glob, sys
from pathlib import Path
rows = []
for f in sorted(glob.glob("/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/runs/*.result.json")):
    r = json.load(open(f))
    fi = r["final"]
    rows.append(dict(tag=r["tag"], mode=r["mode"], base=r["baseline"],
                     corr=r["corr_scale"], tcorr=r["tempo_corr_scale"], tinit=r["tempo_init"],
                     beat_F=fi["beat_phase"], best_F=r["best_beat_F"], db_F=fi["downbeat_phase"],
                     metro=fi["metronome"], ratio=fi["ratio"], logT=fi["lt"],
                     shift=r.get("shift_rad_trained", float("nan")),
                     shift0=r.get("shift_rad_untrained", float("nan"))))
hdr = f"{'tag':<12}{'mode':<7}{'corr':>8}{'tcorr':>7}{'tinit':>6}{'beat_F':>8}{'best_F':>8}{'db_F':>7}{'metro':>7}{'n/n_true':>9}{'logT':>7}{'shift_rad':>10}{'shift0':>8}"
print(hdr); print("-" * len(hdr))
for r in sorted(rows, key=lambda x: (x["mode"], -x["beat_F"])):
    print(f"{r['tag']:<12}{r['mode']:<7}{r['corr']:>8.3f}{r['tcorr']:>7.2f}{r['tinit']:>6}"
          f"{r['beat_F']:>8.3f}{r['best_F']:>8.3f}{r['db_F']:>7.3f}{r['metro']:>7.3f}"
          f"{r['ratio']:>9.3f}{r['logT']:>7.2f}{r['shift']:>10.4f}{r['shift0']:>8.4f}")
print()
print(json.dumps(rows, indent=1))
