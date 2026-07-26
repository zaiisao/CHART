import json
import numpy as np
rows = json.load(open("f1_diag_rows.json"))
sel  = json.load(open("f1_meterselect_rows.json"))
n = len(rows)
f_free = np.array([r["pf_free"]["F"] for r in rows])
f_pk   = np.array([r["peakpick"]["F"] for r in rows])
f_clamp= np.array([r["pf_clamp"]["F"] for r in rows])
meter_ok = np.array([r["meter_ok"] for r in rows], bool)
oct_free = np.array([r["pf_free"]["oct"] for r in rows])
gap = f_free - f_pk

octbad = np.abs(np.log2(np.where(np.isnan(oct_free), 1, oct_free))) > 0.3
# sub-attribution: contribution to the total gap (sum/n)
cats = {
 "meter-wrong (m_pf!=m_gt)": ~meter_ok,
 "  of which 4->2 (half-rate)": np.array([r["m_gt"]==4 and r["m_pf"]==2 for r in rows]),
 "  of which 4->3": np.array([r["m_gt"]==4 and r["m_pf"]==3 for r in rows]),
 "meter-ok & octave-off": meter_ok & octbad,
 "meter-ok & octave-ok": meter_ok & ~octbad,
}
print(f"TOTAL GAP {gap.mean():+.4f}")
for k, msk in cats.items():
    print(f"  {k:30s} n={msk.sum():3d}  contribution {gap[msk].sum()/n:+.4f}  ({100*gap[msk].sum()/n/gap.mean():.0f}% of gap)")

# paired test: map-select PF vs peakpick (matched stems)
smap = {r["stem"]: r for r in sel}
d = np.array([smap[r["stem"]]["map"]["F"] - r["peakpick"]["F"] for r in rows if r["stem"] in smap])
from scipy import stats
w = stats.wilcoxon(d)
t = stats.ttest_rel(np.array([smap[r["stem"]]["map"]["F"] for r in rows]), f_pk)
print(f"\nPAIRED map-select PF vs peakpick: n={len(d)} mean diff {d.mean():+.4f} "
      f"median {np.median(d):+.4f} win/tie/loss={int((d>0.005).sum())}/{int((np.abs(d)<=0.005).sum())}/{int((d<-0.005).sum())}")
print(f"  wilcoxon p={w.pvalue:.4f}  paired-t p={t.pvalue:.4f}")
d2 = f_clamp - f_pk
w2 = stats.wilcoxon(d2)
print(f"PAIRED clamp PF vs peakpick: mean {d2.mean():+.4f} wilcoxon p={w2.pvalue:.4f}")
# downbeats
dbsel = np.nanmean([smap[r["stem"]]["map"]["db_F"] for r in rows])
dbpk = np.nanmean([r["db_pk_F"] for r in rows])
print(f"downbeat F: map-select PF {dbsel:.4f} vs peakpick db-channel {dbpk:.4f}")
