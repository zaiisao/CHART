import json, math
import numpy as np

rows = json.load(open("f1_diag_rows.json"))
n = len(rows)
F = lambda k: np.array([r[k]["F"] for r in rows])
f_free, f_gtm, f_clamp, f_pk = F("pf_free"), F("pf_gtmeter"), F("pf_clamp"), F("peakpick")
f_free_sn, f_clamp_sn = F("pf_free_snap"), F("pf_clamp_snap")
meter_ok = np.array([r["meter_ok"] for r in rows], bool)
m_gt = np.array([r["m_gt"] for r in rows]); m_pf = np.array([r["m_pf"] for r in rows])
ds = np.array([r["dataset"] for r in rows])
dur = np.array([r["dur"] for r in rows]); contrast = np.array([r["contrast"] for r in rows])
oct_free = np.array([r["pf_free"]["oct"] for r in rows])
oct_clamp = np.array([r["pf_clamp"]["oct"] for r in rows])

gap = f_free - f_pk
print(f"n={n}  mean F: free={f_free.mean():.4f} clamp={f_clamp.mean():.4f} pk={f_pk.mean():.4f}")
print(f"TOTAL GAP (free - pk) = {gap.mean():+.4f}")
print(f"meter_ok: {meter_ok.sum()}/{n} = {meter_ok.mean():.3f}")

def sub(name, msk):
    if msk.sum()==0: return
    print(f"  {name:28s} n={msk.sum():3d}  F_free={f_free[msk].mean():.4f} "
          f"F_clamp={f_clamp[msk].mean():.4f} F_pk={f_pk[msk].mean():.4f} "
          f"gap_free={ (f_free-f_pk)[msk].mean():+.4f} gap_clamp={(f_clamp-f_pk)[msk].mean():+.4f}")

print("\n(a) METER subsets")
sub("meter-correct (m_pf==m_gt)", meter_ok)
sub("meter-wrong", ~meter_ok)
# weighted contribution of meter-wrong songs to the total gap
contrib_wrong = gap[~meter_ok].sum()/n
contrib_ok = gap[meter_ok].sum()/n
print(f"  gap contribution: meter-wrong songs {contrib_wrong:+.4f}, meter-correct {contrib_ok:+.4f} (sums to {gap.mean():+.4f})")
# confusion
print("  meter confusion (gt -> pf):")
for g in (2,3,4):
    for p in (2,3,4):
        c = ((m_gt==g)&(m_pf==p)).sum()
        if c: print(f"    {g}->{p}: {c}")

print("\n(b) TIMING offsets (ms), pooled over songs")
def offstats(key):
    o = np.concatenate([np.asarray(r[key]) for r in rows if len(r[key])])*1000
    hit70 = np.mean(np.abs(o)<=70)
    return f"n={len(o)} mean={o.mean():+.1f} med={np.median(o):+.1f} sd={o.std():.1f} |off|<=70ms:{hit70:.3f}"
print("  PF beats  vs GT     :", offstats("off_pf_gt"))
print("  peak-pick vs GT     :", offstats("off_pk_gt"))
print("  PF beats  vs actpeak:", offstats("off_pf_pk"))
print(f"  snap gain: free {f_free_sn.mean()-f_free.mean():+.4f}, clamp {f_clamp_sn.mean()-f_clamp.mean():+.4f}")

print("\n(c) PER-SONG SCATTER (free - pk)")
q = np.percentile(gap, [5,25,50,75,95])
print(f"  quantiles 5/25/50/75/95: {np.round(q,3)}")
big = gap < -0.10
print(f"  songs with gap<-0.10: {big.sum()}/{n}, their summed contribution {gap[big].sum()/n:+.4f} of {gap.mean():+.4f}")
print(f"  songs with gap<-0.10 AND meter-wrong: {(big&~meter_ok).sum()}")
octbad = np.abs(np.log2(np.where(np.isnan(oct_free),1,oct_free)))>0.3
print(f"  songs with octave-ratio off (|log2|>0.3) free: {octbad.sum()}  overlap with big losers: {(big&octbad).sum()}")
sub("big losers (gap<-0.10)", big)
sub("big losers, meter-wrong", big&~meter_ok)
sub("big losers, meter-ok", big&meter_ok)
for d in sorted(set(ds)): sub(f"dataset={d}", ds==d)
# low contrast
lc = contrast < np.median(contrast)
sub("low-contrast half", lc); sub("high-contrast half", ~lc)
sub("short (<60s)", dur<60); sub("long (>=60s)", dur>=60)

# does clamped PF also fix continuity on meter-wrong songs?
C = lambda k,f: np.array([r[k][f] for r in rows])
print("\n(d) CONTINUITY")
for k in ("pf_free","pf_clamp","peakpick"):
    print(f"  {k:10s} CMLc={C(k,'CMLc').mean():.3f} CMLt={C(k,'CMLt').mean():.3f} AMLc={C(k,'AMLc').mean():.3f} AMLt={C(k,'AMLt').mean():.3f}")

print("\nATTRIBUTION of the -0.0610 gap:")
meter_share = f_clamp.mean()-f_free.mean()
loc_share = f_free_sn.mean()-f_free.mean()
print(f"  meter inference (clamp gain)        : {meter_share:+.4f}")
print(f"  localisation (snap gain)            : {loc_share:+.4f}")
print(f"  clamp+snap vs pk                    : {f_clamp_sn.mean()-f_pk.mean():+.4f}")
print(f"  residual PF advantage once meter ok : {f_clamp.mean()-f_pk.mean():+.4f}")

# top-10 worst songs table
print("\nworst 12 songs by gap (free-pk):")
idx = np.argsort(gap)[:12]
for i in idx:
    r = rows[i]
    print(f"  {r['stem'][:40]:40s} ds={r['dataset'][:9]:9s} gap={gap[i]:+.3f} "
          f"F_free={f_free[i]:.3f} F_clamp={f_clamp[i]:.3f} F_pk={f_pk[i]:.3f} "
          f"m_gt={m_gt[i]} m_pf={m_pf[i]} oct_free={oct_free[i]:.2f} oct_clamp={oct_clamp[i]:.2f} contrast={contrast[i]:.2f}")
