"""PREMISE 1(a): are real bar-phase increments ever negative?"""
import sys, numpy as np, collections
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
from common import load_labels, FPS, per_ds, fmt_ds
from phases import wrap, inside_mask, phase_beatlinear, phase_downlinear, phase_pchip, TWO_PI

CONS = {"beatlinear": phase_beatlinear, "downlinear": phase_downlinear}

def run(split):
    L = load_labels(split)
    rows = {k: [] for k in list(CONS)+["pchip"]}
    tot = collections.defaultdict(lambda: [0,0,0,0])  # n, nneg, nzero, nstall
    for s in L:
        T = s["T"]; msk = inside_mask(s, T)
        if msk.sum() < 50: continue
        m = s["meter"]
        for name, fn in CONS.items():
            ph = fn(s, T)
            if ph is None: continue
            dphi = wrap(np.diff(ph))[msk[1:]]
            med = np.median(dphi)
            neg = int((dphi < 0).sum()); z = int((dphi == 0).sum())
            stall = int((dphi < 0.1*med).sum())
            a = tot[name]; a[0]+=len(dphi); a[1]+=neg; a[2]+=z; a[3]+=stall
            rows[name].append(dict(dataset=s["dataset"], stem=s["stem"],
                                   fneg=neg/len(dphi), fstall=stall/len(dphi),
                                   med=med, cv=float(np.std(dphi)/med),
                                   minr=float(dphi.min()/med), maxr=float(dphi.max()/med)))
        ph = phase_pchip(s, T)
        if ph is not None:
            dphi = np.diff(ph)[msk[1:]]
            med = np.median(dphi); neg = int((dphi < 0).sum())
            a = tot["pchip"]; a[0]+=len(dphi); a[1]+=neg; a[2]+=0; a[3]+=int((dphi<0.1*med).sum())
            rows["pchip"].append(dict(dataset=s["dataset"], stem=s["stem"],
                                      fneg=neg/len(dphi), fstall=float((dphi<0.1*med).mean()),
                                      med=med, cv=float(np.std(dphi)/med),
                                      minr=float(dphi.min()/med), maxr=float(dphi.max()/med)))
    print(f"##### SPLIT={split}  ({len(L)} songs)")
    for name in rows:
        n, neg, z, st = tot[name]
        print(f"-- construction={name}: {len(rows[name])} songs, {n} frame-increments")
        print(f"   frac NEGATIVE = {neg/n:.6f} ({neg}/{n});  frac exactly 0 = {z/n:.6f};  frac < 0.1*median (stall) = {st/n:.6f}")
        print(f"   per-song frac_neg   : {fmt_ds(per_ds(rows[name],'fneg'),6)}")
        print(f"   per-song CV(dphi)   : {fmt_ds(per_ds(rows[name],'cv'))}")
        print(f"   per-song min/median : {fmt_ds(per_ds(rows[name],'minr'))}")
        print(f"   per-song max/median : {fmt_ds(per_ds(rows[name],'maxr'))}")
    return rows

for sp in ("train","eval"): run(sp)
