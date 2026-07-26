"""PREMISE 1(b): is bar-phase advance constant within a bar?  + smooth-tempo variance share."""
import sys, numpy as np
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
from common import load_labels, FPS, per_ds, fmt_ds
from phases import inside_mask, phase_beatlinear, bar_knots, frame_t, TWO_PI

def bar_regression(s):
    """Regress TRUE (beat-linear) bar phase on time inside each bar. Returns per-song aggregates."""
    T = s["T"]; ph = phase_beatlinear(s, T)
    if ph is None: return None
    t = frame_t(T); m = s["meter"]
    ss_res = ss_tot = 0.0; res = []
    nbar = 0
    for (a, e, inb) in bar_knots(s):
        msk = (t >= a) & (t < e)
        if msk.sum() < 8 or len(inb) < 1: continue
        x = t[msk]; y = ph[msk]
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ coef
        ss_res += float((r**2).sum()); ss_tot += float(((y-y.mean())**2).sum())
        res.append(r); nbar += 1
    if nbar < 3: return None
    r = np.concatenate(res)
    beat_rad = TWO_PI / m
    return dict(R2=1 - ss_res/ss_tot, resid_beats=float(np.std(r))/beat_rad,
                maxres_beats=float(np.abs(r).max())/beat_rad, nbar=nbar)

def beat_placement(s):
    """Deviation of each within-bar beat from equal subdivision, in UNITS OF A BEAT."""
    devs = []
    for (a, e, inb) in bar_knots(s):
        m = len(inb)
        if m < 2 or e - a < 1e-6: continue
        u = (inb - a) / (e - a)                      # in [0,1)
        devs.append((u - np.arange(m)/m) * m)        # beats
    if not devs: return None
    d = np.concatenate(devs)
    return dict(bp_rms=float(np.sqrt((d**2).mean())), bp_max=float(np.abs(d).max()),
                bp_n=len(d))

def smooth_tempo_share(s, W=8):
    """Variance of beat times about a CONSTANT-tempo grid, share explained by a SMOOTH
    (moving-average, W beats) tempo trajectory. R2 = 1 - SSres_smooth / SSres_const."""
    b = s["beats"]
    if len(b) < 4*W: return None
    k = np.arange(len(b))
    A = np.vstack([k, np.ones_like(k)]).T
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    r_const = b - A @ coef
    ibi = np.diff(b)
    ker = np.ones(W)/W
    ibis = np.convolve(np.pad(ibi, (W//2, W//2), mode='edge'), ker, mode='same')[W//2:W//2+len(ibi)]
    bhat = np.concatenate([[b[0]], b[0] + np.cumsum(ibis)])
    # remove residual affine drift (smooth model only claims the SHAPE)
    coef2, *_ = np.linalg.lstsq(A, b - bhat, rcond=None)
    r_sm = (b - bhat) - A @ coef2
    return dict(share=1 - float((r_sm**2).sum())/max(float((r_const**2).sum()), 1e-12),
                sd_const_ms=float(np.std(r_const))*1000, sd_sm_ms=float(np.std(r_sm))*1000)

for sp in ("train","eval"):
    L = load_labels(sp)
    rowsA=[]; rowsB=[]; rowsC={4:[],8:[],16:[]}
    for s in L:
        a = bar_regression(s)
        if a: rowsA.append(dict(dataset=s["dataset"], **a))
        b = beat_placement(s)
        if b: rowsB.append(dict(dataset=s["dataset"], **b))
        for W in rowsC:
            c = smooth_tempo_share(s, W)
            if c: rowsC[W].append(dict(dataset=s["dataset"], **c))
    print(f"##### SPLIT={sp}: within-bar linear-phase regression ({len(rowsA)} songs, {sum(r['nbar'] for r in rowsA)} bars)")
    print("  R^2                :", fmt_ds(per_ds(rowsA,'R2'),5))
    print("  resid sd [beats]   :", fmt_ds(per_ds(rowsA,'resid_beats'),4))
    print("  resid max [beats]  :", fmt_ds(per_ds(rowsA,'maxres_beats'),4))
    print(f"  beat-vs-equal-subdivision deviation ({sum(r['bp_n'] for r in rowsB)} beats)")
    print("    rms [beats]      :", fmt_ds(per_ds(rowsB,'bp_rms'),4))
    print("    max [beats]      :", fmt_ds(per_ds(rowsB,'bp_max'),4))
    for W in (4,8,16):
        print(f"  smooth-tempo (W={W} beats) variance share of beat timing ({len(rowsC[W])} songs)")
        print("    share            :", fmt_ds(per_ds(rowsC[W],'share'),4))
        print("    sd const [ms]    :", fmt_ds(per_ds(rowsC[W],'sd_const_ms'),1))
        print("    sd smooth [ms]   :", fmt_ds(per_ds(rowsC[W],'sd_sm_ms'),1))
