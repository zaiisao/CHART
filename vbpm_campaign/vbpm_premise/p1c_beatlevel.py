"""PREMISE 1(c) done where the DATA constrains it: the bar-pointer transition residual
integrated over ONE BEAT (frame-level residuals are an artifact of the interpolation).

Advance phi at the previous rate for the true duration of beat k:
  reached = (2pi/m) * IBI_k / IBI_{k-1}   ->  residual r_k = (2pi/m)(IBI_k/IBI_{k-1} - 1)
Fixed-tempo variant uses the song median IBI. Per-frame sigma follows from a random walk:
  sigma_frame = sigma_beat / sqrt(N_frames_per_beat).
"""
import sys, numpy as np
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
from common import load_labels, FPS, per_ds, fmt_ds
from circfit import vm_logpdf, vm_fit, wc_logpdf, wc_fit, wc_gamma
TWO_PI = 2*np.pi

def beat_resid(s, pred):
    b = s["beats"]; m = s["meter"]
    ibi = np.diff(b)
    ibi = ibi[(ibi > 0.1) & (ibi < 2.0)]
    if len(ibi) < 20: return None, None
    br = TWO_PI/m
    if pred == "prevrate":
        r = br*(ibi[1:]/ibi[:-1] - 1.0); npf = ibi[1:]*FPS
    else:
        med = np.median(ibi); r = br*(ibi/med - 1.0); npf = ibi*FPS
    return r, npf

for pred in ("prevrate","meanrate"):
    print(f"\n########## BEAT-LEVEL transition residual, phidot = {pred}")
    tr = load_labels("train"); ev = load_labels("eval")
    rows=[]; R_tr={}; R_ev={}
    for s in tr:
        r, npf = beat_resid(s, pred)
        if r is None: continue
        R_tr[s["stem"]] = r
        k,_ = vm_fit(r); rho,_ = wc_fit(r)
        sdb = float(np.std(r)); sdf = float(np.mean(np.std(r)/np.sqrt(npf)))
        rows.append(dict(dataset=s["dataset"], kappa_beat=k, logk=float(np.log10(k)),
                         gamma=wc_gamma(rho), sd_beatstep=sdb,
                         sd_beatstep_beats=sdb/(TWO_PI/s["meter"]),
                         kappa_frame=1.0/max(sdf,1e-12)**2, logkf=float(np.log10(1/max(sdf,1e-12)**2)),
                         n=len(r)))
    for s in ev:
        r,_ = beat_resid(s, pred)
        if r is not None: R_ev[s["stem"]] = r
    allr = np.concatenate(list(R_tr.values()))
    print(f"  train: {len(R_tr)} songs, {len(allr)} beat-transitions;  eval: {len(R_ev)} songs, {sum(len(v) for v in R_ev.values())}")
    ks = np.array([r["kappa_beat"] for r in rows]); kf = np.array([r["kappa_frame"] for r in rows])
    print(f"  PER-SONG kappa (one-beat step): median={np.median(ks):.4g} p5={np.percentile(ks,5):.4g} "
          f"p95={np.percentile(ks,95):.4g} min={ks.min():.4g} max={ks.max():.4g} | p95/p5={np.percentile(ks,95)/np.percentile(ks,5):.1f}x max/min={ks.max()/ks.min():.1f}x")
    print(f"  PER-SONG kappa (per FRAME, RW-scaled): median={np.median(kf):.4g} p5={np.percentile(kf,5):.4g} p95={np.percentile(kf,95):.4g} | p95/p5={np.percentile(kf,95)/np.percentile(kf,5):.1f}x")
    print("  per-song log10 kappa_beat :", fmt_ds(per_ds(rows,'logk'),3))
    print("  per-song log10 kappa_frame:", fmt_ds(per_ds(rows,'logkf'),3))
    print("  per-song resid sd [beats] :", fmt_ds(per_ds(rows,'sd_beatstep_beats'),4))
    k_pool, mu_k = vm_fit(allr); rho_pool, mu_r = wc_fit(allr)
    print(f"  POOLED train fit: vM kappa={k_pool:.5g} (sd {1/np.sqrt(k_pool):.4f} rad = {1/np.sqrt(k_pool)/(TWO_PI/4):.4f} beat)"
          f" | wC rho={rho_pool:.6f} gamma={wc_gamma(rho_pool):.5f} rad")
    er=[]
    for st, r in R_ev.items():
        ds = st.split("__")[1].split("_")[0]
        lv = float(vm_logpdf(r, k_pool, mu_k).mean()); lw = float(wc_logpdf(r, rho_pool, mu_r).mean())
        ko, muo = vm_fit(r); rho_o, muro = wc_fit(r)
        rho_o = min(rho_o, 1-1e-9)
        lvo = float(vm_logpdf(r, ko, muo).mean()); lwo = float(wc_logpdf(r, rho_o, muro).mean())
        er.append(dict(dataset=ds, vm=lv, wc=lw, gain_wc=lw-lv,
                       cost_fixed_vm=lvo-lv, cost_fixed_wc=lwo-lw,
                       best_fixed=max(lv,lw), best_oracle=max(lvo,lwo)))
    print(f"  HELD-OUT eval ({len(er)} songs), nats per beat-transition:")
    print("    vM fixed-pooled                  :", fmt_ds(per_ds(er,'vm'),3))
    print("    wC fixed-pooled                  :", fmt_ds(per_ds(er,'wc'),3))
    print("    wC - vM  (>0 => heavy tails win) :", fmt_ds(per_ds(er,'gain_wc'),3))
    print("    per-song-oracle vM - fixed vM    :", fmt_ds(per_ds(er,'cost_fixed_vm'),3))
    print("    per-song-oracle wC - fixed wC    :", fmt_ds(per_ds(er,'cost_fixed_wc'),3))
