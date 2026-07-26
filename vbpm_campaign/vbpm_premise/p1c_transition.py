"""PREMISE 1(c): what concentration does p(phi_t | phi_{t-1}, phidot_{t-1}) actually need,
and does a SINGLE fixed concentration hold across songs?  (fit train, score eval)"""
import sys, numpy as np, json
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
from common import load_labels, FPS, per_ds, fmt_ds
from phases import wrap, inside_mask, phase_beatlinear, phase_pchip, TWO_PI
from circfit import vm_logpdf, vm_fit, wc_logpdf, wc_fit, wc_gamma

def residuals(s, cons, pred):
    T = s["T"]
    ph = phase_beatlinear(s, T) if cons == "beatlinear" else phase_pchip(s, T)
    if ph is None: return None
    msk = inside_mask(s, T)
    d = wrap(np.diff(ph)) if cons == "beatlinear" else np.diff(ph)
    if pred == "meanrate":                     # fixed-tempo physical prior
        r = wrap(d - np.median(d[msk[1:]]))
        return r[msk[1:]]
    r = wrap(d[1:] - d[:-1])                   # prev-rate: 2nd difference
    return r[msk[2:]]

def summarize(rs, label):
    r = np.concatenate(rs)
    k_p, mu_k = vm_fit(r); rho_p, mu_r = wc_fit(r)
    return dict(label=label, n=len(r), kappa=k_p, rho=rho_p, mu_vm=mu_k,
                sd_rad=float(np.std(r)), mad_rad=float(np.median(np.abs(r))))

for cons in ("beatlinear","pchip"):
  for pred in ("meanrate","prevrate"):
    print(f"\n########## construction={cons}  phidot={pred}")
    tr = load_labels("train"); ev = load_labels("eval")
    R_tr, R_ev = {}, {}
    rows = []
    for s in tr:
        r = residuals(s, cons, pred)
        if r is None or len(r) < 200: continue
        R_tr[s["stem"]] = r
        k, _ = vm_fit(r); rho, _ = wc_fit(r)
        m = s["meter"]; beat_rad = TWO_PI/m
        rows.append(dict(dataset=s["dataset"], stem=s["stem"], kappa=k, logk=np.log10(k),
                         rho=rho, gamma=wc_gamma(rho), sd_rad=float(np.std(r)),
                         sd_beats=float(np.std(r))/beat_rad,
                         fzero=float((np.abs(r) < 1e-12).mean())))
    for s in ev:
        r = residuals(s, cons, pred)
        if r is not None and len(r) >= 200: R_ev[s["stem"]] = r
    allr_tr = np.concatenate(list(R_tr.values()))
    print(f"  train frames={len(allr_tr)} ({len(R_tr)} songs); eval songs={len(R_ev)}")
    print(f"  frac |resid|<1e-12 (exactly deterministic) = {(np.abs(allr_tr)<1e-12).mean():.4f}")
    ks = np.array([r["kappa"] for r in rows]); gs = np.array([r["gamma"] for r in rows])
    print(f"  PER-SONG von Mises kappa (train fits): median={np.median(ks):.3g} "
          f"p5={np.percentile(ks,5):.3g} p95={np.percentile(ks,95):.3g} "
          f"min={ks.min():.3g} max={ks.max():.3g}  p95/p5={np.percentile(ks,95)/np.percentile(ks,5):.1f}x  max/min={ks.max()/ks.min():.1f}x")
    print(f"  PER-SONG wrapped-Cauchy gamma: median={np.median(gs):.3g} p5={np.percentile(gs,5):.3g} p95={np.percentile(gs,95):.3g}  p95/p5={np.percentile(gs,95)/np.percentile(gs,5):.1f}x")
    print("  per-song log10 kappa   :", fmt_ds(per_ds(rows,'logk'),3))
    print("  per-song resid sd[beat]:", fmt_ds(per_ds(rows,'sd_beats'),5))
    # ---- pooled (single fixed) fit on TRAIN, scored HELD-OUT on EVAL
    k_pool, mu_k = vm_fit(allr_tr); rho_pool, mu_r = wc_fit(allr_tr)
    print(f"  POOLED train fit: vM kappa={k_pool:.4g} (sd={1/np.sqrt(k_pool):.3e} rad)  "
          f"wC rho={rho_pool:.8f} (gamma={wc_gamma(rho_pool):.3e} rad)")
    er = []
    for st, r in R_ev.items():
        lv = vm_logpdf(r, k_pool, mu_k).mean(); lw = wc_logpdf(r, rho_pool, mu_r).mean()
        ko, muo = vm_fit(r); lvo = vm_logpdf(r, ko, muo).mean()
        rho_o, muro = wc_fit(r); lwo = wc_logpdf(r, rho_o, muro).mean()
        ds = st.split("__")[1].split("_")[0]
        er.append(dict(dataset=ds, vm=lv, wc=lw, vm_oracle=lvo, wc_oracle=lwo,
                       gain_wc=lw-lv, cost_fixed_vm=lvo-lv, cost_fixed_wc=lwo-lw,
                       kappa_ev=np.log10(ko)))
    print(f"  HELD-OUT (eval, {len(er)} songs) mean log-lik nats/frame:")
    print("    vM  fixed-pooled  :", fmt_ds(per_ds(er,'vm'),3))
    print("    wC  fixed-pooled  :", fmt_ds(per_ds(er,'wc'),3))
    print("    wC - vM (>0 = heavy tails win):", fmt_ds(per_ds(er,'gain_wc'),3))
    print("    per-song-ORACLE vM minus fixed-pooled vM (cost of one fixed concentration):",
          fmt_ds(per_ds(er,'cost_fixed_vm'),3))
    print("    per-song-ORACLE wC minus fixed-pooled wC:", fmt_ds(per_ds(er,'cost_fixed_wc'),3))
