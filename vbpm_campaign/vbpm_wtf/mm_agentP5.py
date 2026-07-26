"""PROBE 5 (agent P5): is the observation emission evaluated OUT OF DISTRIBUTION at deploy?

(a) z_feat distribution during TRAINING (posterior recursion) vs during PF DEPLOY
(b) trained log-tempo vs the PF's forced [-3.55,-2.18] band
(c) meter: Gumbel-Softmax relaxation (train) vs hard one-hot (PF)
(d) phase support, train vs PF
+ the decisive cross-check: is the emission phase-blind ONLY in the PF regime, or everywhere?
"""
from __future__ import annotations
import argparse, json, math, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_wtf")
from mm_agentP5_lib import (load_some, obs_cache, load_arm, merged_h, train_trace,   # noqa
                            pf_trace, qs, TWO_PI, FPS)

DEV = "cuda:0"
BAND = (-3.55, -2.18)


def targets(beats, downs, start, n):
    b = np.zeros(n, np.float32); db = np.zeros(n, np.float32)
    for t in beats:
        i = int(round(t * FPS)) - start
        if 0 <= i < n: b[i] = 1.0
    for t in downs:
        i = int(round(t * FPS)) - start
        if 0 <= i < n: db[i] = 1.0
    return b, db


def bce(logit, tgt):
    return float(F.binary_cross_entropy_with_logits(logit, tgt, reduction="none").sum(1).mean())


def circ_R(phi):
    phi = np.asarray(phi, float).ravel()
    return float(np.hypot(np.cos(phi).mean(), np.sin(phi).mean()))


def overlap(a, b, lo, hi, nb=120):
    ha, _ = np.histogram(a, bins=nb, range=(lo, hi), density=False)
    hb, _ = np.histogram(b, bins=nb, range=(lo, hi), density=False)
    ha = ha / max(ha.sum(), 1); hb = hb / max(hb.sum(), 1)
    return float(np.minimum(ha, hb).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="i_bern")
    ap.add_argument("--n_train", type=int, default=20)
    ap.add_argument("--n_eval", type=int, default=8)
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--pf_frames", type=int, default=3000)
    ap.add_argument("--K", type=int, default=300)
    a = ap.parse_args()
    arm2 = a.tag.startswith("ii")
    out = {"tag": a.tag}
    torch.manual_seed(0); rng = np.random.default_rng(0)

    print(f"=== PROBE 5  arm tag={a.tag} ===", flush=True)
    model, lw, cfg, ck = load_arm(a.tag, DEV)
    print("  cfg:", {k: cfg[k] for k in ("obs", "steps", "warmup", "bs", "frames", "hidden")})
    print(f"  tempo_dof={float(model.tempo_dof()):.3f} level_ar={float(model.level_ar()):.4f} "
          f"level_offset={float(model.level_offset):.3f}")

    train = load_some("train", a.n_train)
    ev = load_some("eval", a.n_eval)
    otr = obs_cache(train, f"{load_arm.__globals__['ARMS']}/act_train.npz")
    oev = obs_cache(ev, f"{load_arm.__globals__['ARMS']}/act_eval.npz")
    print(f"  loaded train {len(train)} eval {len(ev)}", flush=True)

    # ---------------------------------------------------------------- (A) TRAIN regime
    LT, PHI, LEV, DEV_, MMAX, MENT, MARG, QRHO, PRHO, CROSS = ([] for _ in range(10))
    PLVS, PDVS = [], []
    rec_rows = []
    for it in range(a.batches):
        fe, bb, dd, oo = [], [], [], []
        while len(fe) < a.bs:
            s = train[rng.integers(len(train))]
            T = s["feats"].shape[1]
            if T <= a.frames: continue
            st = int(rng.integers(0, T - a.frames))
            fe.append(s["feats"][:, st:st + a.frames, :])
            b_, d_ = targets(s["beats"], s["downs"], st, a.frames)
            bb.append(b_); dd.append(d_)
            oo.append(otr[s["stem"]][st:st + a.frames])
        f = torch.from_numpy(np.asarray(np.stack(fe), np.float32)).to(DEV)
        b = torch.from_numpy(np.stack(bb)).to(DEV)
        d = torch.from_numpy(np.stack(dd)).to(DEV)
        o = torch.from_numpy(np.stack(oo)).to(DEV)
        h = o if arm2 else torch.einsum("l,bltf->btf", lw, f)
        tr = train_trace(model, h, b, temperature=0.3)
        z = tr["z"]
        LT.append(tr["lt"].cpu().numpy()); PHI.append(tr["phi"].cpu().numpy())
        LEV.append(tr["level"].cpu().numpy()); DEV_.append(tr["dev"].cpu().numpy())
        m = tr["meter"].cpu().numpy()
        MMAX.append(m.max(-1)); MARG.append(m.argmax(-1))
        MENT.append(-(m * np.log(m + 1e-12)).sum(-1))
        QRHO.append(tr["q_rho"].cpu().numpy()); PRHO.append(tr["p_rho"][:, 1:].cpu().numpy())
        CROSS.append(tr["cross"][:, 1:].cpu().numpy())
        PLVS.append(tr["p_lv_s"][:, 1:].cpu().numpy()); PDVS.append(tr["p_dv_s"][:, 1:].cpu().numpy())

        # ---- ablations: what does each decoder actually USE? ----
        B, T, _ = z.shape
        def dec_bce(zz):
            lg = model.decoder(zz)
            return bce(lg[..., 0], b), bce(lg[..., 1], d)
        def obs_nll(zz):
            return float(-model.obs_logp(zz.reshape(-1, 7), o.reshape(-1, 2)).reshape(B, T).sum(1).mean())
        variants = {}
        variants["as_is"] = z
        zr = z.clone(); ph = torch.rand(B, T, device=DEV) * TWO_PI
        zr[..., 0], zr[..., 1] = torch.cos(ph), torch.sin(ph)
        variants["phase_RANDOM"] = zr
        zc = z.clone(); zc[..., 0], zc[..., 1] = 1.0, 0.0
        variants["phase_CONST0"] = zc
        zl = z.clone(); zl[..., 2] = z[..., 2].mean(1, keepdim=True)
        variants["logtempo_CROPMEAN"] = zl
        zb = z.clone(); zb[..., 2] = -2.87
        variants["logtempo_BANDMID"] = zb
        zm = z.clone(); zm[..., 3:] = F.one_hot(z[..., 3:].argmax(-1), 4).float()
        variants["meter_HARD"] = zm
        zm4 = z.clone(); zm4[..., 3:] = 0.0; zm4[..., 6] = 1.0
        variants["meter_CONST4"] = zm4
        zpf = z.clone()                       # full PF-regime z: hard meter + band tempo
        zpf[..., 2] = -2.87
        zpf[..., 3:] = F.one_hot(z[..., 3:].argmax(-1), 4).float()
        variants["PF_REGIME(z lt=band,meter hard)"] = zpf
        row = {}
        for k, zz in variants.items():
            rb, rd = dec_bce(zz)
            row[k] = dict(rec_beat=rb, rec_db=rd, rec_obs=obs_nll(zz))
        # base rates
        pb = float(b.mean()); pd = float(d.mean()); po = o.reshape(-1, 2).mean(0)
        row["_baserate"] = dict(
            rec_beat=float(-(b * math.log(pb + 1e-9) + (1 - b) * math.log(1 - pb + 1e-9)).sum(1).mean()),
            rec_db=float(-(d * math.log(pd + 1e-9) + (1 - d) * math.log(1 - pd + 1e-9)).sum(1).mean()),
            rec_obs=float(-(o * torch.log(po + 1e-9) + (1 - o) * torch.log(1 - po + 1e-9)).sum((1, 2)).mean()))
        rec_rows.append(row)
        print(f"  batch {it+1}/{a.batches} done", flush=True)

    LT = np.concatenate([x.ravel() for x in LT]); PHI = np.concatenate([x.ravel() for x in PHI])
    LEV = np.concatenate([x.ravel() for x in LEV]); DEV_ = np.concatenate([x.ravel() for x in DEV_])
    MMAX = np.concatenate([x.ravel() for x in MMAX]); MENT = np.concatenate([x.ravel() for x in MENT])
    MARG = np.concatenate([x.ravel() for x in MARG])
    QRHO = np.concatenate([x.ravel() for x in QRHO]); PRHO = np.concatenate([x.ravel() for x in PRHO])
    CROSS = np.concatenate([x.ravel() for x in CROSS])
    PLVS = np.concatenate([x.ravel() for x in PLVS]); PDVS = np.concatenate([x.ravel() for x in PDVS])

    print("\n--- (A) TRAINING-REGIME z_feat (posterior recursion, temp=0.3) ---")
    print("  " + qs(LT, "log_tempo "))
    print("  " + qs(LEV, "level     "))
    print("  " + qs(DEV_, "dev       "))
    print("  " + qs(PHI, "phi       "))
    print("  " + qs(QRHO, "q_rho     "))
    print("  " + qs(PRHO, "p_rho     "))
    print("  " + qs(PLVS, "p_lv_sig  "))
    print("  " + qs(PDVS, "p_dv_sig  "))
    print(f"  meter softmax max-prob: mean={MMAX.mean():.4f} med={np.median(MMAX):.4f} "
          f"p1={np.percentile(MMAX,1):.4f} frac>0.99={np.mean(MMAX>0.99):.4f}; "
          f"entropy mean={MENT.mean():.4f}; argmax hist={np.bincount(MARG,minlength=4)/MARG.size}")
    print(f"  bar-crossing rate (train)   = {CROSS.mean():.5f} per frame "
          f"(=> bar every {1/max(CROSS.mean(),1e-9):.1f} frames = {1/max(CROSS.mean(),1e-9)/FPS:.2f} s)")
    print(f"  implied phi-advance exp(lt) : mean={np.exp(np.clip(LT,-12,6)).mean():.4f} "
          f"med={np.median(np.exp(np.clip(LT,-12,6))):.4f} rad/frame "
          f"(physical band = {math.exp(BAND[0]):.4f}..{math.exp(BAND[1]):.4f})")
    frac_band = float(np.mean((LT >= BAND[0]) & (LT <= BAND[1])))
    print(f"  *** fraction of TRAIN log_tempo inside PF band [-3.55,-2.18] = {frac_band:.5f} ***")
    print(f"  circular R of train phi = {circ_R(PHI):.4f}  (0 = uniform)")
    out["train"] = dict(lt=dict(mean=float(LT.mean()), sd=float(LT.std()),
                                p1=float(np.percentile(LT, 1)), med=float(np.median(LT)),
                                p99=float(np.percentile(LT, 99)), min=float(LT.min()), max=float(LT.max())),
                        frac_in_band=frac_band, meter_maxprob=float(MMAX.mean()),
                        meter_entropy=float(MENT.mean()), phi_R=circ_R(PHI),
                        q_rho=float(QRHO.mean()), p_rho=float(PRHO.mean()),
                        cross_rate=float(CROSS.mean()))

    print("\n--- ABLATION: which z dims do the two decoders actually use? (per 256-frame crop) ---")
    keys = list(rec_rows[0].keys())
    for k in keys:
        rb = np.mean([r[k]["rec_beat"] for r in rec_rows])
        rd = np.mean([r[k]["rec_db"] for r in rec_rows])
        ro = np.mean([r[k]["rec_obs"] for r in rec_rows])
        print(f"  {k:32s} rec_beat={rb:8.2f}  rec_db={rd:7.2f}  rec_obs={ro:8.2f}")
        out.setdefault("ablation", {})[k] = dict(rec_beat=float(rb), rec_db=float(rd), rec_obs=float(ro))

    # ---------------------------------------------------------------- (B) PF regime
    print("\n--- (B) PF DEPLOY-REGIME particle z_feat ---", flush=True)
    PLT, PPHI, PM, PW, PRHO2, PDN = [], [], [], [], [], []
    for s in ev:
        T = min(s["T"], a.pf_frames)
        o = torch.from_numpy(oev[s["stem"]][:T]).unsqueeze(0).to(DEV)
        if arm2:
            h = o
        else:
            f = torch.from_numpy(np.asarray(s["feats"][:, :T, :], np.float32)).to(DEV)
            h = torch.einsum("l,ltf->tf", lw, f).unsqueeze(0)
        r = pf_trace(model, h, o, K=a.K, alpha=1.0, seed=1234, record_every=5)
        PLT.append(r["lt"].ravel()); PPHI.append(r["phi"].ravel())
        PM.append(r["m"].ravel()); PW.append(r["w"].ravel())
        PRHO2.append(r["rho"]); PDN.append(r["dphi_noise"])
        print(f"  PF {s['stem'][:44]:44s} T={T}", flush=True)
    PLT = np.concatenate(PLT); PPHI = np.concatenate(PPHI); PM = np.concatenate(PM)
    PW = np.concatenate(PW); PRHO2 = np.concatenate(PRHO2); PDN = np.concatenate(PDN)
    print("  " + qs(PLT, "PF log_tempo"))
    print("  " + qs(PPHI, "PF phi      "))
    print("  " + qs(PRHO2, "PF prior rho"))
    print("  " + qs(PDN, "PF |phi noise| rad/frame"))
    print(f"  PF meter argmax hist = {np.bincount(PM, minlength=4)/PM.size}")
    print(f"  PF circular R of phi = {circ_R(PPHI):.4f}")
    lo = min(LT.min(), PLT.min()); hi = max(LT.max(), PLT.max())
    ov = overlap(LT, PLT, lo, hi)
    ov2 = overlap(LT, PLT, -6, 2)
    frac_pf_in_train = float(np.mean((PLT >= np.percentile(LT, 1)) & (PLT <= np.percentile(LT, 99))))
    frac_tr_in_pf = float(np.mean((LT >= np.percentile(PLT, 1)) & (LT <= np.percentile(PLT, 99))))
    print(f"  *** log-tempo histogram OVERLAP(train, PF) = {ov:.4f} (full range) / {ov2:.4f} ([-6,2]) ***")
    print(f"  *** frac PF lt inside train [p1,p99] = {frac_pf_in_train:.4f} ; "
          f"frac train lt inside PF [p1,p99] = {frac_tr_in_pf:.4f} ***")
    out["pf"] = dict(lt=dict(mean=float(PLT.mean()), sd=float(PLT.std()),
                             p1=float(np.percentile(PLT, 1)), med=float(np.median(PLT)),
                             p99=float(np.percentile(PLT, 99)), min=float(PLT.min()), max=float(PLT.max())),
                     phi_R=circ_R(PPHI), rho=float(PRHO2.mean()),
                     meter_hist=(np.bincount(PM, minlength=4) / PM.size).tolist(),
                     overlap_lt=ov, overlap_lt_clip=ov2,
                     frac_pf_in_train=frac_pf_in_train, frac_train_in_pf=frac_tr_in_pf)

    # ------------------------------------------------- (C) emission phase sensitivity vs tempo
    print("\n--- (C) is the emission phase-blind ONLY at PF tempo, or EVERYWHERE? ---")
    # take real eval observations; sweep phase on a 24-pt grid at several log-tempo values
    s = ev[0]
    T = min(s["T"], 2000)
    o = torch.from_numpy(oev[s["stem"]][:T]).to(DEV)              # [T,2]
    grid = torch.arange(24, device=DEV).float() / 24 * TWO_PI
    lts = dict(train_p1=np.percentile(LT, 1), train_p25=np.percentile(LT, 25),
               train_med=np.median(LT), train_p75=np.percentile(LT, 75),
               train_p99=np.percentile(LT, 99), band_lo=BAND[0], band_mid=-2.87,
               band_hi=BAND[1], zero=0.0)
    print(f"  (sweep on {s['stem'][:40]}, T={T}, meter=one-hot 4/4)")
    print(f"  {'log_tempo':>22s} {'obs sd_over_phase':>18s} {'obs contrast(max-min)':>22s} "
          f"{'beat-logit sd':>14s} {'beat p(min..max)':>22s}")
    sens = {}
    for name, lt in lts.items():
        zz = []
        for g in grid:
            mm = torch.zeros(T, 4, device=DEV); mm[:, 3] = 1.0
            zz.append(model.z_features(mm, g.expand(T), torch.full((T,), float(lt), device=DEV)))
        Z = torch.stack(zz)                                        # [G,T,7]
        with torch.no_grad():
            lp = model.obs_logp(Z.reshape(-1, 7), o.repeat(24, 1)).reshape(24, T)
            bl = model.decoder(Z.reshape(-1, 7)).reshape(24, T, 2)[..., 0]
        sd_obs = float(lp.std(0).mean()); rng_obs = float((lp.max(0).values - lp.min(0).values).mean())
        sd_b = float(bl.std(0).mean())
        pmin = float(torch.sigmoid(bl.min(0).values).mean()); pmax = float(torch.sigmoid(bl.max(0).values).mean())
        print(f"  {name:>12s} {lt:+8.3f} {sd_obs:18.5f} {rng_obs:22.5f} {sd_b:14.5f} "
              f"{pmin:9.4f}..{pmax:.4f}")
        sens[name] = dict(lt=float(lt), obs_sd=sd_obs, obs_range=rng_obs, beat_logit_sd=sd_b,
                          beat_p_min=pmin, beat_p_max=pmax)
    out["phase_sensitivity"] = sens

    # tempo sensitivity of the emission (the side-channel check), phase fixed
    print("\n  --- emission/beat sensitivity to LOG-TEMPO at fixed phase (phi=0) ---")
    lt_grid = torch.linspace(float(np.percentile(LT, 1)), float(np.percentile(LT, 99)), 24, device=DEV)
    zz = []
    for v in lt_grid:
        mm = torch.zeros(T, 4, device=DEV); mm[:, 3] = 1.0
        zz.append(model.z_features(mm, torch.zeros(T, device=DEV), v.expand(T)))
    Z = torch.stack(zz)
    with torch.no_grad():
        lp = model.obs_logp(Z.reshape(-1, 7), o.repeat(24, 1)).reshape(24, T)
        bl = model.decoder(Z.reshape(-1, 7)).reshape(24, T, 2)[..., 0]
    print(f"  obs logp: sd over lt-grid = {float(lp.std(0).mean()):.5f}  "
          f"range = {float((lp.max(0).values-lp.min(0).values).mean()):.5f}")
    print(f"  beat logit: sd over lt-grid = {float(bl.std(0).mean()):.5f}  "
          f"p range = {float(torch.sigmoid(bl.min(0).values).mean()):.4f}.."
          f"{float(torch.sigmoid(bl.max(0).values).mean()):.4f}")
    out["tempo_sensitivity"] = dict(obs_sd=float(lp.std(0).mean()),
                                    obs_range=float((lp.max(0).values - lp.min(0).values).mean()),
                                    beat_sd=float(bl.std(0).mean()))

    # first-layer weight norms per z dim
    W = model.h_dec[0].weight.detach().cpu().numpy()
    Wd = model.decoder[0].weight.detach().cpu().numpy()
    names = ["cos", "sin", "logT", "m1", "m2", "m3", "m4"]
    print("\n  |W| column norms of the FIRST layer (which z dim each decoder listens to):")
    print("    obs  emission: " + "  ".join(f"{n}={np.linalg.norm(W[:, i]):.3f}" for i, n in enumerate(names)))
    print("    beat decoder : " + "  ".join(f"{n}={np.linalg.norm(Wd[:, i]):.3f}" for i, n in enumerate(names)))
    out["w_obs"] = {n: float(np.linalg.norm(W[:, i])) for i, n in enumerate(names)}
    out["w_dec"] = {n: float(np.linalg.norm(Wd[:, i])) for i, n in enumerate(names)}

    json.dump(out, open(f"/home/sogang/jaehoon/VBPM_reintegration/vbpm_wtf/mm_agentP5_{a.tag}.json", "w"),
              indent=1, default=float)
    print("\nWROTE mm_agentP5_%s.json" % a.tag)


if __name__ == "__main__":
    main()
