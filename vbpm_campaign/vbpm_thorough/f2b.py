"""F2 sharpening: K=2000 PF, forward+backward (two-pass) posterior masks,
beat-synchronous downbeat Viterbi. Selection on train fold only; report on eval."""
from __future__ import annotations
import argparse, json, math, sys, time
import numpy as np

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_thorough")
from f2 import (fast_peaks, min_sep, snap, mask_from_hist, fold_hist, beats_with_labels,
                HERE, PF_CFG, load_cache)
from emission import (PhaseEmission, load_act, load_split, song_phase,
                      METERS, TWO_PI, FPS, _estimate_meter)
from run_exp2 import score_events, score_traj, summarize, pr
from pf2 import particle_filter
from vbpm.evaluate import beats_from_activation, downbeats_from_barphase, f_measure

K2 = 2000


def stage_cache2(split):
    tr = load_split("train"); at = load_act("train")
    emis = PhaseEmission(bins_per_beat=24, likelihood="gauss", smooth=0.0).fit(
        tr, at, phase_mode="downbeat")
    prior = np.zeros(5)
    for s in tr:
        m = _estimate_meter(s["beats"], s["downs"])
        if m in METERS:
            prior[m] += 1
    ev = tr if split == "train" else load_split("eval")
    ae = at if split == "train" else load_act("eval")
    store, t0 = {}, time.time()
    kw = dict(meter_prior=prior, fps=FPS, K=K2, alpha=PF_CFG["alpha"],
              sigma_lt=PF_CFG["sigma_lt"], sigma_phi=PF_CFG["sigma_phi"],
              p_switch=PF_CFG["p_switch"], noise=PF_CFG["noise"])
    for i, s in enumerate(ev):
        act = ae.get(s["stem"])
        if act is None:
            continue
        T = min(len(act), s["T"])
        ref = s["beats"][s["beats"] < T / FPS]
        if len(ref) < 3:
            continue
        LL = emis.padded_table(act[:T])
        of = particle_filter(LL, emis.nb, seed=PF_CFG["seed0"] + i, **kw)
        ob = particle_filter(LL[::-1].copy(), emis.nb, seed=91234 + i, **kw)
        st = s["stem"]
        store[st + "|phase_path"] = of["phase_path"].astype(np.float32)
        store[st + "|meter_path"] = of["meter_path"].astype(np.int8)
        store[st + "|hpsi_f"] = of["hist_psi"].astype(np.float16)
        store[st + "|hphi_f"] = of["hist_phi"].astype(np.float16)
        store[st + "|hpsi_b"] = ob["hist_psi"][::-1].astype(np.float16)
        store[st + "|hphi_b"] = ob["hist_phi"][::-1].astype(np.float16)
        store[st + "|meter_b"] = ob["meter_path"][::-1].astype(np.int8)
        store[st + "|ess"] = np.float32(of["ess"])
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(ev)} ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(f"{HERE}/pf2k_cache_{split}.npz", **store)
    print(f"cached2 {split} ({time.time()-t0:.0f}s)", flush=True)


def load_cache2(split):
    d = np.load(f"{HERE}/pf2k_cache_{split}.npz", allow_pickle=True)
    out = {}
    for k in d.files:
        st, key = k.rsplit("|", 1)
        out.setdefault(st, {})[key] = d[k]
    return out


def combine_mask(hf, hb, kappa, mode):
    mf = mask_from_hist(np.asarray(hf, np.float32), kappa)
    if mode == "f":
        return mf
    mb = mask_from_hist(np.asarray(hb, np.float32), kappa)
    if mode == "b":
        return mb
    if mode == "mean":
        return 0.5 * (mf + mb)
    return np.sqrt(np.maximum(mf, 1e-12) * np.maximum(mb, 1e-12))   # "geo"


def db_viterbi(B, act1, m, fps, delta=0.05):
    """Beat-synchronous downbeat Viterbi: states = beat-in-bar 0..m-1, cyclic +1."""
    B = np.asarray(B, float)
    if len(B) < 2 or m < 2:
        return np.array([]), -np.inf
    fr = np.clip((B * fps).astype(int), 0, len(act1) - 1)
    p = np.array([act1[max(0, f - 2): f + 3].max() for f in fr])
    p = np.clip(p, 1e-3, 1 - 1e-3)
    lp1, lp0 = np.log(p), np.log(1 - p)
    em = np.tile(lp0[:, None], (1, m)); em[:, 0] = lp1
    lt_stay = math.log(1 - delta); lt_jump = math.log(delta / max(m - 1, 1))
    V = em[0].copy(); ptr = np.zeros((len(B), m), int)
    for j in range(1, len(B)):
        Vn = np.empty(m); 
        for snew in range(m):
            cand = V + np.where((np.arange(m) + 1) % m == snew, lt_stay, lt_jump)
            k = int(cand.argmax()); ptr[j, snew] = k; Vn[snew] = cand[k] + em[j, snew]
        V = Vn
    j = int(V.argmax()); score = float(V.max())
    states = np.empty(len(B), int); states[-1] = j
    for t in range(len(B) - 1, 0, -1):
        states[t - 1] = ptr[t, states[t]]
    return B[states == 0], score / len(B)


def gather(split):
    songs = load_split(split); acts = load_act(split); pfc = load_cache2(split)
    data = []
    for s in songs:
        act = acts.get(s["stem"]); pf = pfc.get(s["stem"])
        if act is None or pf is None:
            continue
        T = min(len(act), s["T"])
        ref = s["beats"][s["beats"] < T / FPS]
        dref = s["downs"][s["downs"] < T / FPS]
        if len(ref) < 3:
            continue
        data.append(dict(s=s, act=act[:T], T=T, ref=ref, dref=dref,
                         ph=np.asarray(pf["phase_path"], float)[:T],
                         m_pf=int(np.bincount(np.asarray(pf["meter_path"], int)).argmax()),
                         m_gt=_estimate_meter(s["beats"], s["downs"]),
                         hpsi_f=pf["hpsi_f"][:T], hphi_f=pf["hphi_f"][:T],
                         hpsi_b=pf["hpsi_b"][:T], hphi_b=pf["hphi_b"][:T],
                         ess=float(pf["ess"])))
    return data


def mask_beats(d, kappa, eps, thr, mode, oracle_m=False):
    if oracle_m:
        hf, hb = fold_hist(np.asarray(d["hphi_f"], np.float32), d["m_gt"]), \
                 fold_hist(np.asarray(d["hphi_b"], np.float32), d["m_gt"])
        mk = combine_mask(hf, hb, kappa, mode)
    else:
        mk = combine_mask(d["hpsi_f"], d["hpsi_b"], kappa, mode)
    mb = d["act"][:, 0] * (eps + (1 - eps) * mk)
    return fast_peaks(mb, FPS, thr=thr, min_dist_sec=0.15)


def stage_select2():
    data = gather("train")
    print(f"select2 on {len(data)} train songs", flush=True)
    out = {}
    # ---- beats: combine mode x kappa x eps x thr
    best = (-1, None)
    for mode in ("f", "mean", "geo"):
        for kappa in (16.0, 32.0, 64.0):
            for eps in (0.10, 0.30, 0.50):
                for thr in (0.10, 0.20, 0.30, 0.40):
                    m = float(np.mean([f_measure(d["ref"],
                              mask_beats(d, kappa, eps, thr, mode)) for d in data]))
                    if m > best[0]:
                        best = (m, dict(mode=mode, kappa=kappa, eps=eps, thr=thr))
                        print(f"  MASK2 {mode} k={kappa} eps={eps} thr={thr} "
                              f"-> {m:.4f} *", flush=True)
    out["mask2"] = dict(train_F=best[0], **best[1])

    # ---- downbeats: viterbi over mask-beats, delta grid, m source
    bm = out["mask2"]
    Bs = [mask_beats(d, bm["kappa"], bm["eps"], bm["thr"], bm["mode"]) for d in data]
    for msrc in ("pf", "gt"):
        best = (-1, None)
        for delta in (0.01, 0.03, 0.10):
            fs = []
            for B, d in zip(Bs, data):
                m = d["m_pf"] if msrc == "pf" else d["m_gt"]
                est, _ = db_viterbi(B, d["act"][:, 1], m, FPS, delta)
                if len(d["dref"]) >= 2:
                    fs.append(f_measure(d["dref"], est))
            m = float(np.mean(fs))
            print(f"  DBVIT m={msrc} delta={delta} -> train db_F={m:.4f}", flush=True)
            if m > best[0]:
                best = (m, dict(delta=delta))
        out[f"dbvit_{msrc}"] = dict(train_F=best[0], **best[1])

    # ---- downbeats: two-pass phi mask on db channel
    best = (-1, None)
    for mode in ("f", "mean", "geo"):
        for kappa in (8.0, 16.0, 32.0):
            for eps in (0.10, 0.30, 0.50):
                for thr in (0.05, 0.10, 0.20, 0.30):
                    fs = []
                    for d in data:
                        if len(d["dref"]) < 2:
                            continue
                        mk = combine_mask(d["hphi_f"], d["hphi_b"], kappa, mode)
                        mb = d["act"][:, 1] * (eps + (1 - eps) * mk)
                        fs.append(f_measure(d["dref"],
                                  fast_peaks(mb, FPS, thr=thr, min_dist_sec=0.30)))
                    m = float(np.mean(fs))
                    if m > best[0]:
                        best = (m, dict(mode=mode, kappa=kappa, eps=eps, thr=thr))
                        print(f"  DBMASK2 {mode} k={kappa} eps={eps} thr={thr} "
                              f"-> {m:.4f} *", flush=True)
    out["dbmask2"] = dict(train_F=best[0], **best[1])
    json.dump(out, open(f"{HERE}/sel_f2b.json", "w"), indent=1)
    print("WROTE sel_f2b.json:", json.dumps(out), flush=True)


def stage_report2():
    sel = json.load(open(f"{HERE}/sel_f2b.json"))
    bm, dv, dm = sel["mask2"], sel["dbvit_pf"], sel["dbmask2"]
    data = gather("eval")
    rows = {}
    for d in data:
        base = dict(stem=d["s"]["stem"], n_true=len(d["ref"]), n_true_db=len(d["dref"]),
                    ess=d["ess"], obs_contrast=float("nan"),
                    meter_ok=float(d["m_pf"] == d["m_gt"]))
        B = mask_beats(d, bm["kappa"], bm["eps"], bm["thr"], bm["mode"])
        B_om = mask_beats(d, bm["kappa"], bm["eps"], bm["thr"], bm["mode"], oracle_m=True)
        db_v, _ = db_viterbi(B, d["act"][:, 1], d["m_pf"], FPS, dv["delta"])
        db_v_om, _ = db_viterbi(B, d["act"][:, 1], d["m_gt"], FPS, sel["dbvit_gt"]["delta"])
        mk = combine_mask(d["hphi_f"], d["hphi_b"], dm["kappa"], dm["mode"])
        db_m = fast_peaks(d["act"][:, 1] * (dm["eps"] + (1 - dm["eps"]) * mk), FPS,
                          thr=dm["thr"], min_dist_sec=0.30)
        # head-beats + viterbi control (no PF in the beat list; meter from PF)
        Bh = beats_from_activation(d["act"][:, 0], FPS)
        db_hv, _ = db_viterbi(Bh, d["act"][:, 1], d["m_pf"], FPS, dv["delta"])
        # oracle TRUE-phase mask ceiling (diagnostic only)
        ph_true = song_phase(d["s"], "downbeat")
        if ph_true is not None:
            psi_t = (d["m_gt"] * ph_true[:d["T"]]) % TWO_PI
            mko = np.exp(bm["kappa"] * (np.cos(psi_t) - 1.0))
            Bo = fast_peaks(d["act"][:, 0] * (bm["eps"] + (1 - bm["eps"]) * mko), FPS,
                            thr=bm["thr"], min_dist_sec=0.15)
        else:
            Bo = B
        for name, (eb, ed) in dict(
                mask2=(B, db_v), mask2_om=(B_om, db_v_om), mask2_dbmask=(B, db_m),
                headvit=(Bh, db_hv), oracle_phase_mask=(Bo, np.array([]))).items():
            rows.setdefault(name, []).append({**base, **score_events(eb, ed, d["ref"],
                                                                     d["dref"], d["T"])})
    res = {"sel": sel, "K": K2}
    print(f"\n=== eval fold ({len(data)} songs), K=2000 two-pass ===", flush=True)
    for k in rows:
        s = summarize(rows[k], f"ev {k}")
        pr(s)
        res.setdefault("eval", {})[k] = s
    res["rows"] = rows
    json.dump(res, open(f"{HERE}/f2b_report.json", "w"), indent=1, default=float)
    print("WROTE f2b_report.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["cache2", "select2", "report2"])
    ap.add_argument("--split", default="eval")
    a = ap.parse_args()
    if a.stage == "cache2":
        stage_cache2(a.split)
    elif a.stage == "select2":
        stage_select2()
    else:
        stage_report2()
