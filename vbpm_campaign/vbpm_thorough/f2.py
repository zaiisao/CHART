"""STRAND F2: combine the PF (structure) with the activation head (localisation).

Variants: (a) SNAP  (b) MASK  (c) oracle-meter versions  (e) downbeats.
All scores carry the MANDATORY density-matched blind control (reused verbatim
from vbpm_final/run_exp2.py) and n_est/n_true.

Fold-honest: emission fitted on train; ALL hyperparameter selection on the
train fold; eval fold (79 songs) touched only by the final report.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_final")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_thorough")

from emission import (PhaseEmission, load_act, load_split, obs_contrast, song_phase,
                      METERS, TWO_PI, FPS, _estimate_meter)
from run_exp2 import blind_grid_controls, score_events, score_traj, summarize, pr  # controls VERBATIM
from pf2 import particle_filter
from vbpm.evaluate import (beats_from_barphase, downbeats_from_barphase,
                           beats_from_activation, f_measure)

HERE = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_thorough"

# FINAL (already-selected) PF config from vbpm_final/FINAL_eval.json
PF_CFG = dict(K=600, alpha=0.25, sigma_lt=0.05, sigma_phi=0.03, p_switch=0.005,
              noise="gauss", seed0=1234)


# --------------------------------------------------------------------- helpers
def fast_peaks(prob, fps, thr=0.5, min_dist_sec=0.15):
    """Vectorised local-max pick; identical semantics to beats_from_activation."""
    p = np.asarray(prob, float)
    if len(p) < 3:
        return np.array([])
    c = (p[1:-1] >= thr) & (p[1:-1] >= p[:-2]) & (p[1:-1] >= p[2:])
    idx = np.where(c)[0] + 1
    out, last = [], -1e9
    gap = min_dist_sec * fps
    for f in idx:
        if f - last >= gap:
            out.append(f); last = f
    return np.asarray(out, float) / fps


def min_sep(ev, gap):
    out, last = [], -1e9
    for t in ev:
        if t - last >= gap:
            out.append(t); last = t
    return np.asarray(out, float)


def snap(events, peaks, win=0.07, min_gap=0.08):
    """Snap each event to the nearest peak within +-win (keep event if none)."""
    events = np.asarray(events, float)
    if len(peaks) == 0 or len(events) == 0:
        return events
    peaks = np.asarray(peaks, float)
    out = []
    for b in events:
        j = int(np.argmin(np.abs(peaks - b)))
        out.append(peaks[j] if abs(peaks[j] - b) <= win else b)
    return min_sep(np.unique(np.round(np.asarray(out), 5)), min_gap)


def mask_from_hist(hist, kappa):
    """[T,NH] posterior phase histogram -> von-Mises-kernel beat-position mask in (0,1]."""
    NH = hist.shape[1]
    th = (np.arange(NH) + 0.5) * TWO_PI / NH
    k = np.exp(kappa * (np.cos(th) - 1.0))
    return hist.astype(np.float64) @ k


def fold_hist(hist_phi, m):
    """bar-phase histogram -> beat-phase (psi = m*phi) histogram for oracle meter m."""
    T, NH = hist_phi.shape
    out = np.zeros_like(hist_phi)
    for j in range(NH):
        for r in range(m):
            out[:, (m * j + r) % NH] += hist_phi[:, j] / m
    return out


def beats_with_labels(phase, m, fps, min_dist_sec=0.10):
    """Beat times from wraps of psi=(m*phi) plus beat-in-bar label at each beat."""
    phase = np.asarray(phase, float)
    psi = (m * phase) % TWO_PI
    w = np.where(np.diff(psi) < -math.pi)[0] + 1
    out, labs, last = [], [], -1e9
    gap = min_dist_sec * fps
    for f in w:
        if f - last >= gap:
            out.append(f); last = f
            labs.append(int(np.round(phase[f] * m / TWO_PI)) % m)
    return np.asarray(out, float) / fps, np.asarray(labs, int)


# --------------------------------------------------------------------- caching
def stage_cache(split):
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
    for i, s in enumerate(ev):
        act = ae.get(s["stem"])
        if act is None:
            continue
        T = min(len(act), s["T"])
        ref = s["beats"][s["beats"] < T / FPS]
        if len(ref) < 3:
            continue
        LL = emis.padded_table(act[:T])
        out = particle_filter(LL, emis.nb, meter_prior=prior, fps=FPS,
                              K=PF_CFG["K"], alpha=PF_CFG["alpha"],
                              sigma_lt=PF_CFG["sigma_lt"], sigma_phi=PF_CFG["sigma_phi"],
                              p_switch=PF_CFG["p_switch"], noise=PF_CFG["noise"],
                              seed=PF_CFG["seed0"] + i)
        st = s["stem"]
        store[st + "|phase_path"] = out["phase_path"].astype(np.float32)
        store[st + "|meter_path"] = out["meter_path"].astype(np.int8)
        store[st + "|hist_psi"] = out["hist_psi"].astype(np.float16)
        store[st + "|hist_phi"] = out["hist_phi"].astype(np.float16)
        store[st + "|ess"] = np.float32(out["ess"])
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(ev)} ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(f"{HERE}/pf_cache_{split}.npz", **store)
    print(f"cached {split}: {sum(1 for k in store if k.endswith('|ess'))} songs "
          f"({time.time()-t0:.0f}s)", flush=True)


def load_cache(split):
    d = np.load(f"{HERE}/pf_cache_{split}.npz", allow_pickle=True)
    out = {}
    for k in d.files:
        st, key = k.rsplit("|", 1)
        out.setdefault(st, {})[key] = d[k]
    return out


# --------------------------------------------------------------- variant maker
def song_variants(s, act, pf, grids, oracle=True):
    """Return dict name -> (est_beats, est_downs) event arrays, + traj entries."""
    T = min(len(act), s["T"])
    ph = np.asarray(pf["phase_path"], float)[:T]
    mp = np.asarray(pf["meter_path"], int)[:T]
    m_pf = int(np.bincount(mp).argmax())
    m_gt = _estimate_meter(s["beats"], s["downs"])
    a0, a1 = act[:T, 0].astype(float), act[:T, 1].astype(float)
    V = {}

    # -- peak sets
    pk_head = fast_peaks(a0, FPS, thr=0.5, min_dist_sec=0.15)       # head's own beats
    pk_lo = fast_peaks(a0, FPS, thr=grids["snap_thr"], min_dist_sec=0.10)

    # -- base PF read-outs
    b_pf, lab_pf = beats_with_labels(ph, m_pf, FPS)
    d_pf = downbeats_from_barphase(ph, FPS)
    V["pf_base"] = (b_pf, d_pf)

    # -- (a) SNAP
    b_sn = snap(b_pf, pk_lo, win=grids["snap_win"])
    d_sn = snap(d_pf, pk_lo, win=grids["snap_win"])
    V["snap"] = (b_sn, d_sn)
    V["snap_headpk"] = (snap(b_pf, pk_head, win=grids["snap_win"]),
                        snap(d_pf, pk_head, win=grids["snap_win"]))

    # downbeats = the label-0 snapped beats (bar structure from PF, timing from head)
    db_lab = snap(np.asarray([t for t, l in zip(b_pf, lab_pf) if l == 0]),
                  pk_lo, win=grids["snap_win"], min_gap=0.30)
    V["snap_dblab"] = (b_sn, db_lab)

    # -- (b) MASK
    g = grids
    hp = np.asarray(pf["hist_psi"], np.float32)[:T]
    hb = np.asarray(pf["hist_phi"], np.float32)[:T]
    mk = mask_from_hist(hp, g["kappa"])
    if g["norm"]:
        mk = mk / max(mk.max(), 1e-9)
    mb = a0 * (g["eps"] + (1 - g["eps"]) * mk)
    b_mk = fast_peaks(mb, FPS, thr=g["thr"], min_dist_sec=0.15)
    mkd = mask_from_hist(hb, g["kappa_db"])
    if g["norm_db"]:
        mkd = mkd / max(mkd.max(), 1e-9)
    # downbeat mask applied to the BEAT channel (localisation) at bar positions
    mdb = a0 * (g["eps_db"] + (1 - g["eps_db"]) * mkd)
    d_mk = fast_peaks(mdb, FPS, thr=g["thr_db"], min_dist_sec=0.30)
    V["mask"] = (b_mk, d_mk)
    # downbeat mask on the downbeat CHANNEL
    mdb1 = a1 * (g["eps_db"] + (1 - g["eps_db"]) * mkd)
    V["mask_dbch"] = (b_mk, fast_peaks(mdb1, FPS, thr=g["thr_db1"], min_dist_sec=0.30))

    # -- (c) ORACLE-METER versions
    if oracle:
        b_or, lab_or = beats_with_labels(ph, m_gt, FPS)
        V["pf_base_om"] = (b_or, d_pf)
        V["snap_om"] = (snap(b_or, pk_lo, win=grids["snap_win"]), d_sn)
        hp_or = fold_hist(hb, m_gt)
        mk_or = mask_from_hist(hp_or, g["kappa"])
        if g["norm"]:
            mk_or = mk_or / max(mk_or.max(), 1e-9)
        V["mask_om"] = (fast_peaks(a0 * (g["eps"] + (1 - g["eps"]) * mk_or), FPS,
                                   thr=g["thr"], min_dist_sec=0.15), d_mk)
    return V, m_pf, m_gt


DEFAULT_GRIDS = dict(snap_win=0.07, snap_thr=0.10,
                     kappa=8.0, eps=0.10, thr=0.20, norm=1,
                     kappa_db=8.0, eps_db=0.10, thr_db=0.20, thr_db1=0.10, norm_db=1)


def run_variants(split, grids, names=None, oracle=True):
    songs = load_split(split); acts = load_act(split); pfc = load_cache(split)
    rows = {}
    for s in songs:
        act = acts.get(s["stem"]); pf = pfc.get(s["stem"])
        if act is None or pf is None:
            continue
        T = min(len(act), s["T"])
        ref = s["beats"][s["beats"] < T / FPS]
        dref = s["downs"][s["downs"] < T / FPS]
        if len(ref) < 3:
            continue
        V, m_pf, m_gt = song_variants(s, act, pf, grids, oracle=oracle)
        base = dict(stem=s["stem"], n_true=len(ref), n_true_db=len(dref),
                    ess=float(pf["ess"]), obs_contrast=float("nan"),
                    meter_ok=float(m_pf == m_gt))
        for k, (eb, ed) in V.items():
            if names and k not in names:
                continue
            rows.setdefault(k, []).append({**base, **score_events(eb, ed, ref, dref, T)})
    return rows


# -------------------------------------------------------------------- stages
def stage_select():
    """Grid search of SNAP/MASK params on the TRAIN fold only."""
    songs = load_split("train"); acts = load_act("train"); pfc = load_cache("train")
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
        data.append((s, act[:T], np.asarray(pf["phase_path"], float)[:T],
                     np.asarray(pf["meter_path"], int)[:T],
                     np.asarray(pf["hist_psi"], np.float32)[:T],
                     np.asarray(pf["hist_phi"], np.float32)[:T], ref, dref, T))
    print(f"select on {len(data)} train songs", flush=True)
    out = {}

    # ---- SNAP: peak threshold + window
    best = (-1, None)
    for thr in (0.02, 0.05, 0.10, 0.20, 0.50):
        for win in (0.05, 0.07):
            fs = []
            for (s, act, ph, mp, hp, hb, ref, dref, T) in data:
                m_pf = int(np.bincount(mp).argmax())
                b_pf, _ = beats_with_labels(ph, m_pf, FPS)
                pk = fast_peaks(act[:, 0], FPS, thr=thr, min_dist_sec=0.10)
                fs.append(f_measure(ref, snap(b_pf, pk, win=win)))
            m = float(np.mean(fs))
            print(f"  SNAP thr={thr} win={win} -> train beat_F={m:.4f}", flush=True)
            if m > best[0]:
                best = (m, dict(snap_thr=thr, snap_win=win))
    out["snap"] = dict(train_F=best[0], **best[1])

    # ---- MASK beats: kappa x eps x thr x norm
    best = (-1, None)
    for kappa in (4.0, 8.0, 16.0, 32.0):
        for norm in (1, 0):
            masks = [mask_from_hist(hp, kappa) for (_, _, _, _, hp, _, _, _, _) in data]
            if norm:
                masks = [mk / max(mk.max(), 1e-9) for mk in masks]
            for eps in (0.02, 0.10, 0.30):
                for thr in (0.05, 0.10, 0.20, 0.30, 0.50):
                    fs = []
                    for mk, (s, act, ph, mp, hp, hb, ref, dref, T) in zip(masks, data):
                        mb = act[:, 0] * (eps + (1 - eps) * mk)
                        fs.append(f_measure(ref, fast_peaks(mb, FPS, thr=thr,
                                                            min_dist_sec=0.15)))
                    m = float(np.mean(fs))
                    if m > best[0]:
                        best = (m, dict(kappa=kappa, eps=eps, thr=thr, norm=norm))
                        print(f"  MASK k={kappa} eps={eps} thr={thr} norm={norm} "
                              f"-> train beat_F={m:.4f} *", flush=True)
    out["mask"] = dict(train_F=best[0], **best[1])

    # ---- MASK downbeats: on beat channel and downbeat channel
    for ch, tag in ((0, "db_beatch"), (1, "db_dbch")):
        best = (-1, None)
        for kappa in (4.0, 8.0, 16.0, 32.0):
            for norm in (1, 0):
                masks = [mask_from_hist(hb, kappa) for (_, _, _, _, _, hb, _, _, _) in data]
                if norm:
                    masks = [mk / max(mk.max(), 1e-9) for mk in masks]
                for eps in (0.02, 0.10, 0.30):
                    for thr in (0.02, 0.05, 0.10, 0.20, 0.30, 0.50):
                        fs = []
                        for mk, (s, act, ph, mp, hp, hb, ref, dref, T) in zip(masks, data):
                            if len(dref) < 2:
                                continue
                            mb = act[:, ch] * (eps + (1 - eps) * mk)
                            fs.append(f_measure(dref, fast_peaks(mb, FPS, thr=thr,
                                                                 min_dist_sec=0.30)))
                        m = float(np.mean(fs))
                        if m > best[0]:
                            best = (m, dict(kappa=kappa, eps=eps, thr=thr, norm=norm))
                            print(f"  MASK-{tag} k={kappa} eps={eps} thr={thr} norm={norm} "
                                  f"-> train db_F={m:.4f} *", flush=True)
        out[tag] = dict(train_F=best[0], **best[1])

    json.dump(out, open(f"{HERE}/sel_f2.json", "w"), indent=1)
    print("WROTE sel_f2.json:", json.dumps(out), flush=True)


def stage_report():
    sel = json.load(open(f"{HERE}/sel_f2.json"))
    g = dict(DEFAULT_GRIDS)
    g.update(snap_thr=sel["snap"]["snap_thr"], snap_win=sel["snap"]["snap_win"])
    g.update(kappa=sel["mask"]["kappa"], eps=sel["mask"]["eps"],
             thr=sel["mask"]["thr"], norm=sel["mask"]["norm"])
    db = sel["db_beatch"]
    g.update(kappa_db=db["kappa"], eps_db=db["eps"], thr_db=db["thr"],
             norm_db=db["norm"])
    g.update(thr_db1=sel["db_dbch"]["thr"])
    # NOTE: db_dbch selected its own kappa/eps too; report both channels with their own params
    print("grids:", json.dumps(g), flush=True)

    res = {"grids": g, "sel": sel, "pf_cfg": PF_CFG}
    for split in ("eval",):
        rows = run_variants(split, g)
        # db_dbch with its own selected mask params
        g2 = dict(g); g2.update(kappa_db=sel["db_dbch"]["kappa"], eps_db=sel["db_dbch"]["eps"],
                                norm_db=sel["db_dbch"]["norm"])
        rows2 = run_variants(split, g2, names=["mask_dbch"])
        rows["mask_dbch_own"] = rows2["mask_dbch"]
        # head baseline through the SAME harness (verification)
        songs = load_split(split); acts = load_act(split)
        pk = []
        for s in songs:
            act = acts.get(s["stem"])
            if act is None:
                continue
            T = min(len(act), s["T"])
            ref = s["beats"][s["beats"] < T / FPS]
            dref = s["downs"][s["downs"] < T / FPS]
            if len(ref) < 3:
                continue
            e_b = beats_from_activation(act[:T, 0], FPS)
            e_d = beats_from_activation(act[:T, 1], FPS, min_dist_sec=0.30)
            pk.append(dict(stem=s["stem"], n_true=len(ref), n_true_db=len(dref),
                           ess=float("nan"), obs_contrast=float("nan"),
                           meter_ok=float("nan"),
                           **score_events(e_b, e_d, ref, dref, T)))
        rows["act_head"] = pk
        print(f"\n=== {split} fold ({len(pk)} songs) ===", flush=True)
        for k in rows:
            d = summarize(rows[k], f"{split[:2]} {k}")
            pr(d)
            res.setdefault(split, {})[k] = d
        res.setdefault("rows", {})[split] = {k: rows[k] for k in rows}
    json.dump(res, open(f"{HERE}/f2_report.json", "w"), indent=1, default=float)
    print("WROTE f2_report.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["cache", "select", "report"])
    ap.add_argument("--split", default="eval")
    a = ap.parse_args()
    if a.stage == "cache":
        stage_cache(a.split)
    elif a.stage == "select":
        stage_select()
    else:
        stage_report()
