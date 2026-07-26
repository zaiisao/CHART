"""F1 follow-up: is the meter recoverable WITHOUT the oracle?
Per-meter clamped PF + marginal-likelihood (logZ) model selection over m in {2,3,4}.
PF copied from vbpm_final/pf.py (off-limits to edit) with ONE addition: logZ accumulation.
"""
from __future__ import annotations
import json, math, sys, time
import numpy as np

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_final")
from emission import (PhaseEmission, load_act, load_split, METERS, TWO_PI, FPS,
                      _estimate_meter)
from pf import lt_band, _systematic_resample
from vbpm.evaluate import beats_from_barphase, downbeats_from_barphase, f_measure
import mir_eval


def logsumexp(x):
    m = x.max()
    return m + math.log(np.exp(x - m).sum())


def pf_logz(LL, nb, m_fix, K=600, alpha=0.25, sigma_lt=0.05, sigma_phi=0.03,
            fps=50.0, seed=0):
    """Single-meter bootstrap PF, returns MAP-ancestral phase path + log marginal lik."""
    rng = np.random.default_rng(seed)
    T = LL.shape[0]
    nbm = nb[m_fix]
    lo, hi = lt_band(m_fix, fps)
    phi = rng.random(K) * TWO_PI
    lt = lo + (hi - lo) * rng.random(K)

    b = (phi / TWO_PI * nbm).astype(np.int64)
    inc = alpha * LL[0, m_fix - 1, b]
    logZ = logsumexp(inc) - math.log(K)          # w_prev uniform
    logw = inc - inc.max()
    w = np.exp(logw); w /= w.sum()

    phi_hist = np.empty((T, K), np.float32); anc = np.empty((T, K), np.int32)
    map_idx = np.empty(T, np.int64); ess_h = np.empty(T)
    idx = np.arange(K)
    phi_hist[0] = phi; anc[0] = idx
    map_idx[0] = int(w.argmax()); ess_h[0] = 1.0 / float((w**2).sum())

    for t in range(1, T):
        adv = phi + np.exp(lt) + sigma_phi * rng.standard_normal(K)
        phi = adv % TWO_PI
        lt = np.clip(lt + sigma_lt * rng.standard_normal(K), lo, hi)
        b = (phi / TWO_PI * nbm).astype(np.int64)
        inc = alpha * LL[t, m_fix - 1, b]
        logZ += math.log(max(float((w * np.exp(inc - inc.max())).sum()), 1e-300)) + float(inc.max())
        logw = logw + inc
        logw -= logw.max()
        w = np.exp(logw); w /= w.sum()
        phi_hist[t] = phi; anc[t] = idx
        ess_h[t] = 1.0 / float((w**2).sum())
        map_idx[t] = int(w.argmax())
        if ess_h[t] < 0.5 * K:
            a = _systematic_resample(w, rng)
            phi, lt = phi[a], lt[a]
            anc[t] = a
            logw = np.zeros(K); w = np.full(K, 1.0 / K)

    j = int(map_idx[T - 1])
    pp = np.empty(T)
    for t in range(T - 1, -1, -1):
        pp[t] = phi_hist[t, j]
        if t > 0:
            j = int(anc[t - 1][j])
    return pp, float(logZ), float(ess_h.mean())


def blind_grid_controls(ref, T, n_est, n_off=12):
    dur = T / FPS
    if n_est < 2 or len(ref) < 2:
        return float("nan"), float("nan")
    per = dur / n_est
    base = np.arange(n_est) * per
    f0 = f_measure(ref, base)
    best = max(f_measure(ref, base + k * per / n_off) for k in range(n_off))
    return float(f0), float(max(best, f0))


def cont(ref, est):
    if len(ref) < 2 or len(est) < 2:
        return (0.0,) * 4
    r = mir_eval.beat.trim_beats(np.asarray(ref, float))
    e = mir_eval.beat.trim_beats(np.asarray(est, float))
    if len(r) < 2 or len(e) < 2:
        return (0.0,) * 4
    return mir_eval.beat.continuity(r, e)


def main():
    tr = load_split("train"); at = load_act("train")
    ev, ae = load_split("eval"), load_act("eval")
    emis = PhaseEmission(bins_per_beat=24, likelihood="gauss", smooth=0.0).fit(
        tr, at, phase_mode="downbeat")
    # log meter prior from train counts
    cnt = np.zeros(5)
    for s in tr:
        m = _estimate_meter(s["beats"], s["downs"])
        if m in METERS:
            cnt[m] += 1
    logprior = {m: math.log(cnt[m] / cnt.sum()) for m in METERS}
    print("train meter counts:", {m: int(cnt[m]) for m in METERS}, flush=True)

    rows = []
    t0 = time.time()
    for i, s in enumerate(ev):
        act = ae.get(s["stem"])
        if act is None: continue
        T = min(len(act), s["T"])
        ref = s["beats"][s["beats"] < T / FPS]
        dref = s["downs"][s["downs"] < T / FPS]
        if len(ref) < 3: continue
        m_gt = _estimate_meter(s["beats"], s["downs"])
        LL = emis.padded_table(act[:T])
        cand = {}
        for m in METERS:
            pp, lz, ess = pf_logz(LL, emis.nb, m, seed=1234 + i)
            cand[m] = dict(pp=pp, logZ=lz, logZp=lz + logprior[m], ess=ess)
        m_ml  = max(METERS, key=lambda m: cand[m]["logZ"])
        m_map = max(METERS, key=lambda m: cand[m]["logZp"])
        r = dict(stem=s["stem"], dataset=s["dataset"], T=T, n_true=len(ref),
                 n_true_db=len(dref), m_gt=m_gt, m_ml=m_ml, m_map=m_map,
                 logZ={str(m): cand[m]["logZ"] for m in METERS})
        for name, m_sel in (("ml", m_ml), ("map", m_map), ("oracle", m_gt)):
            pp = cand[m_sel]["pp"]
            est = beats_from_barphase(pp, m_sel, FPS)
            dest = downbeats_from_barphase(pp, FPS)
            b0, bb = blind_grid_controls(ref, T, len(est))
            c = cont(ref, est)
            r[name] = dict(F=f_measure(ref, est), n_est=len(est), blind0=b0,
                           blind_best=bb, CMLc=float(c[0]), CMLt=float(c[1]),
                           AMLc=float(c[2]), AMLt=float(c[3]),
                           db_F=f_measure(dref, dest) if len(dref) >= 2 else float("nan"),
                           n_est_db=len(dest))
        rows.append(r)
        if len(rows) % 20 == 0:
            print(f"  {len(rows)} songs ({time.time()-t0:.0f}s)", flush=True)

    json.dump(rows, open("f1_meterselect_rows.json", "w"), default=float)
    n = len(rows)
    for name in ("ml", "map", "oracle"):
        F = np.mean([r[name]["F"] for r in rows])
        dbF = np.nanmean([r[name]["db_F"] for r in rows])
        ne = sum(r[name]["n_est"] for r in rows); nt = sum(r["n_true"] for r in rows)
        bb = np.nanmean([r[name]["blind_best"] for r in rows])
        cml = np.mean([r[name]["CMLt"] for r in rows])
        aml = np.mean([r[name]["AMLt"] for r in rows])
        key = {"ml": "m_ml", "map": "m_map", "oracle": "m_gt"}[name]
        acc = np.mean([r[key] == r["m_gt"] for r in rows])
        print(f"[select={name:6s}] meter_acc={acc:.3f} F={F:.4f} db_F={dbF:.4f} "
              f"n_est/n_true={ne}/{nt} (ratio {ne/nt:.3f}) blind_best={bb:.4f} "
              f"MARGIN={F-bb:+.4f} CMLt={cml:.3f} AMLt={aml:.3f}", flush=True)

if __name__ == "__main__":
    main()
