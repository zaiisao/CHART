#!/usr/bin/env python
"""
alt_tempo_dynamics.py  --  IN-MODEL replacement search for VBPM TEMPO_DYNAMICS.

Faithful VBPM tempo prior (docs/ELBO_for_DBN.md 5.3):
    log phidot_t ~ N( log phidot_{t-1}, sigma^2 )       # first-order random walk (RW1)
phidot is the ONLY driver of beat timing; phase advance is deterministic given phidot.

QUESTION: does real beat timing show a smooth trend + expressive deviation that a
first-order log random walk cannot capture?  Answered by MODEL SELECTION.

Observable proxy for the latent log-tempo:  y_k = log(IBI_k) = log(t_k - t_{k-1}),
= -log(tempo)+const; its DYNAMICS are identical to log-phidot's.

Every candidate is a linear-Gaussian state-space model (LGSSM) = exactly the faithful
drop-in class:
  (1) reparameterizable: Gaussian transition x_t = F x_{t-1} + N(0,Q) -> F x_{t-1} + L eps
  (2) closed-form Gaussian-Gaussian KL (same closed form as 5.3, per component)
  (3) preserves factorization: tempo stays a continuous latent with a Markov transition;
      phase advance stays deterministic-in-mean; meter/phase/decoder untouched.
Extra latent components (slope, per-song mean, fast/slow split) are MINIMAL structural
extensions of the SAME tempo latent, still exactly Kalman-filterable, so the ELBO stays
tractable via the same forward filtering the paper already uses.

CANDIDATES (fit per dataset by MLE of the exact Kalman one-step predictive likelihood):
  M0 RW1-pure     : y_t = y_{t-1} + w                       (1 param)  <-- CURRENT faithful model
  M1 LocalLevel   : level RW + obs noise                    (2 params) (RW1 w/ expressive noise)
  M2 IRW2         : integrated RW (smooth accel, slope RW)  (2 params)
  M3 LLT          : local linear trend (level+slope noise)  (3 params)
  M4 OU           : mean-revert to per-song tempo (AR1 dev) (3 params)
  M5 TwoTimescale : slow RW mean + fast AR1 deviation       (4 params)

Per-song absolute tempo is NEVER scored: every model's level/mean state gets a diffuse
initial covariance, so the filter absorbs the per-song mean for free -> we score DYNAMICS only.
Rank = held-out one-step-ahead predictive LL (nats/beat), corroborated by AIC/BIC on train.

Batched Kalman: vectorized across songs (padded+masked), sequential only in time.
"""
import glob, numpy as np
from scipy.optimize import minimize

np.random.seed(0)
ANN = "/home/sogang/jaehoon/VBPM/dataset_store/beat_this_annotations"
DATASETS = ["ballroom", "asap", "hainsworth", "rwc", "gtzan", "smc"]
MAX_SONGS = 180
MIN_BEATS = 12
MAX_LEN = 1200
BURN = 2
DIFFUSE = 1e4
JIT = 1e-9

def load_logibi(ds):
    seqs = []
    for f in sorted(glob.glob(f"{ANN}/{ds}/annotations/beats/*.beats"))[: MAX_SONGS * 3]:
        try:
            arr = np.loadtxt(f, ndmin=2)
        except Exception:
            continue
        if arr.size == 0:
            continue
        t = np.sort(arr[:, 0].astype(float)[np.isfinite(arr[:, 0])])
        ibi = np.diff(t)
        ibi = ibi[(ibi > 1e-3) & (ibi < 4.0)]
        if len(ibi) >= MIN_BEATS:
            seqs.append(np.log(ibi)[:MAX_LEN])
        if len(seqs) >= MAX_SONGS:
            break
    return seqs

def pack(seqs):
    N = len(seqs); T = max(len(s) for s in seqs)
    Y = np.zeros((N, T)); M = np.zeros((N, T), bool); y0 = np.zeros(N)
    for i, s in enumerate(seqs):
        Y[i, :len(s)] = s; M[i, :len(s)] = True; y0[i] = s[0]
    return Y, M, y0

def batched_predll(Y, M, y0, F, Q, h, r, level_idx, P0diag):
    """One-step-ahead predictive LL, batched over songs. Returns (ll_sum, n_scored)."""
    N, T = Y.shape; d = F.shape[0]
    X = np.zeros((N, d)); X[:, level_idx] = y0
    P = np.tile(np.diag(P0diag).astype(float), (N, 1, 1))
    ll = 0.0; n = 0
    for k in range(T):
        Xp = X @ F.T                                             # (N,d)
        Pp = np.einsum('ij,njk,lk->nil', F, P, F) + Q            # (N,d,d)
        yhat = Xp @ h                                            # (N,)
        PpH = np.einsum('nij,j->ni', Pp, h)                     # (N,d)
        S = PpH @ h + r + JIT                                    # (N,)
        e = Y[:, k] - yhat
        valid = M[:, k]
        if k >= BURN:
            sc = valid
            ll += np.sum(-0.5 * (np.log(2 * np.pi * S[sc]) + e[sc] ** 2 / S[sc]))
            n += int(sc.sum())
        Kg = PpH / S[:, None]
        Xn = Xp + Kg * e[:, None]
        Pn = Pp - np.einsum('ni,nj->nij', Kg, PpH)
        v = valid[:, None]
        X = np.where(v, Xn, X)
        P = np.where(valid[:, None, None], Pn, P)
    return ll, n

# ---- models: return (F, Q, h, r, level_idx, P0diag) given params p and (scalar) placeholder
def build_M0(p):
    q = np.exp(p[0])
    return np.array([[1.0]]), np.array([[q]]), np.array([1.0]), 0.0, 0, np.array([DIFFUSE])
def build_M1(p):
    q, r = np.exp(p[0]), np.exp(p[1])
    return np.array([[1.0]]), np.array([[q]]), np.array([1.0]), r, 0, np.array([DIFFUSE])
def build_M2(p):
    qs, r = np.exp(p[0]), np.exp(p[1])
    F = np.array([[1.0, 1.0], [0.0, 1.0]]); Q = np.array([[0.0, 0.0], [0.0, qs]])
    return F, Q, np.array([1.0, 0.0]), r, 0, np.array([DIFFUSE, 1.0])
def build_M3(p):
    ql, qs, r = np.exp(p[0]), np.exp(p[1]), np.exp(p[2])
    F = np.array([[1.0, 1.0], [0.0, 1.0]]); Q = np.array([[ql, 0.0], [0.0, qs]])
    return F, Q, np.array([1.0, 0.0]), r, 0, np.array([DIFFUSE, 1.0])
def build_M4(p):
    a = np.tanh(p[0]); q, r = np.exp(p[1]), np.exp(p[2])
    F = np.array([[a, 0.0], [0.0, 1.0]]); Q = np.array([[q, 0.0], [0.0, 0.0]])
    return F, Q, np.array([1.0, 1.0]), r, 1, np.array([q / max(1 - a * a, 1e-3), DIFFUSE])
def build_M5(p):
    qg = np.exp(p[0]); a = np.tanh(p[1]); qd, r = np.exp(p[2]), np.exp(p[3])
    F = np.array([[1.0, 0.0], [0.0, a]]); Q = np.array([[qg, 0.0], [0.0, qd]])
    return F, Q, np.array([1.0, 1.0]), r, 0, np.array([DIFFUSE, qd / max(1 - a * a, 1e-3)])

MODELS = {
    "M0_RW1_pure":     (build_M0, 1, np.array([np.log(0.01)])),
    "M1_LocalLevel":   (build_M1, 2, np.array([np.log(0.01), np.log(0.005)])),
    "M2_IRW2_accel":   (build_M2, 2, np.array([np.log(0.001), np.log(0.005)])),
    "M3_LLT":          (build_M3, 3, np.array([np.log(0.005), np.log(0.001), np.log(0.005)])),
    "M4_OU":           (build_M4, 3, np.array([np.arctanh(0.3), np.log(0.01), np.log(0.005)])),
    "M5_TwoTimescale": (build_M5, 4, np.array([np.log(0.001), np.arctanh(0.3), np.log(0.01), np.log(0.005)])),
}

def total_ll(p, build, packed):
    F, Q, h, r, li, P0 = build(p)
    return batched_predll(*packed, F, Q, h, r, li, P0)

def fit_and_eval(name, build, k, p0, tr, te):
    obj = lambda p: -total_ll(p, build, tr)[0]
    best = None
    for jit in range(2):
        pp = p0 + (0.0 if jit == 0 else np.random.randn(len(p0)) * 0.4)
        try:
            res = minimize(obj, pp, method="Nelder-Mead",
                           options={"maxiter": 600, "xatol": 1e-4, "fatol": 1e-3})
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            continue
    p = best.x
    trl, trn = total_ll(p, build, tr)
    tel, ten = total_ll(p, build, te)
    return dict(name=name, k=k, params=p, tel=tel, ten=ten, teper=tel / max(ten, 1),
                aic=2 * k - 2 * trl, bic=k * np.log(max(trn, 1)) - 2 * trl)

def rp(name, p):
    s = lambda i: np.sqrt(np.exp(p[i]))
    if name == "M0_RW1_pure":
        return f"sigma_rw={s(0):.4f}"
    if name == "M1_LocalLevel":
        return f"sig_q={s(0):.4f} sig_r={s(1):.4f}"
    if name == "M2_IRW2_accel":
        return f"sig_slope={s(0):.4f} sig_r={s(1):.4f}"
    if name == "M3_LLT":
        return f"sig_lvl={s(0):.4f} sig_slope={s(1):.4f} sig_r={s(2):.4f}"
    if name == "M4_OU":
        return f"a={np.tanh(p[0]):.3f} sig_dev={s(1):.4f} sig_r={s(2):.4f}"
    if name == "M5_TwoTimescale":
        return f"sig_slow={s(0):.4f} a={np.tanh(p[1]):.3f} sig_fast={s(2):.4f} sig_r={s(3):.4f}"
    return ""

def main():
    print("=" * 104)
    print("VBPM TEMPO_DYNAMICS -- in-model replacement search (held-out one-step predictive LL)")
    print("obs = log inter-beat-interval; higher LL/beat = better; per-song mean absorbed (diffuse)")
    print("=" * 104)
    agg = {n: [0.0, 0] for n in MODELS}; wins = {n: 0 for n in MODELS}
    for ds in DATASETS:
        seqs = load_logibi(ds)
        if len(seqs) < 8:
            print(f"\n[{ds}] too few songs ({len(seqs)}), skip"); continue
        idx = np.random.RandomState(42).permutation(len(seqs))
        ntr = int(0.7 * len(seqs))
        tr = pack([seqs[i] for i in idx[:ntr]]); te = pack([seqs[i] for i in idx[ntr:]])
        nb = sum(len(s) for s in seqs)
        print(f"\n[{ds}]  songs={len(seqs)} (train {ntr}/test {len(seqs)-ntr})  beats={nb}")
        print(f"  {'model':<17}{'k':>2}  {'testLL/beat':>12}  {'testLL':>10}  {'AIC':>10}  {'BIC':>10}   params")
        rows = []
        for name, (build, k, p0) in MODELS.items():
            r = fit_and_eval(name, build, k, p0, tr, te); rows.append(r)
            agg[name][0] += r["tel"]; agg[name][1] += r["ten"]
            print(f"  {name:<17}{k:>2}  {r['teper']:>12.4f}  {r['tel']:>10.1f}  {r['aic']:>10.1f}  {r['bic']:>10.1f}   {rp(name, r['params'])}")
        best = max(rows, key=lambda x: x["teper"]); wins[best["name"]] += 1
        base = next(x for x in rows if x["name"] == "M0_RW1_pure")
        print(f"  -> winner: {best['name']}   (+{best['teper']-base['teper']:.4f} nats/beat vs RW1-pure)")
    print("\n" + "=" * 104)
    print("AGGREGATE across datasets (pooled held-out predictive LL/beat):")
    bp = agg["M0_RW1_pure"][0] / max(agg["M0_RW1_pure"][1], 1)
    ranked = sorted(MODELS, key=lambda n: -(agg[n][0] / max(agg[n][1], 1)))
    for name in ranked:
        per = agg[name][0] / max(agg[name][1], 1)
        print(f"  {name:<17}  LL/beat={per:>9.4f}   delta_vs_RW1={per-bp:+.4f}   dataset_wins={wins[name]}")
    print(f"\n  OVERALL WINNER (pooled): {ranked[0]}")
    print("=" * 104)

if __name__ == "__main__":
    main()
