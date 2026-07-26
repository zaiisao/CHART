"""PROBE 5b -- mechanism: WHAT is the log-tempo dimension carrying, and what does the PF
weighting actually track?  Adversarial follow-ups to mm_agentP5.py."""
from __future__ import annotations
import argparse, json, math, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_wtf")
from mm_agentP5_lib import (load_some, obs_cache, load_arm, train_trace, pf_trace,   # noqa
                            qs, TWO_PI, FPS, ARMS)

DEV = "cuda:0"


def targets(beats, downs, start, n):
    b = np.zeros(n, np.float32); db = np.zeros(n, np.float32)
    for t in beats:
        i = int(round(t * FPS)) - start
        if 0 <= i < n: b[i] = 1.0
    for t in downs:
        i = int(round(t * FPS)) - start
        if 0 <= i < n: db[i] = 1.0
    return b, db


def auc(score, label):
    score = np.asarray(score, float).ravel(); label = np.asarray(label).ravel() > 0.5
    if label.sum() == 0 or (~label).sum() == 0:
        return float("nan")
    r = np.argsort(np.argsort(score)) + 1.0
    n1 = label.sum(); n0 = (~label).sum()
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="i_bern")
    ap.add_argument("--n_train", type=int, default=12)
    ap.add_argument("--n_eval", type=int, default=4)
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--pf_frames", type=int, default=1500)
    a = ap.parse_args()
    arm2 = a.tag.startswith("ii")
    out = {"tag": a.tag}
    torch.manual_seed(0); rng = np.random.default_rng(0)
    model, lw, cfg, ck = load_arm(a.tag, DEV)
    train = load_some("train", a.n_train); ev = load_some("eval", a.n_eval)
    otr = obs_cache(train, f"{ARMS}/act_train.npz"); oev = obs_cache(ev, f"{ARMS}/act_eval.npz")
    print(f"=== PROBE 5b  {a.tag} ===", flush=True)

    # ------------------------------------------------------- 1. is log_tempo a beat CODE?
    LT, B, D, ACT, PHI = [], [], [], [], []
    for it in range(a.batches):
        fe, bb, dd, oo = [], [], [], []
        while len(fe) < a.bs:
            s = train[rng.integers(len(train))]
            T = s["feats"].shape[1]
            if T <= a.frames: continue
            st = int(rng.integers(0, T - a.frames))
            fe.append(s["feats"][:, st:st + a.frames, :])
            b_, d_ = targets(s["beats"], s["downs"], st, a.frames)
            bb.append(b_); dd.append(d_); oo.append(otr[s["stem"]][st:st + a.frames])
        f = torch.from_numpy(np.asarray(np.stack(fe), np.float32)).to(DEV)
        b = torch.from_numpy(np.stack(bb)).to(DEV); d = torch.from_numpy(np.stack(dd)).to(DEV)
        o = torch.from_numpy(np.stack(oo)).to(DEV)
        h = o if arm2 else torch.einsum("l,bltf->btf", lw, f)
        tr = train_trace(model, h, b, temperature=0.3)
        LT.append(tr["lt"].cpu().numpy()); PHI.append(tr["phi"].cpu().numpy())
        B.append(b.cpu().numpy()); D.append(d.cpu().numpy()); ACT.append(o.cpu().numpy())
    LT = np.concatenate(LT); B = np.concatenate(B); D = np.concatenate(D)
    ACT = np.concatenate(ACT); PHI = np.concatenate(PHI)

    print("\n--- 1. does the posterior log-tempo CODE the beat? (train regime) ---")
    print(f"  AUC(log_tempo -> beat frame)      = {auc(LT, B):.4f}")
    print(f"  AUC(log_tempo -> downbeat frame)  = {auc(LT, D):.4f}")
    print(f"  AUC(cos(phi)  -> beat frame)      = {auc(np.cos(PHI), B):.4f}")
    print(f"  AUC(phi       -> beat frame)      = {auc(PHI, B):.4f}")
    print(f"  corr(log_tempo, frozen act[beat]) = {np.corrcoef(LT.ravel(), ACT[...,0].ravel())[0,1]:+.4f}")
    print("  " + qs(LT[B > 0.5], "log_tempo @BEAT frames    "))
    print("  " + qs(LT[B < 0.5], "log_tempo @NON-beat frames"))
    print(f"  lag structure: AUC(lt(t) -> beat(t+k)):  " +
          " ".join(f"k={k}:{auc(LT[:, max(0,-k):LT.shape[1]-max(0,k)], B[:, max(0,k):B.shape[1]-max(0,-k)]):.3f}"
                   for k in (-3, -2, -1, 0, 1, 2, 3)))
    out["auc_lt_beat"] = auc(LT, B); out["auc_phi_beat"] = auc(np.cos(PHI), B)
    out["lt_beat_mean"] = float(LT[B > 0.5].mean()); out["lt_nonbeat_mean"] = float(LT[B < 0.5].mean())

    # ------------------------------------------------------- 2. tanh saturation mechanism
    print("\n--- 2. why is phase inert? first-layer tanh saturation under real z ---")
    zt = torch.from_numpy(np.stack([np.cos(PHI), np.sin(PHI), LT], -1)).float().to(DEV)
    mm = torch.zeros(*zt.shape[:2], 4, device=DEV); mm[..., 2] = 1.0
    z = torch.cat([zt, mm], -1).reshape(-1, 7)
    for name, net in (("obs emission", model.h_dec), ("beat decoder", model.decoder)):
        W = net[0].weight; bias = net[0].bias
        pre = (z @ W.T + bias)
        contrib_ph = (z[:, :2] @ W[:, :2].T)
        contrib_lt = (z[:, 2:3] @ W[:, 2:3].T)
        contrib_m = (z[:, 3:] @ W[:, 3:].T)
        print(f"  {name}: |pre-activation| mean={pre.abs().mean():.3f} "
              f"frac|pre|>2={float((pre.abs()>2).float().mean()):.4f} "
              f"frac|pre|>4={float((pre.abs()>4).float().mean()):.4f}")
        print(f"      sd of the PHASE contribution ={contrib_ph.std():.4f} | "
              f"LOG-TEMPO contribution ={contrib_lt.std():.4f} | METER ={contrib_m.std():.4f}")
        out.setdefault("saturation", {})[name] = dict(
            pre_abs=float(pre.abs().mean()), frac_gt2=float((pre.abs() > 2).float().mean()),
            sd_phase=float(contrib_ph.std()), sd_lt=float(contrib_lt.std()),
            sd_meter=float(contrib_m.std()))

    # ------------------------------------------------------- 3. what does the emission read?
    print("\n--- 3. the emission as a function of log-tempo: is it an ACTIVATION detector? ---")
    s = ev[0]; T = min(s["T"], 2000)
    o = torch.from_numpy(oev[s["stem"]][:T]).to(DEV)
    lt_grid = torch.linspace(-8.0, 2.0, 64, device=DEV)
    mm = torch.zeros(T, 4, device=DEV); mm[:, 2] = 1.0
    Z = torch.stack([model.z_features(mm, torch.zeros(T, device=DEV), v.expand(T)) for v in lt_grid])
    with torch.no_grad():
        lp = model.obs_logp(Z.reshape(-1, 7), o.repeat(64, 1)).reshape(64, T)
    best = lt_grid[lp.argmax(0)].cpu().numpy()
    actb = o[:, 0].cpu().numpy()
    hi = actb > 0.5
    print(f"  argmax-lt of p(o_t|lt): mean@act>0.5 = {best[hi].mean():+.3f} (n={hi.sum()}), "
          f"mean@act<=0.5 = {best[~hi].mean():+.3f}")
    print(f"  corr(argmax-lt, activation) = {np.corrcoef(best, actb)[0,1]:+.4f}   "
          f"AUC(argmax-lt -> act>0.5) = {auc(best, hi):.4f}")
    sl = float((lp[-1] - lp[0]).std())
    print(f"  sd over frames of [logp(lt=+2)-logp(lt=-8)] = {sl:.4f} nats "
          f"(0 => emission cannot discriminate frames at all)")
    print(f"  corr(logp(lt=+2)-logp(lt=-8), activation) = "
          f"{np.corrcoef((lp[-1]-lp[0]).cpu().numpy(), actb)[0,1]:+.4f}")
    out["emission_lt_detector"] = dict(best_hi=float(best[hi].mean()), best_lo=float(best[~hi].mean()),
                                       auc=auc(best, hi), corr=float(np.corrcoef(best, actb)[0, 1]))

    # ------------------------------------------------------- 4. what does the PF track?
    print("\n--- 4. PF deploy: does the particle cloud's log-tempo track the activation? ---")
    rows = []
    for s in ev:
        T = min(s["T"], a.pf_frames)
        ob = torch.from_numpy(oev[s["stem"]][:T]).unsqueeze(0).to(DEV)
        if arm2:
            h = ob
        else:
            f = torch.from_numpy(np.asarray(s["feats"][:, :T, :], np.float32)).to(DEV)
            h = torch.einsum("l,ltf->tf", lw, f).unsqueeze(0)
        r = pf_trace(model, h, ob, K=300, alpha=1.0, seed=1234, record_every=1)
        w = r["w"]; lt = r["lt"]; ph = r["phi"]
        wlt = (w * lt).sum(1) / w.sum(1)
        act = oev[s["stem"]][:T, 0][:len(wlt)]
        bt = np.zeros(T);
        for t in s["beats"]:
            i = int(round(t * FPS))
            if 0 <= i < T: bt[i] = 1
        c = np.corrcoef(wlt, act)[0, 1]
        # circular concentration of the weighted particle cloud (is the PF localising phase?)
        Rp = np.hypot((w * np.cos(ph)).sum(1) / w.sum(1), (w * np.sin(ph)).sum(1) / w.sum(1)).mean()
        rows.append(dict(stem=s["stem"], corr_wlt_act=float(c), auc_wlt_beat=auc(wlt, bt[:len(wlt)]),
                         cloud_R=float(Rp), ess=float((1.0 / (w ** 2).sum(1)).mean())))
        print(f"  {s['stem'][:44]:44s} corr(w-mean lt, act)={c:+.4f} "
              f"AUC(w-mean lt->beat)={rows[-1]['auc_wlt_beat']:.4f} "
              f"cloud phase R={Rp:.4f} ESS={rows[-1]['ess']:.0f}/300")
    out["pf_track"] = rows
    print(f"  MEAN corr={np.mean([r['corr_wlt_act'] for r in rows]):+.4f} "
          f"AUC={np.mean([r['auc_wlt_beat'] for r in rows]):.4f} "
          f"cloud R={np.mean([r['cloud_R'] for r in rows]):.4f}")

    json.dump(out, open(f"/home/sogang/jaehoon/VBPM_reintegration/vbpm_wtf/mm_agentP5b_{a.tag}.json", "w"),
              indent=1, default=float)
    print("WROTE mm_agentP5b_%s.json" % a.tag)


if __name__ == "__main__":
    main()
