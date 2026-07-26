"""PROBE 5c -- (1) KL budget: what did the model BUY with each latent?
(2) is phase dead because the ENCODER never encodes it, or because the wrapped-Cauchy
    concentration rho is so low that the SAMPLE destroys an encoded mean?"""
from __future__ import annotations
import argparse, json, math, sys
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_wtf")
from mm_agentP5_lib import (load_some, obs_cache, load_arm, train_trace, qs, TWO_PI,   # noqa
                            FPS, ARMS)
import variant_b as VB                                                              # noqa

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
    r = np.argsort(np.argsort(score)) + 1.0
    n1 = label.sum(); n0 = (~label).sum()
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="i_bern")
ap.add_argument("--batches", type=int, default=4); ap.add_argument("--bs", type=int, default=16)
a = ap.parse_args()
arm2 = a.tag.startswith("ii")
torch.manual_seed(0); rng = np.random.default_rng(0)
model, lw, cfg, ck = load_arm(a.tag, DEV)
train = load_some("train", 12); otr = obs_cache(train, f"{ARMS}/act_train.npz")
print(f"=== PROBE 5c {a.tag} ===")

infos, QMU, PHI, LT, B, RHO = [], [], [], [], [], []
rows = []
for it in range(a.batches):
    fe, bb, dd, oo = [], [], [], []
    while len(fe) < a.bs:
        s = train[rng.integers(len(train))]
        T = s["feats"].shape[1]
        if T <= 256: continue
        st = int(rng.integers(0, T - 256))
        fe.append(s["feats"][:, st:st + 256, :])
        b_, d_ = targets(s["beats"], s["downs"], st, 256)
        bb.append(b_); dd.append(d_); oo.append(otr[s["stem"]][st:st + 256])
    f = torch.from_numpy(np.asarray(np.stack(fe), np.float32)).to(DEV)
    b = torch.from_numpy(np.stack(bb)).to(DEV); d = torch.from_numpy(np.stack(dd)).to(DEV)
    o = torch.from_numpy(np.stack(oo)).to(DEV)
    h = o if arm2 else torch.einsum("l,bltf->btf", lw, f)
    with torch.no_grad():
        _, info = VB.elbo_b(model, h, b, d, o, temperature=0.3, beta=1.0)
    infos.append(info)
    tr = train_trace(model, h, b, temperature=0.3)
    QMU.append(tr["q_mu"].cpu().numpy()); PHI.append(tr["phi"].cpu().numpy())
    LT.append(tr["lt"].cpu().numpy()); B.append(b.cpu().numpy())
    RHO.append(tr["q_rho"].cpu().numpy())
    # rec_beat if the phase SAMPLE is replaced by the posterior MODE (noiseless phase channel)
    z = tr["z"].clone()
    zm = z.clone(); zm[..., 0] = torch.cos(tr["q_mu"]); zm[..., 1] = torch.sin(tr["q_mu"])
    with torch.no_grad():
        for nm, zz in (("sample", z), ("q_mu(mode)", zm)):
            lg = model.decoder(zz)
            rows.append((nm,
                         float(F.binary_cross_entropy_with_logits(lg[..., 0], b, reduction="none").sum(1).mean()),
                         float(-model.obs_logp(zz.reshape(-1, 7), o.reshape(-1, 2)).reshape(zz.shape[0], -1).sum(1).mean())))

print("\n--- KL budget per 256-frame crop (elbo_b, beta=1, temp=0.3) ---")
for k in ("loss", "recon_beat", "recon_db", "recon_obs", "kl", "kl_phase", "kl_level",
          "kl_dev", "kl_meter", "n_cross", "tempo_dof"):
    print(f"  {k:12s} = {np.mean([i[k] for i in infos]):9.3f}")

QMU = np.concatenate(QMU); PHI = np.concatenate(PHI); LT = np.concatenate(LT)
B = np.concatenate(B); RHO = np.concatenate(RHO)
print("\n--- is the phase MEAN informative (encoder tries) or not (encoder gave up)? ---")
print(f"  AUC(cos q_mu   -> beat) = {auc(np.cos(QMU), B):.4f}")
print(f"  AUC(sin q_mu   -> beat) = {auc(np.sin(QMU), B):.4f}")
print(f"  AUC(cos sample -> beat) = {auc(np.cos(PHI), B):.4f}")
print(f"  AUC(log_tempo  -> beat) = {auc(LT, B):.4f}")
print(f"  circular R of q_mu = {np.hypot(np.cos(QMU).mean(), np.sin(QMU).mean()):.4f}; "
      f"mean q_rho = {RHO.mean():.4f} -> wrapped-Cauchy gamma = {-math.log(RHO.mean()):.2f} rad "
      f"(gamma>1 rad == near-uniform on the circle)")
print("\n--- rec_beat / rec_obs with the NOISELESS phase channel (sample -> posterior mode) ---")
for nm in ("sample", "q_mu(mode)"):
    v = [r for r in rows if r[0] == nm]
    print(f"  {nm:12s} rec_beat={np.mean([x[1] for x in v]):8.3f}  rec_obs={np.mean([x[2] for x in v]):8.3f}")

# ---- decisive: phase channel WITHOUT the tempo side-channel -------------------
print("\n--- rec_beat with the tempo channel DESTROYED (lt := crop mean) ---")
import itertools
res = {}
rng2 = np.random.default_rng(7)
for it in range(a.batches):
    fe, bb, dd, oo = [], [], [], []
    while len(fe) < a.bs:
        s = train[rng2.integers(len(train))]
        T = s["feats"].shape[1]
        if T <= 256: continue
        st = int(rng2.integers(0, T - 256))
        fe.append(s["feats"][:, st:st + 256, :])
        b_, d_ = targets(s["beats"], s["downs"], st, 256)
        bb.append(b_); dd.append(d_); oo.append(otr[s["stem"]][st:st + 256])
    f = torch.from_numpy(np.asarray(np.stack(fe), np.float32)).to(DEV)
    b = torch.from_numpy(np.stack(bb)).to(DEV); d = torch.from_numpy(np.stack(dd)).to(DEV)
    o = torch.from_numpy(np.stack(oo)).to(DEV)
    h = o if arm2 else torch.einsum("l,bltf->btf", lw, f)
    tr = train_trace(model, h, b, temperature=0.3)
    z = tr["z"]
    with torch.no_grad():
        for nm in ("sample+lt", "qmu+lt", "sample+ltCONST", "qmu+ltCONST", "phRAND+ltCONST"):
            zz = z.clone()
            if nm.startswith("qmu"):
                zz[..., 0] = torch.cos(tr["q_mu"]); zz[..., 1] = torch.sin(tr["q_mu"])
            if nm.startswith("phRAND"):
                ph = torch.rand_like(tr["q_mu"]) * TWO_PI
                zz[..., 0] = torch.cos(ph); zz[..., 1] = torch.sin(ph)
            if "CONST" in nm:
                zz[..., 2] = z[..., 2].mean()
            lg = model.decoder(zz)
            v = float(F.binary_cross_entropy_with_logits(lg[..., 0], b, reduction="none").sum(1).mean())
            res.setdefault(nm, []).append(v)
pb = 0.0195
print(f"  base-rate BCE (p={pb}) = {-(256*(pb*math.log(pb)+(1-pb)*math.log(1-pb))):.2f}")
for k, v in res.items():
    print(f"  {k:18s} rec_beat = {np.mean(v):8.3f}")
