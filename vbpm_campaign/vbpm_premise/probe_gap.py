"""PREMISE 3: size of the STRUCTURAL gap between x-only inference q(z|x) and the
per-instance posterior p(z|x,b).

Probes: MLP over a window of the frozen activation (+ optionally the beat / downbeat
target impulses), predicting a CATEGORICAL distribution over B bar-phase bins.
Categorical (not von Mises) so the posterior CAN be multimodal -- that is the object
the tutorial's "definitionally broader" claim is about.

Fit on TRAIN songs (147), scored on EVAL songs (79, fold 0). Nothing outside
vbpm_premise/ is written.
"""
from __future__ import annotations
import argparse, json, math, sys, time
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_final")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
from emission import load_act, load_split, song_phase, inside_mask, TWO_PI, FPS  # noqa
from vbpm.evaluate import _estimate_meter  # noqa

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NB = 48   # phase bins over the BAR (divisible by 2,3,4 -> integer bins per beat)


def impulses(times, T):
    v = np.zeros(T, np.float32)
    idx = np.round(np.asarray(times, float) * FPS - 0.5).astype(int)
    idx = idx[(idx >= 0) & (idx < T)]
    v[idx] = 1.0
    return v


def build(split, feats=("x",)):
    """Returns dict with concatenated per-frame channel matrix, phase, meta."""
    S = load_split(split); A = load_act(split)
    chans, phases, songid, meters, ds, keep = [], [], [], [], [], []
    kept = []
    for si, s in enumerate(S):
        a = A[s["stem"]]
        T = min(len(a), s["T"])
        m = _estimate_meter(s["beats"], s["downs"])
        ph = song_phase(s)
        if ph is None or m not in (2, 3, 4):
            continue
        msk = inside_mask(s, T)
        if msk.sum() < 100:
            continue
        a = a[:T]
        c = []
        if "x" in feats:
            c.append(np.log(a / (1 - a)).astype(np.float32))   # logit activation [T,2]
        if "b" in feats:
            c.append(impulses(s["beats"], T)[:, None])
        if "d" in feats:
            c.append(impulses(s["downs"], T)[:, None])
        chans.append(np.concatenate(c, 1))
        phases.append(ph[:T].astype(np.float32))
        keep.append(msk)
        songid.append(np.full(T, len(kept), np.int32))
        kept.append(dict(stem=s["stem"], dataset=s["dataset"], m=m, T=T,
                         beats=s["beats"], downs=s["downs"]))
    X = np.concatenate(chans, 0)
    ph = np.concatenate(phases, 0)
    sid = np.concatenate(songid, 0)
    msk = np.concatenate(keep, 0)
    starts = np.cumsum([0] + [k["T"] for k in kept])
    return dict(X=X, ph=ph, sid=sid, msk=msk, songs=kept, starts=starts)


class Probe(nn.Module):
    def __init__(self, cin, W, hid, nb=NB):
        super().__init__()
        d = cin * (2 * W + 1)
        self.net = nn.Sequential(nn.Linear(d, hid), nn.ReLU(), nn.Linear(hid, hid),
                                 nn.ReLU(), nn.Linear(hid, nb))

    def forward(self, x):
        return self.net(x.flatten(1))


def windows(Xg, centers, W, starts_g, sid_g):
    """Gather [n,2W+1,C] windows, zero-padded at song boundaries. All on GPU."""
    off = torch.arange(-W, W + 1, device=Xg.device)
    idx = centers[:, None] + off[None, :]
    lo = starts_g[sid_g[centers]][:, None]
    hi = starts_g[sid_g[centers] + 1][:, None]
    ok = (idx >= lo) & (idx < hi)
    out = Xg[idx.clamp(0, Xg.shape[0] - 1)]
    return out * ok[..., None]


def run(feats, W, hid, epochs, seed, tag, out):
    torch.manual_seed(seed); np.random.seed(seed)
    tr = build("train", feats); ev = build("eval", feats)
    mu = tr["X"][tr["msk"]].mean(0); sd = tr["X"][tr["msk"]].std(0) + 1e-6
    for D in (tr, ev):
        D["Xn"] = ((D["X"] - mu) / sd).astype(np.float32)
    C = tr["X"].shape[1]
    G = {}
    for name, D in (("tr", tr), ("ev", ev)):
        G[name] = dict(X=torch.from_numpy(D["Xn"]).to(DEV),
                       starts=torch.from_numpy(D["starts"].astype(np.int64)).to(DEV),
                       sid=torch.from_numpy(D["sid"].astype(np.int64)).to(DEV),
                       cen=torch.from_numpy(np.nonzero(D["msk"])[0].astype(np.int64)).to(DEV),
                       y=torch.from_numpy(((D["ph"] / TWO_PI * NB).astype(int) % NB).astype(np.int64)).to(DEV))
    model = Probe(C, W, hid).to(DEV)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    n = len(G["tr"]["cen"]); bs = 4096
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * max(n // bs, 1))
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV)
        tot = 0.0; cnt = 0
        for i in range(0, n - bs + 1, bs):
            c = G["tr"]["cen"][perm[i:i + bs]]
            xb = windows(G["tr"]["X"], c, W, G["tr"]["starts"], G["tr"]["sid"])
            loss = nn.functional.cross_entropy(model(xb), G["tr"]["y"][c])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss) * len(c); cnt += len(c)
        print(f"  [{tag}] ep{ep} train_nll={tot/cnt:.4f} ({time.time()-t0:.0f}s)", flush=True)
    # ---- eval: full posterior per frame
    model.eval()
    P = np.zeros((len(G["ev"]["cen"]), NB), np.float32)
    nll = 0.0
    with torch.no_grad():
        for i in range(0, len(G["ev"]["cen"]), 8192):
            c = G["ev"]["cen"][i:i + 8192]
            lg = model(windows(G["ev"]["X"], c, W, G["ev"]["starts"], G["ev"]["sid"]))
            lp = torch.log_softmax(lg, -1)
            nll += float(-lp[torch.arange(len(c), device=DEV), G["ev"]["y"][c]].sum())
            P[i:i + len(c)] = lp.exp().cpu().numpy()
    nll /= len(P)
    np.savez_compressed(out, P=P, y=G["ev"]["y"][G["ev"]["cen"]].cpu().numpy(),
                        cen=G["ev"]["cen"].cpu().numpy(),
                        sid=ev["sid"][np.nonzero(ev["msk"])[0]],
                        ph=ev["ph"][np.nonzero(ev["msk"])[0]],
                        m=np.array([s["m"] for s in ev["songs"]]),
                        ds=np.array([s["dataset"] for s in ev["songs"]]),
                        stem=np.array([s["stem"] for s in ev["songs"]]),
                        starts=ev["starts"], nll=nll)
    print(f"  [{tag}] EVAL nll/frame={nll:.4f} nats  (n_frames={len(P)}) -> {out}", flush=True)
    return nll


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="x")      # x | xb | xbd | b
    ap.add_argument("--W", type=int, default=50)
    ap.add_argument("--hid", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    feats = tuple(a.feats)
    tag = a.tag or f"{a.feats}_W{a.W}_h{a.hid}_s{a.seed}"
    out = f"/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise/post_{tag}.npz"
    nll = run(feats, a.W, a.hid, a.epochs, a.seed, tag, out)
    json.dump(dict(tag=tag, feats=a.feats, W=a.W, hid=a.hid, epochs=a.epochs,
                   seed=a.seed, eval_nll=nll),
              open(f"/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise/res_{tag}.json", "w"), indent=1)
