"""Build the cache ONCE, then train every probe with fold-honest early stopping.

TRAIN songs (147) are split 120 inner-train / 27 inner-val (by SONG) to pick the
epoch; the EVAL fold (79 songs, fold 0) is touched only for the final score.
"""
from __future__ import annotations
import copy, json, math, sys, time
import numpy as np, torch, torch.nn as nn

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise")
from probe_gap import build, Probe, windows, NB, DEV, TWO_PI

OUT = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise"
CH = dict(x=[0, 1], b=[2], d=[3])


def gpu(D, cols):
    Xn = D["Xn"][:, cols]
    return dict(X=torch.from_numpy(np.ascontiguousarray(Xn)).to(DEV),
                starts=torch.from_numpy(D["starts"].astype(np.int64)).to(DEV),
                sid=torch.from_numpy(D["sid"].astype(np.int64)).to(DEV),
                y=torch.from_numpy(((D["ph"] / TWO_PI * NB).astype(int) % NB).astype(np.int64)).to(DEV))


def centers(D, songs=None):
    m = D["msk"].copy()
    if songs is not None:
        m &= np.isin(D["sid"], songs)
    return torch.from_numpy(np.nonzero(m)[0].astype(np.int64)).to(DEV)


def nll_of(model, g, cen, W, ret_post=False):
    model.eval(); tot = 0.0
    P = np.zeros((len(cen), NB), np.float32) if ret_post else None
    with torch.no_grad():
        for i in range(0, len(cen), 16384):
            c = cen[i:i + 16384]
            lp = torch.log_softmax(model(windows(g["X"], c, W, g["starts"], g["sid"])), -1)
            tot += float(-lp[torch.arange(len(c), device=DEV), g["y"][c]].sum())
            if ret_post:
                P[i:i + len(c)] = lp.exp().cpu().numpy()
    model.train()
    return tot / len(cen), P


def train_probe(feats, W, hid, epochs, seed, TR, EV, itr_songs, ival_songs, tag, wd=1e-4):
    torch.manual_seed(seed); np.random.seed(seed)
    cols = sum([CH[f] for f in feats], [])
    gtr, gev = gpu(TR, cols), gpu(EV, cols)
    c_itr, c_ival, c_ev = centers(TR, itr_songs), centers(TR, ival_songs), centers(EV)
    model = Probe(len(cols), W, hid).to(DEV)
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=wd)
    n = len(c_itr); bs = 4096
    best = (1e9, -1, None)
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n - bs + 1, bs):
            c = c_itr[perm[i:i + bs]]
            loss = nn.functional.cross_entropy(
                model(windows(gtr["X"], c, W, gtr["starts"], gtr["sid"])), gtr["y"][c])
            opt.zero_grad(); loss.backward(); opt.step()
        v, _ = nll_of(model, gtr, c_ival, W)
        if v < best[0]:
            best = (v, ep, copy.deepcopy(model.state_dict()))
        print(f"  [{tag}] ep{ep} inner_val_nll={v:.4f}{'  *' if best[1]==ep else ''}", flush=True)
    model.load_state_dict(best[2])
    ev_nll, P = nll_of(model, gev, c_ev, W, ret_post=True)
    print(f"  [{tag}] inner_val={best[0]:.4f}@ep{best[1]}  EVAL nll={ev_nll:.4f} nats "
          f"n_frames={len(c_ev)}", flush=True)
    np.savez_compressed(f"{OUT}/post_{tag}.npz", P=P,
                        idx=c_ev.cpu().numpy(), nll=ev_nll, inner_val=best[0], best_ep=best[1])
    return dict(tag=tag, feats="".join(feats), W=W, hid=hid, seed=seed,
                inner_val_nll=best[0], best_ep=best[1], eval_nll=ev_nll, n_eval_frames=len(c_ev))


if __name__ == "__main__":
    t0 = time.time()
    TR = build("train", ("x", "b", "d")); EV = build("eval", ("x", "b", "d"))
    mu = TR["X"][TR["msk"]].mean(0); sd = TR["X"][TR["msk"]].std(0) + 1e-6
    for D in (TR, EV):
        D["Xn"] = ((D["X"] - mu) / sd).astype(np.float32)
    print(f"built in {time.time()-t0:.0f}s  train_frames={TR['msk'].sum()} "
          f"eval_frames={EV['msk'].sum()} songs {len(TR['songs'])}/{len(EV['songs'])}", flush=True)
    # save eval meta once (for the analysis script)
    np.savez_compressed(f"{OUT}/evalmeta.npz", ph=EV["ph"], sid=EV["sid"], msk=EV["msk"],
                        starts=EV["starts"], m=np.array([s["m"] for s in EV["songs"]]),
                        ds=np.array([s["dataset"] for s in EV["songs"]]),
                        stem=np.array([s["stem"] for s in EV["songs"]]),
                        act=EV["X"][:, :2])
    rng = np.random.default_rng(0)
    ns = len(TR["songs"]); perm = rng.permutation(ns)
    ival = np.sort(perm[:27]); itr = np.sort(perm[27:])
    res = []
    grid = []
    for f in (("x",), ("x", "b"), ("x", "b", "d"), ("b",)):
        for W in (25, 50, 100):
            grid.append((f, W, 512))
    grid += [(("x",), 50, 1024), (("x",), 50, 128), (("x", "b"), 50, 1024)]
    for f, W, hid in grid:
        tag = f"{''.join(f)}_W{W}_h{hid}"
        res.append(train_probe(f, W, hid, 20, 0, TR, EV, itr, ival, tag))
        json.dump(res, open(f"{OUT}/results_grid.json", "w"), indent=1)
    for r in res:
        print(f"{r['tag']:16s} inner_val={r['inner_val_nll']:.4f} EVAL={r['eval_nll']:.4f}")
