"""QUICK WIN: restore prior-head leverage by training on SHORT sub-crops.

Bin sweep proved the objective ranks the true tempo bin top-3 in 98.3% of songs at T=150
(vs 61.7% at T=1400) with 9x the NLL span. The 1400-frame training crops divide the
initial-prior term by T, so the head had ~0.3% of the objective to learn from.
This trains the SAME model on random 150-frame sub-crops of the same cached crops.
Usage: train_shortcrop.py <tag> [--input feats|featsmert] [--sub 150] [--steps N]
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np, torch

M = Path("/home/sogang/jaehoon/VBPM/rungs_lab/mert"); sys.path.insert(0, str(M)); sys.path.insert(0, str(M.parent))
from mert_r4_model import R4Conditioned

ap = argparse.ArgumentParser()
ap.add_argument("tag"); ap.add_argument("--input", default="feats", choices=("feats", "featsmert"))
ap.add_argument("--sub", type=int, default=150); ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--batch", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda:1"); ap.add_argument("--select", type=int, default=25)
a = ap.parse_args()
DEV = a.device; FPS = 44100 / 1024
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)

c = torch.load("/disk4/jaehoon/VBPM_cache/mert/cache_mert_fold0_c1400.pt", weights_only=False)
mean, std = c["feat_mean"].to(DEV), c["feat_std"].to(DEV)
mm, ms = c["mert_mean"].to(DEV), c["mert_std"].to(DEV)
MERT_DIM = int(mm.shape[0]); IN = {"feats": 256, "featsmert": 256 + MERT_DIM}[a.input]

def ti_of(f, mt):
    fz = (f - mean) / std
    return fz if a.input == "feats" else torch.cat([fz, (mt - mm) / ms], 1)

crops = [(torch.from_numpy(x["acts"]).to(DEV),
          torch.from_numpy(x["feats"].astype(np.float32)).to(DEV),
          torch.from_numpy(x["mert"].astype(np.float32)).to(DEV)) for x in c["crops"]]
model = R4Conditioned(fps=FPS, input_mode=a.input, device=DEV, input_dim=IN)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

def sel_nll():
    model.eval(); tot = []
    with torch.no_grad():
        for e in c["val_entries"][:24]:
            s = e["stem"]; acts = c["val_acts"][s]; f = c["val_feats"][s].astype(np.float32); mt = c["val_mert"][s].astype(np.float32)
            L = acts.shape[0]
            if L > 1400:
                st = (L - 1400) // 2; acts, f, mt = acts[st:st+1400], f[st:st+1400], mt[st:st+1400]
            A = torch.from_numpy(acts).to(DEV)
            tot.append(-float(model.marginal_ll(A, ti_of(torch.from_numpy(f).to(DEV), torch.from_numpy(mt).to(DEV)))) / A.shape[0])
    model.train(); return float(np.mean(tot))

best = (1e9, -1)
t0 = time.time()
for step in range(a.steps):
    idx = rng.choice(len(crops), a.batch, replace=False)
    opt.zero_grad()
    for i in idx:
        acts, f, mt = crops[i]
        T = acts.shape[0]
        s0 = int(rng.integers(0, max(1, T - a.sub)))
        A = acts[s0:s0+a.sub]
        loss = (-model.marginal_ll(A, ti_of(f[s0:s0+a.sub], mt[s0:s0+a.sub])) / A.shape[0]) / a.batch
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if a.select and step % a.select == 0:
        v = sel_nll()
        if v < best[0]:
            best = (v, step)
            torch.save({"model": model.state_dict(), "input": a.input, "input_dim": IN, "step": step},
                       M / f"runs/short_{a.tag}_bestsel.pt")
        print(f"step {step:4d} sel_nll(T=1400) {v:.5f}  best {best[0]:.5f}@{best[1]}", flush=True)
print(f"TRAIN_DONE {a.tag} best {best[0]:.5f} at step {best[1]} wall {time.time()-t0:.0f}s", flush=True)
