"""Train the Dirac-input VBPM (same recipe as vbpm/probe_dirac.py) and SAVE checkpoints
so the free_run deploy path can be traced offline. Runs on cuda:1. Does NOT modify vbpm/."""
import sys, glob, time, os
import numpy as np, torch
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
from vbpm.model import BarPointerVAE
from vbpm.elbo import strict_elbo

CACHE = "/disk1/jaehoon/vbpm_mert_cache"; dev = "cuda:1"; fps = 50.0; H_DIM = 8
OUT = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_debug"
os.makedirs(OUT, exist_ok=True)

def load(split, cap=None):
    out = []
    for f in sorted(glob.glob(f"{CACHE}/{split}__*.npz")):
        d = np.load(f, allow_pickle=True)
        out.append(dict(T=int(d["feats"].shape[1]), beats=np.asarray(d["beats"], float),
                        downs=np.asarray(d["downs"], float)))
    return out[:cap] if cap else out

train = load("train")
print(f"train songs {len(train)}", flush=True)

def dirac_h(beats, downs, start, n, rng=None):
    h = np.random.randn(n, H_DIM).astype(np.float32) * 0.01
    for t in beats:
        i = int(round(t * fps)) - start
        if 0 <= i < n: h[i, 0] += 1.0
    for t in downs:
        i = int(round(t * fps)) - start
        if 0 <= i < n: h[i, 1] += 1.0
    return h

def targets(beats, downs, start, n):
    b = np.zeros(n, np.float32); db = np.zeros(n, np.float32)
    for t in beats:
        i = int(round(t * fps)) - start
        if 0 <= i < n: b[i] = 1.0
    for t in downs:
        i = int(round(t * fps)) - start
        if 0 <= i < n: db[i] = 1.0
    return b, db

STEPS, WARM, BS, FR = 1200, 600, 16, 256
torch.manual_seed(0); np.random.seed(0); rng = np.random.default_rng(0)
model = BarPointerVAE(h_dim=H_DIM, hidden=128, num_meters=4).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
t0 = time.time()
SNAP = {300, 600, 900, 1200}
for step in range(1, STEPS + 1):
    beta = min(1.0, step / WARM); temp = 1.0 + (0.3 - 1.0) * min(step / STEPS, 1.0)
    hs, bs_, ds = [], [], []
    for _ in range(BS):
        s = train[rng.integers(len(train))]
        if s["T"] <= FR: continue
        st = int(rng.integers(0, s["T"] - FR))
        hs.append(torch.from_numpy(dirac_h(s["beats"], s["downs"], st, FR)))
        b, d = targets(s["beats"], s["downs"], st, FR)
        bs_.append(torch.from_numpy(b)); ds.append(torch.from_numpy(d))
    h = torch.stack(hs).to(dev); b = torch.stack(bs_).to(dev); d = torch.stack(ds).to(dev)
    opt.zero_grad(); loss, info = strict_elbo(model, h, b, d, temperature=temp, beta=beta)
    if not torch.isfinite(loss):
        print("NaN@", step, flush=True); break
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    if step % 50 == 0:
        print(f"s{step:4d} b={beta:.2f} rec_b={info['recon_beat']:6.2f} rec_db={info['recon_db']:6.2f} "
              f"kl(phi={info['kl_phase']:.2f} lv={info['kl_level']:.2f} dv={info['kl_dev']:.2f} m={info['kl_meter']:.2f}) "
              f"nu={info['tempo_dof']:.2f} ncross={info['n_cross']:.1f} | {step/(time.time()-t0):.2f} it/s", flush=True)
    if step in SNAP:
        torch.save({"sd": model.state_dict(), "step": step}, f"{OUT}/dirac_step{step}.pt")
        print(f"  saved snapshot step {step}", flush=True)
print("DONE", flush=True)
