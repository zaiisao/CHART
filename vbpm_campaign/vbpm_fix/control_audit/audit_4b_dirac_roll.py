"""AUDIT 4b -- the DIRAC regime under the roll control, plus the Dirac free-run baseline.

Trains VBPM on Dirac h with vbpm/probe_dirac.py's exact recipe (my own copy, so the
provenance is auditable), reproduces its free-run beat_F, then rolls the Dirac h by
20 s at eval. Because the Dirac h IS the label track, a roll here is the cleanest
possible statement of how much of a Dirac score is input-copying vs open-loop luck.
"""
import sys, time
import numpy as np, torch

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
from vbpm.model import BarPointerVAE
from vbpm.elbo import strict_elbo, free_run
from audit_common import (load_split, truncate, score_phase, agg, ratio, banner,
                          metronome, f_measure, FPS, PRIOR_MAX_FRAMES, PRIOR_N_EVAL)
from roll_control import roll_control

dev = "cuda:0"; H_DIM = 8; ROLL = 1000; CAP = PRIOR_MAX_FRAMES
tr = load_split("train"); ev = load_split("eval")


def dirac_h(beats, downs, start, n, rng=None):
    r = np.random.randn(n, H_DIM) if rng is None else rng.standard_normal((n, H_DIM))
    h = (r * 0.01).astype(np.float32)
    for t in beats:
        i = int(round(t * FPS)) - start
        if 0 <= i < n: h[i, 0] += 1.0
    for t in downs:
        i = int(round(t * FPS)) - start
        if 0 <= i < n: h[i, 1] += 1.0
    return h


def targets(beats, downs, start, n):
    b = np.zeros(n, np.float32); d = np.zeros(n, np.float32)
    for t in beats:
        i = int(round(t * FPS)) - start
        if 0 <= i < n: b[i] = 1.0
    for t in downs:
        i = int(round(t * FPS)) - start
        if 0 <= i < n: d[i] = 1.0
    return b, d


STEPS, WARM, BS, FR = 700, 400, 16, 256
torch.manual_seed(0); rng = np.random.default_rng(0)
model = BarPointerVAE(h_dim=H_DIM, hidden=128, num_meters=4).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)


@torch.no_grad()
def dirac_phase(h_np):
    h = torch.from_numpy(np.ascontiguousarray(h_np)).unsqueeze(0).to(dev)
    return free_run(model, h)["phase_mu"][0].cpu().numpy()


feat_fn = lambda s, T: dirac_h(s["beats"], s["downs"], 0, T)
t0 = time.time()
for step in range(1, STEPS + 1):
    beta = min(1.0, step / WARM); temp = 1.0 + (0.3 - 1.0) * min(step / STEPS, 1.0)
    hs, bs_, ds = [], [], []
    for _ in range(BS):
        s = tr[rng.integers(len(tr))]
        if s["T"] <= FR: continue
        st = int(rng.integers(0, s["T"] - FR))
        hs.append(torch.from_numpy(dirac_h(s["beats"], s["downs"], st, FR, rng)))
        b, d = targets(s["beats"], s["downs"], st, FR)
        bs_.append(torch.from_numpy(b)); ds.append(torch.from_numpy(d))
    h = torch.stack(hs).to(dev); b = torch.stack(bs_).to(dev); d = torch.stack(ds).to(dev)
    opt.zero_grad(); loss, info = strict_elbo(model, h, b, d, temperature=temp, beta=beta)
    if not torch.isfinite(loss):
        print("NaN@", step, flush=True); break
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    if step in (700,):
        model.eval()
        banner(f"DIRAC step {step}  ({step/(time.time()-t0):.2f} it/s)")
        roll_control(dirac_phase, feat_fn, ev[:PRIOR_N_EVAL], roll=ROLL, cap=CAP,
                     label=f"dirac s{step} eval[:30]")
        strat = ev[:10] + ev[32:42] + ev[55:65]   # 10 ballroom + 10 beatles + 10 hainsworth
        roll_control(dirac_phase, feat_fn, strat, roll=ROLL, cap=CAP,
                     label=f"dirac s{step} strat-30 <=1600")
        model.train()
        torch.save({"sd": model.state_dict(), "step": step},
                   "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/dirac_vbpm.pt")
