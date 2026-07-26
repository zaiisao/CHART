"""AUDIT 2 -- "DIRAC IS AN ORACLE": the just-copy-the-input ceiling.

The DIRAC regime sets h[t,0]=1 at true beat frames and h[t,1]=1 at true downbeat
frames -- i.e. the INPUT IS THE LABEL. Any Dirac score must therefore be compared
against how well one can do by TRIVIALLY COPYING the input, not against 0.295.

Three copy ceilings, cheapest first:
  C0  zero parameters : peak-pick channel 0 of h.
  C1  tiny NO-LATENT model : Linear(8->1) / small conv, per-frame, trained on the
      train fold, peak-picked. No phi, no tempo, no meter, no VAE.
  C2  through the OFFICIAL BAR-PHASE read-out : rebuild the bar phase from h's
      downbeat channel alone (pure input arithmetic) and score it with
      beats_from_barphase -- the exact code path the VBPM variants use.
Plus a negative control: same models on Dirac h with the impulses REMOVED.
"""
import sys, math, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
from audit_common import (load_split, ideal_barphase, truncate, score_phase, agg, ratio,
                          banner, beats_from_activation, metronome, f_measure,
                          _estimate_meter, FPS, PRIOR_MAX_FRAMES, PRIOR_N_EVAL)

dev = "cuda:0"
H_DIM = 8   # identical to vbpm/probe_dirac.py


def dirac_h(beats, downs, start, n, rng=None):
    """EXACTLY vbpm/probe_dirac.py's dirac_h (noise 0.01, impulses at label frames)."""
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


tr = load_split("train"); ev = load_split("eval")
PROTOCOLS = [("eval[:30] <=1600 (PRIOR PROTOCOL)", ev[:PRIOR_N_EVAL], PRIOR_MAX_FRAMES),
             ("ALL 79 eval FULL length",           ev,                None)]

# ---------------------------------------------------------------- C0: zero parameters
banner("C0  ZERO-PARAMETER COPY: peak-pick channel 0 of the Dirac h")
for name, songs, cap in PROTOCOLS:
    Fb, Fm, ne, nt = [], [], 0, 0
    for s in songs:
        T, ref, dref = truncate(s, cap)
        if len(ref) < 2: continue
        h = dirac_h(s["beats"], s["downs"], 0, T)
        est = beats_from_activation(h[:, 0], FPS, thr=0.5)
        Fb.append(f_measure(ref, est)); Fm.append(f_measure(ref, metronome(T, FPS)))
        ne += len(est); nt += len(ref)
    print(f"  {name:38s} beat_F={np.mean(Fb):.3f}  metro={np.mean(Fm):.3f}  n_est/n_true={ne/nt:.3f}  N={len(Fb)}")

# ------------------------------------------------- C2: official phase read-out, from h only
banner("C2  COPY THROUGH THE OFFICIAL BAR-PHASE READ-OUT (phase rebuilt from h's channel 1)")
for name, songs, cap in PROTOCOLS:
    rows = []
    for s in songs:
        T, ref, dref = truncate(s, cap)
        if len(ref) < 2 or len(dref) < 2: continue
        h = dirac_h(s["beats"], s["downs"], 0, T)
        d_frames = np.where(h[:, 1] > 0.5)[0]           # read downbeats straight off the input
        if len(d_frames) < 2: continue
        ph = ideal_barphase(d_frames / FPS, T, mode="extrap")
        r = score_phase(ph, ref, dref, T)
        rows.append(r)
    a = agg(rows, ["beat_F", "downbeat_F"])
    print(f"  {name:38s} beat_F={a['beat_F']:.3f}  db_F={a['downbeat_F']:.3f}  "
          f"n_est/n_true={ratio(rows):.3f}  N={len(rows)}")

# ---------------------------------------------------------------- C1: tiny no-latent model
class TinyCopy(nn.Module):
    """NO latents, NO VAE: per-frame h -> beat/downbeat logits."""
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.head = nn.Linear(H_DIM, 2)
        else:
            self.head = nn.Sequential(nn.Conv1d(H_DIM, 32, 5, padding=2), nn.ReLU(), nn.Conv1d(32, 2, 1))
    def forward(self, h):                                   # h [B,T,H]
        if self.kind == "linear":
            return self.head(h)                             # [B,T,2]
        return self.head(h.transpose(1, 2)).transpose(1, 2)


def train_tiny(kind, steps=600, frames=512, bs=16, seed=0, blank=False):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = TinyCopy(kind).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    pw = torch.tensor([12.0, 40.0], device=dev)
    for _ in range(steps):
        hs, ys = [], []
        for _ in range(bs):
            s = tr[rng.integers(len(tr))]
            if s["T"] <= frames: continue
            st = int(rng.integers(0, s["T"] - frames))
            bt = np.array([]) if blank else s["beats"]
            dn = np.array([]) if blank else s["downs"]
            hs.append(torch.from_numpy(dirac_h(bt, dn, st, frames, rng)))
            b, d = targets(s["beats"], s["downs"], st, frames)
            ys.append(torch.from_numpy(np.stack([b, d], -1)))
        h = torch.stack(hs).to(dev); y = torch.stack(ys).to(dev)
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(net(h), y, pos_weight=pw)
        loss.backward(); opt.step()
    return net


@torch.no_grad()
def eval_tiny(net, songs, cap, blank=False):
    net.eval(); Fb, Fd, ne, nt = [], [], 0, 0
    for s in songs:
        T, ref, dref = truncate(s, cap)
        if len(ref) < 2: continue
        bt = np.array([]) if blank else s["beats"]
        dn = np.array([]) if blank else s["downs"]
        h = torch.from_numpy(dirac_h(bt, dn, 0, T)).unsqueeze(0).to(dev)
        p = torch.sigmoid(net(h))[0].cpu().numpy()
        est = beats_from_activation(p[:, 0], FPS, thr=0.5)
        Fb.append(f_measure(ref, est)); ne += len(est); nt += len(ref)
        if len(dref) >= 2:
            Fd.append(f_measure(dref, beats_from_activation(p[:, 1], FPS, thr=0.5, min_dist_sec=0.30)))
    net.train()
    return float(np.mean(Fb)), (float(np.mean(Fd)) if Fd else float("nan")), ne / max(nt, 1), len(Fb)


banner("C1  TINY NO-LATENT MODEL trained on Dirac h (the 'just copy the input' ceiling)")
t0 = time.time()
for kind in ["linear", "conv"]:
    net = train_tiny(kind)
    for name, songs, cap in PROTOCOLS:
        b, d, rr, n = eval_tiny(net, songs, cap)
        print(f"  {kind:6s} | {name:38s} beat_F={b:.3f}  db_F={d:.3f}  n_est/n_true={rr:.3f}  N={n}")
print(f"  ({time.time()-t0:.0f}s)")

banner("NEGATIVE CONTROL: same tiny models, Dirac impulses REMOVED (noise only)")
for kind in ["linear", "conv"]:
    net = train_tiny(kind, blank=True)
    b, d, rr, n = eval_tiny(net, ev, None, blank=True)
    print(f"  {kind:6s} | noise-only h, ALL 79 FULL          beat_F={b:.3f}  db_F={d:.3f}  n_est/n_true={rr:.3f}")
