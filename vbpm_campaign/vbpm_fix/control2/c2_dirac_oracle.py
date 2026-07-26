"""(2) DIRAC IS AN ORACLE -- the "just copy the input" ceiling.

DIRAC h is built FROM THE LABELS: h[t,0]=1 at every true beat frame, h[t,1]=1 at every true
downbeat frame, +N(0,0.01) elsewhere (vbpm/probe_dirac.py's recipe). The input therefore
CONTAINS the answer at eval time too. This script measures how much of any Dirac score is
trivially explained by copying that input:

  copy-0  zero parameters      : peak-pick channel 0 of h
  copy-1  tiny linear 8->2     : no latents, no dynamics, trained on the train fold
  copy-2  tiny conv 8->2       : same, with a 5-frame receptive field
  copy-3  official read-out    : bar phase rebuilt from channel-1 impulses -> beats_from_barphase
  neg     same nets, impulses deleted (noise only) -> must be ~0

Any Dirac result must be quoted against these, not against the metronome.
"""
import json
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/control2")
from cc import (FPS, load_split, truncate, ideal_barphase, score_phase, score_activation,
                agg, ratio, line, banner, metronome, f_measure)

DEV = "cpu"
H_DIM = 8
rng_global = np.random.default_rng(0)

train = load_split("train")
ev = load_split("eval")
print(f"train {len(train)} eval {len(ev)}", flush=True)


def dirac_h(beats, downs, start, n, impulses=True, seed=None):
    r = np.random.default_rng(seed) if seed is not None else rng_global
    h = r.standard_normal((n, H_DIM)).astype(np.float32) * 0.01
    if impulses:
        for t in beats:
            i = int(round(t * FPS)) - start
            if 0 <= i < n:
                h[i, 0] += 1.0
        for t in downs:
            i = int(round(t * FPS)) - start
            if 0 <= i < n:
                h[i, 1] += 1.0
    return h


def targets(beats, downs, start, n):
    b = np.zeros(n, np.float32)
    db = np.zeros(n, np.float32)
    for t in beats:
        i = int(round(t * FPS)) - start
        if 0 <= i < n:
            b[i] = 1.0
    for t in downs:
        i = int(round(t * FPS)) - start
        if 0 <= i < n:
            db[i] = 1.0
    return b, db


PROTOS = [("eval[:30] cap1600", ev[:30], 1600), ("ALL79 FULL", ev, None)]

banner("copy-0  ZERO PARAMETERS: peak-pick Dirac channel 0")
res = {}
for name, songs, cap in PROTOS:
    rows = []
    for s in songs:
        T, ref, dref = truncate(s, cap)
        if len(ref) < 2:
            continue
        h = dirac_h(s["beats"], s["downs"], 0, T, seed=1)
        r = score_activation(h[:, 0], ref, dref, thr=0.5)
        r["dataset"] = s["dataset"]
        rows.append(r)
    line(f"copy-0 peak-pick h[:,0] [{name}]", rows)
    res[f"copy0_{name}"] = dict(beat_F=agg(rows, "beat_F"), ratio=ratio(rows))

banner("copy-1 / copy-2  tiny NO-LATENT nets mapping Dirac h -> beats (trained, train fold)")


class Tiny(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.f = nn.Linear(H_DIM, 2)
        else:
            self.f = nn.Sequential(nn.Conv1d(H_DIM, 16, 5, padding=2), nn.ReLU(),
                                   nn.Conv1d(16, 2, 1))

    def forward(self, h):                       # [B,T,H]
        if self.kind == "linear":
            return self.f(h)                    # [B,T,2]
        return self.f(h.transpose(1, 2)).transpose(1, 2)


def train_tiny(kind, steps=600, frames=512, bs=16):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    net = Tiny(kind).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    pw = torch.tensor([12.0], device=DEV)
    for step in range(1, steps + 1):
        hs, bs_, ds = [], [], []
        for _ in range(bs):
            s = train[rng.integers(len(train))]
            if s["T"] <= frames:
                continue
            st = int(rng.integers(0, s["T"] - frames))
            hs.append(torch.from_numpy(dirac_h(s["beats"], s["downs"], st, frames)))
            b, d = targets(s["beats"], s["downs"], st, frames)
            bs_.append(torch.from_numpy(b))
            ds.append(torch.from_numpy(d))
        h = torch.stack(hs).to(DEV)
        b = torch.stack(bs_).to(DEV)
        d = torch.stack(ds).to(DEV)
        opt.zero_grad()
        o = net(h)
        loss = (F.binary_cross_entropy_with_logits(o[..., 0], b, pos_weight=pw)
                + F.binary_cross_entropy_with_logits(o[..., 1], d, pos_weight=pw))
        loss.backward()
        opt.step()
        if step % 200 == 0:
            print(f"  [{kind}] s{step} loss={loss.item():.4f}", flush=True)
    return net


@torch.no_grad()
def eval_tiny(net, songs, cap, impulses=True):
    rows = []
    for s in songs:
        T, ref, dref = truncate(s, cap)
        if len(ref) < 2:
            continue
        h = torch.from_numpy(dirac_h(s["beats"], s["downs"], 0, T, impulses=impulses, seed=1)).unsqueeze(0)
        o = torch.sigmoid(net(h.to(DEV)))[0].cpu().numpy()
        r = score_activation(o[:, 0], ref, dref, thr=0.5)
        est_db = np.asarray([t for t in range(1, T - 1)
                             if o[t, 1] >= 0.5 and o[t, 1] >= o[t - 1, 1] and o[t, 1] >= o[t + 1, 1]],
                            float) / FPS
        r["downbeat_F"] = f_measure(dref, est_db) if len(dref) >= 2 else float("nan")
        r["dataset"] = s["dataset"]
        rows.append(r)
    return rows


for kind in ["linear", "conv"]:
    net = train_tiny(kind)
    for name, songs, cap in PROTOS:
        rows = eval_tiny(net, songs, cap)
        line(f"copy tiny-{kind} [{name}]", rows)
        res[f"copy_{kind}_{name}"] = dict(beat_F=agg(rows, "beat_F"), db_F=agg(rows, "downbeat_F"),
                                          ratio=ratio(rows))
    neg = eval_tiny(net, ev, None, impulses=False)
    line(f"NEGATIVE control tiny-{kind}, impulses deleted [ALL79 FULL]", neg)
    res[f"neg_{kind}"] = agg(neg, "beat_F")

banner("copy-3  through the OFFICIAL bar-phase read-out (phase rebuilt from the Dirac channels)")
for name, songs, cap in PROTOS:
    rows = []
    for s in songs:
        T, ref, dref = truncate(s, cap)
        if len(ref) < 2 or len(dref) < 2:
            continue
        # bar phase reconstructed from the downbeat impulses that ARE IN THE INPUT
        h = dirac_h(s["beats"], s["downs"], 0, T, seed=1)
        d_frames = np.where(h[:, 1] > 0.5)[0] / FPS
        ph = ideal_barphase(d_frames, T)
        if ph is None:
            continue
        r = score_phase(ph, ref, dref)
        r["dataset"] = s["dataset"]
        rows.append(r)
    line(f"copy-3 bar-phase read-out from h[:,1] [{name}]", rows)
    res[f"copy3_{name}"] = dict(beat_F=agg(rows, "beat_F"), db_F=agg(rows, "downbeat_F"),
                                ratio=ratio(rows))

banner("floor for reference")
for name, songs, cap in PROTOS:
    rows = []
    for s in songs:
        T, ref, dref = truncate(s, cap)
        if len(ref) < 2:
            continue
        rows.append(dict(beat_F=f_measure(ref, metronome(T, FPS)), downbeat_F=float("nan"),
                         n_est=len(metronome(T, FPS)), n_true=len(ref)))
    print(f"  metronome [{name}] {agg(rows,'beat_F'):.3f}", flush=True)

json.dump(res, open("/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/control2/c2_out.json", "w"), indent=1)
print("\nWROTE c2_out.json", flush=True)
