"""(3) HONEST MERT BASELINES on the eval songs + (4) TIME-ROLL leak control on them.

Reproduces vbpm/probe_linear.py exactly (same arch / seed / steps / lr / pos_weight /
threshold) so the previously quoted 0.725 (linear) and 0.804 (conv) are re-derived, then
re-scores each probe under three protocols and under a +1000-frame feature roll.

A probe is the honest bar for any "fixed" VBPM: it uses the same frozen MERT features and
the same 70 ms mir_eval F, and its output is read out by simple peak-picking.
"""
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/control2")
from cc import (FPS, load_split, truncate, score_activation, agg, ratio, by_dataset, line,
                banner, metronome, f_measure, sem)

DEV = "cuda:0"
OUT = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/control2/c3_out.json"

t0 = time.time()
train = load_split("train", with_feats=True)
ev = load_split("eval", with_feats=True)
print(f"loaded train {len(train)} eval {len(ev)} in {time.time()-t0:.0f}s", flush=True)


class Probe(nn.Module):
    """Verbatim copy of vbpm/probe_linear.py::Probe."""

    def __init__(self, kind="linear"):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(13))
        if kind == "linear":
            self.head = nn.Linear(768, 1)
        else:
            self.head = nn.Sequential(nn.Conv1d(768, 128, 5, padding=2), nn.ReLU(),
                                      nn.Conv1d(128, 1, 1))
        self.kind = kind

    def forward(self, feats):                      # [B,13,T,768]
        w = torch.softmax(self.layer_logits, 0)
        m = torch.einsum("l,bltf->btf", w, feats)
        if self.kind == "linear":
            return self.head(m).squeeze(-1)
        return self.head(m.transpose(1, 2)).squeeze(1)


def tgt(beats, start, n):
    y = np.zeros(n, np.float32)
    for t in beats:
        i = int(round(t * FPS)) - start
        if 0 <= i < n:
            y[i] = 1.0
    return y


def train_probe(kind, steps=800, frames=512, bs=16):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    p = Probe(kind).to(DEV)
    opt = torch.optim.AdamW(p.parameters(), lr=3e-4)
    pw = torch.tensor([12.0], device=DEV)
    for step in range(1, steps + 1):
        fe, ys = [], []
        for _ in range(bs):
            s = train[rng.integers(len(train))]
            T = s["feats"].shape[1]
            if T <= frames:
                continue
            st = int(rng.integers(0, T - frames))
            fe.append(torch.from_numpy(s["feats"][:, st:st + frames].astype(np.float32)))
            ys.append(torch.from_numpy(tgt(s["beats"], st, frames)))
        fe = torch.stack(fe).to(DEV)
        ys = torch.stack(ys).to(DEV)
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(p(fe), ys, pos_weight=pw)
        loss.backward()
        opt.step()
        if step % 200 == 0:
            print(f"  [{kind}] s{step} loss={loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
    return p


@torch.no_grad()
def eval_probe(p, songs, cap, thr=0.5, roll=0):
    p.eval()
    rows = []
    for s in songs:
        T, ref, dref = truncate(s, cap)
        if len(ref) < 2:
            continue
        f = torch.from_numpy(s["feats"][:, :T].astype(np.float32)).unsqueeze(0).to(DEV)
        if roll:
            f = torch.roll(f, roll, dims=2)          # slide FEATURES along time, labels stay
        prob = torch.sigmoid(p(f))[0].cpu().numpy()
        r = score_activation(prob, ref, dref, thr=thr)
        r["dataset"] = s["dataset"]
        r["metronome_F"] = f_measure(ref, metronome(T, FPS))
        rows.append(r)
    p.train()
    return rows


res = {}
PROTOS = [("eval[:30] cap1600", ev[:30], 1600), ("ALL79 cap1600", ev, 1600), ("ALL79 FULL", ev, None)]

for kind in ["linear", "conv"]:
    banner(f"MERT baseline: {kind} probe + peak-pick (thr 0.5)")
    p = train_probe(kind)
    for name, songs, cap in PROTOS:
        rows = eval_probe(p, songs, cap)
        line(f"{kind} probe [{name}]", rows,
             extra=f"metro={agg(rows,'metronome_F'):.3f} sem={sem(rows,'beat_F'):.3f}")
        res[f"{kind}_{name}"] = dict(beat_F=agg(rows, "beat_F"), ratio=ratio(rows),
                                     metronome=agg(rows, "metronome_F"), N=len(rows),
                                     sem=sem(rows, "beat_F"))
        if cap is None:
            print("   per dataset:", {k: round(v[0], 3) for k, v in by_dataset(rows, "beat_F").items()},
                  flush=True)
    # ---- (4) TIME-ROLL LEAK CONTROL on the honest baseline (power check for the control) ----
    banner(f"TIME-ROLL control on the {kind} probe (+1000 frames = 20 s, labels fixed)")
    for name, songs, cap in [("ALL79 FULL", ev, None), ("eval[:30] cap1600", ev[:30], 1600)]:
        al = eval_probe(p, songs, cap, roll=0)
        ro = eval_probe(p, songs, cap, roll=1000)
        d = agg(al, "beat_F") - agg(ro, "beat_F")
        print(f"  {kind} [{name}] aligned={agg(al,'beat_F'):.3f} rolled={agg(ro,'beat_F'):.3f} "
              f"floor(metro)={agg(al,'metronome_F'):.3f} drop={d:+.3f}", flush=True)
        res[f"roll_{kind}_{name}"] = dict(aligned=agg(al, "beat_F"), rolled=agg(ro, "beat_F"),
                                          floor=agg(al, "metronome_F"), drop=d)
    torch.save(p.state_dict(),
               f"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/control2/probe_{kind}.pt")
    json.dump(res, open(OUT, "w"), indent=1)

banner("metronome floors (same rows)")
for name, songs, cap in PROTOS:
    rows = []
    for s in songs:
        T, ref, dref = truncate(s, cap)
        if len(ref) < 2:
            continue
        rows.append(dict(beat_F=f_measure(ref, metronome(T, FPS)), downbeat_F=float("nan"),
                         n_est=len(metronome(T, FPS)), n_true=len(ref)))
    print(f"  metronome [{name}] = {agg(rows,'beat_F'):.3f}", flush=True)
    res[f"metronome_{name}"] = agg(rows, "beat_F")

json.dump(res, open(OUT, "w"), indent=1)
print("WROTE", OUT, flush=True)
