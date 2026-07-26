"""AUDIT 6 -- is the honest MERT bar even HIGHER than 0.725/0.804?

The probes are peak-picked at a hard-coded thr=0.5 that was never tuned (n_est/n_true
runs at 1.28-1.35, i.e. over-emission). Here the threshold is chosen on TRAIN songs
only and then applied to eval, so the bar the VBPM variants must clear is not
artificially low.
"""
import sys
import numpy as np, torch, torch.nn as nn

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
from audit_common import (load_split, truncate, banner, beats_from_activation,
                          metronome, f_measure, FPS, PRIOR_MAX_FRAMES, PRIOR_N_EVAL)

dev = "cuda:0"
tr = load_split("train", with_feats=True)
ev = load_split("eval", with_feats=True)


class Probe(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(13))
        self.head = (nn.Linear(768, 1) if kind == "linear" else
                     nn.Sequential(nn.Conv1d(768, 128, 5, padding=2), nn.ReLU(), nn.Conv1d(128, 1, 1)))
        self.kind = kind
    def forward(self, feats):
        m = torch.einsum("l,bltf->btf", torch.softmax(self.layer_logits, 0), feats)
        return self.head(m).squeeze(-1) if self.kind == "linear" else self.head(m.transpose(1, 2)).squeeze(1)


@torch.no_grad()
def probs(p, songs, cap):
    out = []
    for s in songs:
        T, ref, _ = truncate(s, cap)
        if len(ref) < 2: continue
        f = torch.from_numpy(s["feats"][:, :T].astype(np.float32)).unsqueeze(0).to(dev)
        out.append((torch.sigmoid(p(f))[0].cpu().numpy(), ref, T))
    return out


for kind in ["linear", "conv"]:
    ck = f"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/probe_{kind}.pt"
    p = Probe(kind).to(dev); p.load_state_dict(torch.load(ck, map_location=dev)); p.eval()
    banner(f"{kind.upper()} PROBE -- threshold chosen on TRAIN, applied to EVAL")
    P_tr = probs(p, tr[:40], PRIOR_MAX_FRAMES)
    grid = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    tr_F = [float(np.mean([f_measure(r, beats_from_activation(q, FPS, thr=t)) for q, r, T in P_tr])) for t in grid]
    best = grid[int(np.argmax(tr_F))]
    print("  train F by thr: " + "  ".join(f"{t}:{v:.3f}" for t, v in zip(grid, tr_F)) + f"   -> pick {best}")
    for label, songs, cap in [("eval[:30] <=1600", ev[:PRIOR_N_EVAL], PRIOR_MAX_FRAMES),
                              ("ALL 79 FULL", ev, None)]:
        P = probs(p, songs, cap)
        for t in [0.5, best]:
            Fs = [f_measure(r, beats_from_activation(q, FPS, thr=t)) for q, r, T in P]
            ne = sum(len(beats_from_activation(q, FPS, thr=t)) for q, r, T in P)
            nt = sum(len(r) for q, r, T in P)
            tag = " (hard-coded)" if t == 0.5 else " (train-picked)"
            print(f"  {label:20s} thr={t:.2f}{tag:16s} beat_F={np.mean(Fs):.3f}  n_est/n_true={ne/nt:.3f}")
