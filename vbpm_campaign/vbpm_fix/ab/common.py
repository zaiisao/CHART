"""Data / dirac / eval helpers for the A+B variant.  Read-only w.r.t. vbpm/."""
import glob, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from vbpm.evaluate import (beats_from_barphase, downbeats_from_barphase,
                           metronome, f_measure, _estimate_meter)

CACHE = "/disk1/jaehoon/vbpm_mert_cache"
FPS = 50.0
H_DIM_DIRAC = 8
TWO_PI = 2.0 * math.pi


def load_split(split, cap=None, with_feats=True):
    out = []
    for f in sorted(glob.glob(f"{CACHE}/{split}__*.npz")):
        d = np.load(f, allow_pickle=True)
        r = dict(stem=Path(f).stem, beats=np.asarray(d["beats"], float),
                 downs=np.asarray(d["downs"], float), T=int(d["feats"].shape[1]))
        if with_feats:
            r["feats"] = d["feats"]
        out.append(r)
        if cap and len(out) >= cap:
            break
    return out


class LayerMerge(nn.Module):
    """Learnable softmax over the 13 MERT layers -> [.,T,768]."""
    def __init__(self, n_layers=13):
        super().__init__(); self.layer_logits = nn.Parameter(torch.zeros(n_layers))

    def forward(self, feats):
        w = torch.softmax(self.layer_logits, 0)
        return torch.einsum("l,bltf->btf", w, feats)

    def weights(self):
        return torch.softmax(self.layer_logits.detach(), 0).cpu().numpy()


def dirac_h(beats, downs, start, n, shift_frames=0, rng=None, h_dim=H_DIM_DIRAC):
    """h[:,0]=beat impulses, h[:,1]=downbeat impulses (+tiny noise).  ORACLE input."""
    r = rng if rng is not None else np.random.default_rng(0)
    h = r.standard_normal((n, h_dim)).astype(np.float32) * 0.01
    for t in beats:
        i = int(round(t * FPS)) - start + shift_frames
        if 0 <= i < n: h[i, 0] += 1.0
    for t in downs:
        i = int(round(t * FPS)) - start + shift_frames
        if 0 <= i < n: h[i, 1] += 1.0
    return h


def targets(beats, downs, start, n):
    b = np.zeros(n, np.float32); db = np.zeros(n, np.float32)
    for t in beats:
        i = int(round(t * FPS)) - start
        if 0 <= i < n: b[i] = 1.0
    for t in downs:
        i = int(round(t * FPS)) - start
        if 0 <= i < n: db[i] = 1.0
    return b, db


def score_phase(phase, song, T):
    """Deploy read-out exactly as specified, PLUS degeneracy diagnostics.

    bars_advanced / n_wrap / phi_range exist because a bounded audio-conditioned prior-mean
    correction can CANCEL the tempo advance and park the pointer on a wrap boundary: the
    read-out then emits one 'beat' per audio impulse and scores high while the bar pointer
    never advances.  Any beat_F must be read together with bars_advanced ~ n_true/m.
    """
    phase = np.asarray(phase, dtype=float)
    ref = song["beats"][song["beats"] < T / FPS]
    dref = song["downs"][song["downs"] < T / FPS]
    if len(ref) < 2:
        return None
    m = _estimate_meter(ref, dref)
    est = beats_from_barphase(phase, m, FPS)
    dest = downbeats_from_barphase(phase, FPS)
    d = np.diff(phase); d = (d + math.pi) % TWO_PI - math.pi
    return dict(beat_F=f_measure(ref, est),
                db_F=(f_measure(dref, dest) if len(dref) >= 2 else float("nan")),
                metro_F=f_measure(ref, metronome(T, FPS)),
                bars_advanced=float(d.sum() / TWO_PI),
                bars_true=float(len(ref)) / m,
                phi_range=float(phase.max() - phase.min()),
                n_wrap=int((np.diff(phase) < -math.pi).sum()),
                n_est=len(est), n_true=len(ref), meter=m)


def aggregate(rows):
    out = {}
    for k in ("beat_F", "db_F", "metro_F", "bars_advanced", "bars_true", "phi_range", "n_wrap"):
        v = [r[k] for r in rows if r is not None and not (isinstance(r[k], float) and math.isnan(r[k]))]
        out[k] = float(np.mean(v)) if v else float("nan")
    n_est = sum(r["n_est"] for r in rows if r is not None)
    n_true = sum(r["n_true"] for r in rows if r is not None)
    out["n_est/n_true"] = float(n_est) / max(n_true, 1)
    out["n_songs"] = sum(1 for r in rows if r is not None)
    return out


def circ_absdiff(a, b):
    d = (np.asarray(a) - np.asarray(b) + math.pi) % TWO_PI - math.pi
    return np.abs(d)
