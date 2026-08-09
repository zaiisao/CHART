"""Why does swapping the read-out move F by 0.23? Measure the ANCHOR ERROR itself.

For every annotated downbeat inside a window, read the deployed phase at that frame.
A perfect anchor puts phase 0 there; the wrapped value IS the placement error, in ms of
the bar. F has a hard +-70 ms tolerance, so what matters is not whether the head learned
something but whether its error lands inside that window.
"""
from __future__ import annotations

import argparse
import numpy as np
import torch

from phasevae.data.dataset import split_songs
from phasevae.data.excerpts import ExcerptDataset, collate_excerpts, to_model_batch
from phasevae.model import BarPhaseVAE, TWO_PI
from phasevae.scoring.evaluation import scoring_records


def circular_anchor(mu, act, mask=None, center=True, eps=1e-12):
    """[B, T] phase, [B, T] evidence -> [B] shift, s.t. ``mu + shift`` sits on the bar.

    The circular mean of the evidence folded under the model's own phase:
    ``shift = -arg(sum_t a_t e^{i mu_t})``. Zero parameters, nothing learned. This is a
    MEASURING INSTRUMENT for decomposing where F lives, not a model component -- no
    trained variant in this repo uses it.
    """
    w = torch.ones_like(mu) if mask is None else mask
    weight = w * act
    if center:
        total = w.sum(1, keepdim=True).clamp(min=eps)
        weight = weight - w * (weight.sum(1, keepdim=True) / total)
    real = (weight * torch.cos(mu)).sum(1)
    imag = (weight * torch.sin(mu)).sum(1)
    return -torch.atan2(imag, real)

ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=3)
args = ap.parse_args()
device = torch.device(f"cuda:{args.gpu}")

fe_mod = __import__("phasevae.data.frontends.beat_this", fromlist=["FRONTEND"])
frontend = fe_mod.FRONTEND(checkpoint="final0", device=f"cuda:{args.gpu}",
                           output="features+activations")
_, _, test_songs = split_songs(None)
test_set = ExcerptDataset(test_songs, frontend, 45.0, deterministic=True)

model = BarPhaseVAE(frontend.num_channels, emission="triangle", drift_bound=0.01,
                    bar_rate=True, kappa_physical=2000.0)
sd = torch.load("checkpoints/excerpt_base45/seed0.pt", map_location="cpu")
model.load_state_dict(sd, strict=False)
model = model.to(device).eval()

rng = np.random.default_rng(0)
err = {"offset": [], "moment": [], "random": []}
signed = {}
loader = torch.utils.data.DataLoader(test_set, batch_size=8, collate_fn=collate_excerpts)

with torch.no_grad():
    for raw in loader:
        recs = scoring_records(raw)
        keep = [i for i, c in enumerate(recs) if c is not None]
        if not keep:
            continue
        batch = to_model_batch(raw, frontend, device)
        mu_off, _ = model.encoder(batch["h"], batch["delta"])
        act = torch.sigmoid(batch["h"][..., -1])
        shift = circular_anchor(mu_off, act, batch["mask"])
        paths = {"offset": mu_off,
                 "moment": mu_off + shift.unsqueeze(-1),
                 "random": mu_off + torch.as_tensor(
                     rng.uniform(-np.pi, np.pi, size=mu_off.shape[0]),
                     dtype=torch.float32, device=device).unsqueeze(-1)}

        for i in keep:
            c = recs[i]
            fps, t0, period = float(c["fps"]), float(c["t0"]), float(c["bar_period"])
            n = int(batch["mask"][i].sum().item())
            frames = np.round((np.asarray(c["downbeat_times"]) - t0) * fps).astype(int)
            frames = frames[(frames >= 0) & (frames < n)]
            if len(frames) < 2:
                continue
            for name, path in paths.items():
                phi = path[i, frames].cpu().numpy()
                wrapped = np.arctan2(np.sin(phi), np.cos(phi))       # (-pi, pi]
                err[name].append(np.median(np.abs(wrapped)) / TWO_PI * period * 1000)
                signed.setdefault(name, []).append(
                    np.arctan2(np.sin(wrapped).mean(), np.cos(wrapped).mean())
                    / TWO_PI * period * 1000)

print(f"\nanchor placement error, gtzan (n={len(err['offset'])} windows), "
      f"F tolerance is +-70 ms\n")
print(f"{'read-out':10s} {'median':>9s} {'<70ms':>7s} {'<35ms':>7s} {'p90':>9s}")
for name in ("random", "offset", "moment"):
    e = np.asarray(err[name])
    print(f"{name:10s} {np.median(e):7.1f}ms {100*np.mean(e < 70):6.1f}% "
          f"{100*np.mean(e < 35):6.1f}% {np.percentile(e, 90):7.1f}ms")

print("\nSIGNED error -- a systematic bias would show a mean far from 0")
for name in ("offset", "moment"):
    v = np.asarray(signed[name])
    inside = v[np.abs(v) < 250]        # exclude gross flips, which swamp the mean
    print(f"{name:10s} mean {v.mean():+7.1f}ms  median {np.median(v):+7.1f}ms  |  "
          f"non-flip subset (n={len(inside)}): mean {inside.mean():+6.1f}ms  "
          f"sd {inside.std():5.1f}ms")
