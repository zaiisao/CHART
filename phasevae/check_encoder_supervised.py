"""Can the encoder represent bar phase AT ALL? Fit it supervised, with no ELBO.

This separates two explanations that the failing ELBO run cannot tell apart:

  * REPRESENTATION -- the encoder cannot map audio to bar phase, in which case the
    architecture is wrong and no change to the objective will help;
  * OPTIMISATION -- it can, but the ELBO's path never reaches that solution (measured:
    the reconstruction gradient is ~2600x smaller than the KL's at initialisation, and
    the emission flattens from b=1.35 to 0.91 during training, which is the regime where
    a constant-rate metronome beats the true phase).

So: throw the ELBO away, hand the encoder the true phase, and regress. The target is
circular, so the loss is 1 - cos(mu - phi_true) on valid frames -- never a squared error
on the angle, which would punish the wrap. Everything else (data, splits, read-out,
metric) is the deployment pipeline unchanged, so the number is comparable to the ELBO
runs' F. Scoring uses rule g on the phi = 0 crossing.

    PYTHONPATH=. python -m phasevae.check_encoder_supervised --gpu 3
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from vbpm.data import FPS

from .crops import true_phase
from .metrics_db import f_measure, null_times
from .model import BarPhaseVAE, downbeat_frames
from .run import VAL_FOLD, Batches, load_or_build

wrap_readout = downbeat_frames   # alias only: there is ONE read-out and it lives in model.py


def phase_targets(crops):
    """Attach the true per-frame phase and its validity mask. SCORING/SUPERVISION ONLY."""
    for crop in crops:
        phi, valid = true_phase(crop)
        crop["phi_true"] = phi
        crop["phi_valid"] = valid.astype(np.float32)
    return crops


def collate_targets(raw, device):
    """[B, T] true phase and its mask, padded to the batch."""
    length = max(len(c["phi_true"]) for c in raw)
    phi = torch.zeros(len(raw), length)
    ok = torch.zeros(len(raw), length)
    for i, crop in enumerate(raw):
        t = len(crop["phi_true"])
        phi[i, :t] = torch.from_numpy(crop["phi_true"])
        ok[i, :t] = torch.from_numpy(crop["phi_valid"])
    return phi.to(device), ok.to(device)


def evaluate(model, crops, device, batch_size, seed=0):
    """Per-dataset downbeat F for the supervised encoder, beside the nulls."""
    model.eval()
    rows: dict = defaultdict(lambda: defaultdict(list))
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for raw, batch in Batches(crops, batch_size, device)():
            mu = model.infer_phase(batch["h"], batch["delta"])
            wraps = downbeat_frames(mu, batch["mask"]).cpu().numpy()
            for i, crop in enumerate(raw):
                t = len(crop["y"])
                truth = crop["downbeat_times"]
                times = (np.flatnonzero(wraps[i, :t - 1]) + 1) / FPS + crop["t0"]
                rows[crop["dataset"]]["supervised"].append(f_measure(times, truth)[0])
                crop["fps"] = FPS
                for kind in ("random", "zero"):
                    rows[crop["dataset"]][f"null-{kind}"].append(
                        f_measure(null_times(crop, kind, rng), truth)[0])
    return {ds: {k: (float(np.mean(v)), len(v)) for k, v in m.items()}
            for ds, m in rows.items()}


def main() -> None:
    """Fit the encoder to the true phase and report what it can reach."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3, choices=(1, 3))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--limit-per-fold", type=int, default=None)
    ap.add_argument("--crop-cache", default=None)
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    crops, rejects = load_or_build(args.crop_cache, args.limit_per_fold)
    keys = {(c["stem"], round(c["t0"], 3)) for c in crops}
    assert len(keys) == len(crops), "duplicate crops"
    crops = phase_targets(crops)
    print(f"crops: {len(crops)}  rejects: {dict(rejects)}")

    train = [c for c in crops if c["fold"] not in (None, VAL_FOLD)]
    val = [c for c in crops if c["fold"] == VAL_FOLD]
    print(f"train {len(train)} / val {len(val)}")

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = BarPhaseVAE(train[0]["h"].shape[1]).to(device)
    loader = Batches(train, args.batch_size, device)
    opt = torch.optim.Adam(model.encoder.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total, n = 0.0, 0
        for raw, batch in loader(shuffle=True, rng=rng):
            phi, ok = collate_targets(raw, device)
            ok = ok * batch["mask"]
            mu, _kappa = model.encoder(batch["h"])
            loss = ((1.0 - torch.cos(mu - phi)) * ok).sum() / ok.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 5.0)
            opt.step()
            total, n = total + float(loss), n + 1
        if epoch % 5 == 4 or epoch == 0:
            err = np.degrees(np.arccos(np.clip(1.0 - total / n, -1, 1)))
            print(f"  epoch {epoch:2d}  1-cos {total / n:.4f}  "
                  f"(mean angular error ~{err:.1f} deg)", flush=True)

    print("\n==== SUPERVISED encoder, downbeat F (+-70 ms) ====")
    for dataset, modes in sorted(evaluate(model, val, device, args.batch_size).items()):
        line = "  ".join(f"{k} {v:.3f}" for k, (v, _n) in sorted(modes.items()))
        print(f"  {dataset:11s} {line}   (n={next(iter(modes.values()))[1]})")


if __name__ == "__main__":
    main()
