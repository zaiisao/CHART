"""Can the amortised encoder represent the phase AT ALL, given full supervision?

The ELBO arm leaves the reconstruction at the base rate, and the supervised warm start
does not move it either. Two very different explanations: the ELBO's optimum has no
phase in it, or the encoder cannot compute the phase even when it is told the answer.
This isolates the second by dropping the ELBO entirely and regressing q's mean for
phi_1 on the true phase with 1 - cos loss.

If the val error stays near the chance value (1 - cos of a uniform angle = 1.0), the
bottleneck is the inference network, not the objective.

    PYTHONPATH=. python -m phasevae.check_encoder_capacity --gpu 1
"""
from __future__ import annotations

import argparse
import math
import pickle

import numpy as np
import torch

from .model import PhaseVAE
from .run import batches


def main() -> None:
    """Train the encoder head on the supervised phase target and report val error."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1, choices=(1, 3))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-crops", type=int, default=2000)
    parser.add_argument("--cache", default="/disk4/jaehoon/phasevae_crops.pkl")
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    with open(args.cache, "rb") as handle:
        crops, _ = pickle.load(handle)
    train_crops = [c for c in crops if c["fold"] not in (None, 7)][:args.train_crops]
    val_crops = [c for c in crops if c["fold"] == 7]
    print(f"train {len(train_crops)}  val {len(val_crops)}  "
          f"(chance 1 - cos = 1.000, chance |err| < 45 deg = 0.250)")

    torch.manual_seed(0)
    model = PhaseVAE(train_crops[0]["h"].shape[1]).to(device)
    optimiser = torch.optim.Adam(model.encoder.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for _, batch in batches(train_crops, 32, device, True, rng):
            mu = model.encoder(batch["h"], batch["y_channels"])[0][:, 0]
            loss = (1.0 - torch.cos(mu - batch["phi1_target"])).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(float(loss))
        model.eval()
        val_loss, within = [], []
        with torch.no_grad():
            for _, batch in batches(val_crops, 32, device, False):
                mu = model.encoder(batch["h"], batch["y_channels"])[0][:, 0]
                val_loss.append(float((1.0 - torch.cos(mu - batch["phi1_target"])).mean()))
                error = ((mu - batch["phi1_target"] + math.pi) % (2 * math.pi)) - math.pi
                within.append(float((error.abs() < math.pi / 4).float().mean()))
        print(f"  epoch {epoch:2d}  train 1-cos {np.mean(losses):.4f}  "
              f"val 1-cos {np.mean(val_loss):.4f}  val |err| < 45 deg "
              f"{np.mean(within):.3f}", flush=True)


if __name__ == "__main__":
    main()
