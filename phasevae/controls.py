"""Pre-flight controls.

Each one exists because its absence once shipped a wrong number; the measurements
behind them are in docs/phasevae_decisions.md.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from vbpm.data import FPS

from .crops import true_phase
from .metrics_db import f_measure
from .model import downbeat_frames


def assert_readout_recovers_oracle(crops, limit: int = 200, floor: float = 0.95):
    """Score the read-out on the TRUE phase; refuse to train unless it is near 1.0.

    A read-out that cannot score the truth cannot score a model (rule g once looked for
    the atan2 discontinuity at phi = pi and scored F = 0.000 on the ground truth).
    """
    scores = []
    for crop in crops[:limit]:
        phi, _valid = true_phase(crop)
        mu = torch.tensor(np.mod(phi + np.pi, 2 * np.pi) - np.pi)[None]   # encoder's range
        est = (np.flatnonzero(downbeat_frames(mu)[0].numpy()) + 1) / FPS + crop["t0"]
        truth = crop["downbeat_times"]
        scores.append(f_measure(est, truth)[0] if len(est) > 1 else 0.0)
    value = float(np.mean(scores))
    assert value > floor, (
        f"READ-OUT BROKEN: rule g scores F={value:.3f} on the ORACLE trajectory. "
        "Every model number would be meaningless. Fix the read-out before training.")
    return value


def assert_encoder_is_target_blind(model, batch):
    """The encoder may read h and the GIVEN bar rate delta -- never the target.

    Asserted structurally (signature) AND behaviourally (corrupt y, require the
    inferred phase bit-identical). delta being annotation-derived is a recorded
    widening of the deployable surface, not a waiver.
    """
    model.eval()
    allowed = {"self", "h", "delta"}
    named = set(model.encoder.forward.__code__.co_varnames[
        :model.encoder.forward.__code__.co_argcount])
    assert named <= allowed, f"encoder reads {named - allowed}: Point 2 violated"
    clean = model.infer_phase(batch["h"], batch["delta"])
    assert torch.equal(clean, model.infer_phase(batch["h"], batch["delta"])), \
        "encoder is not deterministic"
    poisoned = dict(batch)
    poisoned["y"] = 1.0 - batch["y"]
    assert torch.equal(clean, model.infer_phase(poisoned["h"], batch["delta"])), \
        "the deployed phase moved when the TARGET changed: C2 violated"


def gradient_audit(model, batch):
    """Every parameter must get a non-None, non-zero gradient. Read them; do not assume."""
    model.train()
    out = model(batch["h"], batch["delta"], batch["mask"], batch["y"])
    (-out["elbo"].mean()).backward()
    dead = [n for n, p in model.named_parameters()
            if p.grad is None or not torch.any(p.grad != 0)]
    model.zero_grad(set_to_none=True)
    return dead


def preflight(crops, max_crops: int):
    """Dataset-level controls, printed: duplicates, start-phase flatness, oracle read-out."""
    keys = {(c["stem"], round(c["t0"], 3)) for c in crops}
    assert len(keys) == len(crops), (
        f"DUPLICATE CROPS: {len(crops) - len(keys)} of {len(crops)} share a (stem, t0). "
        "Per-dataset means and seed sds would be computed on replicated data.")
    starts = np.array([c["t0"] for c in crops])
    print(f"  distinct (stem, t0): {len(keys)}/{len(crops)}   "
          f"t0 == 0: {float((starts == 0).mean()):.3f} of crops")
    print(f"  NOTE {max_crops} crops/song is a sampling cap, not a filter")

    phases = [true_phase(c) for c in crops]
    firsts = np.array([float(p[0]) for p, valid in phases if valid[0]])
    hist, _ = np.histogram(firsts, bins=8, range=(0.0, 2 * np.pi))
    print(f"\nCONTROL crop-start phase, 8 bins (want flat): {hist.tolist()} "
          f" n={len(firsts)}")
    print(f"  crops per dataset: {dict(Counter(c['dataset'] for c in crops))}")
    print(f"  bar periods: {min(c['bar_period'] for c in crops):.2f}"
          f"-{max(c['bar_period'] for c in crops):.2f} s")
    print(f"  CONTROL read-out on the oracle: F "
          f"{assert_readout_recovers_oracle(crops):.3f} (must exceed 0.95)")
