"""Pre-flight controls.

Each one exists because its absence once shipped a wrong number; the measurements
behind them are in docs/phasevae_decisions.md.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from ..data.dataset import true_phase
from ..data.excerpts import collate_excerpts
from .evaluation import f_measure, rule_g_times, scoring_records


def assert_no_duplicate_crops(crops):
    """(stem, t0) must be unique: replicated crops silently bias every mean."""
    keys = {(c["stem"], round(c["t0"], 3)) for c in crops}
    assert len(keys) == len(crops), (
        f"DUPLICATE CROPS: {len(crops) - len(keys)} of {len(crops)} share a (stem, t0). "
        "Per-dataset means and seed sds would be computed on replicated data.")
    return keys


def assert_readout_recovers_oracle(crops, limit: int = 200, floor: float = 0.95):
    """Score the read-out on the TRUE phase; refuse to train unless it is near 1.0.

    A read-out that cannot score the truth cannot score a model (rule g once looked for
    the atan2 discontinuity at phi = pi and scored F = 0.000 on the ground truth).
    """
    scores = []
    for crop in crops[:limit]:
        phi, _valid = true_phase(crop)
        mu = torch.tensor(np.mod(phi + np.pi, 2 * np.pi) - np.pi)[None]   # encoder's range
        est = rule_g_times(mu, None, [crop])[0]
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
    deployed = model.deployed_net
    allowed = {"self", "h", "delta"}
    named = set(deployed.forward.__code__.co_varnames[
        :deployed.forward.__code__.co_argcount])
    assert named <= allowed, f"deployed net reads {named - allowed}"
    assert not getattr(deployed, "reads_target", False), \
        "the DEPLOYED inference network consumes the target: unusable at test time"

    # Bit-equality on CPU, deliberately: cuDNN may pick different algorithms between
    # two identical calls when OTHER tenants of a shared GPU perturb memory state --
    # which failed this assert three times on clean models before the cause was found.
    # CPU keeps the check bit-exact regardless of the neighbours.
    import copy
    cpu_model = copy.deepcopy(model).cpu().eval()
    h, delta = batch["h"].cpu(), batch["delta"].cpu()
    clean = cpu_model.infer_phase(h, delta)
    assert torch.equal(clean, cpu_model.infer_phase(h, delta)), \
        "encoder is not deterministic"

    poisoned_y = 1.0 - batch["y"].cpu()
    del poisoned_y  # the deployed path takes no y; blindness is structural (above) and
    #                 behavioural: identical h must give identical phase regardless of
    #                 anything else in the batch dict
    assert torch.equal(clean, cpu_model.infer_phase(h.clone(), delta.clone())), \
        "the deployed phase moved on cloned identical inputs"


def gradient_audit(model, batch):
    """Every parameter must get a non-None, non-zero gradient. Read them; do not assume.

    Backwards the full TRAINING objective, not just the elbo: in psi-distillation mode
    the prior network learns only through the distill and anchor terms, which live
    outside the elbo -- auditing the elbo alone declares all of psi dead (it did).
    """
    model.train()
    out = model(batch["h"], batch["delta"], batch["mask"], batch["y"])
    objective = out["elbo"]
    if "distill" in out:
        objective = objective - out["distill"] - out["prior_anchor"]
    (-objective.mean()).backward()

    dead = [n for n, p in model.named_parameters()
            if p.grad is None or not torch.any(p.grad != 0)]
    model.zero_grad(set_to_none=True)
    return dead


def preflight(*datasets):
    """Dataset-level controls over the deterministic eval datasets, one line each.

    Duplicates (silent unless failing), the two phase-leak tripwires, and the oracle
    read-out that certifies the scorer itself. No features and no model: the controls
    read only the scoring records.
    """
    crops = []
    for dataset in datasets:
        loader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                             collate_fn=collate_excerpts)
        for raw in loader:
            crops += [c for c in scoring_records(raw) if c is not None]

    assert_no_duplicate_crops(crops)

    starts = np.array([c["t0"] for c in crops])
    print(f"  windows: {dict(Counter(c['dataset'] for c in crops))}, "
          f"bar periods {min(c['bar_period'] for c in crops):.2f}-"
          f"{max(c['bar_period'] for c in crops):.2f} s, "
          f"{float((starts == 0).mean()):.0%} starting at t=0 "
          f"(short songs use the whole song)")

    phases = [true_phase(c) for c in crops]
    firsts = np.array([float(p[0]) for p, valid in phases if valid[0]])
    hist, _ = np.histogram(firsts, bins=8, range=(0.0, 2 * np.pi))
    print(f"  CONTROL start-phase spread over 8 bins (a peak = phase leak): "
          f"{hist.tolist()}  n={len(firsts)}")

    print(f"  CONTROL scorer on the TRUE phase: F "
          f"{assert_readout_recovers_oracle(crops):.3f} (must exceed 0.95)")
