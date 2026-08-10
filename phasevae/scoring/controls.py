"""Pre-flight controls.

Each one exists because its absence once shipped a wrong number; the measurements
behind them are in docs/phasevae_decisions.md.
"""
from __future__ import annotations

import torch


def assert_no_duplicate_crops(crops):
    """(song_id, t0) must be unique: replicated crops silently bias every mean."""
    keys = {(c["song_id"], round(c["t0"], 3)) for c in crops}
    assert len(keys) == len(crops), (
        f"DUPLICATE CROPS: {len(crops) - len(keys)} of {len(crops)} share a (song_id, t0). "
        "Per-dataset means and seed sds would be computed on replicated data.")
    return keys


def assert_encoder_is_target_blind(model, batch):
    """The encoder may read h and the GIVEN bar rate delta -- never the target.

    Asserted structurally (signature) AND behaviourally (corrupt y, require the
    inferred phase bit-identical). delta being annotation-derived is a recorded
    widening of the deployable surface, not a waiver.
    """
    model.eval()
    deployed = model.deployed_net
    allowed = {"self", "h", "mask"}
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
    h = batch["h"].cpu()
    clean = cpu_model.infer_phase(h)
    assert torch.equal(clean, cpu_model.infer_phase(h)), \
        "encoder is not deterministic"

    poisoned_y = 1.0 - batch["y"].cpu()
    del poisoned_y  # the deployed path takes no y; blindness is structural (above) and
    #                 behavioural: identical h must give identical phase regardless of
    #                 anything else in the batch dict
    assert torch.equal(clean, cpu_model.infer_phase(h.clone())), \
        "the deployed phase moved on cloned identical inputs"
