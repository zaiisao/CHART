"""The constructor's argument groups: one dataclass per cluster of knobs."""
from __future__ import annotations

import dataclasses

from .constants import KAPPA_PHYSICAL


@dataclasses.dataclass
class EmissionSpec:
    """p(y_t | phi_t): which shape reads the latent, and how big it is."""

    kind: str = "cosine"
    layers: int = 2
    dim: int = 64
    positional: bool = False

    @classmethod
    def coerce(cls, value):
        """Accept either a spec or the bare kind string the configs still pass."""
        return value if isinstance(value, cls) else cls(kind=value)


@dataclasses.dataclass
class WalkSpec:
    """p(phi_t | phi_t-1): the tempo walk's law and the phase prior's tightness."""

    kind: str = "gauss"
    kappa_physical: float = KAPPA_PHYSICAL
    kappa_gate: bool = False


@dataclasses.dataclass
class PlacementSpec:
    """How the placement factor reads phase, and which paths reach it."""

    coord: str = "first"
    lift: float = 0.0
    attach: bool = False


@dataclasses.dataclass
class UpdateSpec:
    """q(phi_t)'s free mean: the update half of the filter."""

    delta_on: bool = False
    gate_cond: bool = True


@dataclasses.dataclass
class DecoderSpec:
    """The knot decoder that emits the rate correction and delta."""

    dim: int = 32
    knot_stride: int = 25
