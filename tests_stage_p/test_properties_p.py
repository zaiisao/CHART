"""Run every Stage-P property against the subject under test.

Against ``oracle`` this is the "proper => passes" half of the iff: a failure here means
the TEST is wrong, not the implementation.
"""
from __future__ import annotations

import pytest

import properties_p as P


@pytest.mark.oracle
@pytest.mark.parametrize("name", list(P.CHECKS))
def test_property(name, subject_factory):
    """Assert one named property holds for the subject under test."""
    try:
        P.CHECKS[name](subject_factory)
    except P.CapacityUnsupportedError as exc:
        pytest.skip(str(exc))


def test_every_property_is_registered():
    """Guard against a check defined but never wired into CHECKS."""
    import inspect

    defined = {n for n, fn in inspect.getmembers(P, inspect.isfunction)
               if n.startswith("check_")}
    missing = defined - set(P.CHECKS)
    assert not missing, f"properties defined but not registered via @check: {sorted(missing)}"
