"""Stage-P suite wiring. One knob: which implementation the properties run against."""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import reference_p as R      # noqa: E402
import subject_p as S        # noqa: E402


def pytest_addoption(parser):
    """Register ``--impl-p``, choosing the Stage-P subject under test."""
    parser.addoption(
        "--impl-p", action="store", default="oracle", choices=("oracle",),
        help="which Stage-P implementation the properties run against. Only 'oracle' "
             "exists today: no Stage-P model has been implemented, and a guessed adapter "
             "that quietly smooths over an API mismatch is exactly how a suite starts "
             "passing for the wrong reason.")


def pytest_configure(config):
    """Declare the suite's markers."""
    for line in (
        "oracle: runs against the known-correct reference; must always pass",
        "mutation: runs against known-broken subjects; each must be caught",
        "gap: settles a question the spec text gets wrong or leaves open",
    ):
        config.addinivalue_line("markers", line)


def pytest_report_header(config):
    """Announce the subject under test."""
    return [f"stage-P subject under test: {config.getoption('--impl-p')} "
            f"(m = {R.STAGE_P_M}, fixed per SPEC_phase.md SS2.1)"]


@pytest.fixture(scope="session")
def subject_factory(request):
    """Return ``(m=None, capacity=None) -> StageP``, the implementation under test.

    ``capacity`` is a testing affordance only (SS4.5); an implementation may ignore it, in
    which case the checks that need a zero-gap q skip via ``CapacityUnsupported``.
    """
    def make(m=None, capacity=None):
        return S.oracle(m=R.STAGE_P_M if m is None else m, capacity=capacity or "full")
    return make


@pytest.fixture
def rng():
    """Seeded per test: a self-test that only passes on a lucky draw is not a self-test."""
    import numpy as np
    return np.random.default_rng(20260803)


@pytest.fixture
def theta():
    """Emission parameters as kwargs, deliberately NOT saturated.

    Near-boundary alpha/beta keep the ELBO identities being checked non-degenerate.
    """
    return {"alpha": 1.4, "beta": -0.9}
