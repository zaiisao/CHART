"""The mainline config schema: the table is well-formed, and a bad key never reaches training."""
from __future__ import annotations

import pathlib

import pytest

from vbpm import config

CONFIG_DIR = pathlib.Path(config.__file__).parent / "configs"
DEFAULT_CONFIG = str(CONFIG_DIR / "baseline.yaml")
SHIPPED_CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))


def test_table_is_well_formed():
    """Every entry declares a type, a default of that type, and a rationale."""
    for key, spec in config.schema().items():          # schema() asserts the rest
        assert spec["description"].strip(), f"{key} has an empty description"


def test_defaults_are_the_schema_defaults():
    """defaults() is the schema with the prose stripped off -- nothing added or lost."""
    known = config.schema()
    assert config.defaults() == {key: spec["default"] for key, spec in known.items()}


@pytest.mark.parametrize("path", SHIPPED_CONFIGS, ids=lambda p: p.name)
def test_shipped_config_loads(path):
    """Every recipe in configs/ validates and names a real hooks module."""
    cfg, hooks = config.load_config(str(path), [])
    assert hooks.__name__ == f"vbpm.variants.{cfg.variant}"
    for hook in ("build_model", "optimizer", "objective", "on_epoch", "epoch_note"):
        assert hasattr(hooks, hook), f"{cfg.variant} is missing the {hook} hook"


def test_overrides_apply_and_normalise_dashes():
    cfg, _ = config.load_config(DEFAULT_CONFIG, ["epochs=2", "beta-warmup=3"])
    assert (cfg.epochs, cfg.beta_warmup) == (2, 3)


@pytest.mark.parametrize("override", [
    "epohcs=2",             # key that is in neither the schema nor the variant
    "emission=triangel",    # not in the enum
    "lr=fast",              # string where a number belongs
    "clip=true",            # bool must not pass as a number
    "epochs=2.5",           # float where an integer belongs
    "epochs=-1",            # below the declared minimum
])
def test_bad_key_refuses(override):
    """Refusal happens at parse time, before a model or a frontend is built."""
    with pytest.raises(AssertionError):
        config.load_config(DEFAULT_CONFIG, [override])


@pytest.mark.parametrize("spec", [
    {"type": "integer", "default": 1},                                    # no description
    {"type": "integer", "default": 1, "description": "d", "min": 0},      # unknown field
    {"type": "int", "default": 1, "description": "d"},                    # unknown type
    {"type": "integer", "default": -1, "description": "d", "minimum": 0},  # default breaks spec
])
def test_malformed_entry_refuses(spec, monkeypatch, tmp_path):
    """A malformed table entry is caught when the schema is read, not at first use."""
    import json
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"k": spec}))
    monkeypatch.setattr(config, "SCHEMA_PATH", path)
    with pytest.raises(AssertionError):
        config.schema()
