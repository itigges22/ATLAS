"""Typed config schema: validation + migration."""

from atlas.cli import config_schema as cs


def test_valid_config_passes():
    env = {"ATLAS_CTX_SIZE": "131072", "ATLAS_BACKEND": "cuda",
           "ATLAS_TRUST_MODE": "trusted", "ATLAS_KEEP_LLAMA_WARM": "1",
           "ATLAS_PROXY_PORT": "8090"}
    r = cs.validate(env)
    assert not r["errors"], r["errors"]


def test_type_range_enum_errors():
    env = {
        "ATLAS_CTX_SIZE": "notanint",       # type
        "ATLAS_PARALLEL_SLOTS": "999",      # range (max 64)
        "ATLAS_BACKEND": "quantum",         # enum
        "ATLAS_TRUST_MODE": "sorta",        # enum
        "ATLAS_PROXY_PORT": "99999",        # port range
        "ATLAS_KEEP_LLAMA_WARM": "maybe",   # bool
    }
    errs = cs.validate(env)["errors"]
    assert len(errs) == 6, errs


def test_unknown_key_warns():
    r = cs.validate({"ATLAS_TYPOED_KEY": "x"})
    assert any("unknown" in w for w in r["warnings"])
    assert not r["errors"]


def test_deprecated_key_warns_not_errors():
    r = cs.validate({"ATLAS_ENABLE_TRAINING": "1"})
    assert any("deprecated" in w for w in r["warnings"])
    assert not r["errors"]


def test_empty_value_is_default_not_error():
    r = cs.validate({"ATLAS_CTX_SIZE": ""})
    assert not r["errors"]


def test_migrate_drops_deprecated_and_stamps_version():
    env = {"ATLAS_CTX_SIZE": "131072", "ATLAS_REGISTRY": "old",
           "ATLAS_ENABLE_TRAINING": "1"}
    migrated, notes = cs.migrate(env)
    assert "ATLAS_REGISTRY" not in migrated
    assert "ATLAS_ENABLE_TRAINING" not in migrated
    assert migrated["ATLAS_CTX_SIZE"] == "131072"
    assert migrated["ATLAS_CONFIG_SCHEMA_VERSION"] == str(
        cs.CONFIG_SCHEMA_VERSION)
    assert len(notes) == 2


def test_env_example_keys_are_in_schema():
    """Every ATLAS_* key documented in .env.example must be known to the
    schema (else it'd falsely warn 'unknown')."""
    import re
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    text = (repo / ".env.example").read_text()
    keys = set(re.findall(r"ATLAS_[A-Z0-9_]+", text))
    unknown = sorted(k for k in keys if k not in cs.SCHEMA)
    assert not unknown, f"keys in .env.example missing from schema: {unknown}"
