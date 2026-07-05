"""Signed artifact manifest verify + bundle snapshot/rollback.

The hash-verification and snapshot/rollback paths need no signing key,
so they run in CI. Signing round-trips are exercised manually where the
release key is present.
"""

import json

from atlas.cli import artifact_manifest as am
from atlas.cli.commands import artifact


def _bundle(tmp_path):
    (tmp_path / "cost_field.pt").write_bytes(b"weights-v1")
    (tmp_path / "cx_normalization.json").write_text('{"midpoint": 1.0}')
    manifest = {
        "schema_version": 1, "model": "m", "embedding_dim": 3840,
        "artifact_sha256": {
            "cost_field.pt": am._sha256(str(tmp_path / "cost_field.pt")),
            "cx_normalization.json":
                am._sha256(str(tmp_path / "cx_normalization.json")),
        },
    }
    (tmp_path / "provenance.json").write_text(json.dumps(manifest))
    return str(tmp_path)


def test_verify_unsigned_intact_is_ok_but_noted(tmp_path):
    d = _bundle(tmp_path)
    ok, problems = am.verify_manifest(d)
    # unsigned but intact: integrity verified (ok), noted as Preview-level
    assert ok
    assert any("unsigned" in p and "Preview-level" in p for p in problems)
    assert not any("mismatch" in p for p in problems)


def test_verify_detects_tamper(tmp_path):
    d = _bundle(tmp_path)
    (tmp_path / "cost_field.pt").write_bytes(b"weights-TAMPERED")
    ok, problems = am.verify_manifest(d)
    assert not ok
    assert any("hash mismatch" in p for p in problems)


def test_verify_detects_missing_file(tmp_path):
    d = _bundle(tmp_path)
    (tmp_path / "cost_field.pt").unlink()
    ok, problems = am.verify_manifest(d)
    assert not ok
    assert any("missing on disk" in p for p in problems)


def test_snapshot_then_rollback_restores(tmp_path):
    d = _bundle(tmp_path)
    assert artifact._snapshot(d) == 0
    # simulate a bad new activation
    (tmp_path / "cost_field.pt").write_bytes(b"bad-new-bundle")
    assert artifact._rollback(d) == 0
    # original content restored
    assert (tmp_path / "cost_field.pt").read_bytes() == b"weights-v1"


def test_rollback_without_snapshot_errors(tmp_path):
    d = _bundle(tmp_path)
    assert artifact._rollback(d) == 1
