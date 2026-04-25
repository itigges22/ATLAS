"""Invariants for the H100/H200 spot-resilient benchmark pipeline.

The cloud-pod entrypoint at `benchmarks/h200/entrypoint.sh` calls a small
set of helpers — `manifest.py`, `snapshot.sh` — to build the tarballs we
ship back from the spot. These tests pin the expectations so a future
edit can't accidentally:

  - Skip the SIGTERM trap (silent data loss on spot reclaim).
  - Drop a globbed file class from the snapshot tar (forgotten artifacts).
  - Break the manifest schema rehydrate / report-builder consume.

Lives next to the rest of the docker-compose / vLLM-cutover invariants.
No GPU / no model needed — everything is static text + small subprocess
runs against the helper scripts.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = PROJECT_ROOT / "benchmarks" / "h200" / "entrypoint.sh"
SNAPSHOT = PROJECT_ROOT / "benchmarks" / "h200" / "snapshot.sh"
MANIFEST_PY = PROJECT_ROOT / "benchmarks" / "h200" / "manifest.py"
REHYDRATE = PROJECT_ROOT / "scripts" / "rehydrate_results.sh"
REPORT_BUILDER = PROJECT_ROOT / "scripts" / "build_v31_report.py"


def test_files_exist():
    """The four scripts the spot pipeline depends on must all exist and be
    executable. A missing file here means the cloud-pod entrypoint will
    cascade-fail the moment it tries to call us."""
    for p in (ENTRYPOINT, SNAPSHOT, MANIFEST_PY, REHYDRATE, REPORT_BUILDER):
        assert p.exists(), f"missing: {p}"
        assert os.access(p, os.X_OK), f"not executable: {p}"


def test_entrypoint_traps_sigterm():
    """The SIGTERM trap is the only thing standing between a spot reclaim
    and lost work. If a future entrypoint edit removes it, the loss is
    silent (no test catches it without us pinning here)."""
    src = ENTRYPOINT.read_text()
    assert re.search(r"trap\s+on_sigterm\s+SIGTERM", src), (
        "entrypoint must trap SIGTERM. Spot pods send SIGTERM ~1-2 min "
        "before reclaim; without the trap, partial results are lost."
    )
    assert "snapshot_now sigterm" in src, (
        "SIGTERM trap must call snapshot_now before exit"
    )


def test_entrypoint_runs_periodic_snapshot_loop():
    """A 24-hour run is too long to rely on only-at-end snapshotting."""
    src = ENTRYPOINT.read_text()
    assert re.search(r"snapshot_loop\s*&", src), (
        "entrypoint must launch snapshot_loop in the background during "
        "the benchmark sweep, otherwise spot reclaim mid-sweep loses "
        "everything since container start."
    )
    assert "SNAPSHOT_LOOP_PID" in src, (
        "entrypoint must capture the snapshot loop PID so the SIGTERM "
        "handler can stop it cleanly before the final snapshot"
    )


def test_entrypoint_snapshots_at_key_transitions():
    """Each phase boundary should produce a labeled snapshot. Good
    diagnostic targets: preflight, baseline-done, atlas-done, final.

    A reviewer pulling the tarballs after a partial run sees exactly
    which phase the container finished."""
    src = ENTRYPOINT.read_text()
    for label in ("preflight", "baseline-done", "atlas-done", "final"):
        assert f"snapshot_now {label}" in src, (
            f"entrypoint should call snapshot_now {label} at the corresponding "
            f"phase transition (was searching for: 'snapshot_now {label}')"
        )


def test_snapshot_globs_required_artifact_classes():
    """The snapshot script is the ONE place we tar results. If a class
    of artifact (REPORT.md, telemetry, manifest, vllm logs, etc.) goes
    missing from the find list, the published tarball is incomplete and
    we can't tell from the outside.

    These globs need to match what the runners actually emit. New runner
    artifact classes added later need to be added here too."""
    src = SNAPSHOT.read_text()
    expected_globs = [
        # Per-benchmark
        ("responses.jsonl", "per-task records"),
        ("results.json", "scoring summary"),
        ("REPORT.md", "human-readable per-benchmark report"),
        ("sample_questions.jsonl", "sample data for reproducibility"),
        # Sweep + aggregate
        ("benchmarks/logs", "sweep log files"),
        ("benchmarks/AGGREGATE_REPORT.md", "cross-benchmark aggregate"),
        # V3 telemetry
        ("benchmark/v3", "V3 pipeline telemetry"),
        # Provenance
        ("manifest.json", "reproducibility manifest"),
        ("pip_freeze.txt", "full pip freeze"),
        # Logs from the live services
        ("/tmp/vllm-gen.log", "vLLM gen log copy"),
        ("/tmp/vllm-embed.log", "vLLM embed log copy"),
        ("/tmp/lens-service.log", "Lens service log copy"),
    ]
    for pattern, why in expected_globs:
        assert pattern in src, (
            f"snapshot.sh missing pattern `{pattern}` ({why}). "
            f"Adding new artifact classes that don't reach the tarball "
            f"makes them invisible to the rehydrated report."
        )


def test_snapshot_keeps_finals_when_pruning():
    """Snapshot prune logic keeps the last 5 periodic tarballs but MUST
    keep all `*final*` tarballs unconditionally. Otherwise a long-running
    sweep deletes its own final archive."""
    src = SNAPSHOT.read_text()
    assert re.search(r"!\s*-name\s+['\"]?\*final\*['\"]?", src), (
        "snapshot.sh prune pass must exclude *final* tarballs from deletion. "
        "If a periodic snapshot fires after the final one (e.g., race), the "
        "final tarball is the keepable one — never delete it."
    )


def test_snapshot_redirects_to_result_dir():
    """The snapshot must respect $RESULT_DIR / $ATLAS_REPO so a dev-box
    rehearsal can sandbox to a temp dir without polluting /workspace."""
    src = SNAPSHOT.read_text()
    assert "RESULT_DIR" in src
    assert "ATLAS_REPO" in src
    assert "/workspace/results" in src or 'RESULT_DIR:-' in src, (
        "snapshot.sh should default RESULT_DIR to /workspace/results "
        "but allow override via env"
    )


def test_manifest_runs_locally_without_workspace():
    """Run manifest.py against this repo with RESULT_DIR pointed at a
    tempdir — must produce valid JSON with the documented schema, NOT
    crash because /workspace doesn't exist on the dev box.

    Catches exactly the bug we hit during the live rehearsal: a hardcoded
    /workspace path leaking through env-resolution."""
    with tempfile.TemporaryDirectory() as td:
        env = os.environ.copy()
        env["ATLAS_REPO"] = str(PROJECT_ROOT)
        env["RESULT_DIR"] = td
        env["MODEL_PATH"] = "/nonexistent/model"  # trigger the absent-model path
        env["ATLAS_RUN_ID"] = "invariant-test"
        result = subprocess.run(
            ["python3", str(MANIFEST_PY)],
            env=env, capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, (
            f"manifest.py crashed on dev-box-style invocation:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        manifest_path = Path(td) / "manifest.json"
        assert manifest_path.exists(), (
            "manifest.json must land at $RESULT_DIR/manifest.json when "
            "ATLAS_MANIFEST_PATH is unset and RESULT_DIR is set"
        )
        manifest = json.loads(manifest_path.read_text())

    # Schema gates the report builder reads
    assert manifest["schema_version"] == 1
    for required in ("run_id", "snapshot_utc", "git", "python", "model", "env", "hardware"):
        assert required in manifest, f"manifest missing field: {required}"
    assert "packages" in manifest["python"]
    assert manifest["model"]["present"] is False  # we pointed at a nonexistent path


def test_manifest_redacts_secrets():
    """env_snapshot must redact HF_TOKEN / SECRET / KEY / PASSWORD vars.
    A snapshot of secrets shipped to the HF dataset would be a quiet
    credential leak."""
    with tempfile.TemporaryDirectory() as td:
        env = os.environ.copy()
        env["ATLAS_REPO"] = str(PROJECT_ROOT)
        env["RESULT_DIR"] = td
        env["MODEL_PATH"] = "/nonexistent"
        env["HF_TOKEN"] = "sk-test-secret-must-not-appear"
        env["ATLAS_API_SECRET"] = "another-secret-must-not-appear"
        env["MODE"] = "atlas_only"  # control: should appear unredacted
        subprocess.run(
            ["python3", str(MANIFEST_PY)],
            env=env, capture_output=True, text=True, timeout=60, check=True,
        )
        manifest = json.loads((Path(td) / "manifest.json").read_text())

    raw = json.dumps(manifest)
    assert "sk-test-secret-must-not-appear" not in raw, (
        "manifest leaked HF_TOKEN value. env_snapshot must redact any var "
        "whose name contains TOKEN/SECRET/KEY/PASSWORD."
    )
    assert "another-secret-must-not-appear" not in raw, (
        "manifest leaked ATLAS_API_SECRET value"
    )
    # Control: MODE captured unredacted
    assert manifest["env"].get("MODE") == "atlas_only"


def test_snapshot_round_trip_via_rehydrate():
    """Build a snapshot from this repo, rehydrate to a temp dir, confirm
    the rehydrated tree has the structure build_v31_report.py expects.

    This is the cheapest end-to-end test of the spot-pipeline path that
    doesn't need a GPU or model load."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        result_dir = td_path / "src_results"
        result_dir.mkdir()

        # Step 1 — snapshot
        env = os.environ.copy()
        env["ATLAS_REPO"] = str(PROJECT_ROOT)
        env["RESULT_DIR"] = str(result_dir)
        env["ATLAS_RUN_ID"] = "round-trip-test"
        env["SNAPSHOT_QUIET"] = "1"
        env.pop("HF_TOKEN", None)  # don't accidentally upload during tests
        env.pop("ATLAS_HF_DATASET", None)
        subprocess.run(
            ["bash", str(SNAPSHOT), "--label=test"],
            env=env, capture_output=True, text=True, timeout=180, check=True,
        )
        tarballs = list(result_dir.glob("atlas_results_*.tar.gz"))
        # latest symlink + actual tarball
        actual = [t for t in tarballs if not t.is_symlink()]
        assert actual, "snapshot.sh did not produce a tarball"
        tarball = actual[0]

        # Step 2 — content sanity
        with tarfile.open(tarball) as tf:
            names = tf.getnames()
        # manifest must be in the tarball
        assert any(n.endswith("manifest.json") for n in names), (
            "tarball missing manifest.json — without it the report builder "
            "can't fingerprint the run"
        )

        # Step 3 — rehydrate
        rehydrate_dest = td_path / "rehydrated"
        subprocess.run(
            ["bash", str(REHYDRATE), str(tarball), "-o", str(rehydrate_dest)],
            capture_output=True, text=True, timeout=60, check=True,
        )
        # Manifest must be findable where build_v31_report.py expects it
        assert (rehydrate_dest / "results" / "manifest.json").exists(), (
            "rehydrate didn't land manifest at the path build_v31_report.py "
            "looks for: <run>/results/manifest.json"
        )

        # Step 4 — report builder runs without error
        report_path = td_path / "report.md"
        result = subprocess.run(
            ["python3", str(REPORT_BUILDER),
             "--baseline", str(rehydrate_dest),
             "--atlas", str(rehydrate_dest),
             "--out", str(report_path)],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, (
            f"build_v31_report.py crashed:\nstdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        assert report_path.exists()
        report = report_path.read_text()
        assert "ATLAS V3.1 Benchmark Report" in report
        assert "Reproducibility" in report
