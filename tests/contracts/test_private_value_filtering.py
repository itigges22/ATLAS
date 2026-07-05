"""Private-value filtering parity contract.

One synthetic fixture corpus (tests/fixtures/private_value_fixtures.json)
must pass against every implementation:

  - Go: proxy/private_values.go (exercised by `go test` in CI; here we
    assert the file exists and pins the same placeholder so drift is
    caught even in Python-only runs)
  - Python x3: the canonical copy in geometric-lens plus its
    byte-identical copies in sandbox/ and v3-service/ (separate
    containers, no shared package — identity is the parity mechanism)

Every fixture value is obviously fake; no real credentials exist in
this repository's test data.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CORPUS = json.loads(
    (REPO / "tests" / "fixtures" / "private_value_fixtures.json").read_text())

PY_COPIES = [
    REPO / "geometric-lens" / "geometric_lens" / "private_values.py",
    REPO / "sandbox" / "private_values.py",
    REPO / "v3-service" / "private_values.py",
    REPO / "atlas" / "cli" / "redact.py",
]


def _load(path):
    spec = importlib.util.spec_from_file_location(
        f"pv_{path.parent.name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_python_copies_are_byte_identical():
    digests = {p: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in PY_COPIES}
    assert len(set(digests.values())) == 1, (
        "private_values.py copies drifted — edit all three together:\n"
        + "\n".join(f"  {p}: {d[:16]}" for p, d in digests.items()))


@pytest.mark.parametrize("copy_path", PY_COPIES, ids=lambda p: p.parent.name)
def test_corpus_passes(copy_path):
    mod = _load(copy_path)
    assert CORPUS["placeholder"] == mod.PLACEHOLDER
    for case in CORPUS["cases"]:
        got = mod.filter_private_values(case["input"])
        for bad in case.get("must_not_contain", []):
            assert bad not in got, (
                f"{case['name']}: {bad!r} survived filtering: {got!r}")
        for keep in case.get("must_contain", []):
            assert keep in got, (
                f"{case['name']}: context {keep!r} lost: {got!r}")
        if case.get("must_not_contain"):
            assert mod.PLACEHOLDER in got, (
                f"{case['name']}: no placeholder: {got!r}")
    for case in CORPUS["negative_cases"]:
        got = mod.filter_private_values(case["input"])
        assert got == case["input"], (
            f"{case['name']}: benign input modified: {got!r}")


def test_logging_filter_masks_records():
    import logging
    mod = _load(PY_COPIES[0])
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    lg = logging.getLogger("pv-contract-test")
    lg.setLevel(logging.INFO)
    h = Capture()
    h.addFilter(mod.PrivateValueLogFilter())
    lg.addHandler(h)
    try:
        lg.info("task failed: DATABASE_URL=postgres://demo:example@localhost/test")
    finally:
        lg.removeHandler(h)
    assert records, "record not captured"
    assert ":example@" not in records[0]
    assert mod.PLACEHOLDER in records[0]


def test_go_implementation_pins_same_placeholder():
    go_src = (REPO / "proxy" / "private_values.go").read_text()
    assert f'privateValuePlaceholder = "{CORPUS["placeholder"]}"' in go_src, (
        "Go placeholder drifted from the fixture corpus")
