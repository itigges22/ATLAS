"""API-version + error-taxonomy contract.

The proxy's error-code set (Go) must match the documented/schema'd
closed set, and the version constants must be present. Prevents a Go
rename from silently desyncing the machine-readable schema clients rely
on. Parsing only.
"""

import json
import re
from pathlib import Path

from tests.contracts import go_source

REPO = Path(__file__).resolve().parents[2]

CANONICAL_CODES = [
    "unauthorized",
    "invalid_input",
    "unsupported_operation",
    "dependency_unavailable",
    "resource_limit",
    "internal_error",
]


def test_go_error_codes_match_canonical_set():
    src = go_source("proxy", 'ErrorCode = "')
    codes = re.findall(r'ErrorCode = "([a-z_]+)"', src)
    assert codes == CANONICAL_CODES, (
        f"Go error codes {codes} != canonical {CANONICAL_CODES}")


def test_error_schema_enum_matches():
    schema = json.loads(
        (REPO / "docs" / "schemas" / "error_envelope.schema.json").read_text())
    enum = schema["properties"]["error"]["enum"]
    assert enum == CANONICAL_CODES, (
        "error_envelope schema enum drifted from the canonical set")


def test_version_constants_present():
    src = go_source("proxy", "const APIVersion")
    assert re.search(r'const APIVersion = "\d+\.\d+\.\d+"', src)
    assert re.search(r'const ProtocolVersion = \d+', src)


def test_version_endpoint_registered():
    src = (REPO / "proxy" / "main.go").read_text()
    assert '"/version"' in src and "handleVersion" in src


def test_sse_envelope_schema_present():
    schema = json.loads(
        (REPO / "docs" / "schemas" / "sse_envelope.schema.json").read_text())
    assert "type" in schema["required"]
    assert "data" in schema["required"]
