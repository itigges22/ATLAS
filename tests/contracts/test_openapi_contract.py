"""OpenAPI ↔ registered-routes parity.

Every discrete route the proxy registers (mux.HandleFunc) must appear in
the OpenAPI spec, and vice versa — so the machine-readable API doc can't
silently drift from the code. The `/` catch-all passthrough is excluded
(it's not a discrete endpoint). Parsing only.
"""

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "docs" / "schemas" / "proxy_openapi.yaml"
MAIN = REPO / "proxy" / "main.go"


def _registered_routes():
    routes = set()
    for m in re.finditer(r'mux\.HandleFunc\("([^"]+)"', MAIN.read_text()):
        path = m.group(1)
        if path == "/":
            continue  # catch-all passthrough, not a discrete endpoint
        routes.add(path)
    return routes


def _documented_paths():
    spec = yaml.safe_load(SPEC.read_text())
    return set(spec["paths"].keys())


def test_every_registered_route_is_documented():
    missing = _registered_routes() - _documented_paths()
    assert not missing, f"routes registered but absent from OpenAPI: {missing}"


def test_no_phantom_documented_paths():
    extra = _documented_paths() - _registered_routes()
    assert not extra, f"OpenAPI documents non-existent routes: {extra}"


def test_spec_version_matches_go_constant():
    spec = yaml.safe_load(SPEC.read_text())
    go = (REPO / "proxy" / "api_version.go").read_text()
    m = re.search(r'const APIVersion = "([\d.]+)"', go)
    assert m and spec["info"]["version"] == m.group(1), (
        "OpenAPI info.version must match the Go APIVersion constant")


def test_error_response_references_envelope_schema():
    text = SPEC.read_text()
    assert "error_envelope.schema.json" in text
