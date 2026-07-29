"""Integration test for the RPG wiring inside main.generate_plan.

Monkeypatches adapters.LLMAdapter so no llama-server is needed, and verifies
the ATLAS_RPG_PLANNING flag routes planning through the two-stage RPG path
(and that it stays off by default).
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "v3-service"))

import adapters  # noqa: E402
import main as v3main  # noqa: E402

PROPOSAL_JSON = json.dumps(
    {"capabilities": [{"id": "c1", "name": "Core", "parent": None},
                      {"id": "c2", "name": "Loader", "parent": "c1"}]}
)
IMPL_JSON = json.dumps(
    {
        "capabilities": [{"id": "c1", "name": "Core", "parent": None},
                         {"id": "c2", "name": "Loader", "parent": "c1"}],
        "files": [
            {"id": "f1", "path": "load.py", "capability": "c2",
             "functions": [{"name": "load", "signature": "def load() -> list", "summary": "read"}]},
        ],
        "edges": [],
        "verify": "pytest",
        "rationale": "single loader file",
    }
)


class _FakeLLM:
    """Stand-in for adapters.LLMAdapter: returns proposal then implementation."""

    _shared = {"n": 0}

    def __init__(self, *a, **k):
        pass

    def __call__(self, prompt, temperature, max_tokens, seed, thinking=None):
        _FakeLLM._shared["n"] += 1
        raw = PROPOSAL_JSON if _FakeLLM._shared["n"] == 1 else IMPL_JSON
        return raw, 100, 1.0


def test_flag_off_uses_flat_planner(monkeypatch):
    monkeypatch.delenv("ATLAS_RPG_PLANNING", raising=False)
    monkeypatch.setattr(adapters, "LLMAdapter", _FakeLLM)
    plan = v3main.generate_plan("build a loader", "/nonexistent-dir", {}, n_candidates=1)
    # Flat planner path: no RPG artifact attached.
    assert "rpg" not in plan


def test_flag_on_routes_through_rpg(monkeypatch):
    _FakeLLM._shared["n"] = 0
    monkeypatch.setenv("ATLAS_RPG_PLANNING", "1")
    monkeypatch.setattr(adapters, "LLMAdapter", _FakeLLM)
    plan = v3main.generate_plan("build a loader", "/nonexistent-dir", {}, n_candidates=1)
    assert "rpg" in plan
    assert plan["winning_index"] == 0
    assert plan["steps"]
    # The RPG file becomes a write step; verify command becomes the final step.
    targets = [s["target"] for s in plan["steps"]]
    assert "load.py" in targets
    assert plan["steps"][-1]["target"] == "pytest"
    assert plan["rpg"]["files"][0]["path"] == "load.py"
