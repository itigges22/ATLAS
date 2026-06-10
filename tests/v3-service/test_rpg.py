"""Tests for the RPG two-stage architecture-first planner (issue #120).

The LLM is faked via a `complete_fn` that returns canned JSON, so the whole
construction path is exercised without a live llama-server.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v3-service"))

from rpg import (  # noqa: E402
    Capability,
    Edge,
    FileSpec,
    FunctionSpec,
    RPG,
    build_implementation_prompt,
    build_proposal_prompt,
    construct_rpg,
    extract_json_object,
    flatten_to_plan,
    defined_names,
    localize,
    missing_planned_signatures,
    node_constraints,
    node_drift,
    parse_capabilities,
    parse_rpg,
    planned_signatures_from_constraints,
    score_rpg,
    validate_rpg,
    verify_node_realization,
)

# ─── canned model outputs ────────────────────────────────────

PROPOSAL_JSON = json.dumps(
    {
        "capabilities": [
            {"id": "c1", "name": "Data", "parent": None},
            {"id": "c2", "name": "Loading", "parent": "c1"},
            {"id": "c3", "name": "Processing", "parent": "c1"},
        ]
    }
)

IMPL_JSON = json.dumps(
    {
        "capabilities": [
            {"id": "c1", "name": "Data", "parent": None},
            {"id": "c2", "name": "Loading", "parent": "c1"},
            {"id": "c3", "name": "Processing", "parent": "c1"},
        ],
        "files": [
            {
                "id": "f1",
                "path": "src/load.py",
                "capability": "c2",
                "functions": [{"name": "load", "signature": "def load(p: str) -> list", "summary": "read"}],
            },
            {
                "id": "f2",
                "path": "src/process.py",
                "capability": "c3",
                "functions": [{"name": "run", "signature": "def run(rows: list) -> list", "summary": "clean"}],
            },
        ],
        "edges": [{"from": "f1", "to": "f2", "kind": "data_flow", "label": "rows"}],
        "verify": "pytest tests/",
        "rationale": "load feeds process",
    }
)


def _fake_complete(proposal=PROPOSAL_JSON, impl=IMPL_JSON):
    calls = {"n": 0}

    def fn(prompt, temperature, max_tokens, seed):
        calls["n"] += 1
        return proposal if calls["n"] == 1 else impl

    return fn


# ─── JSON extraction ─────────────────────────────────────────

class TestExtractJson:
    def test_plain(self):
        assert extract_json_object('{"a": 1}') == {"a": 1}

    def test_fenced(self):
        raw = "```json\n{\"a\": 1}\n```"
        assert extract_json_object(raw) == {"a": 1}

    def test_leading_prose(self):
        assert extract_json_object('Here is the plan: {"a": 1} done') == {"a": 1}

    def test_braces_in_strings(self):
        assert extract_json_object('{"a": "has } brace"}') == {"a": "has } brace"}

    def test_garbage(self):
        assert extract_json_object("no json here") is None
        assert extract_json_object("") is None


# ─── parsing ─────────────────────────────────────────────────

class TestParsing:
    def test_parse_capabilities(self):
        caps = parse_capabilities(PROPOSAL_JSON)
        assert [c.id for c in caps] == ["c1", "c2", "c3"]
        assert caps[0].parent is None
        assert caps[1].parent == "c1"

    def test_parse_capabilities_drops_incomplete(self):
        raw = json.dumps({"capabilities": [{"id": "c1"}, {"name": "x"}, {"id": "c2", "name": "ok"}]})
        caps = parse_capabilities(raw)
        assert [c.id for c in caps] == ["c2"]

    def test_parse_rpg(self):
        g = parse_rpg(IMPL_JSON)
        assert g is not None
        assert [f.path for f in g.files] == ["src/load.py", "src/process.py"]
        assert g.files[0].functions[0].signature.startswith("def load")
        assert g.edges[0].src == "f1" and g.edges[0].dst == "f2"
        assert g.verify == "pytest tests/"

    def test_parse_rpg_carries_capabilities_forward(self):
        impl_no_caps = json.dumps(json.loads(IMPL_JSON) | {"capabilities": []})
        caps = parse_capabilities(PROPOSAL_JSON)
        g = parse_rpg(impl_no_caps, capabilities=caps)
        assert [c.id for c in g.capabilities] == ["c1", "c2", "c3"]

    def test_parse_rpg_unparseable(self):
        assert parse_rpg("not json") is None


# ─── validation / scoring ────────────────────────────────────

class TestValidation:
    def test_valid_graph(self):
        g = parse_rpg(IMPL_JSON)
        ok, issues = validate_rpg(g)
        assert ok is True
        assert issues == []

    def test_cycle_detected(self):
        g = parse_rpg(IMPL_JSON)
        g.edges.append(Edge(src="f2", dst="f1"))
        ok, issues = validate_rpg(g)
        assert ok is False
        assert any("cycle" in i for i in issues)

    def test_unknown_edge_target(self):
        g = parse_rpg(IMPL_JSON)
        g.edges.append(Edge(src="f1", dst="f99"))
        ok, issues = validate_rpg(g)
        assert ok is False
        assert any("unknown file f99" in i for i in issues)

    def test_leaf_without_file(self):
        g = RPG(
            capabilities=[Capability("c1", "Root"), Capability("c2", "Leaf", "c1")],
            files=[FileSpec("f1", "a.py", "c1", [FunctionSpec("x")])],
        )
        ok, issues = validate_rpg(g)
        assert any("leaf capability c2" in i for i in issues)

    def test_empty_graph_not_ok(self):
        ok, issues = validate_rpg(RPG())
        assert ok is False
        assert "no files" in issues

    def test_score_ordering(self):
        good = parse_rpg(IMPL_JSON)
        bad = parse_rpg(IMPL_JSON)
        bad.edges.append(Edge(src="f2", dst="f1"))  # introduces a cycle
        assert score_rpg(good) > score_rpg(bad)
        assert 0.0 <= score_rpg(good) <= 1.0


# ─── flatten to plan ─────────────────────────────────────────

class TestFlatten:
    def test_topological_order(self):
        g = parse_rpg(IMPL_JSON)
        plan = flatten_to_plan(g)
        targets = [s["target"] for s in plan["steps"]]
        # producer (load) precedes consumer (process); verify last.
        assert targets.index("src/load.py") < targets.index("src/process.py")
        assert plan["steps"][-1]["action"] == "run_command"
        assert plan["steps"][-1]["target"] == "pytest tests/"
        assert plan["verify_step"] == plan["steps"][-1]["id"]

    def test_step_ids_sequential(self):
        g = parse_rpg(IMPL_JSON)
        plan = flatten_to_plan(g)
        assert [s["id"] for s in plan["steps"]] == [f"s{i+1}" for i in range(len(plan["steps"]))]

    def test_no_verify_no_verify_step(self):
        g = parse_rpg(IMPL_JSON)
        g.verify = ""
        plan = flatten_to_plan(g)
        assert plan["verify_step"] is None
        assert all(s["action"] == "write_file" for s in plan["steps"])

    def test_duplicate_file_ids_dont_duplicate_or_drop_steps(self):
        # Duplicate ids must not inflate the topological order (which would
        # fake acyclicity and drop a file). Each unique id yields one step.
        g = RPG(
            capabilities=[Capability("c1", "Cap")],
            files=[
                FileSpec("f1", "a.py", "c1", [FunctionSpec("a")]),
                FileSpec("f1", "b.py", "c1", [FunctionSpec("b")]),
                FileSpec("f2", "c.py", "c1", [FunctionSpec("c")]),
            ],
            edges=[Edge("f1", "f2")],
        )
        plan = flatten_to_plan(g)
        write_targets = [s["target"] for s in plan["steps"] if s["action"] == "write_file"]
        assert len(write_targets) == 2  # f1 (once) + f2, not 3
        node_ids = [s["node_id"] for s in plan["steps"] if s.get("node_id")]
        assert len(node_ids) == len(set(node_ids))  # no duplicate node steps

    def test_steps_carry_node_id_and_constraints(self):
        g = parse_rpg(IMPL_JSON)
        plan = flatten_to_plan(g)
        write_steps = [s for s in plan["steps"] if s["action"] == "write_file"]
        for s in write_steps:
            assert s["node_id"]
            assert isinstance(s["constraints"], list)
        # The consumer (process.py) lists its planned signature and its input edge.
        proc = next(s for s in write_steps if s["target"] == "src/process.py")
        joined = " ".join(proc["constraints"])
        assert "def run(rows: list)" in joined
        assert "Consumes rows produced by src/load.py" in joined


class TestNodeConstraints:
    def test_capability_signatures_and_edges(self):
        g = parse_rpg(IMPL_JSON)
        cons = node_constraints(g, "f1")  # src/load.py (producer)
        joined = " ".join(cons)
        assert "Implements capability: Loading" in joined
        assert "Implement `def load(p: str) -> list`" in joined
        assert "Produces rows consumed by src/process.py" in joined

    def test_unknown_file_empty(self):
        g = parse_rpg(IMPL_JSON)
        assert node_constraints(g, "f404") == []


# ─── prompts ─────────────────────────────────────────────────

class TestPrompts:
    def test_proposal_includes_coarse_map(self):
        prompt = build_proposal_prompt("build X", coarse_map=[{"label": "class Foo (a.py)"}])
        assert "class Foo (a.py)" in prompt
        assert "coarse band" in prompt

    def test_proposal_without_coarse_map(self):
        prompt = build_proposal_prompt("build X")
        assert "build X" in prompt
        assert "coarse band" not in prompt

    def test_implementation_includes_capabilities(self):
        caps = parse_capabilities(PROPOSAL_JSON)
        prompt = build_implementation_prompt("build X", caps, project_context={"a.py": "x = 1"})
        assert '"c2"' in prompt
        assert "a.py" in prompt


# ─── two-stage construction ──────────────────────────────────

class TestConstruct:
    def test_happy_path(self):
        res = construct_rpg("build a pipeline", _fake_complete())
        assert res.ok is True
        assert res.stage_reached == "implementation"
        assert res.plan["steps"]
        assert res.rpg is not None
        assert res.score > 0.5

    def test_proposal_empty_falls_back(self):
        res = construct_rpg("x", _fake_complete(proposal='{"capabilities": []}'))
        assert res.ok is False
        assert res.stage_reached == "none"
        assert res.plan is None

    def test_impl_unparseable_falls_back(self):
        res = construct_rpg("x", _fake_complete(impl="garbage, no json"))
        assert res.ok is False
        assert res.stage_reached == "proposal"

    def test_emit_callback_invoked(self):
        events = []
        construct_rpg("x", _fake_complete(), emit=lambda s, d="", **k: events.append(s))
        assert "rpg_proposal_start" in events
        assert "rpg_done" in events

    def test_complete_fn_receives_two_calls(self):
        seeds = []

        def fn(prompt, temperature, max_tokens, seed):
            seeds.append(seed)
            return PROPOSAL_JSON if len(seeds) == 1 else IMPL_JSON

        construct_rpg("x", fn)
        assert len(seeds) == 2  # proposal then implementation


# ─── Phase 3: verification / drift / localization ────────────

class TestDefinedNames:
    def test_python_via_ast(self):
        code = "import os\n\nclass Foo:\n    def bar(self):\n        return 1\n\ndef top():\n    pass\n"
        assert defined_names(code, "m.py") == {"Foo", "bar", "top"}

    def test_python_unparseable_falls_back_to_regex(self):
        code = "def load(:\n  oops syntax\nclass Broken"
        names = defined_names(code, "m.py")
        assert "load" in names  # regex still recovers the names

    def test_go_regex(self):
        code = "package main\nfunc Load() {}\nfunc Run() {}\n"
        assert defined_names(code, "m.go") == {"Load", "Run"}

    def test_go_receiver_method(self):
        code = "package main\nfunc (s *Store) Load() error { return nil }\nfunc Top() {}\n"
        assert defined_names(code, "m.go") == {"Load", "Top"}

    def test_opaque_code_empty(self):
        assert defined_names("x = 1\ny = 2\n", "m.py") == set()


class TestFunctionNameParsing:
    def test_go_receiver_method_name(self):
        from rpg import _function_name_from_signature as fn
        assert fn("func (s *Store) Load() error") == "Load"

    def test_plain_keyword_forms(self):
        from rpg import _function_name_from_signature as fn
        assert fn("def load(p): ...") == "load"
        assert fn("async def go()") == "go"
        assert fn("func Run()") == "Run"
        assert fn("class Foo(Base)") == "Foo"
        assert fn("bare_name(x)") == "bare_name"

    def test_lone_keyword_returns_empty(self):
        # A signature that is just a keyword must not parse as a function named
        # after the keyword (which would cause a false "missing" veto).
        from rpg import _function_name_from_signature as fn
        assert fn("func") == ""
        assert fn("def") == ""
        assert fn("class") == ""

    def test_go_method_not_falsely_missing(self):
        # Regression: a planned Go receiver method that IS defined must not be
        # reported missing (previously parsed as the keyword "func").
        code = "func (s *Store) Load() error { return nil }\n"
        planned = ["func (s *Store) Load() error"]
        assert missing_planned_signatures(code, planned, "store.go") == []


class TestPlannedSignatureExtraction:
    def test_recovers_signatures_from_constraints(self):
        cons = [
            "Implements capability: Loading",
            "Implement `def load(p: str) -> list` — read",
            "Consumes rows produced by src/x.py",
            "Implement `def run(rows)`",
        ]
        assert planned_signatures_from_constraints(cons) == ["def load(p: str) -> list", "def run(rows)"]

    def test_missing_planned_signatures(self):
        code = "def load(p):\n    return []\n"
        planned = ["def load(p: str) -> list", "def run(rows)"]
        assert missing_planned_signatures(code, planned, "m.py") == ["def run(rows)"]

    def test_no_veto_when_code_opaque(self):
        # No parseable defs → don't veto (conservative).
        assert missing_planned_signatures("x = 1", ["def f()"], "m.py") == []


class TestVerifyNodeRealization:
    def test_ok_when_all_present(self):
        g = parse_rpg(IMPL_JSON)
        code = "def load(p):\n    return []\n"
        v = verify_node_realization(g, "f1", code, "src/load.py")
        assert v.ok is True
        assert v.missing_functions == []

    def test_rejects_missing_function(self):
        g = parse_rpg(IMPL_JSON)
        code = "def something_else():\n    return 1\n"
        v = verify_node_realization(g, "f1", code, "src/load.py")
        assert v.ok is False
        assert v.missing_functions

    def test_unknown_node_ok(self):
        g = parse_rpg(IMPL_JSON)
        assert verify_node_realization(g, "f404", "x = 1", "x.py").ok is True


class TestNodeDrift:
    def test_no_drift_when_realized(self):
        g = parse_rpg(IMPL_JSON)
        d = node_drift(g, "f2", "def run(rows):\n    return rows\n", "src/process.py")
        assert d.should_replan is False
        assert d.drift_score == 0.0

    def test_drift_when_function_missing(self):
        g = parse_rpg(IMPL_JSON)
        d = node_drift(g, "f2", "def unrelated():\n    return 0\n", "src/process.py")
        assert d.should_replan is True
        assert d.missing == ["run"]
        assert d.drift_score == 1.0

    def test_opaque_code_no_blind_replan(self):
        g = parse_rpg(IMPL_JSON)
        d = node_drift(g, "f2", "rows = []", "src/process.py")
        assert d.should_replan is False

    def test_threshold_tolerates_small_drift(self):
        # Two planned functions, one realized: drift_score 0.5. A threshold of
        # 0.5 should tolerate it (0.5 > 0.5 is false); 0.0 should not.
        g = RPG(
            capabilities=[Capability("c1", "Cap")],
            files=[FileSpec("f1", "a.py", "c1",
                            [FunctionSpec("keep"), FunctionSpec("drop")])],
        )
        code = "def keep():\n    return 1\n"
        assert node_drift(g, "f1", code, "a.py").should_replan is True
        assert node_drift(g, "f1", code, "a.py", replan_threshold=0.5).should_replan is False


class TestLocalize:
    def test_ranks_relevant_nodes(self):
        g = parse_rpg(IMPL_JSON)
        # "load" should surface f1 (src/load.py, fn load); "process" → f2.
        assert localize(g, "the load function is broken")[0] == "f1"
        assert localize(g, "fix processing of rows")[0] == "f2"

    def test_empty_query(self):
        g = parse_rpg(IMPL_JSON)
        assert localize(g, "") == []

    def test_no_match(self):
        g = parse_rpg(IMPL_JSON)
        assert localize(g, "zzz nonexistent terms") == []
