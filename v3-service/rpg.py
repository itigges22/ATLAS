"""Repository Planning Graph (RPG) — two-stage architecture-first planning.

Implements the V3.2 RPG-style plan-then-fill flow (issue #120), grounded in
*RPG: A Repository Planning Graph for Unified and Scalable Codebase Generation*
(arXiv:2509.16198). See docs/reports/RPG_WAVELET_PLANNING_V3_2.md.

Two stages:
  A. Proposal-level  — *what* to build: a capability tree (modules -> components
     -> leaf capabilities). On L6 (existing repos) the tree is seeded with the
     wavelet coarse band so capabilities map onto real modules.
  B. Implementation  — *how*: expand leaf capabilities into files, function
     signatures, and the data-flow / ordering edges between them -> the RPG.

The module is dependency-free and LLM-agnostic: callers pass a `complete_fn`
that maps (prompt, temperature, max_tokens, seed) -> raw text, so construction
is fully unit-testable with a fake model. The RPG flattens to the existing flat
`Plan` shape (topological file order) so the proxy agent loop and
plan_adherence stay unchanged while the flag is being proven out.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

CompleteFn = Callable[[str, float, int, Optional[int]], str]


# ─── Schema ──────────────────────────────────────────────────

@dataclass
class Capability:
    id: str
    name: str
    parent: Optional[str] = None


@dataclass
class FunctionSpec:
    name: str
    signature: str = ""
    summary: str = ""


@dataclass
class FileSpec:
    id: str
    path: str
    capability: Optional[str] = None
    functions: List[FunctionSpec] = field(default_factory=list)


@dataclass
class Edge:
    src: str  # file id (producer)
    dst: str  # file id (consumer)
    kind: str = "data_flow"  # "data_flow" | "order"
    label: str = ""


@dataclass
class RPG:
    capabilities: List[Capability] = field(default_factory=list)
    files: List[FileSpec] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    verify: str = ""
    rationale: str = ""

    # ─ serialization ─
    def to_dict(self) -> dict:
        return {
            "capabilities": [
                {"id": c.id, "name": c.name, "parent": c.parent} for c in self.capabilities
            ],
            "files": [
                {
                    "id": f.id,
                    "path": f.path,
                    "capability": f.capability,
                    "functions": [
                        {"name": fn.name, "signature": fn.signature, "summary": fn.summary}
                        for fn in f.functions
                    ],
                }
                for f in self.files
            ],
            "edges": [
                {"from": e.src, "to": e.dst, "kind": e.kind, "label": e.label}
                for e in self.edges
            ],
            "verify": self.verify,
            "rationale": self.rationale,
        }


# ─── Tolerant JSON extraction ────────────────────────────────
# Mirrors v3-service main._parse_plan_json (kept here so rpg.py imports nothing
# heavy). Strips ```json fences and leading prose, then brace-matches the first
# top-level object, ignoring braces inside strings.

def extract_json_object(raw: str) -> Optional[dict]:
    if not raw:
        return None
    import re

    # Prefer the first fenced block, but fall back to the full text when it
    # doesn't parse — a fenced non-JSON example in the model's preamble must
    # not mask an unfenced JSON object after it.
    fence = re.search(r"```(?:json)?\s*\n(.*?)\n```", raw, re.DOTALL)
    if fence:
        fenced = _extract_json_from(fence.group(1))
        if fenced is not None:
            return fenced
    return _extract_json_from(raw)


def _extract_json_from(raw: str) -> Optional[dict]:
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    end = -1
    for i in range(start, len(raw)):
        c = raw[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None
    try:
        return json.loads(raw[start:end])
    except (json.JSONDecodeError, ValueError):
        return None


# ─── Parsing ─────────────────────────────────────────────────

def parse_capabilities(raw: str) -> List[Capability]:
    """Parse a Stage-A proposal response into a capability list."""
    obj = extract_json_object(raw)
    if not obj:
        return []
    out: List[Capability] = []
    for c in obj.get("capabilities") or []:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or "").strip()
        name = str(c.get("name") or "").strip()
        if not cid or not name:
            continue
        parent = c.get("parent")
        parent = str(parent).strip() if parent not in (None, "", "null") else None
        out.append(Capability(id=cid, name=name, parent=parent))
    return out


def parse_rpg(raw: str, capabilities: Optional[List[Capability]] = None) -> Optional[RPG]:
    """Parse a Stage-B implementation response into a full RPG. If the model
    omits the capability list, the Stage-A `capabilities` are carried forward."""
    obj = extract_json_object(raw)
    if obj is None:
        return None

    caps = parse_capabilities(raw)
    if not caps and capabilities:
        caps = list(capabilities)

    files: List[FileSpec] = []
    for f in obj.get("files") or []:
        if not isinstance(f, dict):
            continue
        fid = str(f.get("id") or "").strip()
        path = str(f.get("path") or "").strip()
        if not fid or not path:
            continue
        cap = f.get("capability")
        cap = str(cap).strip() if cap not in (None, "", "null") else None
        fns: List[FunctionSpec] = []
        for fn in f.get("functions") or []:
            if not isinstance(fn, dict):
                continue
            fname = str(fn.get("name") or "").strip()
            if not fname:
                continue
            fns.append(
                FunctionSpec(
                    name=fname,
                    signature=str(fn.get("signature") or "").strip(),
                    summary=str(fn.get("summary") or "").strip(),
                )
            )
        files.append(FileSpec(id=fid, path=path, capability=cap, functions=fns))

    edges: List[Edge] = []
    for e in obj.get("edges") or []:
        if not isinstance(e, dict):
            continue
        src = str(e.get("from") or e.get("src") or "").strip()
        dst = str(e.get("to") or e.get("dst") or "").strip()
        if not src or not dst:
            continue
        kind = str(e.get("kind") or "data_flow").strip() or "data_flow"
        edges.append(Edge(src=src, dst=dst, kind=kind, label=str(e.get("label") or "").strip()))

    return RPG(
        capabilities=caps,
        files=files,
        edges=edges,
        verify=str(obj.get("verify") or "").strip(),
        rationale=str(obj.get("rationale") or "").strip(),
    )


# ─── Validation & scoring ────────────────────────────────────

def _topo_order(file_ids: List[str], edges: List[Edge]) -> Tuple[List[str], bool]:
    """Kahn topological sort over file ids (producer -> consumer). Returns
    (order, acyclic). On a cycle, returns the original declaration order and
    acyclic=False."""
    # Dedup while preserving first-seen order: a duplicate id would otherwise be
    # emitted twice, inflating `order` so len(order)==len(file_ids) holds and the
    # graph is falsely reported acyclic (and flatten_to_plan would drop a file).
    file_ids = list(dict.fromkeys(file_ids))
    idset = set(file_ids)
    adj: Dict[str, List[str]] = {fid: [] for fid in file_ids}
    indeg: Dict[str, int] = {fid: 0 for fid in file_ids}
    seen_edge = set()
    for e in edges:
        if e.src not in idset or e.dst not in idset or e.src == e.dst:
            continue
        key = (e.src, e.dst)
        if key in seen_edge:
            continue
        seen_edge.add(key)
        adj[e.src].append(e.dst)
        indeg[e.dst] += 1

    # Stable: process ready nodes in declaration order.
    ready = [fid for fid in file_ids if indeg[fid] == 0]
    order: List[str] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for nxt in adj[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                # insert preserving declaration order
                ready.append(nxt)
        ready.sort(key=lambda x: file_ids.index(x))
    acyclic = len(order) == len(file_ids)
    return (order if acyclic else list(file_ids)), acyclic


def validate_rpg(rpg: RPG) -> Tuple[bool, List[str]]:
    """Structural validation. Returns (ok, issues). `ok` is True when the graph
    is usable (has files, edges resolve, acyclic); issues lists every problem
    found regardless."""
    issues: List[str] = []
    cap_ids = {c.id for c in rpg.capabilities}
    file_ids = [f.id for f in rpg.files]
    file_idset = set(file_ids)

    if not rpg.files:
        issues.append("no files")
    if len(file_idset) != len(file_ids):
        issues.append("duplicate file ids")

    for c in rpg.capabilities:
        if c.parent is not None and c.parent not in cap_ids:
            issues.append(f"capability {c.id} has unknown parent {c.parent}")

    # Leaf capabilities (no children) should map to >=1 file.
    parents = {c.parent for c in rpg.capabilities if c.parent}
    mapped_caps = {f.capability for f in rpg.files if f.capability}
    for c in rpg.capabilities:
        is_leaf = c.id not in parents
        if is_leaf and c.id not in mapped_caps:
            issues.append(f"leaf capability {c.id} ({c.name}) has no file")

    for f in rpg.files:
        if f.capability is not None and f.capability not in cap_ids:
            issues.append(f"file {f.id} references unknown capability {f.capability}")

    for e in rpg.edges:
        if e.src not in file_idset:
            issues.append(f"edge from unknown file {e.src}")
        if e.dst not in file_idset:
            issues.append(f"edge to unknown file {e.dst}")

    _, acyclic = _topo_order(file_ids, rpg.edges)
    if not acyclic:
        issues.append("file dependency graph has a cycle")

    edge_resolves = all(e.src in file_idset and e.dst in file_idset for e in rpg.edges)
    ok = bool(rpg.files) and edge_resolves and acyclic and len(file_idset) == len(file_ids)
    return ok, issues


def score_rpg(rpg: RPG) -> float:
    """Graph-shape heuristic score in [0, 1]. Higher = better-formed plan."""
    ok, issues = validate_rpg(rpg)
    score = 0.0
    if rpg.files:
        score += 0.3
    file_idset = {f.id for f in rpg.files}
    if rpg.edges and all(e.src in file_idset and e.dst in file_idset for e in rpg.edges):
        score += 0.2
    _, acyclic = _topo_order([f.id for f in rpg.files], rpg.edges)
    if acyclic:
        score += 0.2
    if rpg.verify:
        score += 0.15
    # Every file carries at least one planned function signature.
    if rpg.files and all(f.functions for f in rpg.files):
        score += 0.15
    # Penalize unresolved structure.
    score -= min(0.3, 0.05 * len(issues))
    return max(0.0, min(1.0, score))


# ─── Per-node generation constraints ─────────────────────────

def node_constraints(rpg: RPG, file_id: str) -> List[str]:
    """Derive the generation constraints for one RPG node (file).

    Once the graph fixes a node's architectural target, generating it is a
    single-problem task — these constraints are what `/v3/generate` (and the
    PlanSearch / Derivation-Chains pipeline behind it) consume to stay on the
    planned interface: implement these signatures, consume these inputs,
    produce these outputs. Phase 2 of docs/reports/RPG_WAVELET_PLANNING_V3_2.md.
    """
    by_id = {f.id: f for f in rpg.files}
    f = by_id.get(file_id)
    if f is None:
        return []
    out: List[str] = []

    cap = next((c.name for c in rpg.capabilities if c.id == f.capability), "")
    if cap:
        out.append(f"Implements capability: {cap}")

    for fn in f.functions:
        target = fn.signature or fn.name
        if not target:
            continue
        line = f"Implement `{target}`"
        if fn.summary:
            line += f" — {fn.summary}"
        out.append(line)

    for e in rpg.edges:
        if e.dst == file_id and e.src in by_id:
            what = e.label or "output"
            out.append(f"Consumes {what} produced by {by_id[e.src].path}")
        elif e.src == file_id and e.dst in by_id:
            what = e.label or "output"
            out.append(f"Produces {what} consumed by {by_id[e.dst].path}")

    return out


# ─── Flat-Plan projection ────────────────────────────────────

def flatten_to_plan(rpg: RPG) -> dict:
    """Project the RPG onto the existing flat Plan shape (proxy/types.go Plan):
    files in topological order become write/edit steps, then a verify step.
    Producers precede consumers per the data-flow edges."""
    file_ids = [f.id for f in rpg.files]
    order, _ = _topo_order(file_ids, rpg.edges)
    by_id = {f.id: f for f in rpg.files}

    steps: List[dict] = []
    for i, fid in enumerate(order):
        f = by_id[fid]
        cap = next((c.name for c in rpg.capabilities if c.id == f.capability), "")
        sig_hint = f.functions[0].signature or f.functions[0].name if f.functions else ""
        why = cap or (f.functions[0].summary if f.functions else "implement file")
        if sig_hint:
            why = f"{why} ({sig_hint})" if why else sig_hint
        steps.append(
            {
                "id": f"s{i + 1}",
                "action": "write_file",
                "target": f.path,
                "why": (why or "implement file")[:200],
                # Phase 2: the proxy threads these into the per-node
                # /v3/generate call so generation stays on the planned interface.
                "node_id": f.id,
                "constraints": node_constraints(rpg, f.id),
            }
        )

    verify_id = None
    if rpg.verify:
        verify_id = f"s{len(steps) + 1}"
        steps.append(
            {
                "id": verify_id,
                "action": "run_command",
                "target": rpg.verify,
                "why": "verify the generated repository builds / tests pass",
            }
        )

    return {
        "steps": steps,
        "verify_step": verify_id,
        "rationale": rpg.rationale or "RPG-projected plan (topological file order)",
    }


# ─── Graph-guided verification, drift & localization (Phase 3) ───

_DECL_KEYWORDS = frozenset({
    "def", "func", "function", "fn", "fun", "class", "struct",
    "interface", "trait", "type", "async", "impl", "object",
    # Modifiers and type-ish tokens that lead non-Python/Go signatures
    # ("public static void main(...)", "const handleClick = ...",
    # "virtual int run() override"). The bare-name fallback must never
    # return these as the "function name" — a wrong name here makes every
    # candidate look drifted forever, and each false drift costs a full
    # V3 regeneration.
    "public", "private", "protected", "static", "final", "abstract",
    "virtual", "override", "const", "let", "var", "export", "extern",
    "inline", "unsigned", "signed", "void", "int", "float", "double",
    "bool", "boolean", "string", "char", "long", "short", "auto", "new",
    "return", "if", "for", "while", "switch", "catch", "else", "elif",
})

# Control-flow keywords that look like `name(...)` call sites in C-style
# code; never function names.
_CALLISH_NON_NAMES = frozenset({
    "if", "for", "while", "switch", "catch", "return", "throw", "with",
    "assert", "elif", "except", "sizeof", "typeof", "new", "delete",
    "super", "print",
})


def _function_name_from_signature(sig: str) -> str:
    """Pull the declared name out of a signature string. Handles
    `def load(...)`, `async def load(...)`, `func Load(...)`, a Go receiver
    method `func (r *T) Load(...)`, `fn run(...)`, `function go(...)`,
    `class Foo`, or a bare name. Returns "" when the only token is a bare
    declaration keyword (e.g. signature is just `func`), so callers treat it
    as unknown rather than checking for a function literally named `func`."""
    import re

    s = sig.strip()
    # Go-style receiver method: `func (r *T) Name(...)` — the name follows the
    # receiver parens, so the plain `keyword name` pattern would miss it.
    m = re.search(r"\bfunc\s*\([^)]*\)\s*([A-Za-z_]\w*)", s)
    if m:
        return m.group(1)
    m = re.search(r"\b(?:async\s+)?(?:def|func|function|fn|fun|class|struct|interface|trait|type)\s+([A-Za-z_]\w*)", s)
    if m:
        return m.group(1)
    # JS/TS arrow or function-expression assignment:
    # `handleClick = () =>`, `handleClick = async function`, `handleClick: (x) =>`.
    m = re.search(r"\b([A-Za-z_]\w*)[ \t]{0,16}[:=][ \t]{0,16}(?:async[ \t]{1,16})?(?:function\b|\([^)]{0,200}\)[ \t]{0,16}=>|[A-Za-z_]\w*[ \t]{0,16}=>)", s)
    if m and m.group(1) not in _DECL_KEYWORDS:
        return m.group(1)
    # C-style `modifiers type name(args)`: the name is the token directly
    # attached to the parameter list, so scan for `name(` and take the first
    # hit that isn't a keyword ("public static void main(String[] a)" → main).
    for cm in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", s):
        if cm.group(1) not in _DECL_KEYWORDS and cm.group(1) not in _CALLISH_NON_NAMES:
            return cm.group(1)
    # Bare "name", but never a lone declaration keyword or modifier —
    # returning "" makes the caller skip enforcement rather than hunt for a
    # function literally named "public".
    m = re.match(r"[ \t]{0,16}([A-Za-z_]\w*)[ \t]{0,16}$", s)
    if m and m.group(1) not in _DECL_KEYWORDS:
        return m.group(1)
    return ""


def defined_names(code: str, filename: str) -> set:
    """Names of functions/classes defined in `code`. Python uses stdlib `ast`
    (precise, methods included); other languages use a keyword+name regex.
    Returns an empty set when nothing parses — callers treat that as "unknown,"
    not "missing.\""""
    names, _ = _defined_names_ex(code, filename)
    return names


def _defined_names_ex(code: str, filename: str) -> tuple:
    """(names, confident) — `confident` means the extraction actually saw the
    file's structure: a successful Python AST parse is confident even when it
    defines nothing (a script that defines nothing genuinely misses every
    planned def), while the regex path is confident only when it found
    definitions (an empty regex result may just mean an unsupported syntax
    style, so enforcement must not fire on it)."""
    import re

    names: set = set()
    if filename.endswith((".py", ".pyi", ".pyx")):
        import ast

        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError):
            tree = None
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(node.name)
            return names, True
        # fall through to regex on unparseable Python
    for m in re.finditer(
        r"\b(?:def|func|function|fn|fun|class|struct|interface|trait|type)\s+([A-Za-z_][\w]*)",
        code,
    ):
        names.add(m.group(1))
    # Go-style receiver methods: `func (r *T) Name(...)`.
    for m in re.finditer(r"\bfunc\s*\([^)]*\)\s*([A-Za-z_]\w*)", code):
        names.add(m.group(1))
    # JS/TS arrow functions and function expressions bound to a name:
    # `const handleClick = () => ...`, `handleClick = async function ...`,
    # `handleClick: (x) => ...` (object-literal methods).
    for m in re.finditer(
        r"\b([A-Za-z_]\w*)[ \t]{0,16}[:=][ \t]{0,16}(?:async[ \t]{1,16})?(?:function\b|\([^)]{0,200}\)[ \t]{0,16}=>|[A-Za-z_]\w*[ \t]{0,16}=>)",
        code,
    ):
        if m.group(1) not in _DECL_KEYWORDS:
            names.add(m.group(1))
    # C-style / class-method definitions: `name(args) {` at definition
    # position (Java/C/C++/C#, JS method shorthand). Control-flow keywords
    # that look call-shaped are excluded.
    for m in re.finditer(r"\b([A-Za-z_]\w*)[ \t]{0,16}\([^;{})]{0,300}\)[ \t\r\n]{0,16}\{", code):
        if (m.group(1) not in _CALLISH_NON_NAMES
                and m.group(1) not in _DECL_KEYWORDS):
            names.add(m.group(1))
    return names, bool(names)


def planned_signatures_from_constraints(constraints: List[str]) -> List[str]:
    """Recover planned signatures from the `Implement \\`...\\`` constraint
    strings that flatten_to_plan emits, so the generation veto can check them
    without the structured RPG object."""
    import re

    out: List[str] = []
    for c in constraints or []:
        m = re.search(r"Implement\s+`([^`]+)`", c)
        if m:
            out.append(m.group(1))
    return out


def missing_planned_signatures(code: str, planned: List[str], filename: str) -> List[str]:
    """Planned signatures whose declared name is NOT defined in `code`.

    Conservative in exactly one direction: when the extraction was not
    confident about the file's structure (see _defined_names_ex), returns []
    — we don't veto what we can't see. A confidently-parsed file that defines
    nothing is NOT the escape case: such a candidate genuinely misses every
    planned signature, and exempting it would veto a half-right candidate
    while keeping an all-wrong one.
    """
    if not planned:
        return []
    defined, confident = _defined_names_ex(code, filename)
    if not confident:
        return []
    missing: List[str] = []
    for sig in planned:
        name = _function_name_from_signature(sig)
        if name and name not in defined:
            missing.append(sig)
    return missing


@dataclass
class RealizationVerdict:
    ok: bool
    missing_functions: List[str]
    defined: List[str]


def verify_node_realization(rpg: RPG, file_id: str, code: str, filename: str) -> RealizationVerdict:
    """Does `code` realize the node's planned function signatures? Extends the
    #39-pt-1 structural veto from "imports survive" to "the planned interface
    exists." `ok=False` means a planned function is missing (reject candidate)."""
    by_id = {f.id: f for f in rpg.files}
    f = by_id.get(file_id)
    if f is None or not f.functions:
        return RealizationVerdict(True, [], sorted(defined_names(code, filename)))
    planned = [fn.signature or fn.name for fn in f.functions if (fn.signature or fn.name)]
    missing = missing_planned_signatures(code, planned, filename)
    return RealizationVerdict(
        ok=not missing,
        missing_functions=missing,
        defined=sorted(defined_names(code, filename)),
    )


@dataclass
class DriftReport:
    file_id: str
    missing: List[str]      # planned functions absent from the generated code
    drift_score: float      # fraction of planned functions missing, [0, 1]
    should_replan: bool     # True when structure drifted from the plan


def node_drift(rpg: RPG, file_id: str, code: str, filename: str,
               replan_threshold: float = 0.0) -> DriftReport:
    """Compare a node's planned structure against generated `code`. A planned
    function that didn't get realized is structural drift away from the RPG —
    the signal to re-plan the affected subgraph (Phase 3 drift loop).

    `should_replan` fires when the drift fraction exceeds `replan_threshold`.
    The default of 0.0 means any unrealized planned function triggers a re-plan;
    callers that tolerate small gaps (e.g. an omitted helper) can raise it."""
    by_id = {f.id: f for f in rpg.files}
    f = by_id.get(file_id)
    planned_names = [fn.name for fn in f.functions if fn.name] if f else []
    if not planned_names:
        return DriftReport(file_id=file_id, missing=[], drift_score=0.0, should_replan=False)
    defined = defined_names(code, filename)
    if not defined:
        # Unparseable / opaque — can't assess; don't trigger a re-plan blindly.
        return DriftReport(file_id=file_id, missing=[], drift_score=0.0, should_replan=False)
    missing = [n for n in planned_names if n not in defined]
    drift = len(missing) / len(planned_names)
    return DriftReport(file_id=file_id, missing=missing, drift_score=drift,
                       should_replan=drift > replan_threshold)


def _tokenize_query(text: str) -> set:
    import re

    return {t for t in re.split(r"[^A-Za-z0-9]+", text.lower()) if len(t) > 1}


def localize(rpg: RPG, query: str, k: int = 5) -> List[str]:
    """Map a request / failing test to the most relevant RPG node ids by token
    overlap of capability name + file path + function names/summaries. The
    graph-aware replacement for symbol-name-only matching (#39 pt 4)."""
    qtokens = _tokenize_query(query)
    if not qtokens:
        return []
    cap_name = {c.id: c.name for c in rpg.capabilities}
    scored: List[tuple] = []
    for f in rpg.files:
        hay = " ".join(
            [f.path, cap_name.get(f.capability, "")]
            + [fn.name for fn in f.functions]
            + [fn.summary for fn in f.functions]
        )
        overlap = len(qtokens & _tokenize_query(hay))
        if overlap:
            scored.append((overlap, f.id))
    scored.sort(key=lambda t: (-t[0], rpg.files.index(next(f for f in rpg.files if f.id == t[1]))))
    return [fid for _, fid in scored[:k]]


# ─── Prompts ─────────────────────────────────────────────────

_PROPOSAL_TEMPLATE = """You are a software architect doing PROPOSAL-LEVEL planning.
Decide WHAT to build: the capabilities (modules -> components -> leaf capabilities).
Do NOT decide files or code yet.

User goal: {user_message}
{coarse_section}
Output ONLY a JSON object, no markdown fences, no prose:
{{
  "capabilities": [
    {{"id": "c1", "name": "<high-level module>", "parent": null}},
    {{"id": "c2", "name": "<component or leaf capability>", "parent": "c1"}}
  ]
}}

Rules:
- Use a small hierarchy: top-level modules with null parent, refined into children.
- Leaf capabilities are concrete enough to map to one file each in the next stage.
- Cover the user's goal fully; do not invent unrelated scope.
- 3 to 15 capabilities total.

JSON:"""

_IMPLEMENTATION_TEMPLATE = """You are a software architect doing IMPLEMENTATION-LEVEL planning.
Given the capability tree, decide HOW to build it: the files, the key function
signatures in each file, and the data-flow / ordering edges between files.

User goal: {user_message}

Capability tree:
{capability_json}
{context_section}
Output ONLY a JSON object, no markdown fences, no prose:
{{
  "capabilities": <the same capability list, unchanged>,
  "files": [
    {{"id": "f1", "path": "<relative path>", "capability": "<capability id>",
      "functions": [{{"name": "<fn>", "signature": "<def/func signature>", "summary": "<one line>"}}]}}
  ],
  "edges": [
    {{"from": "f1", "to": "f2", "kind": "data_flow", "label": "<what flows>"}}
  ],
  "verify": "<a command that proves it works, e.g. pytest>",
  "rationale": "<one sentence on the structure>"
}}

Rules:
- Every LEAF capability maps to at least one file.
- Edges express dependencies: "from" is the producer/dependency, "to" is the
  consumer. The graph MUST be acyclic.
- Reference only file ids you define and capability ids from the tree.
- Prefer many small files over a few large ones.

JSON:"""


def _coarse_section(coarse_map: Optional[List[dict]]) -> str:
    """Render the wavelet coarse band (decompose_project output) as a grounding
    hint for the proposal stage on existing repos (L6)."""
    if not coarse_map:
        return ""
    lines = ["", "Existing repository structure (wavelet coarse band — ground capabilities on these):"]
    for p in coarse_map[:30]:
        label = p.get("label") if isinstance(p, dict) else None
        if label:
            lines.append(f"- {label}")
    lines.append("")
    return "\n".join(lines)


def _context_section(project_context: Optional[Dict[str, str]]) -> str:
    if not project_context:
        return "\n(no existing files — generating from scratch)\n"
    parts = ["", "Existing files (truncated):"]
    for path, content in list(project_context.items())[:12]:
        preview = content[:200]
        if len(content) > 200:
            preview += "\n..."
        parts.append(f"### {path}\n```\n{preview}\n```")
    parts.append("")
    return "\n".join(parts)


def build_proposal_prompt(user_message: str, coarse_map: Optional[List[dict]] = None) -> str:
    return _PROPOSAL_TEMPLATE.format(
        user_message=user_message,
        coarse_section=_coarse_section(coarse_map),
    )


def build_implementation_prompt(
    user_message: str,
    capabilities: List[Capability],
    project_context: Optional[Dict[str, str]] = None,
) -> str:
    cap_json = json.dumps(
        {"capabilities": [{"id": c.id, "name": c.name, "parent": c.parent} for c in capabilities]},
        indent=2,
    )
    return _IMPLEMENTATION_TEMPLATE.format(
        user_message=user_message,
        capability_json=cap_json,
        context_section=_context_section(project_context),
    )


# ─── Two-stage construction ──────────────────────────────────

@dataclass
class RPGResult:
    rpg: Optional[RPG]
    plan: Optional[dict]
    ok: bool
    score: float
    issues: List[str]
    stage_reached: str  # "proposal" | "implementation" | "none"


def construct_rpg(
    user_message: str,
    complete_fn: CompleteFn,
    project_context: Optional[Dict[str, str]] = None,
    coarse_map: Optional[List[dict]] = None,
    max_tokens: int = 2048,
    emit=None,
) -> RPGResult:
    """Run the two-stage RPG construction. `complete_fn(prompt, temperature,
    max_tokens, seed) -> raw text` isolates the LLM so this is unit-testable.

    Returns an RPGResult; callers fall back to the flat planner when `ok` is
    False. Never raises on model output — only on a broken `complete_fn`.
    """

    def _emit(stage: str, detail: str = "", **data):
        if emit:
            try:
                emit(stage, detail, **data)
            except TypeError:
                emit(stage, detail)

    # Stage A — proposal.
    _emit("rpg_proposal_start", "planning capabilities (what to build)")
    prop_prompt = build_proposal_prompt(user_message, coarse_map)
    prop_raw = complete_fn(prop_prompt, 0.4, max_tokens, 42)
    capabilities = parse_capabilities(prop_raw)
    if not capabilities:
        _emit("rpg_proposal_empty", "no capabilities parsed — falling back")
        return RPGResult(None, None, False, 0.0, ["proposal stage produced no capabilities"], "none")
    _emit("rpg_proposal_done", f"{len(capabilities)} capabilities", count=len(capabilities))

    # Stage B — implementation.
    _emit("rpg_impl_start", "expanding into files, signatures, and edges (how)")
    impl_prompt = build_implementation_prompt(user_message, capabilities, project_context)
    impl_raw = complete_fn(impl_prompt, 0.3, max_tokens, 43)
    rpg = parse_rpg(impl_raw, capabilities=capabilities)
    if rpg is None:
        _emit("rpg_impl_unparseable", "implementation stage didn't parse — falling back")
        return RPGResult(None, None, False, 0.0, ["implementation stage unparseable"], "proposal")

    ok, issues = validate_rpg(rpg)
    score = score_rpg(rpg)
    plan = flatten_to_plan(rpg)
    _emit(
        "rpg_done",
        f"RPG built: {len(rpg.files)} files, {len(rpg.edges)} edges, score={score:.2f}, ok={ok}",
        files=len(rpg.files),
        edges=len(rpg.edges),
        score=score,
        ok=ok,
    )
    return RPGResult(rpg=rpg, plan=plan, ok=ok, score=score, issues=issues, stage_reached="implementation")
