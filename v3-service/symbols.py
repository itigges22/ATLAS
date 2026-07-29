"""Tree-sitter structural tooling: friendly-selector structural edits, symbol
indexing, direct-call resolution (structural_score), call-chain context, and
cyclomatic complexity."""

import re


# --- structural_edit (GH #39 v1) ----------------------------------------------------
#
# Friendly-selector-driven structural edits. Replaces the model's edit_file
# old_str/new_str pair (which truncates on long blocks: 2716-char Flask
# template hit max_tokens mid-JSON in the May 7 session) with a tree-sitter
# syntax-tree node selector.
#
# v1 supports:
#   - Python: function:NAME, class:NAME
#   - HTML:   <tag>
# Single-match enforcement: ambiguous selectors fail with a clear error so
# the model knows to be more specific. Returns new content for the proxy to
# write, preserving the lens-score-before-write pattern that write_file uses.

try:
    import tree_sitter as _ts
    import tree_sitter_python as _tsp
    import tree_sitter_html as _tsh
    _PY_LANG = _ts.Language(_tsp.language())
    _HTML_LANG = _ts.Language(_tsh.language())
    _STRUCTURAL_EDIT_AVAILABLE = True
except ImportError as _e:
    print(f"[structural_edit] tree-sitter not available: {_e} — endpoint will return 501", flush=True)
    _STRUCTURAL_EDIT_AVAILABLE = False
    _PY_LANG = None
    _HTML_LANG = None


def _ast_language_for_path(path: str):
    p = path.lower()
    if p.endswith(".py"):
        return "python", _PY_LANG
    if p.endswith((".html", ".htm")):
        return "html", _HTML_LANG
    return None, None


def _ast_selector_to_query(selector: str, language: str):
    """Translate friendly selector → (tree-sitter query string, target capture).
    Returns (None, None, error_message) for unknown selectors.
    """
    s = selector.strip()
    if language == "python":
        if s.startswith("function:"):
            name = s[len("function:"):].strip()
            if not name:
                return None, None, "selector 'function:' missing name (e.g. 'function:dashboard')"
            return (
                f'(function_definition name: (identifier) @_name (#eq? @_name "{name}")) @target',
                "target", None,
            )
        if s.startswith("class:"):
            name = s[len("class:"):].strip()
            if not name:
                return None, None, "selector 'class:' missing name (e.g. 'class:UserModel')"
            return (
                f'(class_definition name: (identifier) @_name (#eq? @_name "{name}")) @target',
                "target", None,
            )
        return None, None, (
            f"unknown selector '{selector}' for python. Supported: function:NAME, class:NAME"
        )
    if language == "html":
        if s.startswith("<") and s.endswith(">") and len(s) > 2:
            tag = s[1:-1].strip().lower()
            if not tag.replace("-", "").replace("_", "").isalnum():
                return None, None, (
                    f"selector '{selector}' has invalid tag name — use a bare "
                    f"tag like <script> or <body>, not attributes"
                )
            # tree-sitter-html parses <script> and <style> as dedicated
            # script_element / style_element nodes (their bodies are raw
            # JS/CSS, not HTML), NOT generic `element` nodes — so the generic
            # element query matches them 0 times. Target their real node type.
            if tag == "script":
                return "(script_element) @target", "target", None
            if tag == "style":
                return "(style_element) @target", "target", None
            return (
                f'(element (start_tag (tag_name) @_tag (#eq? @_tag "{tag}"))) @target',
                "target", None,
            )
        return None, None, (
            f"unknown selector '{selector}' for html. Supported: <tag> (e.g. <body>, <head>, <h1>, <script>, <style>)"
        )
    return None, None, f"unsupported language: {language}"


# GH #39 point 4: project-aware symbol resolution. Caller (proxy) extracts
# candidate symbols from the user message and ships a file_map of relevant
# project files; we tree-sitter-walk each, build a symbol index, return
# snippets for the symbols that are actually defined in the project.
# Stateless — no caching, fresh index per call. v1 supports Python only.

def _symbol_index_for_python_source(source: bytes):
    """Return list of (name, kind, start_byte, end_byte) for each top-level
    function/class definition in source. Decorator-aware: function with
    @app.route(...) returns the byte range that includes the decorator,
    so callers paste the whole decorated unit."""
    try:
        parser = _ts.Parser(_PY_LANG)
        tree = parser.parse(source)
    except Exception:
        return []
    out = []
    # Walk root children only — top-level definitions. Skip nested functions
    # and methods inside classes for v1 (they'd noise up the index without
    # adding much value for the kinds of references users actually make).
    for node in tree.root_node.children:
        target = node
        kind = None
        if node.type == "function_definition":
            kind = "function"
        elif node.type == "class_definition":
            kind = "class"
        elif node.type == "decorated_definition":
            for child in node.children:
                if child.type == "function_definition":
                    target = child
                    kind = "function"
                    break
                if child.type == "class_definition":
                    target = child
                    kind = "class"
                    break
            # Use the wrapper's byte range so the decorator is included
        if not kind:
            continue
        # Find name child of the function/class itself
        name = None
        for child in target.children:
            if child.type == "identifier":
                name = source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
                break
        if not name:
            continue
        # Use outer node's byte range (decorator wrapper if present)
        out.append((name, kind, node.start_byte, node.end_byte))
    return out


def symbol_index(file_map: dict, candidate_symbols: list, max_snippets: int = 3, max_lines_per_snippet: int = 200) -> dict:
    """Resolve candidate_symbols against a project's Python files.

    file_map: {path: source_text} of project .py files
    candidate_symbols: ['dashboard', 'UserModel', ...] extracted from user msg
    Returns:
        matched: [{name, kind, file, snippet, n_lines}] for symbols defined in the project
        skipped: [{name, reason}] for symbols mentioned but not found
    """
    if not _STRUCTURAL_EDIT_AVAILABLE:
        return {"matched": [], "skipped": [{"name": s, "reason": "tree-sitter not installed"} for s in candidate_symbols]}

    # Build {symbol_name: [(file, kind, start_byte, end_byte)]} index
    index: dict = {}
    for path, source_text in (file_map or {}).items():
        if not path.lower().endswith(".py"):
            continue
        try:
            source_bytes = source_text.encode("utf-8")
        except (UnicodeEncodeError, AttributeError):
            continue
        for name, kind, sb, eb in _symbol_index_for_python_source(source_bytes):
            index.setdefault(name, []).append((path, kind, sb, eb, source_bytes))

    matched, skipped, seen = [], [], set()
    for sym in candidate_symbols:
        if sym in seen:
            continue
        seen.add(sym)
        if len(matched) >= max_snippets:
            skipped.append({"name": sym, "reason": "snippet cap reached"})
            continue
        hits = index.get(sym)
        if not hits:
            skipped.append({"name": sym, "reason": "not defined in scanned project files"})
            continue
        if len(hits) > 1:
            # Ambiguous — multiple files define the same symbol. Skip
            # rather than guess; the model can read_file directly if the
            # context matters.
            skipped.append({"name": sym, "reason": f"ambiguous ({len(hits)} definitions)"})
            continue
        path, kind, sb, eb, source_bytes = hits[0]
        snippet_bytes = source_bytes[sb:eb]
        snippet = snippet_bytes.decode("utf-8", errors="replace")
        # Trim very long snippets — keep the head only. The model can
        # read_file for the full content if it actually needs it.
        snippet_lines = snippet.split("\n")
        truncated = False
        if len(snippet_lines) > max_lines_per_snippet:
            snippet = "\n".join(snippet_lines[:max_lines_per_snippet]) + f"\n# ... ({len(snippet_lines) - max_lines_per_snippet} more lines truncated)"
            truncated = True
        matched.append({
            "name": sym,
            "kind": kind,
            "file": path,
            "snippet": snippet,
            "n_lines": len(snippet_lines),
            "truncated": truncated,
        })
    return {"matched": matched, "skipped": skipped}


# GH #39 point 1: structural verification of V3 candidates.
#
# Sandbox tests whether code RUNS; structural verification tests whether
# the candidate's calls actually resolve. The two answer different
# questions — sandbox can pass for code with try/except ImportError
# fallbacks, lazy imports, or dead branches that never execute the
# unresolved call. Tree-sitter sees what sandbox can't.
#
# v1 supports Python only. Direct-identifier calls only (skips method
# calls like `obj.foo()` and chained calls — they'd need import-graph
# resolution that's a v2 problem). Resolution order:
#   1. Local function/class definition in the same file
#   2. Imported name (top-of-file imports only, no conditional imports)
#   3. Python builtin
#   4. Project-wide symbol (any function/class in any scanned file)
# Anything that doesn't match → unresolved. Strict: 1+ unresolved → veto.

# The COMPLETE builtin namespace, derived from the interpreter rather
# than hand-curated. A previous curated subset was missing real builtins
# (TimeoutError, ConnectionError, memoryview, breakpoint, ...), and any
# gap here is a false VETO of valid code — `exit(1)` in a new file was
# rejected as a would-be NameError. Site builtins (exit/quit/help/...)
# are added explicitly so the set doesn't depend on how this interpreter
# was started; over-crediting a shadowed builtin only makes the veto
# more lenient, never blocks valid code.
import builtins as _builtins_mod

PY_BUILTINS = frozenset(
    {n for n in dir(_builtins_mod) if not n.startswith("_")}
    | {"exit", "quit", "help", "license", "copyright", "credits",
       "__import__", "__build_class__"}
)


def _extract_python_imports(source: bytes) -> set:
    """Names introduced into the file's namespace by import statements.

    Handles `import foo`, `import foo.bar`, `import foo as bar`,
    `from foo import bar`, `from foo import bar as baz`. Doesn't track
    star imports — `from foo import *` returns nothing because we don't
    know what's in `foo` without resolving the import. Star imports are
    a known v1 gap; conservative behavior is "treat the file's calls
    as more likely unresolved" rather than silently passing them.
    """
    if not _STRUCTURAL_EDIT_AVAILABLE:
        return set()
    try:
        parser = _ts.Parser(_PY_LANG)
        tree = parser.parse(source)
    except Exception:
        return set()

    imported = set()

    def text_of(node):
        return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def walk(node):
        if node.type == "import_statement":
            for child in node.children:
                if child.type == "dotted_name":
                    # `import foo.bar` introduces `foo` into namespace
                    imported.add(text_of(child).split(".")[0])
                elif child.type == "aliased_import":
                    # `import foo as bar` — alias is the trailing identifier
                    last_ident = None
                    for c in child.children:
                        if c.type == "identifier":
                            last_ident = c
                    if last_ident is not None:
                        imported.add(text_of(last_ident))
        elif node.type == "import_from_statement":
            past_import_kw = False
            for child in node.children:
                if not past_import_kw:
                    if child.type == "import" or text_of(child) == "import":
                        past_import_kw = True
                    continue
                # After `import` keyword: dotted_name, identifier,
                # aliased_import, or wildcard_import
                if child.type == "dotted_name":
                    imported.add(text_of(child).split(".")[0])
                elif child.type == "identifier":
                    imported.add(text_of(child))
                elif child.type == "aliased_import":
                    last_ident = None
                    for c in child.children:
                        if c.type == "identifier":
                            last_ident = c
                    if last_ident is not None:
                        imported.add(text_of(last_ident))
                elif child.type == "wildcard_import":
                    # `from foo import *` — can't enumerate without
                    # resolving the import. Best we can do: bail out
                    # of strict mode for this file by adding a sentinel.
                    imported.add("*")
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return imported


def _extract_python_call_targets(source: bytes) -> list:
    """All direct-identifier call targets. Skips attribute / subscript /
    chained calls — those need full import-graph resolution and are out
    of scope for v1. Returns a list (not set) because duplicate calls
    matter when reporting — caller may dedup later."""
    if not _STRUCTURAL_EDIT_AVAILABLE:
        return []
    try:
        parser = _ts.Parser(_PY_LANG)
        tree = parser.parse(source)
    except Exception:
        return []

    out = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "call":
            # `function:` field is the first non-paren child
            for child in node.children:
                if child.type == "identifier":
                    out.append(source[child.start_byte:child.end_byte].decode("utf-8", errors="replace"))
                    break
                # attribute / subscript / lambda → skip silently
                # so we don't false-positive on `obj.method()`
                if child.type not in ("(",):
                    break
        stack.extend(node.children)
    return out


def _extract_python_top_level_defs(source: bytes) -> set:
    """Top-level function and class names defined in the file. Used as
    one input to call resolution. Skips nested functions and class
    methods — those don't introduce names into the file's top-level
    namespace."""
    if not _STRUCTURAL_EDIT_AVAILABLE:
        return set()
    try:
        parser = _ts.Parser(_PY_LANG)
        tree = parser.parse(source)
    except Exception:
        return set()

    names = set()
    for node in tree.root_node.children:
        target = node
        if node.type == "decorated_definition":
            for c in node.children:
                if c.type in ("function_definition", "class_definition"):
                    target = c
                    break
        if target.type in ("function_definition", "class_definition"):
            for c in target.children:
                if c.type == "identifier":
                    names.add(source[c.start_byte:c.end_byte].decode("utf-8", errors="replace"))
                    break
    return names


def _extract_python_bound_names(source: bytes) -> set:
    """Every name BOUND anywhere in the file — assignment targets, function
    and lambda parameters, for / with-as / except-as / comprehension
    targets, walrus, global/nonlocal, and def/class names at any nesting.

    Used to credit local callables so the structural resolver does NOT flag
    a call to a local variable, function parameter, or loop variable as
    unresolved (which would false-reject valid code — the #147 review's
    top finding). Deliberately scope-BLIND: a name bound inside one
    function is credited when called from another, which can miss a rare
    genuine cross-function NameError. That false-negative is the correct
    trade for a gate that BLOCKS writes — wrongly rejecting valid code is
    far worse than letting an uncommon bug through.
    """
    if not _STRUCTURAL_EDIT_AVAILABLE:
        return set()
    try:
        parser = _ts.Parser(_PY_LANG)
        tree = parser.parse(source)
    except Exception:
        return set()

    names = set()

    def add_pattern(node):
        # Recursively pull bare-identifier targets from a binding pattern
        # (a, (b, c), *rest = ...). Skip subscript/attribute targets
        # (a[0]=, a.b=) — those don't bind a NEW bare name.
        if node is None:
            return
        if node.type == "identifier":
            names.add(source[node.start_byte:node.end_byte].decode("utf-8", "replace"))
            return
        if node.type in ("subscript", "attribute"):
            return
        for c in node.children:
            add_pattern(c)

    stack = [tree.root_node]
    while stack:
        n = stack.pop()
        t = n.type
        if t in ("function_definition", "class_definition"):
            nm = n.child_by_field_name("name")
            if nm is not None and nm.type == "identifier":
                names.add(source[nm.start_byte:nm.end_byte].decode("utf-8", "replace"))
        if t in ("function_definition", "lambda"):
            params = n.child_by_field_name("parameters")
            if params is not None:
                pstack = list(params.children)
                while pstack:
                    p = pstack.pop()
                    if p.type == "identifier":
                        names.add(source[p.start_byte:p.end_byte].decode("utf-8", "replace"))
                    elif p.type in ("subscript", "attribute", "default_parameter",
                                    "typed_parameter", "typed_default_parameter",
                                    "list_splat_pattern", "dictionary_splat_pattern"):
                        # descend, but a default_parameter's VALUE isn't a binding —
                        # take only its leading identifier target.
                        for c in p.children:
                            if c.type == "identifier":
                                names.add(source[c.start_byte:c.end_byte].decode("utf-8", "replace"))
                                break
                            pstack.append(c)
        elif t in ("assignment", "augmented_assignment", "named_expression"):
            add_pattern(n.child_by_field_name("left") or n.child_by_field_name("name"))
        elif t in ("for_statement", "for_in_clause"):
            add_pattern(n.child_by_field_name("left"))
        elif t == "as_pattern":  # with ... as X / except ... as X
            add_pattern(n.child_by_field_name("alias") or (n.children[-1] if n.children else None))
        elif t in ("global_statement", "nonlocal_statement"):
            for c in n.children:
                if c.type == "identifier":
                    names.add(source[c.start_byte:c.end_byte].decode("utf-8", "replace"))
        stack.extend(n.children)
    return names


def build_project_symbols(file_map: dict) -> set:
    """Aggregate top-level function/class names across every .py file
    in file_map. Built once per V3 run, reused across all candidates."""
    out = set()
    for path, source_text in (file_map or {}).items():
        if not path.lower().endswith(".py"):
            continue
        try:
            out |= _extract_python_top_level_defs(source_text.encode("utf-8"))
        except Exception:
            continue
    return out


def structural_score(project_symbols, candidate_code: str,
                     max_names: int = 10) -> dict:
    """Check a candidate for unresolved direct-identifier calls.

    project_symbols: set built by build_project_symbols(file_map). Pass
    {} or set() if the project is empty / unavailable — every call
    will fall through to imports/builtins/unresolved.

    max_names caps the reported unresolved_calls list (telemetry-friendly
    default); 0 returns every name — required by callers that DIFF the
    lists (the proxy structural gate).

    Returns:
        ok: True if parse succeeded
        n_calls_total / n_unresolved: aggregate counts
        unresolved_calls: list of unique unresolved names (capped at
                          max_names unless max_names=0)
        wildcard_imports: True if the candidate has `from x import *`
                          (unresolved reporting is suppressed in that
                          case, so the list is always empty then)
    """
    if not _STRUCTURAL_EDIT_AVAILABLE:
        return {"ok": False, "error": "tree-sitter not installed"}
    try:
        candidate_bytes = candidate_code.encode("utf-8")
    except (UnicodeEncodeError, AttributeError) as e:
        return {"ok": False, "error": f"candidate not utf-8: {e}"}

    try:
        local_defs = _extract_python_top_level_defs(candidate_bytes)
        imports = _extract_python_imports(candidate_bytes)
        calls = _extract_python_call_targets(candidate_bytes)
        # Bound names (locals, params, loop/with targets, nested defs) credit
        # local callables so `fn = build(); fn()` is not flagged as a
        # NameError — #147 review finding #4/#5. Scope-blind on purpose.
        bound = _extract_python_bound_names(candidate_bytes)
    except Exception as e:
        return {"ok": False, "error": f"parse failed: {type(e).__name__}: {e}"}

    has_wildcard = "*" in imports
    if has_wildcard:
        # Star import in scope → can't reliably mark anything unresolved.
        # Be lenient and only flag calls that aren't obviously local /
        # builtin — wildcard might supply the rest.
        pass

    unresolved = []
    seen_unresolved = set()
    for name in calls:
        if name in seen_unresolved:
            continue
        if name in local_defs:
            continue
        if name in imports:
            continue
        if name in bound:  # local var / param / loop target / nested def
            continue
        if name in PY_BUILTINS:
            continue
        if name in (project_symbols or set()):
            continue
        if has_wildcard:
            # Wildcard import might supply this — treat as resolved-by-
            # wildcard rather than unresolved. False negatives possible
            # but better than blocking valid code.
            continue
        seen_unresolved.add(name)
        unresolved.append(name)

    return {
        "ok": True,
        "n_calls_total": len(calls),
        "n_unresolved": len(unresolved),
        # max_names=0 returns the FULL list. The proxy's structural gate
        # diffs original-vs-edited name lists; a truncated list makes that
        # comparison unsound in both directions on files with more
        # unresolved names than the cap.
        "unresolved_calls": unresolved[:max_names] if max_names else unresolved,
        "wildcard_imports": has_wildcard,
        "n_local_defs": len(local_defs),
        "n_imports": len(imports),
    }


# GH #39 point 3: Phase 3 repair with call-chain context.
#
# When all candidates fail sandbox and we drop to PR-CoT / refinement,
# the repair model gets `error` (raw stderr) + `code` (the failing
# candidate). It has to guess from the traceback alone what the
# failing function does inside the project, who calls it, what it
# depends on. With a call graph we can hand it that context directly.
#
# v1 approach: parse the deepest frame from a Python traceback to get
# the failing function name, then walk file_map to find:
#   - which file defines that function
#   - which other project functions call it (direct callers, 1 hop)
#   - which other project functions IT calls (direct callees, 1 hop)
# Format as a markdown block, append to the error field passed to
# PR-CoT / refinement so the repair LLM sees it as part of failure
# context.

# Python traceback frame: `File "path", line N, in funcname`
_TRACEBACK_FRAME_RE = re.compile(r'File "[^"]+", line \d+, in (\S+)')


def _failing_function_from_stderr(stderr: str):
    """Return the deepest function name in a Python traceback, or None
    if stderr doesn't look like a traceback. The deepest frame is the
    one nearest the actual error; earlier frames are callers."""
    if not stderr:
        return None
    matches = _TRACEBACK_FRAME_RE.findall(stderr)
    if not matches:
        return None
    # Filter sentinels — `<module>`, `<lambda>`, `<genexpr>` aren't
    # callable names we can look up. Walk back until we find one.
    for name in reversed(matches):
        if not name.startswith("<"):
            return name
    return None


def _python_call_targets_per_function(source: bytes):
    """Return {function_name: list[called_identifier_names]} for the
    file. Top-level functions only; class methods aggregate under their
    class name (we don't track method-level callers in v1)."""
    if not _STRUCTURAL_EDIT_AVAILABLE:
        return {}
    try:
        parser = _ts.Parser(_PY_LANG)
        tree = parser.parse(source)
    except Exception:
        return {}

    out = {}

    def text_of(node):
        return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    for node in tree.root_node.children:
        target = node
        if node.type == "decorated_definition":
            for c in node.children:
                if c.type in ("function_definition", "class_definition"):
                    target = c
                    break
        if target.type not in ("function_definition", "class_definition"):
            continue
        # Find function/class name
        name = None
        for c in target.children:
            if c.type == "identifier":
                name = text_of(c)
                break
        if not name:
            continue
        # Extract direct-identifier calls from the function body
        calls = []
        stack = list(target.children)
        while stack:
            n = stack.pop()
            if n.type == "call":
                for child in n.children:
                    if child.type == "identifier":
                        calls.append(text_of(child))
                        break
                    if child.type not in ("(",):
                        break
            stack.extend(n.children)
        out[name] = calls
    return out


def call_chain_context(file_map: dict, function_name: str, max_callers: int = 6, max_callees: int = 6) -> str:
    """Build a markdown block describing direct callers + callees of
    function_name across file_map's project. Returns empty string when
    the function isn't found anywhere — caller should skip injection
    in that case rather than dilute the error context with a useless
    'no matches' block."""
    if not function_name or not file_map or not _STRUCTURAL_EDIT_AVAILABLE:
        return ""

    # Pass 1: per-file map of {func: callees}. Also locate definition.
    per_file = {}  # path -> {func: [calls]}
    defined_in = None
    for path, source_text in file_map.items():
        if not path.lower().endswith(".py"):
            continue
        try:
            src_bytes = source_text.encode("utf-8")
        except (UnicodeEncodeError, AttributeError):
            continue
        funcs = _python_call_targets_per_function(src_bytes)
        per_file[path] = funcs
        if defined_in is None and function_name in funcs:
            defined_in = path

    if defined_in is None:
        return ""

    # Pass 2: callers — any (path, func) where func's body calls function_name
    callers = []
    for path, funcs in per_file.items():
        for fname, calls in funcs.items():
            if fname == function_name and path == defined_in:
                continue  # don't list the function as its own caller
            if function_name in calls:
                callers.append((path, fname))

    # Callees: the target function's own calls
    callees = per_file.get(defined_in, {}).get(function_name, [])
    # Dedup callees while preserving order
    seen = set()
    unique_callees = []
    for c in callees:
        if c in seen:
            continue
        seen.add(c)
        unique_callees.append(c)

    sb = [f"## Call-chain context for failing function `{function_name}`"]
    sb.append("")
    sb.append(f"Defined in: `{defined_in}`")
    sb.append("")

    if callers:
        capped = callers[:max_callers]
        sb.append(f"**Direct callers in project ({len(callers)} found):**")
        for path, fname in capped:
            sb.append(f"- `{fname}` in {path}")
        if len(callers) > max_callers:
            sb.append(f"- ... and {len(callers) - max_callers} more")
        sb.append("")
    else:
        sb.append("**Direct callers in project:** (none found — this function may be an entry point or only called by external code)")
        sb.append("")

    if unique_callees:
        capped = unique_callees[:max_callees]
        sb.append(f"**Functions called by `{function_name}` ({len(unique_callees)} unique):**")
        for c in capped:
            sb.append(f"- `{c}`")
        if len(unique_callees) > max_callees:
            sb.append(f"- ... and {len(unique_callees) - max_callees} more")
        sb.append("")
    else:
        sb.append(f"**Functions called by `{function_name}`:** (none — leaf function)")
        sb.append("")

    sb.append("Use this map to scope your fix: changing what `" + function_name + "` calls may require updating its callers; changing its callees may not.")

    return "\n".join(sb)


def cyclomatic_complexity(path: str, source_text: str) -> dict:
    """McCabe-style cyclomatic complexity from the tree-sitter syntax tree.

    Counts decision points across the whole file (sum of per-function CC,
    not strictly McCabe's per-function definition — we want one number for
    tier classification, not a per-symbol map). Decision-point set targets
    the things that actually predict V3-pipeline benefit: branches, loops,
    exception handlers, short-circuit booleans, comprehensions with filters,
    match/case clauses.

    v1 supports Python only. HTML CC isn't meaningful (markup, no real
    branching in tree-sitter's view of it — Jinja control blocks parse as
    text content). Other languages return {"ok": False} so the proxy's
    regex-based classifyFileTier stays the fallback floor.
    """
    if not _STRUCTURAL_EDIT_AVAILABLE:
        return {"ok": False, "error": "tree-sitter not installed in this build"}

    p = (path or "").lower()
    if not p.endswith(".py"):
        return {"ok": False, "error": f"cyclomatic_complexity v1 supports .py only (got {path})"}

    try:
        parser = _ts.Parser(_PY_LANG)
        tree = parser.parse(source_text.encode("utf-8"))
    except Exception as e:
        return {"ok": False, "error": f"parse failed: {type(e).__name__}: {e}"}

    # Decision-point node types in Python's tree-sitter grammar.
    # Each adds 1 to CC. `if_clause` inside a comprehension is the
    # filter clause (e.g. `[x for x in xs if x > 0]`) and counts as a branch.
    DECISION = {
        "if_statement", "elif_clause",
        "for_statement", "while_statement",
        "except_clause",
        "conditional_expression",  # ternary x if cond else y
        "boolean_operator",        # and / or short-circuit
        "case_clause",             # match-case
        "if_clause",               # comprehension filter
    }

    cc = 1  # base path
    stack = [tree.root_node]
    while stack:
        n = stack.pop()
        if n.type in DECISION:
            cc += 1
        stack.extend(n.children)

    return {"ok": True, "language": "python", "cyclomatic_complexity": cc}


def structural_edit(path: str, source_text: str, selector: str, content: str) -> dict:
    """Apply a friendly-selector structural edit. Stateless transform — caller provides
    the source bytes (read from their own filesystem) and gets back new content.
    v3-service does no file IO; the proxy reads + writes via its existing
    workspace mount, which keeps lens-score-before-write intact."""
    if not _STRUCTURAL_EDIT_AVAILABLE:
        return {"success": False, "error": "structural_edit unavailable: tree-sitter not installed in this v3-service build"}

    # Empty-content guard (defense-in-depth; the proxy also checks). Splicing
    # empty content over a node deletes it — a model that omits `content`
    # would silently remove the function instead of fixing it.
    if not content.strip():
        return {"success": False, "error": (
            f"structural_edit: content is empty — that would DELETE '{selector}', not edit it. "
            f"Provide the full replacement body of the node."
        )}

    language, lang_obj = _ast_language_for_path(path)
    if not language:
        return {"success": False, "error": (
            f"unsupported file type for structural_edit: {path}. v1 supports .py, .html, .htm — "
            f"use edit_file for other languages."
        )}

    query_str, target_cap, err = _ast_selector_to_query(selector, language)
    if err:
        return {"success": False, "error": err}

    try:
        source = source_text.encode("utf-8")
    except (UnicodeEncodeError, AttributeError) as e:
        return {"success": False, "error": f"source not valid utf-8 string: {e}"}

    try:
        parser = _ts.Parser(lang_obj)
        tree = parser.parse(source)
        query = _ts.Query(lang_obj, query_str)
        # tree_sitter ≥0.23 moved captures off Query onto QueryCursor; older
        # versions exposed Query.captures directly. Support both so the
        # service works whichever wheel pip resolves.
        if hasattr(_ts, "QueryCursor"):
            captures = _ts.QueryCursor(query).captures(tree.root_node)
        else:
            captures = query.captures(tree.root_node)
    except Exception as e:
        return {"success": False, "error": f"tree-sitter parse/query error: {type(e).__name__}: {e}"}

    targets = captures.get(target_cap, [])
    if len(targets) == 0:
        # Ground the retry in the file's REAL symbols. A weak model
        # hallucinates selectors for functions that don't exist
        # (observed: function:get_inventory_count, function:calculate_inventory
        # against a file that defines item_subtotal / total_value). The lens
        # can't catch this — the replacement text is plausible code; the
        # TARGET is the problem. Listing what's actually defined turns a
        # dead-end "verify the symbol exists" into an actionable retry.
        available = ""
        if language == "python":
            try:
                names = []
                for name, kind, _sb, _eb in _symbol_index_for_python_source(source_text.encode("utf-8")):
                    names.append(f"{kind}:{name}")
                if names:
                    available = " This file defines: " + ", ".join(names[:30]) + ". Use one of these exact selectors, or read the file to confirm."
            except Exception:
                available = ""
        return {"success": False, "error": (
            f"selector '{selector}' matched 0 nodes in {path} — that symbol does not exist in this file."
            + (available or " Read the file first to see what's defined.")
        )}
    if len(targets) > 1:
        return {"success": False, "error": (
            f"selector '{selector}' matched {len(targets)} nodes in {path}. "
            f"structural_edit requires exactly one match — use a more specific selector."
        )}

    target = targets[0]
    # Python grammar wraps decorated functions/classes in decorated_definition.
    # function:dashboard matches the inner function_definition; if its parent
    # is decorated_definition we want THAT byte range so @app.route(...) lines
    # get replaced too. Otherwise the model writes new @decorator lines and
    # the old ones stay, double-decorating the function.
    if language == "python" and target.parent is not None and target.parent.type == "decorated_definition":
        target = target.parent
    try:
        new_bytes = source[:target.start_byte] + content.encode("utf-8") + source[target.end_byte:]
        new_content = new_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        return {"success": False, "error": f"replacement produced invalid utf-8: {e}"}

    # Post-splice syntax gate (Python). Tree-sitter is error-tolerant: it
    # happily locates the node and splices in replacement content that is
    # not valid Python — observed live: a model emitted `item["id""]` and
    # `&quot;`-escaped quotes, structural_edit reported success, and a previously
    # runnable Flask app shipped with a SyntaxError. Refuse to hand back a
    # broken file; return the parse error so the model can fix its quoting
    # on the retry. Keyed off file type, not the model.
    if language == "python":
        try:
            compile(new_content, path, "exec")
        except SyntaxError as e:
            snippet = (e.text or "").strip()
            return {"success": False, "error": (
                f"structural_edit: the replacement makes {path} invalid Python — "
                f"SyntaxError at line {e.lineno}: {e.msg}"
                + (f" (offending line: {snippet})" if snippet else "")
                + '. The file was NOT modified. Check your quoting (no doubled '
                  'quotes like ["id""], no escaped \\" inside the content, no '
                  'HTML entities like &quot;) and re-emit the full node.'
            )}

    return {
        "success": True,
        "language": language,
        "selector": selector,
        "new_content": new_content,
        "byte_range": [target.start_byte, target.end_byte],
        "old_size": len(source),
        "new_size": len(new_bytes),
    }
