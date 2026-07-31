#!/usr/bin/env python3
"""
ATLAS V3 Pipeline Service — HTTP wrapper around the V3 benchmark pipeline.

Exposes the full V3 pipeline (PlanSearch, DivSampling, BudgetForcing, BlendASC,
S*, PR-CoT, RefinementLoop, etc.) as an HTTP service that the Go proxy can
call for T2/T3 tasks.

Test cases are generated via SelfTestGen since we don't have benchmark
ground truth. The sandbox runs syntax/runtime checks on all candidates.

Streams progress events back as SSE for real-time feedback.

The pipeline itself lives in flat sibling modules: adapters.py (service
clients), scoring.py (candidate verification), symbols.py (tree-sitter
tooling), planning.py (/v3/plan), and pipeline.py (the orchestrator).
"""

import json
import os
import sys
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

# Force line-buffered stdout
sys.stdout.reconfigure(line_buffering=True)

import adapters
from adapters import ClientDisconnected, _post_pattern_outcome
from pipeline import V3PipelineService, _build_problem_from_request
from planning import generate_plan
from symbols import (structural_edit, structural_score, build_project_symbols,
                     symbol_index, cyclomatic_complexity,
                     _symbol_index_for_python_source, _STRUCTURAL_EDIT_AVAILABLE)
# The handler does not call these; tests exercise them through `import main`.
from pipeline import _candidate_by_index, _make_self_test
from planning import _score_plan
from scoring import (verify_build_command, smoke_compile_check,
                     score_candidate_per_step, _project_relative_path)
from symbols import _ast_selector_to_query

PORT = int(os.environ.get("ATLAS_V3_PORT", "8070"))


# --- HTTP Handler (SSE streaming) --------------------------------------------

pipeline = V3PipelineService()


class V3Handler(BaseHTTPRequestHandler):
    def _authorized(self) -> bool:
        """Token check for every route except /health. Constant-time
        compare; 401 bodies never echo token material."""
        if not adapters.SERVICE_TOKEN or self.path == "/health":
            return True
        import hmac
        got = self.headers.get("Authorization", "")
        want = f"Bearer {adapters.SERVICE_TOKEN}"
        if hmac.compare_digest(got, want):
            return True
        self._json_response(401, {
            "error": "unauthorized",
            "detail": "internal service auth is enabled; send "
                      "Authorization: Bearer <service-token> "
                      "(secrets/service-token)"})
        return False

    def do_POST(self):
        from structured_log import set_request_id as _set_rid
        _set_rid(self.headers.get("X-ATLAS-Request-ID", ""))
        if not self._authorized():
            return
        if self.path == "/v3/generate":
            self._handle_generate()
        elif self.path == "/v3/plan":
            self._handle_plan()
        elif self.path == "/internal/structural_edit":
            self._handle_structural_edit()
        elif self.path == "/internal/cyclomatic_complexity":
            self._handle_cyclomatic_complexity()
        elif self.path == "/internal/symbol_index":
            self._handle_symbol_index()
        elif self.path == "/internal/outline":
            self._handle_outline()
        elif self.path == "/internal/pycheck":
            self._handle_pycheck()
        elif self.path == "/internal/structural_check":
            self._handle_structural_check()
        elif self.path == "/health":
            self._json_response(200, {"status": "ok"})
        else:
            self._json_response(404, {"error": "not found"})

    def do_GET(self):
        from structured_log import set_request_id as _set_rid
        _set_rid(self.headers.get("X-ATLAS-Request-ID", ""))
        if not self._authorized():
            return
        if self.path == "/health":
            self._json_response(200, {"status": "ok", "service": "v3-pipeline"})
        else:
            self._json_response(404, {"error": "not found"})

    def _handle_generate(self):
        """Handle /v3/generate — accepts arbitrary file generation requests from Go proxy.

        Request format (V3GenerateRequest):
            file_path: str          — target file path
            baseline_code: str      — model's initial content (candidate #0)
            project_context: dict   — other files in project {path: content}
            framework: str          — detected framework
            build_command: str      — build verification command
            constraints: list[str]  — extracted requirements
            tier: int               — 2 or 3
            working_dir: str        — project root

        Response format (V3GenerateResponse):
            code: str               — winning candidate
            passed: bool            — whether it passed verification
            phase_solved: str       — which phase solved it
            candidates_tested: int
            winning_score: float
            total_tokens: int
            total_time_ms: float
        """
        content_len = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(content_len) or b"{}")
        except json.JSONDecodeError as e:
            self._json_response(400, {"error": f"invalid JSON body: {e}"})
            return

        file_path = body.get("file_path", "")
        baseline_code = body.get("baseline_code", "")
        project_context = body.get("project_context", {})
        framework = body.get("framework", "")
        build_command = body.get("build_command", "")
        constraints = body.get("constraints", [])
        tier = body.get("tier", 2)
        working_dir = body.get("working_dir", "")

        if not file_path and not baseline_code:
            self._json_response(400, {"error": "file_path or baseline_code required"})
            return

        # Build problem description from the adapter request
        problem = _build_problem_from_request(
            file_path, baseline_code, project_context,
            framework, build_command, constraints,
        )

        # Build file context for the pipeline
        files = dict(project_context) if project_context else {}

        print(f"[generate] file={file_path} framework={framework} tier=T{tier}", flush=True)
        if build_command:
            print(f"[generate] requested build command: {build_command}", flush=True)
        print(f"[generate] constraints: {constraints}", flush=True)

        # Stream V3 pipeline progress as SSE events, then final result as JSON
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit_progress(stage, detail="", **data):
            """Stream progress events to the Go proxy."""
            payload = {"stage": stage, "detail": detail}
            if data:
                payload["data"] = data
            event = json.dumps(payload)
            try:
                self.wfile.write(f"data: {event}\n\n".encode())
                self.wfile.flush()
                # Also log for debugging
                print(f"  [SSE] {stage}: {detail[:80]}", flush=True)
            except (BrokenPipeError, ConnectionResetError):
                # Client gone — flag it so pipeline.run aborts at the
                # next phase boundary instead of grinding on.
                emit_progress.disconnected = True
            except Exception as e:
                print(f"  [SSE ERROR] {e}", flush=True)

        # Run V3 pipeline with streaming progress
        try:
            result = pipeline.run(
                problem=problem,
                task_id=f"gen-{Path(file_path).stem}",
                progress_callback=emit_progress,
                files=files,
                file_path=file_path,  # PC-048: language-aware smoke check
                build_command=build_command,
                working_dir=working_dir or "/workspace",
            )
        except ClientDisconnected as e:
            print(f"[generate] pipeline aborted: {e}", flush=True)
            return

        # If baseline code was provided and pipeline didn't produce anything better,
        # use the baseline
        if not result.get("code") and baseline_code:
            result["code"] = baseline_code
            result["phase_solved"] = "baseline"

        # After baseline substitution, not before — the pattern cache must
        # see the solution that is actually returned (it saw solution=""
        # on baseline-only results when this fired earlier).
        _post_pattern_outcome(problem, result)

        # Send final result
        response = {
            "code": result.get("code", ""),
            "passed": result.get("passed", False),
            "phase_solved": result.get("phase_solved", "none"),
            "candidates_tested": result.get("candidates_generated", 0),
            "winning_score": result.get("winning_score", 0.0),
            "total_tokens": result.get("total_tokens", 0),
            "total_time_ms": result.get("total_time_ms", 0.0),
            "verification_evidence": result.get("verification_evidence", []),
        }

        final = json.dumps(response)
        try:
            self.wfile.write(f"event: result\ndata: {final}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # Client closed mid-stream (timed out, cancelled, etc).
            # See ISSUES.md PC-026.
            pass

    def _handle_plan(self):
        """Handle /v3/plan — generate a structured plan for an agent task.

        Same SSE shape as /v3/generate so the Go proxy's SSE parser can
        reuse its frame-reading logic; only the stage names and the
        final result envelope differ.

        Request:
            user_message: str       — the prompt the user typed
            working_dir: str        — proxy's container working dir
            project_context: dict   — files the agent has read so far
            tier: int               — 2 or 3
            n_candidates: int       — optional; default 3

        Response (event: result):
            steps: list[dict]
            verify_step: str | null
            candidates_tested: int
            winning_score: float
            winning_index: int
            rationale: str
            reasons: list[str]
        """
        content_len = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(content_len) or b"{}")
        except json.JSONDecodeError as e:
            self._json_response(400, {"error": f"invalid JSON body: {e}"})
            return

        user_message = body.get("user_message", "")
        working_dir = body.get("working_dir", "")
        project_context = body.get("project_context", {}) or {}
        n_candidates = int(body.get("n_candidates", 3))

        if not user_message:
            self._json_response(400, {"error": "user_message required"})
            return

        print(f"[plan] msg={user_message[:80]!r} cwd={working_dir} files={len(project_context)} n={n_candidates}",
              flush=True)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit_progress(stage, detail="", **data):
            payload = {"stage": stage, "detail": detail}
            if data:
                payload["data"] = data
            event = json.dumps(payload)
            try:
                self.wfile.write(f"data: {event}\n\n".encode())
                self.wfile.flush()
                print(f"  [SSE plan] {stage}: {detail[:80]}", flush=True)
            except BrokenPipeError:
                # best-effort: swallow on failure (caller continues)
                pass
            except Exception as e:
                print(f"  [SSE plan ERROR] {e}", flush=True)

        try:
            plan = generate_plan(
                user_message=user_message,
                working_dir=working_dir,
                project_context=project_context,
                n_candidates=n_candidates,
                progress_callback=emit_progress,
            )
        except Exception as e:
            print(f"  [plan ERROR] {e}", flush=True)
            plan = {
                "steps": [], "verify_step": None,
                "candidates_tested": 0, "winning_score": 0.0,
                "winning_index": -1,
                "rationale": f"planner failed: {e}",
                "reasons": [str(e)],
            }

        final = json.dumps(plan)
        try:
            self.wfile.write(f"event: result\ndata: {final}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # best-effort: swallow on failure (caller continues)
            pass

    def _handle_structural_edit(self):
        """POST /internal/structural_edit — friendly-selector structural replacement.

        Request:
            {"path": "...",  "source": "<full file content>",
             "selector": "function:foo" | "<body>" | ..., "content": "..."}
        Response (success):
            {"success": true, "new_content": "...", "byte_range": [start, end],
             "old_size": int, "new_size": int, "language": "python" | "html"}
        Response (failure):
            {"success": false, "error": "..."}

        Stateless transform — caller (the proxy) reads the file, sends content
        in, gets new content out. v3-service does no file IO; proxy writes
        after lens-scoring, matching write_file's flow so PC-207 lens-veto
        can still reject stub-shaped replacements.
        """
        content_len = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(content_len) or b"{}")
        except json.JSONDecodeError as e:
            self._json_response(400, {"success": False, "error": f"invalid JSON body: {e}"})
            return

        path = body.get("path", "")
        source_text = body.get("source", "")
        selector = body.get("selector", "")
        content = body.get("content", "")
        if not path or not selector or not source_text:
            self._json_response(400, {"success": False, "error": "missing required field(s): path, source, selector"})
            return

        result = structural_edit(path, source_text, selector, content)
        # Log per-call signal — matches the verbose-logging pattern we added
        # to score_candidate_per_step. Lets `docker logs atlas-v3-service-1`
        # answer "what did the model ask structural_edit to do" without SSE capture.
        if result.get("success"):
            print(
                f"  [structural_edit] {result['language']} {path} selector={selector!r} "
                f"matched bytes [{result['byte_range'][0]}-{result['byte_range'][1]}] "
                f"old={result['old_size']}B new={result['new_size']}B",
                flush=True,
            )
        else:
            print(f"  [structural_edit] FAIL path={path} selector={selector!r}: {result['error']}", flush=True)
        self._json_response(200, result)

    def _handle_symbol_index(self):
        """POST /internal/symbol_index — resolve candidate symbols to project snippets.

        Request:
            {"file_map": {"app.py": "...", "utils.py": "..."},
             "symbols": ["dashboard", "UserModel", ...],
             "max_snippets": 3, "max_lines_per_snippet": 200}
        Response:
            {"matched": [{name, kind, file, snippet, n_lines, truncated}],
             "skipped": [{name, reason}]}

        Caps default to 3 snippets / 200 lines each. Caller is the proxy,
        which extracts symbols from the user message via regex and walks
        the working directory for .py files (with its own size cap).
        """
        content_len = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(content_len) or b"{}")
        except json.JSONDecodeError as e:
            self._json_response(400, {"matched": [], "skipped": [], "error": f"invalid JSON body: {e}"})
            return

        file_map = body.get("file_map") or {}
        symbols = body.get("symbols") or []
        max_snippets = int(body.get("max_snippets", 3))
        max_lines = int(body.get("max_lines_per_snippet", 200))

        result = symbol_index(file_map, symbols, max_snippets=max_snippets, max_lines_per_snippet=max_lines)
        n_matched = len(result.get("matched", []))
        n_skipped = len(result.get("skipped", []))
        n_files = len(file_map)

        # Phase 3 (#39 point 4): when the call graph is enabled, attach each
        # matched symbol's graph neighborhood (callers / callees / impact) so
        # the proxy can inject structurally related code instead of name-matched
        # snippets alone. Additive: the "matched"/"skipped" shape is unchanged,
        # so flag-off callers see exactly today's response.
        try:
            from graph import call_graph_enabled, symbol_neighborhood, build_graph
            if call_graph_enabled() and result.get("matched"):
                # Build the project graph ONCE, then neighborhood each matched
                # symbol against it (not once per symbol).
                g = build_graph(file_map)
                related = []
                for m in result["matched"]:
                    nb = symbol_neighborhood(file_map, m["name"], graph=g)
                    if nb["callers"] or nb["callees"] or nb["impact"]:
                        related.append(nb)
                if related:
                    result["graph"] = related
        except Exception as cge:
            print(f"  [symbol_index] graph neighborhood skipped: {cge}", flush=True)

        print(
            f"  [symbol_index] {n_files} files, {len(symbols)} candidates → "
            f"matched={n_matched} skipped={n_skipped}",
            flush=True,
        )
        self._json_response(200, result)

    def _handle_pycheck(self):
        """POST /internal/pycheck — does this Python source parse?

        Request:  {"path": "app.py", "source": "<file text>"}
        Response: {"ok": bool, "error": "...", "line": N}

        Used by the proxy's edit_file path to refuse writing a .py file the
        edit would break — the same gate structural_edit applies post-splice. Pure
        compile() check, no execution.
        """
        content_len = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(content_len) or b"{}")
        except json.JSONDecodeError as e:
            self._json_response(400, {"ok": False, "error": f"invalid JSON body: {e}"})
            return
        path = body.get("path", "") or "<edit>"
        source = body.get("source", "") or ""
        try:
            compile(source, path, "exec")
            self._json_response(200, {"ok": True})
        except SyntaxError as e:
            snippet = (e.text or "").strip()
            msg = f"SyntaxError at line {e.lineno}: {e.msg}"
            if snippet:
                msg += f" (offending line: {snippet})"
            self._json_response(200, {"ok": False, "error": msg, "line": e.lineno or 0})

    def _handle_structural_check(self):
        """POST /internal/structural_check — does every direct-identifier call
        in this Python source resolve (local def, import, builtin, or a
        supplied project symbol)?

        Request:  {"path": "app.py", "source": "<file text>",
                   "project_context": {"other.py": "..."}}   # optional
        Response: {"ok": bool, "unresolved": ["render_template", ...],
                   "wildcard_imports": bool}

        Used by the proxy's structural gate on edit_file, structural_edit, and the
        write_file branches to refuse landing a .py file whose change
        introduces a NameError — a direct call to a name the file neither
        imports, defines, nor gets from builtins (#147: render_template
        called with only render_template_string imported landed as verified
        because the in-pipeline veto was skipped when no project_context was
        sent). Same resolver as the V3 structural veto; no execution.
        Returns the FULL unresolved list (no cap) — the proxy diffs
        original-vs-edited lists and a truncated list makes that comparison
        unsound. `ok:false` means the check couldn't run (tree-sitter
        missing / non-UTF-8) — the caller treats that as pass (fail-open).
        Malformed Python does NOT produce ok:false; tree-sitter parses it
        tolerantly and returns a best-effort extraction.
        """
        content_len = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(content_len) or b"{}")
        except json.JSONDecodeError as e:
            self._json_response(400, {"ok": False, "error": f"invalid JSON body: {e}"})
            return
        source = body.get("source", "") or ""
        project_symbols = build_project_symbols(body.get("project_context") or {})
        struct = structural_score(project_symbols, source, max_names=0)
        if not struct.get("ok"):
            # couldn't parse / tree-sitter unavailable → fail-open
            self._json_response(200, {"ok": False, "error": struct.get("error", "unavailable")})
            return
        self._json_response(200, {
            "ok": True,
            "unresolved": struct.get("unresolved_calls", []),
            "n_unresolved": struct.get("n_unresolved", 0),
            "wildcard_imports": struct.get("wildcard_imports", False),
        })

    def _handle_outline(self):
        """POST /internal/outline — list a file's top-level functions/classes.

        Request:  {"path": "app.py", "source": "<file text>"}
        Response: {"symbols": [{name, kind, start_line, end_line}], "supported": bool}

        Reuses the same decorator-aware tree-sitter walk structural_edit uses, so a
        symbol the outline names is selectable by structural_edit with the same name.
        Bodies are NOT returned — this is the cheap "what's in here" probe so
        the model can then read just the one function's line range instead of
        the whole file. .py only here; the proxy regex-falls-back for the rest.

        When ATLAS_CALL_GRAPH is on, each symbol also carries its intra-file
        call-graph neighborhood (`calls` / `called_by`). The outline is the
        artifact the model inspects right before it decides WHICH symbol to
        edit, so this is where structural context earns its keep: it lets the
        model follow `total_value -> item_subtotal` to a callee-rooted bug
        instead of editing the function where the symptom merely surfaces
        (issue #39). Scoped to this one file — no project-wide scan — so it's
        cheap and never misses the file in a large repo. Additive: the
        symbols/supported shape is unchanged, so flag-off callers see exactly
        today's response.
        """
        content_len = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(content_len) or b"{}")
        except json.JSONDecodeError as e:
            self._json_response(400, {"symbols": [], "supported": False, "error": f"invalid JSON body: {e}"})
            return
        path = body.get("path", "") or ""
        source = body.get("source", "") or ""
        symbols = []
        supported = False
        if path.endswith(".py") and _STRUCTURAL_EDIT_AVAILABLE:
            supported = True
            src = source.encode("utf-8")
            for name, kind, sb_, eb in _symbol_index_for_python_source(src):
                symbols.append({
                    "name": name,
                    "kind": kind,
                    "start_line": src[:sb_].count(b"\n") + 1,
                    "end_line": src[:eb].count(b"\n") + 1,
                })

        # Call-graph neighborhood (issue #39, flag-gated). Build the single-file
        # graph once and attach callers/callees to each symbol the model can see.
        if symbols:
            try:
                from graph import call_graph_enabled, symbol_neighborhood, build_graph
                if call_graph_enabled():
                    file_map = {path: source}
                    g = build_graph(file_map)
                    for s in symbols:
                        nb = symbol_neighborhood(file_map, s["name"], graph=g)
                        if nb["callees"]:
                            s["calls"] = nb["callees"]
                        if nb["callers"]:
                            s["called_by"] = nb["callers"]
            except Exception as cge:  # pragma: no cover - import/extract guard
                print(f"  [outline] call-graph neighborhood skipped: {cge}", flush=True)

        self._json_response(200, {"symbols": symbols, "supported": supported})

    def _handle_cyclomatic_complexity(self):
        """POST /internal/cyclomatic_complexity — McCabe CC for tier classification.

        Request:  {"path": "...", "source": "<full file content>"}
        Response: {"ok": true, "language": "python", "cyclomatic_complexity": 12}
                  or {"ok": false, "error": "..."}
        """
        content_len = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(content_len) or b"{}")
        except json.JSONDecodeError as e:
            self._json_response(400, {"ok": False, "error": f"invalid JSON body: {e}"})
            return

        path = body.get("path", "")
        source_text = body.get("source", "")
        if not path or source_text == "":
            self._json_response(400, {"ok": False, "error": "missing required field(s): path, source"})
            return

        result = cyclomatic_complexity(path, source_text)
        # Per-call signal — same pattern as structural_edit. Lets us correlate
        # tier upgrades to the file that triggered them in docker logs.
        if result.get("ok"):
            print(
                f"  [cc] {result['language']} {path} cc={result['cyclomatic_complexity']}",
                flush=True,
            )
        # Don't log the not-supported case — it'd flood the log on every HTML/JSON write.
        self._json_response(200, result)

    def _json_response(self, code, data):
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        except (BrokenPipeError, ConnectionResetError):
            # Client closed before we finished writing — typically a
            # docker healthcheck that hit its timeout. Not actionable.
            # See ISSUES.md PC-026.
            pass

    def log_message(self, format, *args):
        # Suppress default HTTP logging
        pass


# --- Main --------------------------------------------------------------------

class _PrivateValueStream:
    """Line-filtering wrapper for stdout/stderr: the v3 service logs
    via print(), so the stream is the serialization choke point (the
    equivalent of the root-handler filter in the FastAPI services)."""

    def __init__(self, stream):
        self._stream = stream

    def write(self, text):
        from private_values import filter_private_values
        filtered = filter_private_values(text)
        if os.environ.get("ATLAS_LOG_FORMAT", "").lower() == "json" \
                and filtered.strip():
            # Wrap each non-empty print line as a structured record so v3
            # matches the other services' JSON logs (it logs via print()).
            import json as _json
            from structured_log import get_request_id as _get_rid
            for line in filtered.splitlines():
                if not line.strip():
                    continue
                rec = {"service": "v3-service", "level": "info", "msg": line}
                rid = _get_rid()
                if rid:
                    rec["request_id"] = rid
                self._stream.write(_json.dumps(rec) + "\n")
            return
        self._stream.write(filtered)

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


if __name__ == "__main__":
    sys.stdout = _PrivateValueStream(sys.stdout)
    sys.stderr = _PrivateValueStream(sys.stderr)
    from structured_log import install as _install_logging
    _install_logging("v3-service")
    print(f"ATLAS V3 Pipeline Service starting on :{PORT}")
    print(f"  Inference:     {adapters.INFERENCE_URL}")
    print(f"  Geometric Lens: {adapters.LENS_URL}")
    print(f"  Sandbox: {adapters.SANDBOX_URL}")

    # ThreadingHTTPServer: a long pipeline call must not starve /health and
    # the /internal/* endpoints. Shared state is thread-safe by construction:
    # LLMAdapter._lock serializes the llama.cpp backend, pipeline components
    # are read-only after __init__ except telemetry appends (each stage event
    # is one small O_APPEND write; the per-task pipeline summary takes
    # pipeline._SUMMARY_LOCK), per-run state (events, candidates, adapters)
    # is local to each request, and the graph package's FileGraphCache
    # carries its own lock.
    server = ThreadingHTTPServer(("0.0.0.0", PORT), V3Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
