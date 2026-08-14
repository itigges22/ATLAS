"""The V3 pipeline orchestrator: probe, candidate generation, sandbox
verification, the lens/structural/call-graph vetoes, candidate selection,
the repair phases, stage telemetry, and the /v3/generate problem builder."""

import base64
import fcntl
import hashlib
import json
import os
import re
import stat
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from stages.llm_client import extract_code
from stages.budget_forcing import BudgetForcing, BudgetForcingConfig
from stages import cxgx_gate
from stages.plan_search import PlanSearch, PlanSearchConfig
from stages.div_sampling import DivSampling, DivSamplingConfig
from stages.failure_analysis import FailingCandidate
from stages.pr_cot import PRCoT, PRCoTConfig
from stages.refinement_loop import (
    RefinementLoop, RefinementLoopConfig,
    can_afford_iteration, estimate_iteration_ms,
)
from stages.self_test_gen import (SelfTestGen, SelfTestGenConfig,
                                  PROVENANCE_GENERATED, PROVENANCE_TRUSTED)
from stages.candidate_selection import CandidateInfo, select_candidate

import adapters
import contract
import scoring
import symbols

BASE_TEMPERATURE = 0.6
DIVERSITY_TEMPERATURE = 0.8
MAX_TOKENS = 8192


# --- Stage telemetry ---------------------------------------------------------

# Serializes pipeline-summary appends across the ThreadingHTTPServer's
# request threads. Stage JSONL appends need no lock: each event is one
# small O_APPEND write.
_SUMMARY_LOCK = threading.Lock()

_TELEMETRY_DISABLE_VALUES = {"0", "off", "none", "disabled", "false"}

# stage name -> summary phase. Stages not listed (token/llm_*/task_type/…)
# don't contribute a phase row.
_STAGE_PHASE = {}
for _phase, _stages in {
    "probe": ("probe", "probe_light", "probe_retry", "probe_failed",
              "probe_error", "probe_scored", "probe_sandbox", "probe_pass",
              "probe_unverifiable"),
    "self_test": ("self_test_gen", "self_test_done", "self_test_error",
                  "self_test_skip", "self_test_inconclusive",
                  "self_test_untrusted"),
    "allocation": ("phase2", "phase2_allocated"),
    "generation": ("phase1", "plansearch", "plansearch_done",
                   "plansearch_error", "divsampling", "divsampling_done",
                   "divsampling_error", "divsampling_stop", "lens_per_step"),
    "sandbox": ("sandbox_test", "sandbox_pass", "sandbox_fail",
                "sandbox_done", "smoke_check", "interactive_lint",
                "self_test_verify", "build_verify",
                "build_verify_unavailable"),
    "veto": ("lens_veto", "structural_veto", "call_graph_veto"),
    "selection": ("selected", "consensus", "consensus_ranking"),
    "repair_pr_cot": ("phase3", "call_chain_context", "pr_cot",
                      "pr_cot_pass", "pr_cot_failed", "pr_cot_error"),
    "repair_refinement": ("refinement", "refinement_pass",
                          "refinement_failed", "refinement_error",
                          "refinement_verify_failed", "refinement_skip"),
    "fallback": ("fallback", "fallback_all_vetoed", "fallback_unverified",
                 "budget_exhausted", "budget_no_verified_candidate"),
}.items():
    for _s in _stages:
        _STAGE_PHASE[_s] = _phase

_VETO_STAGES = frozenset(("lens_veto", "structural_veto", "call_graph_veto"))


def _remaining_budget_ms(start: float) -> Optional[float]:
    """Remaining wall-clock (ms) in this run's ATLAS_V3_TIMEOUT budget.

    The proxy's V3 bridge abandons a live pipeline call after
    ``ATLAS_V3_TIMEOUT`` seconds (default 300; 0 disables the cap).
    The service reads the same knob so late phases can skip work the
    bridge would abandon mid-flight anyway. Returns None when the cap
    is disabled.
    """
    raw = os.environ.get("ATLAS_V3_TIMEOUT", "").strip()
    try:
        seconds = int(raw) if raw else 300
    except ValueError:
        seconds = 300
    if seconds <= 0:
        return None
    return seconds * 1000.0 - (time.time() - start) * 1000.0


def _resolve_telemetry_dir() -> Optional[Path]:
    """Resolve the stage-telemetry directory for the live service.

    ``ATLAS_V3_TELEMETRY_DIR`` names the directory; a disable value
    (``0``/``off``/``none``/``disabled``/``false``) turns telemetry off;
    unset/empty falls back to ``/data/telemetry`` when writable (the
    compose volume), else telemetry is disabled. Resolution never
    raises — telemetry must not break generation.
    """
    configured = os.environ.get("ATLAS_V3_TELEMETRY_DIR", "").strip()
    if configured.lower() in _TELEMETRY_DISABLE_VALUES:
        return None
    candidate = Path(configured) if configured else Path("/data/telemetry")
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / ".write_probe"
        probe.touch()
        probe.unlink()
        return candidate
    except OSError as e:
        if configured:
            print(f"  [telemetry] {candidate} not writable ({e}) — "
                  f"stage telemetry disabled", flush=True)
        return None


# --- Benchmark-only candidate-pool capture -----------------------------------
#
# Verification hands the rest of the pipeline one bit: accepted or not. A
# candidate that passed 9 of 10 generated cases and one that passed none
# arrive at the contract record identically, and the rejected candidates'
# bytes are gone the moment the run returns. That is what made the measured
# "every suite scored 0/N" pattern unattributable — nothing retained could
# separate a wrong candidate from a wrong answer key.
#
# This writes the pool, the per-case expected/actual pairs and the selection
# identities to one append-only file so the question can be settled offline.
# It is a measurement instrument, not a feature: unset means no file is
# opened, capture decides nothing, and every failure mode disables capture
# rather than touching the run. Candidate source appears here and nowhere
# else — never in a response, an SSE frame, telemetry or a log line.

CAPTURE_ENV = "ATLAS_V3_CAPTURE_POOL"
CAPTURE_SCHEMA = "atlas.v3_candidate_capture/1"

# Sizing, from the Suite A run this instrument exists to explain: candidates
# averaged 1.9 KB (p95 4.1 KB, max 13 KB), at most 3 per generation call and
# roughly 2 calls per task. A candidate_evaluation is the base64 of that plus
# ≤5 cases and the contract record — ~10 KB typical, ~25 KB worst — so a
# 12-task diagnostic is a few megabytes. 64 MiB is ~25x that headroom and
# still bounds a runaway loop to something the telemetry volume absorbs.
CAPTURE_DEFAULT_MAX_BYTES = 64 * 1024 * 1024
# One pathological candidate must not consume the whole budget. Its identity
# is still recorded; only the bytes are dropped, and the record says so.
CAPTURE_MAX_RECORD_BYTES = 2 * 1024 * 1024
# Execution output is diagnostic context, not evidence: tails are enough.
CAPTURE_MAX_FIELD_BYTES = 4096

_CAPTURE_GOT = re.compile(r"got\s+(.*)", re.DOTALL)
_CAPTURE_TIMEOUT = re.compile(r"tim(?:ed?\s*)?out", re.IGNORECASE)


def _capture_clip(text: str) -> str:
    text = text or ""
    if len(text) <= CAPTURE_MAX_FIELD_BYTES:
        return text
    return text[:CAPTURE_MAX_FIELD_BYTES] + "…[clipped]"


def _capture_case(index, tc, ran_ok, out, err, harness_error=""):
    """One generated case as observed, with the failure kinds kept apart.

    A wrong answer, a crash, a timeout and a case the harness could not even
    build are four different findings about a run that all reduce to "not
    passed" in production. Classification is read off what the sandbox
    already returned; nothing extra is executed.
    """
    inp = (getattr(tc, "input_str", "") or "").strip()
    exp = (getattr(tc, "expected_output", "") or "").strip()
    out, err = out or "", err or ""
    passed = bool(ran_ok) and "SELF_TEST_PASS" in out
    # The generated assertion reports the value it saw as `got <actual>`, so
    # the candidate's real output survives in the failure text.
    match = None if passed else _CAPTURE_GOT.search(err)
    actual = match.group(1).strip() if match else None
    if harness_error:
        outcome = "harness_error"
    elif passed:
        outcome = "pass"
    elif not inp or not exp:
        outcome = "generated_test_malformed"
    elif _CAPTURE_TIMEOUT.search(err):
        outcome = "timeout"
    elif actual is not None or "AssertionError" in err:
        outcome = "wrong_answer"
    else:
        outcome = "execution_error"
    return {"index": index, "input": inp, "expected": exp, "actual": actual,
            "passed": passed, "outcome": outcome,
            "stdout": _capture_clip(out), "stderr": _capture_clip(err),
            "harness_error": harness_error}


class _PoolCapture:
    """An append-only JSONL sink for one run's candidate pool.

    Disabled is the default and the only state that costs anything: with no
    path configured every method is a no-op. Once enabled, no error it can
    hit — encoding, a full disk, a revoked directory — is allowed to reach
    the pipeline; capture turns itself off and the diagnostic record is
    invalid instead of the run being different.
    """

    def __init__(self, path: Optional[Path] = None, fd: Optional[int] = None,
                 error: str = ""):
        self.path = path
        self._fd = fd
        self.enabled = fd is not None
        self.write_error = error
        self.records_written = 0
        self.bytes_written = 0
        self.limit_reached = False
        self._lock = threading.Lock()
        self._seen = set()
        self._oracle: Dict[str, Dict[str, Any]] = {}
        self._selection: Optional[Dict[str, Any]] = None
        self._next_index = 0
        self._session_id = ""

    # -- lifecycle ----------------------------------------------------------

    @classmethod
    def disabled(cls) -> "_PoolCapture":
        return cls()

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "_PoolCapture":
        """Open the configured sink, or stay inert.

        Every rejection here is a refusal to write, never an exception: a
        relative path, a missing parent, a symlink at the final component, or
        anything that is not a regular file. O_NOFOLLOW is what stops the
        configured path being aimed at something else through a link that was
        planted first.
        """
        configured = (env or os.environ).get(CAPTURE_ENV, "").strip()
        if not configured:
            return cls()
        path = Path(configured)
        if not path.is_absolute():
            return cls(error="capture path must be absolute")
        if not path.parent.is_dir():
            return cls(error="capture parent directory does not exist")
        try:
            fd = os.open(str(path),
                         os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW,
                         0o600)
        except OSError as exc:
            return cls(error=f"open: {exc}")
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise OSError("capture path is not a regular file")
            # A file that already existed may carry wider permissions than
            # the create mode would have given it.
            if stat.S_IMODE(info.st_mode) != 0o600:
                os.fchmod(fd, 0o600)
        except OSError as exc:
            os.close(fd)
            return cls(error=f"validate: {exc}")
        return cls(path=path, fd=fd)

    def close(self, result: Optional[Dict[str, Any]] = None) -> None:
        """Write the delivered artifact, the selection summary and the status
        line, then release the descriptor."""
        if not self.enabled:
            return
        try:
            if result:
                code = result.get("code") or ""
                if code:
                    self.note_candidate(
                        role="delivered", index=None, code=code,
                        accepted=bool(result.get("passed")),
                        record=result.get("evidence_record"), phase="delivered")
            self._write_selection(result)
        except Exception as exc:                       # noqa: BLE001
            self.write_error = self.write_error or f"close: {exc}"
        self._write_status()
        fd, self._fd = self._fd, None
        self.enabled = False
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    # -- collection ---------------------------------------------------------

    def bind(self, session_id: str) -> None:
        self._session_id = session_id or ""

    def note_oracle(self, code: str, cases, passed: int, total: int) -> None:
        """Keep a verification run's per-case detail, keyed by the bytes it
        judged. Production keeps only the boolean this discards."""
        if not self.enabled:
            return
        self._oracle[contract.content_hash(code)] = {
            "suite_available": True, "cases": list(cases),
            "cases_passed": passed, "cases_total": total}

    def note_candidate(self, *, role: str, index, code: str, accepted: bool,
                       record, phase: str, lens=None) -> None:
        """One candidate_evaluation record.

        Deduplicated on (role, bytes): candidate zero is evaluated twice by
        design — once as the probe, once as pool member 0 — and one record
        per role is the honest count of distinct candidates, not of visits.
        """
        if not self.enabled or not code:
            return
        raw = code.encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        key = (role, digest)
        if key in self._seen:
            return
        self._seen.add(key)
        if index is None:
            index = self._next_index
        self._next_index = max(self._next_index, int(index) + 1)
        oracle = self._oracle.get(digest) or {
            "suite_available": False, "cases": [],
            "cases_passed": 0, "cases_total": 0}
        payload = {
            "type": "candidate_evaluation",
            "session_id": self._session_id, "phase": phase,
            "candidate_index": index, "role": role,
            "code_b64": base64.b64encode(raw).decode("ascii"),
            "code_sha256": digest, "code_bytes": len(raw),
            "adapter_id": (record or {}).get("adapter_id", ""),
            "contract_record": record,
            "contract_record_source":
                "production" if record else "not_built_in_production",
            "oracle": oracle,
            "accepted": bool(accepted),
            "lens": lens or {},
        }
        self.write(payload)

    def note_incumbent(self, *, code: str, record, adapter: str,
                       evaluation: str = "evaluated") -> None:
        """The artifact V3 was asked to improve on, measured the same way.

        It lives in a SHADOW pool of its own. The live pool, the lens
        choice, the contract selection, the returned code, the envelope and
        Go's authorization never see it: this exists to answer "was the
        selected candidate better than the thing it would replace", which
        selection cannot answer today because the incumbent has never
        carried a record.
        """
        if not self.enabled or not code:
            return
        raw = code.encode("utf-8")
        self.write({
            "type": "incumbent_observation",
            "session_id": self._session_id,
            "role": "incumbent_baseline",
            "pool": "shadow_comparison",
            "code_b64": base64.b64encode(raw).decode("ascii"),
            "code_sha256": hashlib.sha256(raw).hexdigest(),
            "code_bytes": len(raw),
            "adapter_id": adapter,
            "contract_record": record,
            "evaluation": evaluation,
            "influences_live_selection": False,
        })

    def note_consensus(self, record) -> None:
        """Per-input clusters with their exact members.

        Agreement counts alone cannot say WHICH candidates agreed, so a
        ranking could never be checked against an independent verdict. The
        generated expected outputs are not part of this and are not read to
        build it.
        """
        if not self.enabled or not record:
            return
        payload = dict(record)
        payload["type"] = "consensus_clusters"
        payload["session_id"] = self._session_id
        self.write(payload)

    def note_pool(self, *, phase: str, pool, lens_index=None,
                  evidence_index=None, verified_index=None,
                  status: str = "", reason: str = "", tied: int = 0,
                  incomparable: int = 0, ineligible: int = 0) -> None:
        """What the selectable pool was and what each selector picked."""
        if not self.enabled:
            return
        self._selection = {
            "phase": phase,
            "pool": [contract.content_hash(c.get("code") or "") for c in pool],
            "pool_indices": [c.get("index") for c in pool],
            "lens_index": lens_index, "evidence_index": evidence_index,
            "verified_index": verified_index,
            "selection_status": status, "selection_reason": reason,
            "tied_count": tied, "incomparable_count": incomparable,
            "ineligible_count": ineligible,
        }

    def next_index(self) -> int:
        return self._next_index

    # -- writing ------------------------------------------------------------

    def write(self, record: Dict[str, Any]) -> bool:
        """Append one complete record, or nothing at all."""
        if not self.enabled or self._fd is None:
            return False
        record = dict(record)
        record.setdefault("schema", CAPTURE_SCHEMA)
        try:
            blob = (json.dumps(record, separators=(",", ":"), default=str)
                    + "\n").encode("utf-8")
        except (TypeError, ValueError) as exc:
            self.write_error = self.write_error or f"encode: {exc}"
            return False
        if len(blob) > CAPTURE_MAX_RECORD_BYTES:
            # Identity survives; the bytes do not, and the record says which.
            trimmed = dict(record)
            trimmed["code_b64"] = ""
            trimmed["omitted"] = "record_exceeds_per_record_limit"
            try:
                blob = (json.dumps(trimmed, separators=(",", ":"), default=str)
                        + "\n").encode("utf-8")
            except (TypeError, ValueError) as exc:
                self.write_error = self.write_error or f"encode: {exc}"
                return False
        return self._append(blob)

    def _append(self, blob: bytes) -> bool:
        """The cap is checked and the bytes are written under one exclusive
        lock, so a second worker process cannot slip a record in between and
        neither can interleave a partial line."""
        with self._lock:
            fd = self._fd
            if fd is None:
                return False
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                try:
                    size = os.lseek(fd, 0, os.SEEK_END)
                    if size + len(blob) > CAPTURE_DEFAULT_MAX_BYTES:
                        self.limit_reached = True
                        return False
                    os.write(fd, blob)
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError as exc:
                self.write_error = self.write_error or f"append: {exc}"
                self.enabled = False
                return False
        self.records_written += 1
        self.bytes_written += len(blob)
        return True

    def _write_selection(self, result: Optional[Dict[str, Any]]) -> None:
        summary = dict(self._selection or {
            "phase": (result or {}).get("phase_solved", "none"),
            "pool": [], "pool_indices": [], "lens_index": None,
            "evidence_index": None, "verified_index": None,
            "selection_status": "not_run", "selection_reason": "",
            "tied_count": 0, "incomparable_count": 0, "ineligible_count": 0})
        summary["type"] = "selection_summary"
        summary["session_id"] = self._session_id
        # The service does not know what Go finally wrote. This names the
        # bytes the service RETURNED; the delivered artifact is joined
        # offline from the runner's own authorization telemetry.
        code = (result or {}).get("code") or ""
        summary["service_returned_candidate_hash"] = \
            contract.content_hash(code) if code else ""
        self.write(summary)

    def _write_status(self) -> None:
        status = {"type": "capture_status", "session_id": self._session_id,
                  "max_bytes": CAPTURE_DEFAULT_MAX_BYTES,
                  "bytes_written": self.bytes_written,
                  "records_written": self.records_written,
                  "write_error": self.write_error,
                  "limit_reached": self.limit_reached}
        if not self.write(status):
            # The cap refused the full line; a bare marker still tells the
            # reader the file is truncated rather than complete.
            self.limit_reached = True
            self._append(b'{"type":"capture_status","limit_reached":true}\n')


def _capture_pool_member(capture: "_PoolCapture", candidate, probe_code: str) -> None:
    """Record a pool member under the role that explains where it came from.

    Index 0 is candidate zero only when a probe actually produced code; with
    no probe the pool is generated candidates all the way down, and calling
    the first of them "candidate zero" would misname the comparison the
    diagnostic exists to make.
    """
    capture.note_candidate(
        role=("candidate_zero" if probe_code and candidate.get("index") == 0
              else "generated"),
        index=candidate.get("index"), code=candidate.get("code") or "",
        accepted=bool(candidate.get("passed")),
        record=candidate.get("contract_record"), phase="sandbox",
        lens={"energy": candidate.get("energy"),
              "energy_norm": candidate.get("energy_norm"),
              "energy_calibrated": candidate.get("energy_calibrated"),
              "per_step": candidate.get("per_step")})


def _summarize_phases(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fold the run's progress events into ordered per-phase rows.

    Each row carries the phase name, the stage that closed it (its
    outcome marker), that stage's detail, and the span between the
    phase's first and last event. Derived purely from the events the
    run already emits — no extra instrumentation in the hot path.
    """
    rows: List[Dict[str, Any]] = []
    by_phase: Dict[str, Dict[str, Any]] = {}
    for ev in events:
        phase = _STAGE_PHASE.get(ev.get("stage", ""))
        if phase is None:
            continue
        row = by_phase.get(phase)
        if row is None:
            row = {"phase": phase, "first_ms": round(ev.get("t", 0.0) * 1000)}
            by_phase[phase] = row
            rows.append(row)
        row["outcome"] = ev.get("stage", "")
        row["detail"] = str(ev.get("detail", ""))[:120]
        row["duration_ms"] = round(ev.get("t", 0.0) * 1000) - row["first_ms"]
    return rows


# --- V3 Pipeline Orchestrator ------------------------------------------------

def _candidate_by_index(candidates: List[Dict[str, Any]], index: int) -> Optional[Dict[str, Any]]:
    """Return the candidate dict whose original ``index`` field matches.

    Selection reports the winner by the candidate's original index, but
    the ``passing`` list has been sorted and filtered — positional
    indexing would pick the wrong candidate (or IndexError). Returns None
    when no candidate carries that index.
    """
    return next((c for c in candidates if c.get("index") == index), None)


def _entry_function(code: str) -> Optional[str]:
    """Name the function a self-test should call.

    The generator used to take the first `def` in the file, which is the
    entry point only when the solution happens to define no helpers above
    it. `def parse(...)` followed by `def solve(...)` meant every test
    called parse with the case input and compared it to the final answer —
    a guaranteed failure that says nothing about the code.

    Measured across a 28-session run: 0 of 44 candidates passed, and the
    self-test results were 0/5, 0/4, 0/3 — never partial. Imperfect code
    fails some cases; only a harness fault fails all of them uniformly.

    The entry point is the top-level function nothing else in the file
    calls. Where several qualify a conventional name wins, then the last
    one, since helpers are conventionally defined above their caller.
    Unparseable code falls back to the first def, which is what the old
    behaviour was.
    """
    import ast as _ast
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        m = re.search(r'^def (\w+)\(', code, re.MULTILINE)
        return m.group(1) if m else None
    top = [n for n in tree.body if isinstance(n, _ast.FunctionDef)]
    if not top:
        return None
    names = {n.name for n in top}
    called = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name):
            if node.func.id in names:
                called.add(node.func.id)
    roots = [n.name for n in top if n.name not in called]
    for preferred in ("solve", "main", "run"):
        if preferred in roots:
            return preferred
    if roots:
        return roots[-1]
    return top[-1].name


def _entry_takes_case_input(code: str, name: str) -> bool:
    """Whether the case input can be passed to `name` and its answer read
    back as a return value.

    The two self-test shapes are not interchangeable. A function-shaped
    solution takes the case input as arguments and returns the answer. A
    script-shaped one reads stdin and prints — calling it with the case
    input raises TypeError, and there is no return value to compare.

    Measured on aoc_sonar: candidates are `def main():` reading sys.stdin
    and printing, so the function path called main(case) and every case
    failed. The stdin/stdout path below handles exactly this and was never
    reached, because the choice keyed on a function merely existing.

    Requires all three: the function takes at least one parameter, it
    returns a value somewhere, and the file does not read stdin itself.
    """
    import ast as _ast
    if "sys.stdin" in code or "input()" in code:
        return False
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return True  # old behaviour for code that does not parse
    for node in _ast.walk(tree):
        if isinstance(node, _ast.FunctionDef) and node.name == name:
            args = node.args
            takes = bool(args.args or args.posonlyargs or args.kwonlyargs
                         or args.vararg)
            returns = any(isinstance(n, _ast.Return) and n.value is not None
                          for n in _ast.walk(node))
            return takes and returns
    return False


def _reads_input_file(code: str):
    """The literal filename a candidate opens for reading, if it opens one.

    Returns None when the program takes no named file, which is the case the
    stdin path below already covers.

    Only string literals count. A computed path cannot be materialised without
    running the program, and guessing one wrong is worse than falling through
    to the existing behaviour.
    """
    import ast as _ast
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return None
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Call):
            continue
        func = node.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        if name not in ("open", "read_text", "read_bytes", "Path"):
            continue
        if not node.args:
            continue
        target = node.args[0]
        if not (isinstance(target, _ast.Constant) and isinstance(target.value, str)):
            continue
        # A write/append target is an output file, not the program's input.
        mode = node.args[1] if len(node.args) > 1 else None
        if isinstance(mode, _ast.Constant) and isinstance(mode.value, str):
            if any(ch in mode.value for ch in ("w", "a", "x")):
                continue
        for kw in node.keywords:
            if kw.arg == "mode" and isinstance(kw.value, _ast.Constant):
                if any(ch in str(kw.value.value) for ch in ("w", "a", "x")):
                    break
        else:
            if "/" not in target.value and "\\" not in target.value:
                return target.value
    return None


_SOURCE_SUFFIXES = (".py", ".js", ".ts", ".go", ".java", ".rb", ".php", ".rs",
                    ".c", ".cpp", ".h", ".html", ".css", ".md", ".json",
                    ".yaml", ".yml", ".toml")


def _task_input_file(project_files) -> str:
    """The data file the task supplies for the program to read, if any.

    Verification has to hold a candidate to the environment's contract, not
    to whichever contract the candidate happens to prefer. The caller runs
    `python solve.py` with this file on disk and NO stdin, so a candidate
    that reads stdin must fail here rather than be handed a stdin the real
    run will never provide.

    Measured: with the self-test choosing its shape from the candidate's own
    code, stdin-reading candidates passed verification and were selected in 3
    sessions — every one of which then failed the task, printing 0 because no
    stdin arrived.
    """
    if not project_files:
        return ""
    for name in project_files:
        if not name.lower().endswith(_SOURCE_SUFFIXES):
            return name
    return ""


# ------------------------------------------------------------------ mode ---
# One valid mode, because two independent flags allowed an invalid one:
# selection enabled with probing disabled turned on the evidence ranker while
# collecting no evidence for it to rank.
MODE_OFF = "off"          # no probing, legacy selection
MODE_SHADOW = "shadow"    # probing + telemetry, legacy selection
MODE_ENFORCE = "enforce"  # probing + telemetry + evidence selection


def _selection_mode(env: Optional[Dict[str, str]] = None) -> str:
    # `env is None` means "read the process environment"; an explicitly
    # supplied empty dict means "this environment has nothing set", and the
    # two must not collapse via truthiness.
    import os as _os
    source = _os.environ if env is None else env
    raw = str(source.get("ATLAS_EVIDENCE_MODE", MODE_OFF)).strip().lower()
    return raw if raw in (MODE_OFF, MODE_SHADOW, MODE_ENFORCE) else MODE_OFF


def _probing_enabled(mode: str) -> bool:
    return mode in (MODE_SHADOW, MODE_ENFORCE)


def _selection_enabled(mode: str) -> bool:
    return mode == MODE_ENFORCE


# The candidate is staged beside its input rather than spliced into the
# wrapper, so the bytes that run are the bytes that were generated.
_CANDIDATE_FILE = "candidate.py"


def _trusted_oracle(self_tests) -> bool:
    """Whether these cases may decide anything about a candidate.

    Only a suite whose every case declares trusted provenance is an oracle.
    Model-generated cases are not: the same model wrote the code and the
    answer, from the problem statement alone, and for these tasks producing
    the expected output IS solving the problem. Measured on the captured
    pool: 21 of 36 valid generated keys disagreed with the task's own
    reference, and a correct candidate scored 2/5 against its own suite.

    Unknown provenance fails closed. A case that cannot say where it came
    from gets no authority, so a future producer must opt in deliberately
    rather than inherit trust by omission.
    """
    cases = getattr(self_tests, "test_cases", None) or []
    if not cases:
        return False
    return all(getattr(tc, "provenance", None) == PROVENANCE_TRUSTED
               for tc in cases)


def _staged_candidate_run(code: str, inp: str, infile: str):
    """How a case runs a candidate, for both builders that need it.

    Returns ``(header, run, files)``: the files the sandbox stages, the
    source that attaches stdin and captures stdout, and the one statement
    that executes the candidate. The self-test and the output probe differ
    only in what they do with the captured output and with an exception, so
    they share this and nothing else -- two copies of "how to run a
    candidate" is how one of them was repaired and the other was not.

    The case's input is staged, never written by executable code: that write
    lands in the candidate's working directory, which is read-only in the
    sandbox, so it raised before the candidate ran. The candidate executes
    once, from a file, under the name a program is really run with; exec'ing
    it inside the imported ``solution`` module left ``__name__`` as
    ``'solution'`` and a ``if __name__ == "__main__":`` body never ran.
    """
    files = {_CANDIDATE_FILE: code}
    if infile:
        # Empty stdin, not absent stdin. The caller runs the program with
        # stdin at EOF, so a candidate that reads sys.stdin must terminate
        # immediately and fail fast. With no stdin attached it BLOCKS until
        # the sandbox timeout instead: measured live, one stdin candidate
        # turned the probe into a 300s hang and the dead-oracle fast return
        # never fired, because the probe "failed" by timeout rather than by
        # inconclusive.
        files[infile] = inp
        stdin_setup = "_s.stdin=_o.StringIO('')\n"
    else:
        stdin_setup = f"_s.stdin=_o.StringIO({repr(inp)})\n"
    header = ("import sys as _s,io as _o,runpy as _rp\n" + stdin_setup
              + "_c=_o.StringIO()\n_old=_s.stdout\n_s.stdout=_c\n")
    run = f"_rp.run_path({repr(_CANDIDATE_FILE)},run_name='__main__')\n"
    return header, run, files


def _make_self_test(code: str, tc, task_input_file: str = ""):
    """Build one case's executable check, and the files the sandbox stages.

    Returns ``(wrapper_source, files)``. The wrapper creates nothing: the
    case's input file is staged through the sandbox's request files map, the
    same mechanism that already carries project context. Writing it from
    executable code instead put the write in the candidate's working
    directory, which is read-only in the sandbox container, so the case died
    with ``OSError: [Errno 30]`` before the candidate ran. Measured on the
    captured candidate pool: 50 of 50 generated cases failed there, which is
    why every suite in a 50-task run scored 0/N and a partial score was
    structurally impossible.

    The candidate then runs through ``runpy.run_path(..., run_name="__main__")``
    -- once, from a file, with the name a program is actually run under. The
    previous form exec'd it inside the imported ``solution`` module, where
    ``__name__`` is ``'solution'``, so a ``if __name__ == "__main__":`` body
    never executed and the case compared against empty output.

    Uses ast.literal_eval (safe — only parses Python literals) to convert
    I/O string representations to actual values for comparison.
    All code runs inside the sandboxed container.
    """
    inp = tc.input_str.strip()
    exp = tc.expected_output.strip()
    name = _entry_function(code)
    if name and _entry_takes_case_input(code, name):
        return (code + "\nimport ast as _a\n"
            + f"_i={repr(inp)}\n_e={repr(exp)}\n"
            + "try:\n _p=_a.literal_eval(_i)\nexcept:\n _p=_i\n"  # noqa: E722  -- bare except inside generated user code, intentional
            + f"_r={name}(*_p) if isinstance(_p,tuple) else {name}(_p) if isinstance(_p,list) else {name}(_p)\n"
            + "try:\n _ev=_a.literal_eval(_e)\nexcept:\n _ev=_e\n"  # noqa: E722  -- bare except inside generated user code, intentional
            + "assert str(_r)==str(_ev) or _r==_ev,f'got {_r}'\nprint('SELF_TEST_PASS')\n"), {}
    # A program that reads a named file has to be given that file. Feeding it
    # stdin instead tests a contract the task never stated, and the verdict
    # comes out backwards: a candidate that correctly reads input.txt finds no
    # such file in the sandbox and FAILS, while one that reads stdin passes.
    # Verification then selects for the shape that cannot work when the caller
    # runs `python solve.py` with no stdin.
    #
    # Measured on the two AoC tasks whose input is a file: 9 of 12 candidates
    # ATLAS wrote read stdin, against a prompt that says "reads input.txt",
    # and both tasks sat at 7/26. The same model asked directly, with no
    # pipeline, wrote input.txt readers and scored 12/12.
    infile = task_input_file or _reads_input_file(code)
    header, run, files = _staged_candidate_run(code, inp, infile)
    # A SystemExit the program raised itself is ordinary termination when its
    # code is 0 or None, and an execution failure otherwise; anything else
    # propagates and the process exits non-zero.
    return (
        header
        + "try:\n"
        + "    " + run
        + "except SystemExit as _e:\n"
        "    if _e.code not in (0,None):\n"
        "        raise\n"
        "finally:\n"
        " _s.stdout=_old\n"
        f"assert _c.getvalue().strip()=={repr(exp)},f'got {{_c.getvalue().strip()}}'\n"
        "print('SELF_TEST_PASS')\n"), files


_CONSENSUS_MARK = "V3_OUT:"




def _evaluate_candidate(file_path, code, smoke_passed, has_oracle, emit,
                       sandbox=None, *, task=None):
    """THE canonical contract record for ONE artifact.

    Behavioural adapters actually run here; an unsupported or inconclusive
    candidate stays available as a fallback but is never represented as
    behaviourally verified.

    One record, bound to the exact bytes it describes. No parallel strength,
    score or coverage field is kept beside it: a second authoritative copy is
    how two answers about one candidate start to disagree.
    """
    adapter = adapters.select_adapter(file_path, code, has_oracle)
    probe_ev = None
    # Behavioural probing costs real budget, so it is opt-in. Leaving it on
    # while only the DECISION was shadowed changed latency and could consume
    # enough budget to alter later pipeline behaviour -- that is
    # decision-shadowing, not passive shadowing.
    if _probing_enabled(_selection_mode()) and \
            adapter in (adapters.ADAPTER_BROWSER_CANVAS_JS, adapters.ADAPTER_BROWSER_INLINE_SCRIPT):
        target = code
        if adapter == adapters.ADAPTER_BROWSER_INLINE_SCRIPT:
            target = adapters.extract_inline_script(code)
        try:
            probe_ev = run_browser_probe(target, sandbox)
        except Exception as exc:                      # noqa: BLE001
            emit("behavior_probe_error", str(exc)[:120])
            probe_ev = None
    task = task or _task_identity("", "")
    return adapters.contract_record(
        adapter=adapter, accepted=bool(smoke_passed), probe=probe_ev,
        contract_id=task["contract_id"], contract_version=task["contract_version"],
        artifact_scope=task["artifact_scope"],
        evaluation_context_hash=task["evaluation_context_hash"],
        candidate_content_hash=contract.content_hash(code))



def _ensure_delivered_evidence(result, *, file_path, problem):
    """Every successful exit describes THE BYTES IT RETURNS.

    The pipeline has six ways to return code, and only two of them ran the
    structured evaluation on the artifact they hand back: the probe's early
    return and phase-one selection. Repair, refinement, the dead-oracle
    consensus and the budget fallback returned code with no record of it at
    all, or with the record of a different candidate, so the envelope either
    went missing or described bytes the caller never received.

    This evaluates the delivered bytes through the SAME canonical
    adapter->contract path every other candidate goes through. No probe is
    dispatched here -- an artifact whose behaviour was never observed reports
    exactly that -- and the verifier's own accept/reject is the observation
    input, never a claim of complete evidence: `accepted` on an interactive
    artifact still yields syntax-level, unsupported evidence, and only the
    oracle adapter, which exists only where an oracle ran, reaches oracle
    strength.

    It changes no path's `passed`, selection or return value. Where a legacy
    `passed=true` sits on evidence that is not closure-eligible, the envelope
    says so plainly rather than hiding the contradiction.
    """
    if not result:
        return
    code = result.get("code") or ""
    if not code:
        return
    task = _task_identity(file_path, problem)
    record = result.get("evidence_record")
    if not record or record.get("candidate_content_hash") != contract.content_hash(code):
        record = adapters.contract_record(
            adapter=adapters.select_adapter(
                file_path, code, bool(result.get("has_oracle"))),
            accepted=bool(result.get("passed")),
            probe=None,
            contract_id=task["contract_id"],
            contract_version=task["contract_version"],
            artifact_scope=task["artifact_scope"],
            evaluation_context_hash=task["evaluation_context_hash"],
            candidate_content_hash=contract.content_hash(code))
        result["evidence_record"] = record
        # The selection that produced the old record described a different
        # pool; re-state it over the record actually delivered.
        result["contract_selection"] = None
    selection = result.get("contract_selection")
    if not selection or selection.get("best_record") is not record:
        try:
            result["contract_selection"] = contract.select([record], record)
        except contract.ContractError as exc:
            result["contract_selection"] = {
                "best_record": None, "verified_winner": None, "tied": [],
                "incomparable": [], "ineligible": [],
                "selection_reason": f"identity error: {exc}"}


def _task_identity(file_path, problem):
    """What every record in this run is measured under.

    The rubric is the TASK's, not a candidate's: same contract, same artifact
    scope, same evaluation context for every candidate, so records that
    disagree about any of them are incomparable rather than silently ranked
    against each other.
    """
    ext = (file_path or "").rsplit(".", 1)
    suffix = ext[-1].lower() if len(ext) == 2 else "unknown"
    return {"contract_id": f"generate:{suffix}",
            "contract_version": "1",
            "artifact_scope": file_path or "",
            "evaluation_context_hash": contract.content_hash(problem or "")}


def _record_closes(record, code):
    """Closure is contract.select's verdict, on a record proven to describe
    these exact bytes. A record whose hash does not match the candidate it is
    attached to may neither close nor win -- stale evidence about other bytes
    is the one way a verified claim becomes a false one.
    """
    if not record or record.get("candidate_content_hash") != contract.content_hash(code):
        return False
    try:
        selection = contract.select([record], record)
    except contract.ContractError:
        return False
    return selection.get("verified_winner") is record


# Execution budget for ONE probe arm. Kept modest: two arms run per
# candidate, and the harness is a bounded virtual-clock drain, not a wall
# clock wait.
BROWSER_PROBE_TIMEOUT_S = 20


def run_browser_probe(code: str, sandbox=None, timeout_s: int = BROWSER_PROBE_TIMEOUT_S):
    """Run the deterministic harness INSIDE THE ISOLATED SANDBOX.

    This previously shelled out to `node` directly from the V3 service, which
    executes model-generated JavaScript in the service's own environment --
    its filesystem, env vars, network and process capabilities. The
    instrumentability regex is a routing hint, not a containment boundary,
    and treating it as one was a security defect. The sandbox already runs
    untrusted candidate code under its own restrictions and supports
    javascript, so the probe belongs there.

    Returns None for "inconclusive" -- never a behavioural verdict -- when
    the sandbox is unavailable, times out, or the artifact is not
    instrumentable.
    """
    if sandbox is None or not adapters.js_is_instrumentable(code):
        return None
    runs = {}
    for mode in ("baseline", "input"):
        # The harness reads its artifact from a file and its mode from argv;
        # inside the sandbox there is one code blob, so both are inlined.
        blob = (
            "const __MODE__ = " + json.dumps(mode) + ";\n"
            "const __ARTIFACT__ = " + json.dumps(code) + ";\n"
            + adapters.js_probe_source_inline()
        )
        try:
            ok, stdout, _stderr = sandbox(blob, language="javascript",
                                          timeout=timeout_s)
        except TypeError:
            return None          # adapter without language support
        except Exception:        # noqa: BLE001
            return None
        if not ok:
            # A failed or timed-out execution can still have emitted
            # parseable stdout; trusting it would let a crash produce
            # behavioural evidence.
            return None
        runs[mode] = adapters.parse_probe_output(stdout)
    return adapters.combine_runs(runs.get("baseline"), runs.get("input"))


def _make_output_probe(code: str, tc, task_input_file: str = ""):
    """Run a candidate on a case's INPUT and report what it printed.

    Returns ``(wrapper_source, files)`` — the same staged shape the self-test
    uses, through the same ``_staged_candidate_run``.

    The self-test compares that output to the model's predicted answer. For a
    problem the model cannot reliably solve, producing the answer IS the
    problem, so the prediction is wrong and correct code fails its own suite.
    Measured across 42 verifications in one run: every one scored 0/N, while a
    candidate taken from those same logs passed immediately when given a
    correct expected value. Uniform zero is a broken answer key, not broken
    code.

    CodeT (Chen et al., 2022), which this pipeline already cites, does not use
    generated tests as an oracle, for exactly this reason. It runs candidates
    on the generated INPUTS and clusters them by agreement, so the signal is
    the answer candidates converge on rather than the answer the model
    guessed. This probe produces the raw material for that.
    """
    inp = tc.input_str.strip()
    name = _entry_function(code)
    if name and _entry_takes_case_input(code, name):
        return (code + "\nimport ast as _a\n"
                + f"_i={repr(inp)}\n"
                + "try:\n _p=_a.literal_eval(_i)\nexcept:\n _p=_i\n"
                + f"_r={name}(*_p) if isinstance(_p,tuple) else {name}(_p)\n"
                + f"print({repr(_CONSENSUS_MARK)}+repr(str(_r).strip()))\n")
    infile = task_input_file or _reads_input_file(code)
    header, run, files = _staged_candidate_run(code, inp, infile)
    # The marker is emitted ONLY on clean completion, and only for non-empty
    # output. Swallowing exceptions and printing the marker regardless let two
    # CRASHING candidates agree: repr('') is the two-character string "''",
    # which is truthy, so their empty outputs clustered and crash consensus
    # won (third-party audit reproduction: two ordinary-exception candidates,
    # WINNERS [0, 1]). A crash prints a CRASH line the clustering explicitly
    # refuses, and silence emits nothing.
    #
    # The probe reports rather than fails, so unlike the self-test it catches
    # what the candidate raised instead of letting it end the process. A
    # SystemExit the program raised itself is ordinary termination at code 0
    # or None -- a candidate that prints its answer and calls sys.exit() has
    # answered, and calling that a crash discarded a real output.
    return (header
            + "_crashed=False\n"
            + "try:\n"
            + "    " + run
            + "except SystemExit as _e:\n    _crashed=_e.code not in (0,None)\n"
            + "except BaseException:\n    _crashed=True\n"
            + "finally:\n _s.stdout=_old\n"
            + "_out=_c.getvalue().strip()\n"
            + f"if _crashed:\n    print({repr(_CONSENSUS_MARK)}+'CRASH')\n"
            + f"elif _out:\n    print({repr(_CONSENSUS_MARK)}+repr(_out))\n"), files


def _consensus_record(candidates, test_cases, sandbox, task_input_file=""):
    """What each candidate printed on each generated INPUT, and who agreed.

    The generated expected outputs are never read here — only the inputs are
    used, so a wrong answer key cannot reach this signal at all. What comes
    back is correlated ranking evidence: these candidates came from one
    model, so agreement between them is not independence and may never
    become closure, a verified winner, or delivery authorization. It is
    recorded so a future policy can be argued from measurement.
    """
    per_case = []
    outputs = {}
    for i, tc in enumerate(test_cases):
        clusters, crashed, timed_out, silent = {}, [], [], []
        raw_input = getattr(tc, "input_str", "") or ""
        for c in candidates:
            digest = contract.content_hash(c.get("code") or "")
            try:
                probe_code, probe_files = _make_output_probe(
                    c["code"], tc, task_input_file)
                ok, out, err = sandbox(probe_code, files=probe_files)
            except Exception:                          # noqa: BLE001
                ok, out, err = False, "", "probe error"
            marker = ""
            if ok and _CONSENSUS_MARK in (out or ""):
                marker = out.split(_CONSENSUS_MARK)[-1].strip()
            if marker == "CRASH":
                crashed.append(digest)
                marker = ""
            elif not marker:
                (timed_out if _CAPTURE_TIMEOUT.search(err or "") else
                 silent).append(digest)
            else:
                clusters.setdefault(marker, []).append(digest)
            outputs.setdefault(digest, []).append(marker)
        # Cluster ids and output hashes, so a ranking can be checked against
        # an independent verdict afterwards. Agreement counts alone cannot
        # say WHICH candidates agreed.
        ordered = sorted(clusters.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        rows = [{"cluster_id": f"c{i}-{n}",
                 "output_sha256": contract.content_hash(k),
                 "members": v, "size": len(v)}
                for n, (k, v) in enumerate(ordered)]
        top = max((r["size"] for r in rows), default=0)
        winners = [r["cluster_id"] for r in rows if r["size"] == top and top]
        per_case.append({
            "input_index": i,
            "input_sha256": contract.content_hash(raw_input),
            "input": raw_input[:400],
            "clusters": rows,
            "winning_cluster_id": winners[0] if len(winners) == 1 else None,
            "tied_cluster_ids": winners if len(winners) > 1 else [],
            "crashed": crashed, "timed_out": timed_out, "no_output": silent,
        })
    # A candidate ranks only if it answered every input: partial validity is
    # not agreement material.
    signatures = {}
    for digest, outs in outputs.items():
        if all(outs):
            signatures.setdefault(tuple(outs), []).append(digest)
    groups = sorted(signatures.values(), key=len, reverse=True)
    agreement = len(groups[0]) if groups else 0
    tied = [g for g in groups if len(g) == agreement] if groups else []
    return {
        "cases": per_case,
        "candidates": [contract.content_hash(c.get("code") or "")
                       for c in candidates],
        "groups": [{"members": g, "size": len(g)} for g in groups],
        "agreement": agreement,
        "tied_groups": len(tied),
        "ranked": [d for g in groups for d in g],
        "reads_expected_output": False,
        "authority": "ranking_only",
    }


def _consensus_winners(candidates, test_cases, sandbox, emit,
                       task_input_file=""):
    """CodeT agreement: candidates whose outputs match the largest cluster.

    Returns [] when there is nothing to agree on — fewer than two candidates,
    no case produced output, or every candidate disagreed with every other.
    Agreement between independently generated programs is evidence; one
    program agreeing with itself is not, so a lone cluster does not win.

    A candidate enters clustering only if it produced a real answer on EVERY
    probe case. Partial validity is not agreement material: with any(), two
    candidates that crashed on most cases but happened to match on one could
    form the winning cluster, promoting code proven broken on the majority of
    the very inputs the consensus ran (third-party audit finding).
    """
    if len(candidates) < 2 or not test_cases:
        return []
    sigs = {}
    for c in candidates:
        outs = []
        for tc in test_cases:
            try:
                probe_code, probe_files = _make_output_probe(
                    c["code"], tc, task_input_file)
                ok, out, _ = sandbox(probe_code, files=probe_files)
            except Exception:
                ok, out = False, ""
            marker = ""
            if ok and _CONSENSUS_MARK in (out or ""):
                marker = out.split(_CONSENSUS_MARK)[-1].strip()
            if marker == "CRASH" or marker == "''":
                # A crash or empty output is not an answer to agree on.
                marker = ""
            outs.append(marker)
        if all(outs):
            sigs.setdefault(tuple(outs), []).append(c)
    if not sigs:
        return []
    best = max(sigs.values(), key=len)
    if len(best) < 2:
        return []
    emit("consensus", f"{len(best)}/{len(candidates)} candidates agree",
         cluster=len(best), clusters=len(sigs))
    return best


def _dead_oracle_consensus(problem, task_id, llm, plan_search, probe_code,
                           test_cases, sandbox, emit, task_input_file, start):
    """Bounded input-only consensus for the dead-oracle condition.

    When the generated oracle scores uniformly 0/N, the fast return
    preserves latency at the cost of making consensus — the mechanism built
    for exactly this condition — unreachable (third-party audit finding).
    This is the bounded middle: at most two extra candidates, at most four
    generated inputs, every step gated on the remaining wall-clock, and an
    immediate unverified return when no cluster forms.

    Feature-flagged (ATLAS_V3_DEAD_ORACLE_CONSENSUS=1) and OFF by default:
    the unbounded ancestor of this path was the ~300s dead-oracle tax, and
    this variant has to earn the default in an A/B before it gets it.

    Returns (winning candidate or None, extra tokens spent).
    """
    left = _remaining_budget_ms(start)
    if left is not None and left < 45000:
        emit("dead_oracle_skip",
             f"only {round(left)}ms of budget left — not starting")
        return None, 0
    tokens = 0
    cands = []
    if probe_code:
        cands.append({"index": 0, "code": probe_code})
    try:
        ps = plan_search.generate(problem, task_id, llm, num_plans=2)
        tokens += ps.total_tokens
        for code in ps.candidates:
            if code:
                cands.append({"index": len(cands), "code": code})
    except Exception as exc:
        emit("dead_oracle_generation_failed", str(exc)[:120])
    left = _remaining_budget_ms(start)
    if len(cands) < 2 or (left is not None and left < 15000):
        emit("dead_oracle_skip",
             f"{len(cands)} candidate(s), {round(left) if left is not None else 'unlimited'}ms left — nothing to compare")
        return None, tokens
    # _consensus_winners already enforces the hard rules: a candidate joins
    # clustering only with a real answer on EVERY case, and a cluster needs
    # at least two members. "No strong cluster" falls out as [].
    agreed = _consensus_winners(cands, test_cases[:4], sandbox, emit,
                                task_input_file)
    if not agreed:
        emit("dead_oracle_no_cluster",
             f"{len(cands)} candidates, no agreement — returning unverified")
        return None, tokens
    return agreed[0], tokens


class V3PipelineService:
    """Full V3 pipeline for a single coding task, with streaming progress."""

    def __init__(self):
        # ALL V3 components enabled — same as benchmark runner with all phases
        # active. Stage telemetry mirrors the bench runner's telemetry/*.jsonl
        # into ATLAS_V3_TELEMETRY_DIR so live-orchestrator runs are measurable.
        self.telemetry_dir = _resolve_telemetry_dir()
        t = self.telemetry_dir
        self.budget_forcing = BudgetForcing(BudgetForcingConfig(enabled=True),
                                            telemetry_dir=t)
        self.plan_search = PlanSearch(PlanSearchConfig(enabled=True),
                                      telemetry_dir=t)
        self.div_sampling = DivSampling(DivSamplingConfig(enabled=True),
                                        telemetry_dir=t)
        self.pr_cot = PRCoT(PRCoTConfig(enabled=True), telemetry_dir=t)
        self.refinement_loop = RefinementLoop(RefinementLoopConfig(enabled=True),
                                              telemetry_dir=t)
        self.self_test_gen = SelfTestGen(SelfTestGenConfig(enabled=True),
                                         telemetry_dir=t)

    def run(self, problem: str, task_id: str = "cli",
            progress_callback=None, files: Dict[str, str] = None,
            file_path: str = "", build_command: str = "",
            working_dir: str = "/workspace",
            baseline_code: str = "") -> Dict[str, Any]:
        """Run the full V3 pipeline on a coding problem.

        Args:
            problem: Problem description
            task_id: Task identifier
            progress_callback: SSE progress emitter
            files: Dict of filename→content from Aider's existing file context
            file_path: Target file path (used by PC-048 to detect language
                for the smoke check — `.html` files use HTML parser, not
                Python compile, etc.)
            build_command: Optional project build command to run against an
                ephemeral candidate overlay after syntax/self-tests pass.
            working_dir: Container workspace root used by the sandbox overlay.
            baseline_code: The incumbent's EXACT bytes, as the request sent
                them, before prompt construction. Read only by the
                diagnostic pool capture; with capture off nothing touches
                it, and it never enters a live list or decision.

        Writes one pipeline-summary telemetry line per task (fail-soft;
        see _write_pipeline_summary) around the actual pipeline body.
        """
        start = time.time()
        result: Optional[Dict[str, Any]] = None
        error = ""
        # Benchmark-only: inert unless ATLAS_V3_CAPTURE_POOL names a file.
        # Opened per run so a diagnostic can be turned on and off without
        # restarting the service, and closed here so the delivered artifact
        # and the selection summary are written from the finished result.
        capture = _PoolCapture.from_env()
        capture.bind(task_id)
        try:
            result = self._run_impl(
                problem, task_id=task_id, progress_callback=progress_callback,
                files=files, file_path=file_path, build_command=build_command,
                working_dir=working_dir, baseline_code=baseline_code,
                _capture=capture)
            _ensure_delivered_evidence(result, file_path=file_path, problem=problem)
            return result
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            capture.close(result)
            self._write_pipeline_summary(task_id, result, error, start)

    def _write_pipeline_summary(self, task_id: str,
                                result: Optional[Dict[str, Any]],
                                error: str, start: float) -> None:
        """Append one summary line to telemetry/pipeline_summary.jsonl.

        Carries the per-task shape the bench runner gets for free from its
        per-task JSON files: phases run (outcome + duration, folded from the
        run's progress events), veto events, and the final result fields.
        Fail-soft by construction — a telemetry error never reaches the
        caller, so it can never break generation.
        """
        if self.telemetry_dir is None:
            return
        try:
            events = (result or {}).get("events") or []
            line = {
                "schema": "v3_pipeline_summary_v1",
                "ts": datetime.now(timezone.utc).isoformat(),
                "task_id": task_id,
                "passed": bool((result or {}).get("passed")),
                "phase_solved": (result or {}).get("phase_solved", "none"),
                "task_type": (result or {}).get("task_type", ""),
                "candidates_generated": (result or {}).get("candidates_generated", 0),
                "total_tokens": (result or {}).get("total_tokens", 0),
                "total_time_ms": round(
                    (result or {}).get("total_time_ms")
                    or (time.time() - start) * 1000),
                "phases": _summarize_phases(events),
                "veto_events": [
                    {"stage": ev.get("stage", ""),
                     "index": (ev.get("data") or {}).get("index", -1),
                     "detail": str(ev.get("detail", ""))[:120]}
                    for ev in events if ev.get("stage") in _VETO_STAGES
                ],
            }
            if error:
                line["error"] = error[:300]
            with _SUMMARY_LOCK:
                with open(self.telemetry_dir / "pipeline_summary.jsonl", "a") as f:
                    f.write(json.dumps(line) + "\n")
        except Exception as e:
            print(f"  [telemetry] pipeline summary write failed (non-fatal): {e}",
                  flush=True)

    def _run_impl(self, problem: str, task_id: str = "cli",
                  progress_callback=None, files: Dict[str, str] = None,
                  file_path: str = "", build_command: str = "",
                  working_dir: str = "/workspace", baseline_code: str = "",
                  _capture: Optional["_PoolCapture"] = None) -> Dict[str, Any]:
        """The pipeline body — see run() for the argument contract.

        `_capture` is the benchmark-only pool sink run() owns; it observes
        and decides nothing, and defaults to an inert one.
        """
        start = time.time()
        events = []
        files = files or {}
        capture = _capture if _capture is not None else _PoolCapture.disabled()

        # PC-048: derive language from the target file's extension. Used
        # only by smoke_compile_check below to pick the right syntax
        # checker. Defaults to Python when no file_path is supplied.
        _ext = Path(file_path).suffix.lower() if file_path else ""
        # Only languages scoring.smoke_compile_check can actually verify —
        # an entry here that the checker rejects would fail every candidate
        # with "verification unavailable" instead of checking anything.
        _ext_to_lang = {
            ".py": "python", ".pyw": "python",
            ".html": "html", ".htm": "html",
            ".json": "json",
            ".yaml": "yaml", ".yml": "yaml",
            ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
            ".ts": "typescript", ".tsx": "typescript",
            ".xml": "xml",
            ".sh": "bash", ".bash": "bash",
            ".go": "go",
            ".java": "java",
            ".kt": "kotlin",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
        }
        smoke_language = _ext_to_lang.get(_ext, "python")

        # If existing file context is provided, prepend it to the problem
        # so all V3 modules (PlanSearch, PR-CoT, etc.) can see the code
        if files:
            file_context_parts = []
            for fname, content in files.items():
                file_context_parts.append(f"### Existing file: {fname}\n```\n{content}\n```")
            problem = (
                "The following files already exist in the project:\n\n"
                + "\n\n".join(file_context_parts)
                + "\n\n---\n\nTask:\n" + problem
            )

        def emit(stage, detail="", **data):
            ev = {"stage": stage, "detail": detail, "t": time.time() - start}
            if data:
                ev["data"] = data
            # Token deltas stream live through the callback but are not
            # stored: one dict per token would make the final `event: result`
            # frame multi-MB on long generations.
            if stage != "token":
                events.append(ev)
            if progress_callback:
                try:
                    progress_callback(stage, detail, **data)
                except TypeError:
                    progress_callback(stage, detail)

        def check_client():
            """Abort at phase boundaries once the SSE client disconnects.
            The handler sets `disconnected` on the callback when a write
            hits BrokenPipeError; a dead client must not keep burning GPU."""
            if getattr(progress_callback, "disconnected", False):
                raise adapters.ClientDisconnected(f"client disconnected during task {task_id}")

        # The adapter refuses to start a generation that cannot finish
        # before the cap, so every phase and every loop inside one is
        # covered by a single check rather than a boundary guard each.
        _budget_ms = _remaining_budget_ms(start)
        llm = adapters.LLMAdapter(progress_callback=emit)
        # Assigned rather than passed to the constructor: tests substitute
        # their own LLM doubles here, and a new required kwarg would break
        # every one of them for a value only the real adapter reads.
        if _budget_ms is not None:
            llm.deadline = start + _budget_ms / 1000.0
        # PC-046: ship the user's other project files into the sandbox so
        # multi-file imports resolve. `files` is the same Dict that V3
        # already prepends to the LLM prompt above; passing it to the
        # sandbox closes the gap where the model writes
        # `from utils import helper` and the sandbox imports a workspace
        # that contains only solution.py.
        sandbox = adapters.SandboxAdapter(project_files=files)
        # The contract verification must hold candidates to: the caller runs
        # the program with these files on disk and no stdin.
        task_input_file = _task_input_file(files)
        embed = adapters.EmbedAdapter()

        result = {
            "task_id": task_id,
            "passed": False,
            "code": "",
            "phase_solved": "none",
            "candidates_generated": 0,
            "total_tokens": 0,
            "total_time_ms": 0.0,
            "events": [],
            "verification_evidence": [],
        }

        # ===== PHASE 0: PROBE =====
        emit("probe", "Generating probe candidate...")
        # Light probe first (1024 thinking tokens), retry with standard if fails
        try:
            chatml = self.budget_forcing.format_chatml(problem, "light")
            response, tokens, t_ms = llm(chatml, BASE_TEMPERATURE, MAX_TOKENS, 42)
            probe_code = extract_code(response)
            if probe_code:
                emit("probe_light", f"Light probe: {len(probe_code)} chars, {tokens} tokens, {t_ms:.0f}ms")
        except Exception as e:
            emit("probe_error", str(e))
            probe_code = ""

        if not probe_code:
            emit("probe_retry", "Light probe failed — retrying with standard budget")
            try:
                chatml = self.budget_forcing.format_chatml(problem, "standard")
                response, tokens, t_ms = llm(chatml, BASE_TEMPERATURE, MAX_TOKENS, 42)
                probe_code = extract_code(response)
            except Exception as e:
                emit("probe_error", str(e))

        if not probe_code:
            emit("probe_failed", "No code extracted from probe")
            # Generate with the minimal reasoning budget
            chatml = self.budget_forcing.format_chatml(problem, "nothink")
            response, tokens, t_ms = llm(chatml, BASE_TEMPERATURE, MAX_TOKENS, 42)
            probe_code = extract_code(response)

        # Classify task type. Interactive tasks (games, UIs, framework code)
        # skip synthetic I/O self-tests entirely — those tests would fail by
        # construction, falsely triggering PR-CoT/refinement on working code.
        # See ISSUES.md PC-022.
        task_type = scoring.classify_task_type(problem)
        emit("task_type", task_type)
        result["task_type"] = task_type

        # Generate self-tests (algorithmic tasks only) — used for sandbox verification
        self_tests = None
        if task_type == "algorithmic":
            emit("self_test_gen", "Generating verification tests...")
            try:
                self_tests = self.self_test_gen.generate(problem, llm, task_id)
                emit("self_test_done", f"{len(self_tests.test_cases)} test cases")
                result["total_tokens"] += self_tests.generation_tokens
            except Exception as e:
                emit("self_test_error", str(e)[:200])
        else:
            emit("self_test_skip", "Interactive task — using compile smoke-test")

        # Set before verified_sandbox is ever called (phase 0 probe is the
        # first caller, after the self-test generation below).
        _has_trusted_oracle = _trusted_oracle(self_tests)

        def verified_sandbox(code, extra_test=""):
            """Sandbox + verification. Algorithmic tasks: I/O self-tests; interactive: compile smoke."""
            verification_evidence: List[Dict[str, Any]] = []

            def verify_build_if_requested(out="", err=""):
                ok, build_out, build_err, evidence = scoring.verify_build_command(
                    code=code,
                    sandbox=sandbox,
                    build_command=build_command,
                    file_path=file_path,
                    project_files=files,
                    working_dir=working_dir or "/workspace",
                    emit=emit,
                )
                if evidence:
                    verification_evidence.append(evidence)
                if not ok:
                    return False, build_out, build_err, verification_evidence
                return True, out, err, verification_evidence

            # Non-Python candidates always use the language-aware syntax path.
            # Python self-tests cannot establish correctness for another language.
            if smoke_language not in ("python", "py"):
                ok, out, err = scoring.smoke_compile_check(code, sandbox, language=smoke_language)
                emit("smoke_check", f"compile={'OK' if ok else 'FAIL'} ({smoke_language})")
                if not ok:
                    return ok, out, err, verification_evidence
                return verify_build_if_requested(out, err)

            # Interactive tasks: skip the run-and-test; just verify the code
            # parses and compiles. Running curses/pygame/flask in the sandbox
            # would fail for environmental reasons (no TTY, no display) even
            # when the code is correct — see PC-022.
            if task_type == "interactive":
                # PC-048: pass the detected language so HTML/JSON/etc. files
                # don't get parsed as Python (which produces spurious
                # SYNTAX_ERROR cascades into PR-CoT repair + LLM timeouts).
                ok, out, err = scoring.smoke_compile_check(code, sandbox, language=smoke_language)
                emit("smoke_check", f"compile={'OK' if ok else 'FAIL'} ({smoke_language})")
                if not ok:
                    return ok, out, err, verification_evidence
                # Interactive lint is Python-AST based — only meaningful for
                # Python files. Skip for HTML/CSS/JSON/etc.
                if smoke_language not in ("python", "py"):
                    return True, out, err, verification_evidence
                # Interactive lint: catch raw stdin reads / blocking input loops
                # that compile fine but don't actually work for keystroke
                # handling (PC-034).
                lint_ok, lint_reason = scoring.interactive_lint(code)
                if lint_ok:
                    emit("interactive_lint", "OK")
                    return verify_build_if_requested(out, err)
                emit("interactive_lint", f"FAIL: {lint_reason}")
                return False, out, f"interactive_lint: {lint_reason}", verification_evidence

            ok, out, err = sandbox(code)
            if not ok:
                return False, out, err, verification_evidence
            if self_tests and self_tests.test_cases:
                p, fails = 0, []
                observed = []
                for i, tc in enumerate(self_tests.test_cases):
                    try:
                        tc_code, tc_files = _make_self_test(
                            code, tc, task_input_file)
                        tp, to, te = sandbox(tc_code, files=tc_files)
                        if tp and "SELF_TEST_PASS" in to:
                            p += 1
                        else:
                            fails.append(f"TC{i+1}:{te[:60] if te else 'wrong'}")
                        observed.append(_capture_case(i, tc, tp, to, te))
                    except Exception as ex:
                        fails.append(f"TC{i+1}:{str(ex)[:40]}")
                        observed.append(_capture_case(i, tc, False, "", "",
                                                      harness_error=str(ex)[:200]))
                total = len(self_tests.test_cases)
                # p and total are computed here and discarded at the return
                # below, which hands back a boolean. Capture keeps them,
                # and the per-case pairs behind them, without touching the
                # verdict.
                capture.note_oracle(code, observed, p, total)
                emit("self_test_verify", f"{p}/{total} passed")
                if not _has_trusted_oracle:
                    # Observed, recorded, and given no authority. These cases
                    # were written by the same model as the candidate, from
                    # the problem statement alone; letting them reject is how
                    # a candidate that matched the task's own reference on
                    # every input scored 2/5 and was discarded. The score and
                    # every per-case pair stay in telemetry and in the pool
                    # capture for offline analysis -- what changes is that
                    # nothing downstream reads them as a verdict.
                    #
                    # The verdict below is the part that does not depend on
                    # the generated cases: the candidate executed, and the
                    # project's own build command still gets to speak.
                    emit("self_test_untrusted",
                         f"{p}/{total} against model-generated cases — "
                         f"diagnostic only, no rejection authority",
                         cases=total, passed_cases=p,
                         provenance=PROVENANCE_GENERATED)
                    return verify_build_if_requested(out, err)
                # A suite the candidate passes NO case of says nothing about
                # the candidate. The cases come from the same model that
                # writes the code, and for these tasks producing an expected
                # output IS solving the problem — so a wrong answer key and
                # wrong code are indistinguishable at zero.
                #
                # Measured: 0 of 44 candidates across a 28-session run, with
                # scores of 0/5, 0/4, 0/3 and never a partial. A check that
                # rejects every candidate it sees is not evidence, and it
                # cost the pipeline every winner it might have had — every
                # passing session was won by the model's own direct write.
                #
                # A partial score is different: the suite has demonstrated
                # some case is passable, so falling below half is the
                # candidate underperforming a bar something else cleared.
                if total > 0 and p == 0:
                    # Inconclusive is not a pass. An earlier version emitted
                    # this and fell through to the build check, which made
                    # "it compiles" the whole of verification: a candidate
                    # that failed every one of its own cases was marked
                    # passed, joined the selection pool, and could be written
                    # over the model's work. Measured on the two tasks whose
                    # answer is computed from a file the program has to read:
                    # 0/4 sessions correct on the run that shipped it,
                    # against 1-2 of 4 on each of the four runs before.
                    #
                    # "Cannot condemn" and "therefore promote" are different
                    # claims. A suite that passes nothing still establishes
                    # nothing, so the candidate stays unverified and the
                    # caller's own write stands — the rule the fallback path
                    # below already states, that executing but wrong is worse
                    # than an honest failure.
                    emit("self_test_inconclusive",
                         f"0/{total} — no case passed, so the suite cannot "
                         f"separate a wrong answer key from wrong code. Not "
                         f"verified; leaving the caller's own write in place",
                         cases=total)
                    return (False, out,
                            f"Self-test:0/{total} inconclusive — nothing verified",
                            verification_evidence)
                elif total > 0 and p < total / 2:
                    return False, out, f"Self-test:{p}/{total}. "+";".join(fails[:3]), verification_evidence
            return verify_build_if_requested(out, err)

        # Score and test probe with self-generated tests. The probe is the
        # only candidate the CxGx gate below can see, so it is scored with
        # the combined C(x)+G(x) call — one embedding extraction, both
        # models — rather than C(x) alone.
        probe_scores = dict(scoring.NEUTRAL_COMBINED)
        probe_energy_raw, probe_energy_norm = 0.0, 0.5
        probe_cx_calibrated = False
        probe_passed = False
        probe_stdout = probe_stderr = ""
        if probe_code:
            probe_scores = scoring.score_candidate_combined(probe_code)
            probe_energy_raw = probe_scores["cx_energy"]
            probe_energy_norm = probe_scores["cx_normalized"]
            probe_cx_calibrated = probe_scores["cx_calibrated"]
            norm_label = f"{probe_energy_norm:.2f}" if probe_cx_calibrated else "uncalibrated"
            emit("probe_scored",
                 f"C(x)={probe_energy_raw:.2f} norm={norm_label} "
                 f"G(x)={probe_scores['gx_score']:.2f} "
                 f"({probe_scores['verdict']})",
                 gx_score=probe_scores["gx_score"],
                 gx_available=probe_scores["gx_available"],
                 verdict=probe_scores["verdict"])
            probe_passed, probe_stdout, probe_stderr, probe_evidence = verified_sandbox(probe_code)
            emit("probe_sandbox", f"passed={probe_passed} stderr={probe_stderr[:80] if probe_stderr else ''}")
            result["total_tokens"] += tokens

        # Evidence strength decides whether phase 0 may close the pipeline.
        # A compile smoke on interactive code demonstrates only that the file
        # parses, and returning on it made candidates_generated=1 with
        # PlanSearch, DivSampling, consensus and ranking never running — the
        # whole test-time-compute apparatus skipped for any browser artifact.
        # Strength comes from the VERIFIER THAT RAN. Keying it off the file
        # extension mapped every .py to behavioural completeness, which is
        # wrong for Pygame/Tkinter/Flask — those get a compile smoke and
        # nothing more, and would have closed the pipeline claiming behaviour
        # nobody demonstrated.
        # A suite exists; whether it may DECIDE anything is a separate
        # question, and only a trusted one may. Model-generated cases route
        # the artifact to the no-oracle adapter, so nothing claims oracle
        # strength on evidence that never had it.
        _has_oracle = _trusted_oracle(self_tests)
        # Recorded for the finaliser: which verifier the artifact was eligible
        # for is a fact of the run, not something to re-derive afterwards.
        result["has_oracle"] = _has_oracle
        _task = _task_identity(file_path, problem)
        # The incumbent, measured the same way and kept apart. Only when the
        # diagnostic sink is on: with capture off this is not built, not
        # executed and not written, so default behaviour is unchanged. It
        # enters no live list and no live decision.
        if capture.enabled and baseline_code:
            try:
                _inc = _evaluate_candidate(
                    file_path, baseline_code,
                    scoring.smoke_compile_check(
                        baseline_code, sandbox, language=smoke_language)[0],
                    _has_oracle, emit, sandbox, task=_task)
                capture.note_incumbent(code=baseline_code, record=_inc,
                                       adapter=_inc["adapter_id"])
            except Exception as _exc:                  # noqa: BLE001
                # An incumbent that cannot be evaluated is recorded as
                # exactly that. Never synthesise a result for it.
                capture.note_incumbent(
                    code=baseline_code, record=None, adapter="",
                    evaluation=f"unevaluated: {str(_exc)[:120]}")

        probe_result = _evaluate_candidate(
            file_path, probe_code, probe_passed, _has_oracle, emit, sandbox,
            task=_task)
        probe_adapter = probe_result["adapter_id"]
        result["evidence_record"] = probe_result
        capture.note_candidate(
            role="candidate_zero", index=0, code=probe_code,
            accepted=probe_passed, record=probe_result, phase="probe",
            lens={"energy": probe_energy_raw, "energy_norm": probe_energy_norm,
                  "energy_calibrated": probe_cx_calibrated,
                  "gx_score": probe_scores.get("gx_score"),
                  "verdict": probe_scores.get("verdict")})
        emit("probe_evidence",
             f"adapter={probe_adapter} strength={probe_result['evidence_strength']} "
             f"supported={probe_result['supported']}",
             adapter=probe_adapter, strength=probe_result["evidence_strength"])

        # Early return is a LIVE decision, so it is mode-aware. In shadow the
        # probe runs and its verdict is recorded, but only the probe-free
        # judgement may act -- otherwise a behaviourally complete browser
        # candidate would return early and skip candidate generation, which
        # is a control-flow change, not observation.
        #
        # The judgement itself is contract.select over the candidate's own
        # record: it closes only when that exact record is the VERIFIED WINNER
        # under its own rubric. A best record that is not closure-eligible,
        # anything unsupported, failed or incomparable, and any record whose
        # hash does not match these bytes, all leave the pipeline open. The
        # strength floor comes from the contract, so an artifact class whose
        # contract closes on syntax legitimately may, and one that demands an
        # oracle still cannot close on a compile.
        _mode = _selection_mode()
        _probe_free = adapters.contract_record(
            adapter=probe_result["adapter_id"], accepted=bool(probe_passed),
            probe=None, contract_id=_task["contract_id"],
            contract_version=_task["contract_version"],
            artifact_scope=_task["artifact_scope"],
            evaluation_context_hash=_task["evaluation_context_hash"],
            candidate_content_hash=contract.content_hash(probe_code))
        probe_free_early = probe_passed and _record_closes(_probe_free, probe_code)
        evidence_early = probe_passed and _record_closes(probe_result, probe_code)
        result["evidence_early_return"] = {
            "mode": _mode,
            "probe_free_would_return_early": probe_free_early,
            "evidence_would_return_early": evidence_early,
            "agreement": probe_free_early == evidence_early,
            "adapter": probe_adapter,
            "strength": probe_result["evidence_strength"],
            "closure_eligible": probe_result["closure_eligible"],
            "minimum_closure_strength": adapters.closure_floor(probe_adapter),
        }
        emit("evidence_early_return", f"mode={_mode} probe_free={probe_free_early} "
             f"evidence={evidence_early}", **result["evidence_early_return"])

        if _selection_enabled(_mode):
            _take_early = evidence_early
        else:
            _take_early = probe_free_early

        if _take_early:
            emit("probe_pass", "Probe passed — returning early")
            result["passed"] = True
            result["code"] = probe_code
            result["phase_solved"] = "probe"
            result["candidates_generated"] = 1
            result["total_time_ms"] = (time.time() - start) * 1000
            result["verification_evidence"] = probe_evidence
            result["winning_score"] = probe_energy_norm
            result["events"] = events
            return result

        # The oracle condemned nothing and certified nothing (0/N), and the
        # probe executes. Every candidate this pipeline could generate faces
        # the same broken answer key, so none can be verified either —
        # measured across every logged run: zero candidates ever selected on
        # this path, every session ending in fallback_unverified after
        # burning the full budget (~300s per write), leaving too little
        # session time for the model's own write-run-fix loop. When
        # verification cannot distinguish candidates, generating them buys
        # nothing: return unverified FAST and let the caller's own draft
        # stand. A partial oracle score (some case passed) still runs the
        # full pipeline, because there the suite can actually rank.
        # Untrusted cases never produce the inconclusive verdict above, and
        # the guard says so rather than relying on that: a 0/N from
        # model-generated cases must not skip candidate generation, which is
        # how every session on this path returned nothing at all.
        if _has_oracle and "inconclusive" in (probe_stderr or ""):
            # Flagged recovery before the fast return: bounded input-only
            # consensus (see _dead_oracle_consensus). Default off until an
            # A/B shows the bounded version pays for its latency.
            if (os.environ.get("ATLAS_V3_DEAD_ORACLE_CONSENSUS", "0") == "1"
                    and self_tests and self_tests.test_cases):
                chosen, extra_tokens = _dead_oracle_consensus(
                    problem, task_id, llm, self.plan_search, probe_code,
                    self_tests.test_cases, sandbox, emit, task_input_file,
                    start)
                result["total_tokens"] += extra_tokens
                if chosen is not None:
                    emit("dead_oracle_consensus",
                         "consensus cluster formed under a dead oracle — "
                         "returning the agreed candidate")
                    result["passed"] = True
                    result["code"] = chosen["code"]
                    result["phase_solved"] = "dead_oracle_consensus"
                    result["candidates_generated"] = max(
                        1, chosen["index"] + 1)
                    result["total_time_ms"] = (time.time() - start) * 1000
                    result["events"] = events
                    return result
            emit("probe_unverifiable",
                 "self-test cannot certify anything (0/N) and the probe "
                 "executes — skipping candidate generation, returning "
                 "unverified without spending the budget")
            result["passed"] = False
            result["code"] = ""
            result["phase_solved"] = "oracle_inconclusive"
            result["candidates_generated"] = 1
            result["total_time_ms"] = (time.time() - start) * 1000
            result["events"] = events
            return result

        # ===== PHASE 2: CxGx K ALLOCATION =====
        # The probe failed verification, so this task is not trivial: C(x)
        # picks a base tier, G(x) escalates it, and k never drops below the
        # gate's k=3 floor (what this phase allocated unconditionally
        # before the gate existed).
        #
        # Live-path difference from the bench the gate was measured on: the
        # proxy's V3 bridge abandons this call after ATLAS_V3_TIMEOUT
        # (default 180s), a cap the bench never had. An unbounded escalation
        # to k=8 here would spend the whole budget on generation and hand
        # the user a timeout fallback instead of the k=3 answer the clock
        # could have produced — the failure mode the phase-3 refinement gate
        # already fixes. So the remaining wall-clock and the per-call
        # latency observed on THIS task go into the allocation, and the gate
        # lowers the tier to what the budget can actually generate. The
        # floor is not budget-dependent: k=3 is what would have run anyway.
        check_client()
        emit("phase2", "Allocating compute budget...")
        alloc = cxgx_gate.allocate(
            cx_normalized=probe_energy_norm,
            cx_calibrated=probe_cx_calibrated,
            gx_score=probe_scores["gx_score"],
            gx_available=probe_scores["gx_available"],
            gx_verdict=probe_scores["verdict"],
            remaining_ms=_remaining_budget_ms(start),
            observed_llm_call_ms=getattr(llm, "avg_call_ms", 0.0),
        )
        k, budget_tier = alloc.k, alloc.tier
        bf_tier = budget_tier
        emit("phase2_allocated", f"k={k} tier={budget_tier}",
             k=k, tier=budget_tier, base_tier=alloc.base_tier,
             gx_escalation=alloc.gx_escalation,
             capped_from=alloc.capped_from, reason=alloc.reason)

        # ===== PHASE 1: CONSTRAINT-DIVERSE CANDIDATE GENERATION =====
        emit("phase1", f"Generating {k} diverse candidates...", k=k)
        candidates = []

        def out_of_budget(reserve_ms: Optional[float] = None) -> bool:
            """True when too little of ATLAS_V3_TIMEOUT is left to start more
            work and still hand back a result.

            The reserve defaults to the cost of one more LLM call as observed
            on this run, plus room to serialize the result. A flat reserve is
            useless here: measured 2026-08-03, a phase-3 entry check with 20s
            left passed, PR-CoT then spent 31s on one call and the cap landed
            mid-way through the next. The unit of work is a generation, so
            that is what has to fit.

            The refinement loop was the only phase that checked at all, and it
            checked once. Everything ahead of it — probe, self-tests,
            PlanSearch, sandbox, PR-CoT — ran unguarded.
            """
            left = _remaining_budget_ms(start)
            if left is None:
                return False
            if reserve_ms is None:
                observed = getattr(llm, "avg_call_ms", 0.0) or 0.0
                reserve_ms = max(20000.0, observed * 1.2 + 10000.0)
            return left < reserve_ms

        def finish_with_best(reason: str) -> Dict[str, Any]:
            """Hand back the best candidate found so far.

            V3 is an anytime algorithm — the refinement loop is budget-gated,
            so it expands to fill the budget rather than converging early.
            Measured 2026-08-03 on one task: a 180s budget produced 9 LLM
            calls and 5 sandbox verifications, a 420s budget 21 calls and 10.
            It is not interrupted mid-something-it-would-finish; the clock is
            its terminal condition.

            An anytime algorithm whose clock expires has to return its best
            answer. This one returned nothing: the caller's deadline cancelled
            the request, every verified candidate was discarded, and the write
            fell back to the model's own output — 16 of 41 write calls in one
            28-session run, ~48 of the 56 minutes spent in them.

            Vetoed candidates stay excluded for the same reason the end-of-run
            fallback excludes them: "executes but is wrong" is worse than an
            honest failure.
            """
            emit("budget_exhausted", reason,
                 candidates=len(candidates),
                 remaining_ms=round(_remaining_budget_ms(start) or 0))
            pool = [c for c in candidates if not c.get("vetoed_by")]
            passing = [c for c in pool if c.get("passed")]
            chosen = None
            if passing:
                passing.sort(key=lambda c: c.get("energy", 999))
                chosen = passing[0]
                result["passed"] = True
                result["phase_solved"] = "budget"
            else:
                # No code rather than an unverified candidate. The caller's
                # baseline is the model's own write, which is syntax- and
                # structure-gated; a candidate that failed the sandbox is
                # not better than that, and ranking the failures by energy
                # picks among them without evidence.
                #
                # Measured across a 28-session run: 7 of 8 returns took the
                # unverified path and task success fell 20/28 to 17/28 —
                # aoc_course and aoc_slope went 2/2 to 0/2 — while harness
                # integrity reached 28/28. The same reasoning the vetoed
                # path already uses: "executes but is wrong" is worse than
                # an honest failure.
                emit("budget_no_verified_candidate",
                     f"{len(pool)} candidate(s), none verified — "
                     f"leaving the caller's gated baseline in place")
            capture.note_pool(
                phase="budget", pool=passing,
                lens_index=(chosen or {}).get("index"),
                status="budget_exhausted", reason=reason,
                ineligible=len(pool) - len(passing))
            if chosen is not None:
                result["code"] = chosen["code"]
                result["verification_evidence"] = chosen.get("verification_evidence", [])
                result["winning_score"] = chosen.get("energy_norm", 0.0)
            result["total_time_ms"] = (time.time() - start) * 1000
            result["events"] = events
            return result
        try:

            # Start with probe if it produced code
            if probe_code:
                candidates.append({
                    # Phase-zero evidence is cached here so candidate zero is
                    # never re-probed, and never enters selection with defaults.
                    # One canonical record, no parallel copies of its fields.
                    "contract_record": probe_result,
                    "index": 0, "code": probe_code,
                    "energy": probe_energy_raw, "energy_norm": probe_energy_norm,
                    "energy_calibrated": probe_cx_calibrated,
                    "passed": probe_passed, "stdout": "", "stderr": "",
                })

            remaining_k = max(0, k - len(candidates))

            # Step 1A: PlanSearch
            if remaining_k > 0:
                emit("plansearch", f"Generating {remaining_k} plans...",
                     plans=remaining_k)
                try:
                    ps_result = self.plan_search.generate(
                        problem, task_id, llm, num_plans=remaining_k,
                        # The allocator's tier, not the signature default:
                        # generate() was silently running "standard" thinking
                        # depth whatever the CxGx gate decided (audit finding).
                        budget_tier=budget_tier,
                    )
                    for i, code in enumerate(ps_result.candidates):
                        if code:
                            energy_raw, energy_norm, energy_calibrated = scoring.score_candidate(code)
                            per_step = scoring.score_candidate_per_step(code)  # PC-207
                            cand_index = len(candidates)
                            candidates.append({
                                "index": cand_index, "code": code,
                                "energy": energy_raw, "energy_norm": energy_norm,
                                "energy_calibrated": energy_calibrated,
                                "passed": False, "stdout": "", "stderr": "",
                                "per_step": per_step,
                            })
                            if per_step:
                                emit("lens_per_step",
                                     f"cand {cand_index}: gx_min={per_step['gx_score_min']:.2f} "
                                     f"first_off_rails={per_step['first_off_rails_idx']}",
                                     index=cand_index,
                                     source="plansearch",
                                     first_off_rails_idx=per_step["first_off_rails_idx"],
                                     gx_score_min=per_step["gx_score_min"],
                                     gx_score_mean=per_step["gx_score_mean"],
                                     cx_norm_max=per_step["cx_norm_max"],
                                     n_tokens=per_step["n_tokens"])
                    result["total_tokens"] += ps_result.total_tokens
                    emit("plansearch_done",
                         f"{len(ps_result.candidates)} candidates from PlanSearch",
                         candidates=len(ps_result.candidates),
                         tokens=ps_result.total_tokens)
                except Exception as e:
                    emit("plansearch_error", str(e)[:200])

            # Step 1B: DivSampling to fill remaining slots
            remaining_k = max(0, k - len(candidates))
            if remaining_k > 0:
                emit("divsampling", f"Filling {remaining_k} slots with diverse sampling...",
                     slots=remaining_k)
                for idx in range(remaining_k):
                    check_client()
                    if out_of_budget():
                        emit("divsampling_stop",
                             f"budget spent after {len(candidates)} candidate(s)")
                        break
                    try:
                        perturbed = self.div_sampling.apply(problem, len(candidates) + idx, task_id)
                        chatml = self.budget_forcing.format_chatml(perturbed, bf_tier)
                        response, tokens, t_ms = llm(
                            chatml, DIVERSITY_TEMPERATURE,
                            self.budget_forcing.get_max_tokens(bf_tier),
                            42 + len(candidates) + idx,
                        )
                        code = extract_code(response)
                        if code:
                            energy_raw, energy_norm, energy_calibrated = scoring.score_candidate(code)
                            per_step = scoring.score_candidate_per_step(code)  # PC-207
                            cand_index = len(candidates)
                            candidates.append({
                                "index": cand_index, "code": code,
                                "energy": energy_raw, "energy_norm": energy_norm,
                                "energy_calibrated": energy_calibrated,
                                "passed": False, "stdout": "", "stderr": "",
                                "per_step": per_step,
                            })
                            if per_step:
                                emit("lens_per_step",
                                     f"cand {cand_index}: gx_min={per_step['gx_score_min']:.2f} "
                                     f"first_off_rails={per_step['first_off_rails_idx']}",
                                     index=cand_index,
                                     source="divsampling",
                                     first_off_rails_idx=per_step["first_off_rails_idx"],
                                     gx_score_min=per_step["gx_score_min"],
                                     gx_score_mean=per_step["gx_score_mean"],
                                     cx_norm_max=per_step["cx_norm_max"],
                                     n_tokens=per_step["n_tokens"])
                        result["total_tokens"] += tokens
                    except Exception as e:
                        emit("divsampling_error", str(e)[:200])
                emit("divsampling_done", f"{len(candidates)} total candidates",
                     total=len(candidates))

            result["candidates_generated"] = len(candidates)

            # ===== SANDBOX TESTING =====
            emit("sandbox_test", f"Testing {len(candidates)} candidates...",
                 candidates=len(candidates))
            # Sort by energy (easy first) for early-exit potential
            candidates.sort(key=lambda c: c.get("energy", 0))

            passing = []
            for c in candidates:
                check_client()
                if c.get("passed"):
                    # Cached execution result -- do NOT skip evidence. This
                    # `continue` is exactly how candidate zero escaped the
                    # "one evaluation path": compile smoke set passed=True for
                    # an interactive artifact, so the real Snake candidate
                    # entered the pool with no behavioural evidence at all and
                    # rank_key saw defaults for it.
                    if "contract_record" not in c:
                        c["contract_record"] = _evaluate_candidate(
                            file_path, c["code"], True, _has_oracle, emit, sandbox,
                            task=_task)
                    _capture_pool_member(capture, c, probe_code)
                    passing.append(c)
                    continue
                sb_start = time.time()
                passed, stdout, stderr, verification_evidence = verified_sandbox(c["code"])
                sb_ms = int((time.time() - sb_start) * 1000)
                c["passed"] = passed
                c["stdout"] = stdout
                c["stderr"] = stderr
                c["verification_evidence"] = verification_evidence
                # ONE structured evaluation path: candidate zero and every
                # generated candidate get the same adapter, the same probe and
                # the same evidence record. Two sets of verification semantics
                # is how the boolean survived into the candidate path, letting
                # ATLAS generate alternatives it could not rank.
                c["contract_record"] = _evaluate_candidate(
                    file_path, c["code"], passed, _has_oracle, emit, sandbox,
                    task=_task)
                _capture_pool_member(capture, c, probe_code)
                if passed:
                    passing.append(c)
                    emit("sandbox_pass", f"Candidate {c['index']} passed",
                         index=c["index"], elapsed_ms=sb_ms,
                         energy=c.get("energy_norm", 0.0))
                else:
                    emit("sandbox_fail", f"Candidate {c['index']} failed",
                         index=c["index"], elapsed_ms=sb_ms,
                         stderr=(stderr or "")[:120])

            emit("sandbox_done", f"{len(passing)}/{len(candidates)} passed",
                 passed=len(passing), total=len(candidates))

            # Counterfactual ranking, recorded and acted on by nothing. It
            # runs the candidates on the generated INPUTS and reports who
            # agreed; the generated expected outputs never enter it. Off
            # unless ATLAS_EVIDENCE_MODE turns probing on, because it costs
            # one sandbox run per candidate per case. Candidate zero is in
            # this pool like any other candidate and may rank first.
            if (_probing_enabled(_selection_mode()) and len(passing) >= 2
                    and self_tests and getattr(self_tests, "test_cases", None)):
                try:
                    result["consensus"] = _consensus_record(
                        passing, self_tests.test_cases, sandbox,
                        task_input_file)
                    capture.note_consensus(result["consensus"])
                    emit("consensus_ranking",
                         f"{result['consensus']['agreement']}/{len(passing)} "
                         f"agree on every input — ranking evidence only",
                         agreement=result["consensus"]["agreement"],
                         groups=len(result["consensus"]["groups"]),
                         tied_groups=result["consensus"]["tied_groups"])
                except Exception as exc:               # noqa: BLE001
                    emit("consensus_ranking", f"unavailable: {str(exc)[:120]}")

            # Nothing passed, which for these tasks usually means the answer
            # key was wrong rather than every candidate. Measured across 42
            # verifications in one run: all 42 scored 0/N, and a candidate
            # pulled from those logs passed immediately against a correct
            # expected value. The pipeline has never selected a candidate in
            # any measured run — every session shipped the model's own draft.
            #
            # Fall back to the agreement signal CodeT actually uses: run the
            # candidates on the generated INPUTS and take the largest cluster
            # that produced the same answers. Narrow on purpose — this only
            # runs where the oracle has already condemned everything, so a
            # working suite keeps deciding.
            if not passing and self_tests and self_tests.test_cases:
                agreed = _consensus_winners(
                    candidates, self_tests.test_cases, sandbox, emit,
                    task_input_file)
                for c in agreed:
                    c["passed"] = True
                    passing.append(c)
                if agreed:
                    emit("sandbox_done",
                         f"{len(passing)}/{len(candidates)} by consensus",
                         passed=len(passing), total=len(candidates))

            # ===== LENS VETO =====
            # PC-207 alignment fix: hard-reject sandbox-passing candidates whose
            # geometric-lens gx_min sits below THIS model's calibrated severe band.
            # Sandbox is an ORM (does it execute?), lens is a PRM (is the
            # generation pattern collapsing into a stub?) — they answer
            # different questions. The May 7 dashboard.html session shipped
            # a 10-line `<h1>Dashboard</h1>` stub because sandbox said pass
            # while lens said gx_min=0.069. Without this filter, V3 returns
            # passed=True and the proxy's PC-044 nudges the agent to done.
            #
            # Language-agnostic by construction: the lens runs on the model's
            # residual stream; gx values don't depend on whether the file
            # being scored is HTML, Python, Rust, or Java.
            if passing:
                kept, vetoed = [], []
                for c in passing:
                    per_step = c.get("per_step") or {}
                    # Vetoes read the MEAN, not the MIN.
                    #
                    # gx_score_min is a minimum over every token, so it falls
                    # with length whatever the content: measured 2026-08-04 on
                    # one function repeated, 0.468 at 20 tokens down to 0.305
                    # at 305. It cannot separate — real code 0.325, a
                    # repetition loop 0.320, stub spam 0.286 all sit together,
                    # which is why severe=0.28 never fired once in 56 sessions.
                    #
                    # gx_score_mean holds across the same 15x length change
                    # (0.577 to 0.517) and does separate: real code 0.594,
                    # repetition 0.485, stub spam 0.467. Across 188 scores
                    # from live runs it ranged 0.547-0.651, so severe_mean at
                    # 0.52 sits below every clean sample observed and above
                    # both pathologies.
                    gx_mean = per_step.get("gx_score_mean")
                    gx_min = per_step.get("gx_score_min")
                    thresholds = per_step.get("thresholds") or {}
                    severe_mean = thresholds.get("severe_mean")
                    severe = thresholds.get("severe")
                    if (gx_mean is not None
                            and isinstance(severe_mean, (int, float))):
                        vetoed_now = gx_mean < severe_mean
                        gx_min, severe = gx_mean, severe_mean
                    else:
                        # Artifacts predating severe_mean keep the old check.
                        vetoed_now = (gx_min is not None
                                      and isinstance(severe, (int, float))
                                      and gx_min < severe)
                    if vetoed_now:
                        # A vetoed candidate is a failing candidate: mark it so
                        # the phase-3 pool (`not c.get("passed")`) picks it up
                        # and the energy fallback can never return it. The veto
                        # reason replaces the (empty) passing-run stderr so
                        # repair sees WHY it was rejected.
                        c["passed"] = False
                        c["vetoed_by"] = "lens"
                        c["stderr"] = (
                            f"lens veto: gx_min={gx_min:.3f} below the severe "
                            f"threshold {severe:.3f} — generation pattern "
                            f"collapsed toward a stub; the code executes but "
                            f"likely does not implement the task")
                        vetoed.append(c)
                        emit("lens_veto",
                             f"Candidate {c['index']} sandbox-passed but lens-vetoed "
                             f"(gx_min={gx_min:.3f} < {severe:.3f}) — likely a stub",
                             index=c["index"], gx_score_min=gx_min,
                             first_off_rails_idx=per_step.get("first_off_rails_idx", -1))
                    else:
                        kept.append(c)
                if vetoed:
                    print(
                        f"  [lens] vetoed {len(vetoed)}/{len(passing)} sandbox-passing "
                        f"candidates using per-model severe thresholds — falling "
                        f"{'through to phase-3 repair' if not kept else 'back to remaining %d' % len(kept)}",
                        flush=True,
                    )
                passing = kept

            # ===== STRUCTURAL VETO =====
            # GH #39 point 1: hard-reject candidates whose direct-identifier
            # calls don't resolve against (local defs, imports, builtins,
            # project symbols). Sandbox can pass for code where the unresolved
            # call is in a try/except ImportError fallback or a dead branch
            # that doesn't execute under the tests; tree-sitter sees the
            # surface bug regardless. Same architecture as lens veto.
            #
            # Language-agnostic fit: v1 supports Python only (matches the
            # rest of the GH #39 stack), but the resolution-order pattern
            # generalizes to any language with explicit imports + named
            # functions (Go, Rust, JS/TS modules). Adding a language adds
            # implementation surface, not model-facing API surface.
            if passing:
                # #147: gate on `passing` alone, not `passing and files`. The
                # edit path (improveContentWithV3) frequently sends no
                # project_context, so `files` was empty and the whole veto was
                # skipped — a NameError edit (render_template called with only
                # render_template_string imported) sailed through and landed as
                # verified. structural_score resolves against the candidate's
                # OWN imports/defs/builtins, so it catches an unresolved direct
                # call with empty project_symbols; project symbols only add
                # lenient cross-file crediting.
                project_symbols = symbols.build_project_symbols(files or {})
                kept = []
                for c in passing:
                    struct = symbols.structural_score(project_symbols, c.get("code", ""))
                    if struct.get("ok") and struct.get("n_unresolved", 0) >= 1:
                        # Same contract as the lens veto: vetoed = failing.
                        c["passed"] = False
                        c["vetoed_by"] = "structural"
                        c["stderr"] = (
                            "structural veto: unresolved direct call(s) that "
                            "would raise NameError at runtime: "
                            + ", ".join(struct["unresolved_calls"][:5]))
                        emit("structural_veto",
                             f"Candidate {c['index']} sandbox-passed but "
                             f"{struct['n_unresolved']} unresolved call(s): "
                             f"{', '.join(struct['unresolved_calls'][:3])}",
                             index=c["index"],
                             n_unresolved=struct["n_unresolved"],
                             unresolved_calls=struct["unresolved_calls"][:5],
                             n_calls_total=struct["n_calls_total"])
                        print(
                            f"  [structural] vetoed cand {c['index']} — "
                            f"{struct['n_unresolved']} unresolved: {struct['unresolved_calls'][:5]}",
                            flush=True,
                        )
                        continue
                    if struct.get("ok"):
                        c["structural"] = struct  # stash for phase 3 / repair
                    kept.append(c)
                if len(kept) < len(passing):
                    print(
                        f"  [structural] kept {len(kept)}/{len(passing)} candidates after structural veto"
                        f"{' — falling through to phase-3 repair' if not kept else ''}",
                        flush=True,
                    )
                passing = kept

            # ===== CALL-GRAPH VETO (issue #39, Phase 1) =====
            # Deepens the structural veto using the import graph: reject a candidate
            # whose direct calls don't resolve to a real, in-scope definition (local,
            # builtin, imported, or supplied by a resolved wildcard) — not merely
            # "some project file defines that name." Catches broken cross-file
            # references the shipped veto accepts. Flag-gated by ATLAS_CALL_GRAPH;
            # conservative — stays lenient on opaque wildcards and never empties the
            # candidate set (a fully-failing set falls through intact to repair).
            if passing and files and file_path:
                try:
                    from graph import call_graph_enabled, unresolved_calls
                    _cg_on = call_graph_enabled()
                except Exception:
                    _cg_on = False
                if _cg_on:
                    cg_kept, cg_vetoed = [], []
                    for c in passing:
                        try:
                            res = unresolved_calls(
                                file_path, c.get("code", ""), files, strict=True)
                        except Exception as cge:
                            print(f"  [call_graph] veto skipped for cand {c.get('index')}: {cge}",
                                  flush=True)
                            cg_kept.append(c)
                            continue
                        if res.get("ok") and res.get("unresolved"):
                            cg_vetoed.append((c, res["unresolved"]))
                            continue
                        cg_kept.append(c)
                    if cg_kept:  # only prune when at least one candidate survives
                        # Marking happens only when the prune actually applies —
                        # the conservative all-vetoed case keeps the full set
                        # (and its passed flags) intact.
                        for c, unresolved in cg_vetoed:
                            c["passed"] = False
                            c["vetoed_by"] = "call_graph"
                            c["stderr"] = (
                                "call-graph veto: cross-file call(s) that resolve "
                                "to no in-scope definition: "
                                + ", ".join(unresolved[:5]))
                            emit("call_graph_veto",
                                 f"Candidate {c.get('index')} has unresolved call(s): "
                                 f"{', '.join(unresolved[:3])}",
                                 index=c.get("index"), unresolved=unresolved[:5])
                            print(f"  [call_graph] vetoed cand {c.get('index')} — "
                                  f"unresolved: {unresolved[:5]}", flush=True)
                        passing = cg_kept

            # ===== CANDIDATE SELECTION =====
            # Lens selection: minimum C(x) energy among the passing candidates.
            # (S* tiebreaking used to run first for 2+ passers; across 118 H200
            # tiebreaks every pair scored 0-0 and 110/110 winners equaled the
            # lens min-energy pick, so it carried zero discriminating signal.)
            if passing:
                ci_list = [
                    CandidateInfo(c["index"], c["code"], c["energy"], c["passed"])
                    for c in passing
                ]
                selected = select_candidate(ci_list, strategy="lens")

                # Evidence-based ranking. SHADOW BY DEFAULT: it records what it
                # would have chosen and changes nothing, so uplift can be shown
                # across task families before it touches a live decision.
                # ATLAS_EVIDENCE_SELECTION=1 promotes it to the real selector.
                # Evidence-based selection. SHADOW BY DEFAULT: it records what
                # it would have chosen and changes nothing, so uplift can be
                # shown across task families before it touches a live decision.
                # The choice is contract.select over the candidates' own
                # records, under the rubric the BASELINE was measured with --
                # records that disagree about contract, artifact or context are
                # incomparable rather than silently ranked, so a foreign
                # majority cannot outvote one matching record.
                pool = [c for c in passing if c.get("contract_record")]
                expected = result.get("evidence_record") or (
                    pool[0]["contract_record"] if pool else None)
                contract_pick = None
                if pool and expected:
                    try:
                        picked = contract.select(
                            [c["contract_record"] for c in pool], expected)
                    except contract.ContractError as exc:
                        picked = {"best_record": None, "verified_winner": None,
                                  "tied": [], "incomparable": [], "ineligible": [],
                                  "selection_reason": f"identity error: {exc}"}
                    by_record = {id(c["contract_record"]): c for c in pool}
                    best = by_record.get(id(picked.get("best_record")))
                    verified = by_record.get(id(picked.get("verified_winner")))
                    # A record must describe the bytes it is attached to before
                    # it may win anything.
                    if verified and not _record_closes(
                            verified["contract_record"], verified["code"]):
                        verified = None
                    contract_pick = verified or best
                    status = contract.selection_status(picked)
                    result["contract_selection"] = picked
                    result["evidence_selection"] = {
                        "lens_index": getattr(selected, "index", None),
                        "evidence_index": (contract_pick or {}).get("index"),
                        "verified_index": (verified or {}).get("index"),
                        "agree": getattr(selected, "index", None)
                                 == (contract_pick or {}).get("index"),
                        "status": status,
                        "reason": picked.get("selection_reason", ""),
                        "tied": len(picked.get("tied") or []),
                        "incomparable": len(picked.get("incomparable") or []),
                        "ineligible": len(picked.get("ineligible") or []),
                        "candidates": [
                            {"index": c.get("index"),
                             "strength": c["contract_record"]["evidence_strength"],
                             "adapter": c["contract_record"]["adapter_id"],
                             "supported": c["contract_record"]["supported"],
                             "closure_eligible": c["contract_record"]["closure_eligible"],
                             "quality": c["contract_record"]["overall_quality_score"],
                             "missing_required": c["contract_record"]["missing_required"],
                             "energy": c.get("energy")}
                            for c in pool],
                    }
                    capture.note_pool(
                        phase="phase1", pool=pool,
                        lens_index=getattr(selected, "index", None),
                        evidence_index=(contract_pick or {}).get("index"),
                        verified_index=(verified or {}).get("index"),
                        status=status, reason=picked.get("selection_reason", ""),
                        tied=len(picked.get("tied") or []),
                        incomparable=len(picked.get("incomparable") or []),
                        ineligible=len(picked.get("ineligible") or []))
                    emit("evidence_shadow",
                         f"lens picked {getattr(selected, 'index', None)}, "
                         f"contract would pick {(contract_pick or {}).get('index')} "
                         f"({status})",
                         **{k: v for k, v in result["evidence_selection"].items()
                            if k != "candidates"})
                    # Only a VERIFIED winner may replace the lens choice. A best
                    # record that is not closure-eligible is diagnostic: turning
                    # it into the delivered artifact is exactly how a partial
                    # result becomes a success claim.
                    if _selection_enabled(_selection_mode()) and verified:
                        selected = CandidateInfo(verified["index"], verified["code"],
                                                 verified.get("energy", 0.0), True)

                if selected:
                    emit("selected", f"Lens selected candidate {selected.index}",
                         index=selected.index, energy=getattr(selected, "energy", 0.0))
                    result["passed"] = True
                    result["code"] = selected.code
                    result["phase_solved"] = "phase1"
                    result["total_time_ms"] = (time.time() - start) * 1000
                    winner = _candidate_by_index(passing, selected.index)
                    result["verification_evidence"] = (winner or {}).get("verification_evidence", [])
                    result["winning_score"] = (winner or {}).get("energy_norm", 0.0)
                    result["evidence_record"] = ((winner or {}).get("contract_record")
                                                 or result.get("evidence_record"))
                    result["events"] = events
                    return result

            # ===== PHASE 3: VERIFIED ITERATIVE REFINEMENT =====
            check_client()
            if out_of_budget():
                return finish_with_best("budget spent before the repair phase")
            emit("phase3", "All candidates failed — entering repair phase...",
                 failing=len([c for c in candidates if not c.get("passed")]))

            failing = [
                FailingCandidate(
                    index=c["index"], code=c["code"],
                    error_output=c.get("stderr", ""),
                )
                for c in candidates if not c.get("passed")
            ]

            # Repair verifies against the SAME self-tests phase 0 generated —
            # verified_sandbox closes over them. Regenerate only when phase 0
            # produced none (e.g. a transient LLM failure); a failed retry here
            # must not downgrade an existing good set to None. Interactive
            # tasks repair against compile-smoke (PC-022).
            if task_type == "algorithmic" and not (self_tests and self_tests.test_cases):
                emit("self_test_gen", "Generating self-tests...")
                try:
                    self_tests = self.self_test_gen.generate(problem, llm, task_id)
                    emit("self_test_done", f"{len(self_tests.test_cases)} test cases generated")
                    result["total_tokens"] += self_tests.generation_tokens
                except Exception as e:
                    emit("self_test_error", str(e)[:200])

            # GH #39 point 3: build call-graph context for the failing
            # function once, reuse across PR-CoT + refinement. Skips
            # cleanly when stderr isn't a Python traceback or the failing
            # function isn't defined in the project — both arms get plain
            # error_output in that case. When ATLAS_CALL_GRAPH is on the
            # block is a multi-hop reachability slice (entry-point path,
            # transitive impact, callees); flag-off it stays at direct
            # callers/callees (1 hop). Fail-soft on any graph failure.
            chain_context_block = ""
            if failing:
                failing_func = symbols._failing_function_from_stderr(failing[0].error_output)
                if failing_func and files:
                    try:
                        from graph import call_graph_enabled as _cg_on, repair_context as _cg_repair
                        chain_context_block = _cg_repair(files, failing_func, transitive=_cg_on())
                    except Exception as cge:
                        print(f"  [phase3] graph repair-context skipped: {cge}", flush=True)
                    if chain_context_block:
                        emit("call_chain_context",
                             f"Built call-chain for failing `{failing_func}`",
                             function=failing_func)
                        print(
                            f"  [phase3] call-chain context built for `{failing_func}`",
                            flush=True,
                        )

            def _enriched_error(stderr: str) -> str:
                """Append call-chain context to a candidate's stderr if available."""
                if not chain_context_block:
                    return stderr
                return (stderr or "") + "\n\n" + chain_context_block

            # Strategy 1: PR-CoT Quick Repair
            if failing and out_of_budget():
                return finish_with_best("budget spent before PR-CoT repair")
            if failing:
                emit("pr_cot", "Attempting PR-CoT repair...",
                     strategy="pr_cot", failing=len(failing))
                best_failing = failing[0]
                try:
                    pr_result = self.pr_cot.repair(
                        problem=problem,
                        code=best_failing.code,
                        error=_enriched_error(best_failing.error_output),
                        llm_call=llm,
                        task_id=task_id,
                    )
                    result["total_tokens"] += pr_result.total_tokens
                    for repair_code in pr_result.repairs:
                        passed, stdout, stderr, repair_evidence = verified_sandbox(repair_code)
                        capture.note_candidate(
                            role="repair", index=None, code=repair_code,
                            accepted=passed, record=None, phase="repair_pr_cot")
                        if passed:
                            emit("pr_cot_pass", "PR-CoT repair succeeded!",
                                 strategy="pr_cot", tokens=pr_result.total_tokens)
                            result["passed"] = True
                            result["code"] = repair_code
                            result["phase_solved"] = "pr_cot"
                            result["total_time_ms"] = (time.time() - start) * 1000
                            result["verification_evidence"] = repair_evidence
                            result["events"] = events
                            return result
                    emit("pr_cot_failed", "PR-CoT repair did not produce passing code")
                except Exception as e:
                    emit("pr_cot_error", str(e)[:200])

            # Strategy 2: Refinement Loop — entered only when the remaining
            # wall-clock can afford one iteration. H200 join: 453/487
            # refinement entries timed out with ZERO completed iterations
            # while burning ~6 minutes each; one iteration is ~3 sequential
            # LLM calls, estimated at the per-call latency observed on THIS
            # run. The budget is the ATLAS_V3_TIMEOUT cap the proxy's V3
            # bridge enforces — starting work the bridge will abandon only
            # delays the fallback the user ends up with.
            run_refinement = bool(failing)
            if run_refinement:
                est_ms = estimate_iteration_ms(getattr(llm, "avg_call_ms", 0.0))
                remaining_ms = _remaining_budget_ms(start)
                if (remaining_ms is not None
                        and not can_afford_iteration(remaining_ms, est_ms)):
                    run_refinement = False
                    emit("refinement_skip",
                         f"remaining budget {remaining_ms / 1000:.0f}s cannot "
                         f"afford one iteration (~{est_ms / 1000:.0f}s) — "
                         f"skipping to fallback",
                         strategy="refinement",
                         remaining_ms=round(remaining_ms),
                         estimated_iteration_ms=round(est_ms))
            if run_refinement and out_of_budget():
                run_refinement = False
            if run_refinement:
                check_client()
                emit("refinement", "Starting refinement loop...",
                     strategy="refinement", failing=len(failing))
                # GH #39 point 3: enrich each failing candidate's error_output
                # with call-chain context so the refinement loop sees it on
                # every iteration. Cheap (chain_context_block is built once
                # above and reused).
                failing_for_refinement = failing
                if chain_context_block:
                    failing_for_refinement = [
                        FailingCandidate(
                            index=c.index,
                            code=c.code,
                            error_output=_enriched_error(c.error_output),
                        )
                        for c in failing
                    ]
                try:
                    ref_result = self.refinement_loop.run(
                        problem=problem,
                        failing_candidates=failing_for_refinement,
                        original_constraints=[],
                        llm_call=llm,
                        sandbox_run=sandbox,
                        embed_call=embed,
                        task_id=task_id,
                    )
                    result["total_tokens"] += ref_result.total_tokens
                    if ref_result.solved:
                        passed, stdout, stderr, refinement_evidence = verified_sandbox(ref_result.winning_code)
                        capture.note_candidate(
                            role="refinement", index=None,
                            code=ref_result.winning_code, accepted=passed,
                            record=None, phase="refinement")
                        if passed:
                            emit("refinement_pass",
                                 f"Refinement solved in {ref_result.total_iterations} iterations!",
                                 strategy="refinement",
                                 iterations=ref_result.total_iterations,
                                 tokens=ref_result.total_tokens)
                            result["passed"] = True
                            result["code"] = ref_result.winning_code
                            result["phase_solved"] = "refinement"
                            result["total_time_ms"] = (time.time() - start) * 1000
                            result["verification_evidence"] = refinement_evidence
                            result["events"] = events
                            return result
                        emit("refinement_verify_failed", (stderr or "")[:200])
                    emit("refinement_failed", f"Exhausted {ref_result.total_iterations} iterations")
                except Exception as e:
                    emit("refinement_error", str(e)[:200])

            # ===== FALLBACK: Return best candidate even if none passed =====
            # Vetoed candidates are excluded outright: a veto means "executes
            # but is wrong" (stub, NameError-in-waiting), which is worse than
            # an honest sandbox failure — and returning one is exactly the
            # May 7 dashboard-stub failure mode. If every candidate was
            # vetoed, return no code; the caller falls back to its baseline.
            # Nothing verified, so nothing is returned. The caller's
            # baseline is the model's own write, which is syntax- and
            # structure-gated; a candidate that failed the sandbox is not
            # better than that, and ranking failures by energy picks among
            # them without evidence.
            #
            # Measured: across one 28-session run, 0 of 44 candidates passed
            # the sandbox, and this path still handed back a failing one 11
            # times — the proxy logged each as a V3 write and put it on disk
            # over the model's own. The run that shipped that behaviour
            # scored 20/28 to 17/28 against the run that did not.
            #
            # Same reasoning the vetoed branch below already used, and the
            # same as the budget boundary: "executes but is wrong" is worse
            # than an honest failure.
            unverified = [c for c in candidates if not c.get("vetoed_by")]
            if unverified:
                emit("fallback_unverified",
                     f"{len(unverified)} candidate(s), none passed verification — "
                     f"leaving the caller's gated baseline in place")
            elif candidates:
                emit("fallback_all_vetoed",
                     "Every candidate was vetoed — returning no code")
            result["total_time_ms"] = (time.time() - start) * 1000
            result["events"] = events
            return result
        except adapters.BudgetExhausted as exc:
            # An anytime algorithm whose clock expires owes its caller the
            # best answer it has. Raised from the adapter rather than
            # checked at phase boundaries: every phase runs its own loop —
            # PR-CoT alone issues two calls — so a boundary check that
            # reserves one call is already wrong by the second.
            return finish_with_best(f"budget exhausted mid-pipeline ({exc})")


# --- Problem Builder for /v3/generate ----------------------------------------

def _build_problem_from_request(
    file_path: str, baseline_code: str, project_context: Dict[str, str],
    framework: str, build_command: str, constraints: List[str],
    user_message: str = "",
) -> str:
    """Build a problem description for the V3 pipeline from a generate request.

    This is the ONLY place `baseline_code` is used. The incumbent becomes
    prose in the prompt; it receives no adapter, no contract record, no
    execution, no consensus probe, no lens score, and no place in `passing`
    or the selection pool. Pool index 0 is the phase-zero probe candidate --
    a fresh generation -- not the incumbent, whatever the "candidate #0"
    comments elsewhere say. So a selection that concludes
    `best_not_closure_eligible` has ranked the V3-generated candidates
    against each other and has said nothing about whether the winner beats
    the artifact it would replace.

    The user's own request leads, when the caller sends one. Without it the
    pipeline saw only "Create the file X", the project context and the
    baseline, under an instruction to improve on the baseline "preserving all
    functionality" — so every candidate could only mimic a draft whose
    requirement it had never been shown, and a baseline that misread the task
    was reproduced rather than corrected.

    Measured on the AoC tasks, whose prompt states "reads input.txt": 9 of the
    12 solutions ATLAS produced read stdin instead, and the caller runs
    `python solve.py` with no stdin. The same model given the task directly
    wrote file readers 12 times out of 12.
    """
    parts = []

    if user_message.strip():
        parts.append("## The request\n\n")
        parts.append(user_message.strip() + "\n\n")

    parts.append(f"Create the file `{file_path}`")
    if framework:
        parts.append(f" for a {framework} project")
    parts.append(".\n\n")

    # Project context
    if project_context:
        parts.append("## Existing project files:\n\n")
        for path, content in project_context.items():
            if len(content) < 500:
                parts.append(f"### {path}\n```\n{content}\n```\n\n")
            else:
                parts.append(f"### {path} (truncated)\n```\n{content[:300]}\n...\n```\n\n")

    # Constraints
    if constraints:
        parts.append("## Requirements:\n")
        for c in constraints:
            parts.append(f"- {c}\n")
        parts.append("\n")

    # Build command
    if build_command:
        parts.append(f"## Build verification:\nThe file must pass: `{build_command}`\n\n")

    # Baseline as reference
    if baseline_code:
        parts.append("## Reference implementation:\n")
        parts.append("Improve upon this baseline if possible, preserving all functionality.\n\n")
        parts.append(f"```\n{baseline_code}\n```\n")

    return "".join(parts)
