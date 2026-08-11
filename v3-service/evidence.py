"""Evidence strength, and the behavioural probe that can produce it.

The defect this exists to fix: one `passed` boolean carried every kind of
verification, so a compile-smoke result on browser JavaScript was
indistinguishable from a passing I/O oracle. Phase 0 returned early on it,
`candidates_generated` was 1, and PlanSearch / DivSampling / consensus /
ranking never ran. ATLAS paid orchestration latency without applying any of
its test-time compute.

Two ideas here:

  * STRENGTH — what a verifier actually demonstrated, on an ordered scale.
    "It parses" and "it plays" are no longer the same fact.
  * COVERAGE — which required behaviours were demonstrated. Strength alone
    is not enough: a probe that shows a loop ticks has not shown collision
    works, and must not close the pipeline as if it had.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

# Ordered weakest -> strongest. Compare with STRENGTH_ORDER.index().
NONE = "none"
SYNTAX = "syntax"
RUNTIME = "runtime"
BEHAVIORAL_PARTIAL = "behavioral_partial"
BEHAVIORAL_COMPLETE = "behavioral_complete"

STRENGTH_ORDER = [NONE, SYNTAX, RUNTIME, BEHAVIORAL_PARTIAL, BEHAVIORAL_COMPLETE]


def at_least(strength: str, floor: str) -> bool:
    try:
        return STRENGTH_ORDER.index(strength) >= STRENGTH_ORDER.index(floor)
    except ValueError:
        return False


def may_return_early(strength: str, missing_required: List[str],
                     behavior_score: float = 0.0) -> bool:
    """Early return needs COMPLETE behavioural evidence, not merely strong.

    Accepting BEHAVIORAL_PARTIAL was a contradiction: the real ATLAS artifact
    grades temporal_progress + input_causality + collision but no score
    transition -- 0.75, no *required* criterion missing, so it returned early.
    The pipeline would have measured itself inferior to the bare model's 1.00
    and then declined to generate a single alternative. Detecting a weakness
    and acting on it are different things, and only the second one amplifies.

    A partial candidate is still perfectly usable as candidate #0; it just
    does not get to close the pipeline.
    """
    if missing_required:
        return False
    if strength != BEHAVIORAL_COMPLETE:
        return False
    return behavior_score >= 1.0


# --------------------------------------------------------------- JS probe ---

# Instrumentation only. It never inspects the artifact's identifiers, so it
# does not care what the author named the snake, the loop, or the score.
_JS_HARNESS = r"""
// Fully deterministic instrumentation: a VIRTUAL clock, no wall time.
//
// The previous version used real setTimeout across two separate processes,
// so OS scheduling jitter could give the baseline and keyed runs different
// frame counts -- and a raw trace diff then read that as "input caused a
// change". An animation whose key handler does nothing could be scored
// input-causal. Correctness must not depend on machine load, so nothing here
// touches real time: callbacks go in a priority queue keyed by (due time,
// insertion order), both runs advance through identical virtual timestamps,
// input is injected at an exact virtual instant, and both runs execute the
// same bounded number of callbacks.
const MODE = process.argv[3] || 'baseline';
const INPUT_AT = 300;          // virtual ms
const MAX_VT   = 6000;         // virtual ms ceiling
const MAX_CB   = 4000;         // callback ceiling (runaway scheduling)

let __vt = 0, __seq = 0, __cbs = 0;
const __q = [];
const __push = (fn, delay) => {
  const d = Math.max(0, Number(delay) || 0);
  const id = ++__seq;
  __q.push({ due: __vt + d, seq: id, fn });
  return id;
};
const __cancel = (id) => { const i = __q.findIndex(t => t.seq === id); if (i >= 0) __q.splice(i, 1); };
global.setTimeout = (fn, d) => __push(fn, d);
global.setInterval = (fn, d) => { const self = { id: 0 };
  const tick = () => { try { fn(); } catch (e) { __err(e); } self.id = __push(tick, d); };
  self.id = __push(tick, d); return self.id; };
global.clearTimeout = __cancel; global.clearInterval = __cancel;
global.requestAnimationFrame = (fn) => __push(() => fn(__vt), 16);
global.cancelAnimationFrame = __cancel;
global.Date = class extends Date { constructor(...a){ super(...(a.length?a:[0])); }
  static now(){ return __vt; } };
global.performance = { now: () => __vt };

const __ev = { runtime_clean:true, supported:true, error:null, ended:false, textSets:0 };
const __err = (e) => { __ev.runtime_clean = false; __ev.error = String(e && e.message || e).slice(0,200); };
const __rects = [];
let __seed = 12345;
Math.random = () => { __seed = (__seed * 1103515245 + 12345) & 0x7fffffff; return __seed / 0x7fffffff; };

function __ctx() {
  return new Proxy({}, { get: (_, p) => {
    if (typeof p === 'symbol') return undefined;
    if (['fillStyle','strokeStyle','font','lineWidth','textAlign','textBaseline','globalAlpha'].includes(p)) return '';
    return (...a) => {
      // Record any positioned draw, not just rects: path and image games
      // must not read as inert.
      if (p === 'fillRect' || p === 'strokeRect' || p === 'rect' || p === 'arc' ||
          p === 'moveTo' || p === 'lineTo' || p === 'drawImage' || p === 'fillText')
        __rects.push(p + ':' + a.slice(0,2).map(v => Math.round(Number(v)||0)).join(','));
    };
  }, set: () => true });
}
const __canvas = { width:400, height:400, getContext:__ctx, addEventListener:(e,f)=>{(__L[e] ||= []).push(f);},
                   getBoundingClientRect:()=>({left:0,top:0,width:400,height:400}), style:{} };
const __L = {};
function __el(id){
  if (String(id).toLowerCase().includes('canvas')) return __canvas;
  return new Proxy({ style:{}, classList:{add(){},remove(){},toggle(){}},
                     addEventListener:(e,f)=>{(__L[e] ||= []).push(f);}, appendChild(){}, focus(){} },
    { get:(t,p)=> p in t ? t[p] : '',
      set:(t,p,v)=>{ if((p==='textContent'||p==='innerHTML'||p==='innerText') && t[p] !== undefined && String(t[p]) !== String(v)) __ev.textSets++; t[p]=v; return true; } });
}
global.document = { getElementById:__el, querySelector:(s)=>__el(String(s)), querySelectorAll:()=>[],
                    createElement:()=>__el('x'), body:{appendChild(){},style:{}},
                    addEventListener:(e,f)=>{(__L[e] ||= []).push(f);} };
global.window = { addEventListener:(e,f)=>{(__L[e] ||= []).push(f);}, innerWidth:800, innerHeight:600,
                  document: global.document, location:{ reload:()=>{ __ev.ended = true; }, href:'' } };
global.location = global.window.location;
global.alert = () => { __ev.ended = true; };
process.on('uncaughtException', __err);

const __src = require('fs').readFileSync(process.argv[2], 'utf8');
try { (0, eval)(__src); } catch (e) { __err(e);
  console.log(JSON.stringify({ ...__ev, trace:'', early:'' })); process.exit(0); }

const __fire = (k, code) => (__L['keydown']||[]).forEach(f => {
  try { f({ key:k, code:k, keyCode:code, which:code, preventDefault(){}, stopPropagation(){} }); } catch(e){ __err(e); }
});

// Deterministic drain: always the same virtual instants, same budget.
let __early = '';
let __injected = false, __drove = false;
while (__q.length && __vt <= MAX_VT && __cbs < MAX_CB) {
  __q.sort((a,b) => a.due - b.due || a.seq - b.seq);
  const t = __q.shift();
  __vt = Math.max(__vt, t.due);
  if (!__injected && __vt >= INPUT_AT) { __injected = true; if (MODE === 'input') __fire('ArrowUp', 38); }
  if (__early === '' && __vt >= 900) __early = __rects.slice(0, 60).join('|');
  if (!__drove && __vt >= 2500) { __drove = true; for (let i=0;i<6;i++) __fire('ArrowRight', 39); }
  try { t.fn(); } catch (e) { __err(e); }
  __cbs++;
}
console.log(JSON.stringify({ ...__ev, early: __early, trace: __rects.slice(0, 600).join('|'), cbs: __cbs }));
"""


def js_probe_source() -> str:
    return _JS_HARNESS


# Artifacts the shim can meaningfully instrument. Anything else reports
# supported=false — unverified, NOT failed.
_CANVAS_RE = re.compile(r"getContext\s*\(|requestAnimationFrame|addEventListener\s*\(\s*['\"]keydown", re.I)
_NODE_ONLY_RE = re.compile(r"\brequire\s*\(|\bmodule\.exports\b|\bprocess\.(argv|stdin)\b")


def js_is_instrumentable(code: str) -> bool:
    if not code or not code.strip():
        return False
    if _NODE_ONLY_RE.search(code):
        return False        # a Node script, not browser code
    return bool(_CANVAS_RE.search(code))


def combine_runs(baseline: Optional[Dict], keyed: Optional[Dict]) -> Optional[Dict]:
    """Turn two controlled runs into behavioural evidence.

    Causality is a DIFFERENCE between an unkeyed and a keyed run from an
    identical deterministic start. A single run cannot tell "input changed
    the world" from "a timer moved pixels" — an animation that ignores
    input passed an earlier single-run version of this check.
    """
    if not baseline or not keyed:
        return None
    if not baseline.get("runtime_clean", True) or not keyed.get("runtime_clean", True):
        return {"supported": True, "runtime_clean": False,
                "error": baseline.get("error") or keyed.get("error"),
                "temporal_progress": False, "input_causality": False,
                "collision_transition": False, "food_or_score_transition": False}
    # Temporal progress is derived from the trace itself, not from a fixed
    # virtual timestamp: a game that dies in nine callbacks (a snake starting
    # next to a wall) never reaches a timestamp snapshot, and read as inert.
    # Splitting the recorded draws in half and comparing is timing-free.
    trace = baseline.get("trace", "")
    parts = [x for x in trace.split("|") if x]
    half = len(parts) // 2
    first, second = parts[:half], parts[half:2 * half]
    return {
        "supported": True,
        "runtime_clean": True,
        # Rendering kept changing on its own.
        "temporal_progress": half > 0 and first != second,
        # The keyed run diverged from the unkeyed one.
        "input_causality": bool(baseline.get("trace")) and baseline.get("trace") != keyed.get("trace"),
        "collision_transition": bool(baseline.get("ended") or keyed.get("ended")),
        "food_or_score_transition": (baseline.get("textSets", 0) or 0) > 0,
    }


def parse_probe_output(stdout: str) -> Optional[Dict]:
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


# Required behaviours for an interactive game artifact. Coverage is judged
# against these; anything absent keeps the pipeline open.
INTERACTIVE_REQUIRED = ["temporal_progress", "input_causality"]
INTERACTIVE_OPTIONAL = ["collision_transition", "food_or_score_transition"]


def grade_interactive(ev: Optional[Dict]) -> Tuple[str, List[str], float]:
    """(strength, missing_required, behavioural score in [0,1]).

    An unsupported artifact is unverified, not failed: strength SYNTAX with
    every required behaviour still missing, so the pipeline stays open.
    """
    if not ev or not ev.get("supported", True):
        return SYNTAX, list(INTERACTIVE_REQUIRED), 0.0
    if not ev.get("runtime_clean", True):
        return SYNTAX, list(INTERACTIVE_REQUIRED), 0.0

    missing = [k for k in INTERACTIVE_REQUIRED if not ev.get(k)]
    hits = sum(1 for k in INTERACTIVE_REQUIRED + INTERACTIVE_OPTIONAL if ev.get(k))
    score = hits / float(len(INTERACTIVE_REQUIRED) + len(INTERACTIVE_OPTIONAL))

    if missing:
        # It ran without throwing, which is more than syntax and less than
        # working.
        return RUNTIME, missing, score
    if all(ev.get(k) for k in INTERACTIVE_OPTIONAL):
        return BEHAVIORAL_COMPLETE, [], score
    return BEHAVIORAL_PARTIAL, [], score


def rank_key(cand: Dict):
    """Selection order: required behaviours, then breadth of demonstrated
    behaviour, then runtime health, then lens. The lens is a code-embedding
    proxy and must never outrank direct behavioural evidence."""
    ev = cand.get("behavior") or {}
    strength = cand.get("evidence_strength", NONE)
    return (
        0 if cand.get("missing_required") else 1,
        cand.get("behavior_score", 0.0),
        STRENGTH_ORDER.index(strength) if strength in STRENGTH_ORDER else 0,
        1 if ev.get("runtime_clean", False) else 0,
        -float(cand.get("energy", 999)),      # lens: tie-break only
    )


# --------------------------------------------------------------- adapters ---
#
# Evidence strength must come from the VERIFIER THAT RAN, never from the file
# extension. The first cut keyed off extension and mapped every .py to
# behavioral_complete, which is wrong for Pygame, Tkinter, curses and Flask:
# those receive a compile smoke and nothing more, and would have closed the
# pipeline claiming behaviour nobody demonstrated. It also sent .css through a
# JavaScript probe and treated every .js as a canvas game.
#
# Adapters carry the domain knowledge. Everything above them -- the strength
# ordering, coverage, early-return policy, ranking, and the unsupported vs
# failed distinction -- stays prompt-agnostic.

BROWSER_CANVAS_JS = "browser_canvas_js"
BROWSER_INLINE_SCRIPT = "browser_inline_script"
JAVASCRIPT_COMPILE = "javascript_compile"
CSS_SYNTAX = "css_syntax"
ALGORITHMIC_IO = "algorithmic_io"
PYTHON_COMPILE = "python_compile"
INTERACTIVE_PYTHON_UNSUPPORTED = "interactive_python_unsupported"
UNSUPPORTED = "unsupported"

# Python UI/server frameworks a compile check cannot speak for.
_INTERACTIVE_PY_RE = re.compile(
    r"\b(import\s+pygame|from\s+pygame|import\s+tkinter|from\s+tkinter|"
    r"import\s+curses|from\s+curses|Flask\s*\(|FastAPI\s*\(|"
    r"QApplication|import\s+PySide|import\s+PyQt)", re.I)

_INLINE_SCRIPT_RE = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", re.S | re.I)


def select_adapter(file_path: str, code: str, has_io_oracle: bool = False) -> str:
    """Which verifier can speak for this artifact. Capability, not keywords."""
    ext = (file_path or "").lower().rsplit(".", 1)
    ext = ("." + ext[-1]) if len(ext) == 2 else ""
    code = code or ""

    if ext in (".py",):
        if _INTERACTIVE_PY_RE.search(code):
            return INTERACTIVE_PYTHON_UNSUPPORTED
        return ALGORITHMIC_IO if has_io_oracle else PYTHON_COMPILE
    if ext in (".js", ".mjs"):
        # A plain Node script or a module of helpers is NOT a canvas game.
        return BROWSER_CANVAS_JS if js_is_instrumentable(code) else JAVASCRIPT_COMPILE
    if ext in (".jsx", ".tsx", ".ts"):
        return UNSUPPORTED          # needs transpilation first
    if ext in (".html", ".htm"):
        for m in _INLINE_SCRIPT_RE.finditer(code):
            if js_is_instrumentable(m.group(1)):
                return BROWSER_INLINE_SCRIPT
        return UNSUPPORTED
    if ext == ".css":
        return CSS_SYNTAX
    return UNSUPPORTED


def extract_inline_script(html: str) -> str:
    return "\n".join(m.group(1) for m in _INLINE_SCRIPT_RE.finditer(html or ""))


def result(accepted: bool, strength: str, adapter: str, supported: bool = True,
           behavior: Optional[Dict] = None, behavior_score: float = 0.0,
           missing_required: Optional[List[str]] = None) -> Dict:
    """The structured verification record every verifier returns."""
    return {
        "accepted": bool(accepted),
        "strength": strength,
        "adapter": adapter,
        "supported": bool(supported),
        "behavior": behavior or {},
        "behavior_score": float(behavior_score),
        "missing_required": list(missing_required or []),
    }


def result_from_adapter(adapter: str, smoke_passed: bool,
                        probe_evidence: Optional[Dict] = None) -> Dict:
    """Map a verifier's actual outcome onto evidence strength.

    Only two adapters can ever yield behavioural evidence: the generated I/O
    oracle, and the browser behaviour probe. Everything else tops out at
    syntax, no matter what the file is called.
    """
    if adapter == ALGORITHMIC_IO:
        return result(smoke_passed,
                      BEHAVIORAL_COMPLETE if smoke_passed else NONE,
                      adapter, behavior_score=1.0 if smoke_passed else 0.0)

    if adapter in (BROWSER_CANVAS_JS, BROWSER_INLINE_SCRIPT):
        if probe_evidence is None:
            # Probe could not run: unverified, never "failed".
            return result(smoke_passed, SYNTAX if smoke_passed else NONE, adapter,
                          supported=False, missing_required=list(INTERACTIVE_REQUIRED))
        strength, missing, score = grade_interactive(probe_evidence)
        return result(smoke_passed, strength, adapter,
                      supported=bool(probe_evidence.get("supported", True)),
                      behavior=probe_evidence, behavior_score=score,
                      missing_required=missing)

    if adapter in (JAVASCRIPT_COMPILE, PYTHON_COMPILE, CSS_SYNTAX):
        return result(smoke_passed, SYNTAX if smoke_passed else NONE, adapter,
                      missing_required=[] if adapter == CSS_SYNTAX
                      else list(INTERACTIVE_REQUIRED))

    # INTERACTIVE_PYTHON_UNSUPPORTED and UNSUPPORTED
    return result(smoke_passed, SYNTAX if smoke_passed else NONE, adapter,
                  supported=False, missing_required=list(INTERACTIVE_REQUIRED))


def may_return_early_result(res: Dict) -> bool:
    """Early return decided from the structured record, not a bare boolean."""
    if not res.get("accepted"):
        return False
    return may_return_early(res.get("strength", NONE),
                            res.get("missing_required") or [],
                            float(res.get("behavior_score", 0.0)))


def js_probe_source_inline() -> str:
    """The harness, adapted to run as ONE blob inside the sandbox.

    The sandbox executes a single code string with no argv and no artifact
    file, so mode and artifact arrive as pre-declared constants instead.
    """
    src = _JS_HARNESS
    src = src.replace("const MODE = process.argv[3] || 'baseline';",
                      "const MODE = __MODE__;")
    src = src.replace("const __src = require('fs').readFileSync(process.argv[2], 'utf8');",
                      "const __src = __ARTIFACT__;")
    return src
