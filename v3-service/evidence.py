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
// Deterministic instrumentation. Runs the artifact once in a given MODE and
// prints a render trace plus flags. Causality is decided by the CALLER, by
// diffing a keyed run against an unkeyed one from an identical start — a
// single run cannot distinguish "input changed the world" from "a timer
// moved some pixels", which is how an animation that ignores input passed
// an earlier version of this probe.
const MODE = process.argv[3] || 'baseline';   // baseline | input
const __ev = { runtime_clean:true, supported:true, error:null, ended:false, textSets:0 };
const __rects = [];
let __seed = 12345;
Math.random = () => { __seed = (__seed * 1103515245 + 12345) & 0x7fffffff; return __seed / 0x7fffffff; };
function __ctx() {
  return new Proxy({}, { get: (_, p) => {
    if (typeof p === 'symbol') return undefined;
    if (['fillStyle','strokeStyle','font','lineWidth','textAlign','textBaseline','globalAlpha'].includes(p)) return '';
    return (...a) => { if (p === 'fillRect' || p === 'strokeRect' || p === 'rect')
                         __rects.push(a.slice(0,2).join(',')); };
  }, set: () => true });
}
const __canvas = { width:400, height:400, getContext:__ctx, addEventListener:()=>{},
                   getBoundingClientRect:()=>({left:0,top:0,width:400,height:400}), style:{} };
const __L = {};
function __el(id){
  if (String(id).toLowerCase().includes('canvas')) return __canvas;
  return new Proxy({ style:{}, classList:{add(){},remove(){},toggle(){}},
                     addEventListener:(e,f)=>{(__L[e] ||= []).push(f);}, appendChild(){}, focus(){} },
    { get:(t,p)=> p in t ? t[p] : '', set:(t,p,v)=>{ if(p==='textContent'||p==='innerHTML'||p==='innerText') __ev.textSets++; t[p]=v; return true; } });
}
global.document = { getElementById:__el, querySelector:()=>__el('x'), querySelectorAll:()=>[],
                    createElement:()=>__el('x'), body:{appendChild(){},style:{}},
                    addEventListener:(e,f)=>{(__L[e] ||= []).push(f);} };
global.window = { addEventListener:(e,f)=>{(__L[e] ||= []).push(f);}, innerWidth:800, innerHeight:600,
                  location:{ reload:()=>{ __ev.ended = true; }, href:'' } };
global.location = global.window.location;
global.alert = () => { __ev.ended = true; };
global.requestAnimationFrame = (fn) => setTimeout(fn, 16);
global.cancelAnimationFrame = (id) => clearTimeout(id);
process.on('uncaughtException', e => { __ev.runtime_clean = false; __ev.error = String(e.message).slice(0,200); });

const __src = require('fs').readFileSync(process.argv[2], 'utf8');
try { (0, eval)(__src); }
catch (e) { __ev.runtime_clean = false; __ev.error = String(e.message).slice(0,200);
            console.log(JSON.stringify({...__ev, trace:'', early:''})); process.exit(0); }

const __fire = (k, code) => (__L['keydown']||[]).forEach(f => {
  try { f({ key:k, code:k, keyCode:code, which:code, preventDefault(){}, stopPropagation(){} }); } catch(e){}
});

// Input goes in EARLY, before a game can reach a wall on its own.
setTimeout(() => { if (MODE === 'input') { __fire('ArrowUp', 38); } }, 120);
// Snapshot the first stretch for temporal progress.
setTimeout(() => { __ev.early = __rects.slice(0, 40).join('|'); }, 260);
// Then drive one direction hard to reach a terminal state.
setTimeout(() => { for (let i = 0; i < 6; i++) __fire('ArrowRight', 39); }, 900);
setTimeout(() => {
  __ev.trace = __rects.slice(0, 400).join('|');
  console.log(JSON.stringify(__ev));
  process.exit(0);
}, 2200);
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
    early, late = baseline.get("early", ""), baseline.get("trace", "")
    return {
        "supported": True,
        "runtime_clean": True,
        # Rendering kept changing on its own.
        "temporal_progress": bool(early) and bool(late) and not late.startswith(early * 2) and early != late,
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
