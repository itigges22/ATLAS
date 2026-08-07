#!/usr/bin/env python3
"""Run the freshly authored task set through the full ATLAS agent.

The other half of the comparison scripts/novel-baseline.py starts: identical
statements, identical reference answers, identical holdout check — but this
arm gets the whole harness (agent loop, tools, gates, V3 pipeline), driven
through /v1/agent exactly the way a user session is.

The task registry and workspace plumbing are reused from the e2e suite so a
session here behaves byte-for-byte like a reliability-run session; only the
task source differs.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from novel_tasks import build_tasks  # noqa: E402


def _load_e2e():
    spec = importlib.util.spec_from_file_location(
        "e2e_suite", REPO / "scripts" / "e2e-reliability.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["e2e_suite"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--url", default="")
    ap.add_argument("--workspace", required=True,
                    help="host path the proxy has mounted (e.g. ~/demo2/e2e)")
    ap.add_argument("--subdir", default="e2e")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--start", type=int, default=0,
                    help="skip the first N tasks (resume support)")
    ap.add_argument("--json", dest="json_out", default="")
    ap.add_argument("--save-events", default="",
                    help="directory for per-session event dumps")
    args = ap.parse_args()

    import os
    e2e = _load_e2e()
    url = args.url or os.environ.get("ATLAS_PROXY_URL", "http://localhost:8090")

    tasks = build_tasks(args.count, args.seed)[args.start:]
    workspace = Path(args.workspace).expanduser()

    results = []
    for i, nt in enumerate(tasks):
        def make_check(nt):
            def check(ws: Path, s=None):
                prog = ws / "solve.py"
                if not prog.exists():
                    return False, "solve.py was never created"
                import subprocess
                def run(data: str):
                    (ws / "input.txt").write_text(data)
                    try:
                        p = subprocess.run([sys.executable, "solve.py"],
                                           cwd=str(ws), capture_output=True,
                                           text=True, timeout=60)
                    except subprocess.TimeoutExpired:
                        return False, "timeout"
                    if p.returncode != 0:
                        return False, f"failed: {p.stderr.strip()[:120]}"
                    return True, p.stdout.strip()
                ok, got = run(nt.input_text)
                if not ok:
                    return False, got
                if got != nt.expected:
                    return False, f"got {got!r}, want {nt.expected!r}"
                ok2, got2 = run(nt.holdout_text)
                (ws / "input.txt").write_text(nt.input_text)
                if not ok2 or got2 != nt.holdout_expected:
                    return False, (f"holdout mismatch: got {got2!r}, want "
                                   f"{nt.holdout_expected!r}")
                return True, f"{got} correct, and correct on the holdout"
            return check

        task = e2e.Task(
            name=nt.name,
            prompt=nt.prompt,
            files={"input.txt": nt.input_text},
            check=make_check(nt),
            must_exist=("input.txt",),
            immutable=(),  # solve.py may legitimately rewrite nothing else
        )
        print(f"[{args.start + i + 1}/{args.start + len(tasks)}] {nt.name} ...",
              flush=True)
        t0 = time.time()
        session = e2e.run_session(task, 1, url, workspace, args.subdir,
                                  args.timeout)
        if args.save_events:
            evdir = Path(args.save_events)
            evdir.mkdir(parents=True, exist_ok=True)
            with open(evdir / f"{nt.name}.jsonl", "w") as fh:
                for ev in getattr(session, "events", []):
                    fh.write(json.dumps(ev) + "\n")
        ok, detail = task.check(workspace)
        results.append({
            "task": nt.name, "family": nt.family, "passed": ok,
            "detail": detail, "turns": getattr(session, "turns", 0),
            "wall_s": round(time.time() - t0, 1),
        })
        print(f"      {'PASS' if ok else 'fail'} turns="
              f"{getattr(session, 'turns', '?')} "
              f"{results[-1]['wall_s']}s — {detail[:80]}", flush=True)
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(results, indent=1))

    p = sum(1 for r in results if r["passed"])
    print(f"\nATLAS TOTAL {p}/{len(results)} ({100.0 * p / len(results):.0f}%)")
    byfam = {}
    for r in results:
        got, tot = byfam.get(r["family"], (0, 0))
        byfam[r["family"]] = (got + (1 if r["passed"] else 0), tot + 1)
    for fam in sorted(byfam):
        got, tot = byfam[fam]
        print(f"   {fam:10} {got}/{tot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
