#!/usr/bin/env python3
"""Generate the AoC-style puzzle fixtures and their reference answers.

Advent of Code asks that puzzle inputs and text not be redistributed, so these
are original problems in the same shape — parse a text input, simulate a rule,
print one number — rather than copies. That keeps the property that matters for
measurement: the answer is a specific integer, so "did it work" needs no
judgement.

Each problem emits TWO inputs. The model only ever sees `input.txt`; the
harness re-runs its program against `holdout.txt` and checks that answer too,
so a solution that hardcodes the number it was shown fails. Deterministic
(fixed seeds) so the committed fixtures and answers stay in agreement.

Run:  python scripts/fixtures/aoc/generate.py
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


# --- problem 1: sonar (Day-1 shape) --------------------------------------

def gen_sonar(seed: int, n: int = 2000) -> str:
    rng = random.Random(seed)
    depth = rng.randint(100, 200)
    out = []
    for _ in range(n):
        depth += rng.randint(-8, 9)
        out.append(str(depth))
    return "\n".join(out) + "\n"


def solve_sonar(text: str) -> int:
    nums = [int(x) for x in text.split()]
    # Count sliding-window-of-3 sums that exceed the previous window.
    sums = [sum(nums[i:i + 3]) for i in range(len(nums) - 2)]
    return sum(1 for a, b in zip(sums, sums[1:]) if b > a)


# --- problem 2: course plotting (Day-2 shape, with aim) ------------------

def gen_course(seed: int, n: int = 1200) -> str:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        # Weighted so aim trends positive: a real course never ends at a
        # negative depth, and a negative product would be a puzzle artefact
        # rather than a property of the solution.
        cmd = rng.choices(["forward", "down", "up"], weights=[5, 3, 2])[0]
        out.append(f"{cmd} {rng.randint(1, 9)}")
    return "\n".join(out) + "\n"


def solve_course(text: str) -> int:
    horiz = depth = aim = 0
    for line in text.split("\n"):
        if not line.strip():
            continue
        cmd, val = line.split()
        val = int(val)
        if cmd == "forward":
            horiz += val
            depth += aim * val
        elif cmd == "down":
            aim += val
        elif cmd == "up":
            aim -= val
    return horiz * depth


# --- problem 3: trajectory over a repeating grid (Day-3 shape) -----------

def gen_slope(seed: int, rows: int = 400, cols: int = 31) -> str:
    rng = random.Random(seed)
    grid = []
    for _ in range(rows):
        grid.append("".join("#" if rng.random() < 0.34 else "." for _ in range(cols)))
    return "\n".join(grid) + "\n"


def solve_slope(text: str) -> int:
    grid = [ln for ln in text.split("\n") if ln.strip()]
    width = len(grid[0])
    # Product of trees hit on five slopes — the part-2 shape.
    total = 1
    for dx, dy in ((1, 1), (3, 1), (5, 1), (7, 1), (1, 2)):
        x = 0
        hits = 0
        for y in range(0, len(grid), dy):
            if grid[y][x % width] == "#":
                hits += 1
            x += dx
        total *= hits
    return total


# --- problem 4: exponential population (Day-6 shape) --------------------
# Naive per-individual simulation blows up; the counting solution is the point.

def gen_shoal(seed: int, n: int = 300) -> str:
    rng = random.Random(seed)
    return ",".join(str(rng.randint(1, 5)) for _ in range(n)) + "\n"


def solve_shoal(text: str, days: int = 256) -> int:
    counts = [0] * 9
    for tok in text.strip().split(","):
        counts[int(tok)] += 1
    for _ in range(days):
        spawning = counts.pop(0)
        counts.append(spawning)      # new individuals at timer 8
        counts[6] += spawning        # parents reset to 6
    return sum(counts)


# Explicit seeds, not hash(name): str hashing is salted per process, so
# hash-derived seeds would regenerate different inputs on every run and the
# committed answers would stop matching the committed fixtures.
PROBLEMS = {
    "sonar": (gen_sonar, solve_sonar, 1101, 7101),
    "course": (gen_course, solve_course, 1202, 7202),
    "slope": (gen_slope, solve_slope, 1303, 7303),
    "shoal": (gen_shoal, solve_shoal, 1406, 7406),
}


def main() -> None:
    answers = {}
    for name, (gen, solve, seed_main, seed_hold) in PROBLEMS.items():
        d = HERE / name
        d.mkdir(parents=True, exist_ok=True)
        main_text = gen(seed=seed_main)
        hold_text = gen(seed=seed_hold)
        (d / "input.txt").write_text(main_text)
        (d / "holdout.txt").write_text(hold_text)
        answers[name] = {
            "input": solve(main_text),
            "holdout": solve(hold_text),
        }
        print(f"{name:8s} input={answers[name]['input']:<24} "
              f"holdout={answers[name]['holdout']}")
    (HERE / "answers.json").write_text(json.dumps(answers, indent=2) + "\n")
    print(f"\nwrote {HERE / 'answers.json'}")


if __name__ == "__main__":
    main()
