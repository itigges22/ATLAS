"""Second wave of freshly authored tasks: eight families, zero overlap with
the first set's problem shapes.

The first set (novel_tasks.py) has been the validation target for a long
run of harness fixes, which makes it useless for answering "did those fixes
generalize or did we tune to the set?" — the question this file exists for.
Same construction discipline as v1:

  * the statement fully specifies the rules, so a careful reader can solve it
  * a REFERENCE implementation defines truth, so expected answers are computed
    rather than guessed
  * every instance carries a HOLDOUT input the solution never sees, so an
    answer hardcoded from the visible input dies
  * instances are seeded, so a run is reproducible and both arms get exactly
    the same problems — and a NEW seed regenerates every input, so repeated
    runs never reuse data

The family genres are deliberately disjoint from v1 (running balances,
signal debouncing, interval coverage, dependency resolution, grid walks,
byte checksums, login sessions, stack machines): here it is scan scheduling,
cache eviction, ranked-choice elimination, inline-markup parsing, weighted
shortest paths, spreadsheet evaluation, first-fit packing, and path
self-intersection.
"""

from __future__ import annotations

import heapq
import random
from typing import List

from novel_tasks import NovelTask, _statement


# --- families ---------------------------------------------------------------


def _elevator_input(rng: random.Random) -> str:
    lines = [str(rng.randint(1, 40)) for _ in range(rng.randint(14, 22))]
    return "\n".join(lines) + "\n"


def _elevator_solve(text: str) -> str:
    """An elevator starts at floor 20 moving up. It first visits every
    requested floor at or above 20 in increasing order, then reverses and
    visits every requested floor below 20 in decreasing order. Duplicate
    requests are a single visit. If no requests are at or above 20 the first
    phase is skipped entirely; likewise the second phase when none are below.
    Answer: the total travel distance (the sum of absolute differences
    between consecutive positions, starting from 20), then the floor the
    elevator ends on."""
    floors = sorted({int(l) for l in text.split() if l.strip()})
    up = [f for f in floors if f >= 20]
    down = [f for f in reversed(floors) if f < 20]
    pos, dist = 20, 0
    for f in up + down:
        dist += abs(f - pos)
        pos = f
    return f"{dist} {pos}"


def _lru_input(rng: random.Random) -> str:
    cap = rng.randint(3, 4)
    keys = ["ka", "kb", "kc", "kd", "ke", "kf"]
    lines = [str(cap)]
    for _ in range(rng.randint(50, 80)):
        op = rng.choice(["GET", "GET", "PUT"])
        lines.append(f"{op} {rng.choice(keys)}")
    return "\n".join(lines) + "\n"


def _lru_solve(text: str) -> str:
    """The first line is the cache capacity; the rest are operations on an
    LRU cache of keys. GET is a hit when the key is present (a hit makes the
    key most-recently-used); a GET miss changes nothing. PUT inserts the key
    as most-recently-used, or refreshes it if already present. A key that has
    accumulated 3 GET hits is PINNED and can never be evicted. When a PUT
    would exceed capacity, the least-recently-used unpinned key is evicted;
    if every resident key is pinned, the PUT is refused and the cache is
    unchanged. Answer: total GET hits, then the last evicted key (the word
    none if nothing was ever evicted)."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    cap = int(lines[0])
    order: List[str] = []  # least-recent first
    hits = {}
    total_hits = 0
    last_evicted = "none"
    for line in lines[1:]:
        op, key = line.split()
        if op == "GET":
            if key in order:
                total_hits += 1
                hits[key] = hits.get(key, 0) + 1
                order.remove(key)
                order.append(key)
        else:  # PUT
            if key in order:
                order.remove(key)
                order.append(key)
                continue
            if len(order) >= cap:
                victim = next((k for k in order if hits.get(k, 0) < 3), None)
                if victim is None:
                    continue  # refused
                order.remove(victim)
                last_evicted = victim
            order.append(key)
    return f"{total_hits} {last_evicted}"


def _votes_input(rng: random.Random) -> str:
    names = ["iris", "jude", "kane", "lila", "milo"]
    lines = []
    for _ in range(rng.randint(25, 40)):
        k = rng.randint(1, len(names))
        lines.append(" ".join(rng.sample(names, k)))
    return "\n".join(lines) + "\n"


def _votes_solve(text: str) -> str:
    """Each line is one ranked-choice ballot, most preferred candidate
    first. Counting proceeds in rounds. In a round, each ballot counts for
    its highest-ranked candidate still in the race; a ballot whose candidates
    are all eliminated is exhausted and stops counting toward the total. If a
    candidate holds a strict majority of the non-exhausted ballots, they win.
    Otherwise the candidate with the fewest votes this round is eliminated
    (ties broken by eliminating the alphabetically LAST of the tied) and a
    new round begins; when only one candidate remains, they win. Answer: the
    winner, then the number of eliminations performed."""
    ballots = [l.split() for l in text.split("\n") if l.strip()]
    alive = sorted({c for b in ballots for c in b})
    eliminations = 0
    while True:
        counts = {c: 0 for c in alive}
        live_ballots = 0
        for b in ballots:
            for c in b:
                if c in counts:
                    counts[c] += 1
                    live_ballots += 1
                    break
        for c in alive:
            if counts[c] * 2 > live_ballots:
                return f"{c} {eliminations}"
        if len(alive) == 1:
            return f"{alive[0]} {eliminations}"
        fewest = min(counts.values())
        out = sorted([c for c in alive if counts[c] == fewest])[-1]
        alive.remove(out)
        eliminations += 1


def _markup_input(rng: random.Random) -> str:
    words = ["gate", "lamp", "reed", "silt", "moss", "kiln", "opal", "fern"]
    lines = []
    for _ in range(rng.randint(10, 16)):
        toks = []
        for _ in range(rng.randint(4, 9)):
            w = rng.choice(words)
            r = rng.random()
            if r < 0.18:
                toks.append("*")
            elif r < 0.36:
                toks.append("_")
            toks.append(w)
        lines.append(" ".join(toks))
    return "\n".join(lines) + "\n"


def _markup_solve(text: str) -> str:
    """Text contains two span markers: * opens and closes a star span, _
    opens and closes an underscore span. Scan each line left to right,
    character by character. A marker seen while no span is open opens its
    span; the SAME marker seen while its span is open closes it, completing
    one valid span whose inner text is everything between the two markers.
    The OTHER marker seen inside an open span is plain text. A span still
    open at the end of its line is discarded (markers never span lines).
    Answer: the number of valid spans in the whole file, then the character
    length of the longest span's inner text (0 if there are no valid
    spans)."""
    spans = 0
    longest = 0
    for line in text.split("\n"):
        open_mark = None
        start = 0
        for i, ch in enumerate(line):
            if ch in "*_":
                if open_mark is None:
                    open_mark, start = ch, i + 1
                elif ch == open_mark:
                    spans += 1
                    longest = max(longest, i - start)
                    open_mark = None
    return f"{spans} {longest}"


def _routes_input(rng: random.Random) -> str:
    n = rng.randint(10, 14)
    nodes = [f"{chr(97 + i)}{chr(97 + i)}" for i in range(n)]
    nodes[0], nodes[-1] = "aa", "zz"
    lines = []
    # A guaranteed open chain aa -> ... -> zz (weights never divisible by 7).
    chain = nodes[:]
    rng.shuffle(chain)
    chain.remove("aa"); chain.insert(0, "aa")
    chain.remove("zz"); chain.append("zz")
    for a, b in zip(chain, chain[1:]):
        w = rng.randint(1, 20)
        while w % 7 == 0:
            w = rng.randint(1, 20)
        lines.append(f"{a} {b} {w}")
    for _ in range(rng.randint(12, 20)):
        a, b = rng.sample(nodes, 2)
        lines.append(f"{a} {b} {rng.randint(1, 21)}")
    rng.shuffle(lines)
    return "\n".join(lines) + "\n"


def _routes_solve(text: str) -> str:
    """Each line is a one-way road from one town to another with a toll.
    Roads whose toll is divisible by 7 are closed and cannot be used at all.
    Answer: the cheapest total toll to get from town aa to town zz over open
    roads, then the number of towns reachable from aa over open roads
    (counting aa itself)."""
    adj = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        a, b, w = line.split()
        w = int(w)
        if w % 7 == 0:
            continue
        adj.setdefault(a, []).append((b, w))
    dist = {"aa": 0}
    pq = [(0, "aa")]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        for v, w in adj.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return f"{dist['zz']} {len(dist)}"


def _cells_input(rng: random.Random) -> str:
    names = []
    lines = []
    for i in range(rng.randint(12, 18)):
        name = f"{chr(65 + i % 6)}{i // 6 + 1}"
        if len(names) < 2 or rng.random() < 0.3:
            lines.append(f"{name} = {rng.randint(-9, 15)}")
        else:
            a, b = rng.sample(names, 2)
            op = rng.choice("+-*")
            if rng.random() < 0.4:
                lines.append(f"{name} = {a} {op} {rng.randint(2, 9)}")
            else:
                lines.append(f"{name} = {a} {op} {b}")
        names.append(name)
    return "\n".join(lines) + "\n"


def _cells_solve(text: str) -> str:
    """Each line defines a spreadsheet cell, in order: either `CELL = N` for
    an integer, or `CELL = X OP Y` where OP is +, - or * and X and Y are each
    either an earlier-defined cell name or an integer. Every reference is to
    a cell defined on an earlier line. Answer: the value of the cell defined
    on the LAST line, then the number of cells whose value is negative."""
    vals = {}
    last = 0
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        name, rhs = [s.strip() for s in line.split("=", 1)]
        parts = rhs.split()
        def resolve(tok: str) -> int:
            return vals[tok] if tok in vals else int(tok)
        if len(parts) == 1:
            v = resolve(parts[0])
        else:
            x, op, y = parts
            a, b = resolve(x), resolve(y)
            v = a + b if op == "+" else a - b if op == "-" else a * b
        vals[name] = v
        last = v
    negative = sum(1 for v in vals.values() if v < 0)
    return f"{last} {negative}"


def _freight_input(rng: random.Random) -> str:
    lines = ["100"]
    for _ in range(rng.randint(18, 28)):
        if rng.random() < 0.25:
            lines.append(str(rng.randint(51, 95)))
        else:
            lines.append(str(rng.randint(5, 50)))
    return "\n".join(lines) + "\n"


def _freight_solve(text: str) -> str:
    """The first line is the truck capacity; every other line is a crate
    weight, loaded in order. A crate heavier than half the capacity gets a
    fresh truck to itself, and that truck is then sealed: it accepts nothing
    more. Any other crate goes into the FIRST unsealed truck that still has
    room for it, or a fresh truck if none does. Answer: the number of trucks
    used, then the free space left in the last truck that was opened."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    cap = int(lines[0])
    trucks: List[int] = []   # free space per truck, in open order
    sealed: List[bool] = []
    for w in map(int, lines[1:]):
        if w * 2 > cap:
            trucks.append(cap - w)
            sealed.append(True)
            continue
        for i, free in enumerate(trucks):
            if not sealed[i] and free >= w:
                trucks[i] = free - w
                break
        else:
            trucks.append(cap - w)
            sealed.append(False)
    return f"{len(trucks)} {trucks[-1]}"


def _turtle_input(rng: random.Random) -> str:
    lines = []
    for _ in range(rng.randint(16, 26)):
        r = rng.random()
        if r < 0.55:
            lines.append(f"F {rng.randint(1, 9)}")
        elif r < 0.775:
            lines.append("L")
        else:
            lines.append("R")
    return "\n".join(lines) + "\n"


def _turtle_solve(text: str) -> str:
    """A turtle starts on an infinite grid at position 0 0, facing north.
    `F N` moves it N steps forward one grid cell at a time; `L` and `R` turn
    it 90 degrees left or right in place. A cell is REVISITED each time the
    turtle steps onto a cell it has already stood on at any earlier moment
    (the start cell counts as stood on). Answer: the total number of
    revisits, then the Manhattan distance from the final position back to
    0 0."""
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # N E S W
    facing = 0
    x = y = 0
    seen = {(0, 0)}
    revisits = 0
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line == "L":
            facing = (facing - 1) % 4
        elif line == "R":
            facing = (facing + 1) % 4
        else:
            _, n = line.split()
            dx, dy = dirs[facing]
            for _ in range(int(n)):
                x, y = x + dx, y + dy
                if (x, y) in seen:
                    revisits += 1
                else:
                    seen.add((x, y))
    return f"{revisits} {abs(x) + abs(y)}"


FAMILIES = [
    ("elevator", _elevator_input, _elevator_solve),
    ("lru", _lru_input, _lru_solve),
    ("votes", _votes_input, _votes_solve),
    ("markup", _markup_input, _markup_solve),
    ("routes", _routes_input, _routes_solve),
    ("cells", _cells_input, _cells_solve),
    ("freight", _freight_input, _freight_solve),
    ("turtle", _turtle_input, _turtle_solve),
]


# Line formats, stated explicitly in every prompt — same reasoning as v1:
# a bare model cannot open input.txt, so an unstated format measures
# format-guessing rather than coding.
FORMATS = {
    "elevator": "Each line is one requested floor, an integer, e.g. `27`.",
    "lru": "The first line is the capacity, an integer. Every other line is `GET KEY` or `PUT KEY`, e.g. `GET kb`.",
    "votes": "Each line is one ballot: candidate names separated by spaces, most preferred first, e.g. `lila kane iris`.",
    "markup": "Each line is text in which the characters * and _ appear between words.",
    "routes": "Each line is `FROM TO TOLL`, two-letter town names and an integer, e.g. `aa cc 12`.",
    "cells": "Each line is `CELL = N` or `CELL = X OP Y`, e.g. `A1 = 5` or `B2 = A1 * 3`.",
    "freight": "The first line is the truck capacity. Every other line is a crate weight, an integer.",
    "turtle": "Each line is `F N` (an integer step count), `L`, or `R`.",
}


def build_tasks(count: int = 40, seed: int = 20260808) -> List[NovelTask]:
    """`count` instances spread evenly over the families, deterministic in
    `seed` so both arms see identical problems. A different seed regenerates
    every input and every answer."""
    out: List[NovelTask] = []
    i = 0
    while len(out) < count:
        fam, make_input, solve = FAMILIES[i % len(FAMILIES)]
        variant = i // len(FAMILIES)
        data = make_input(random.Random(f"{seed}:{fam}:{variant}"))
        hold = make_input(random.Random(f"{seed}:{fam}:{variant}:holdout"))
        prompt = (
            f"input.txt holds one record per line. {FORMATS[fam]} "
            f"{_statement(solve)} "
            f"Write solve.py that reads input.txt and prints the answer on a "
            f"single line, fields separated by single spaces. Then run it and "
            f"confirm the answer."
        )
        out.append(NovelTask(
            name=f"{fam}{variant + 1}", family=fam, prompt=prompt,
            input_text=data, holdout_text=hold,
            expected=solve(data), holdout_expected=solve(hold),
        ))
        i += 1
    return out


if __name__ == "__main__":
    tasks = build_tasks()
    print(f"{len(tasks)} tasks over {len(FAMILIES)} families")
    # Sanity: deterministic in seed, distinct across seeds, holdout differs.
    again = build_tasks()
    assert [t.expected for t in tasks] == [t.expected for t in again]
    other = build_tasks(seed=1)
    assert [t.input_text for t in tasks] != [t.input_text for t in other]
    differing = sum(1 for t in tasks if t.expected != t.holdout_expected)
    print(f"expected != holdout on {differing}/{len(tasks)} instances")
    for t in tasks[:len(FAMILIES)]:
        print(f"\n--- {t.name}")
        print(f"    {t.prompt[:150]}...")
        print(f"    expected={t.expected!r}  holdout={t.holdout_expected!r}")
