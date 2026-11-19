"""
Santa 2026 - Elf Calculator: v1  (backward beam search)

Change vs v0: v0 built every big number digit by digit in ONE fixed base R (2 instructions
per digit). v1 works BACKWARDS from the number and, at every step, picks among many moves:

    x = y * m            ->  continue from y = x / m           1 instruction  (m divides x)
    x = y * m + r        ->  continue from y = (x - r) / m     2 instructions (r held)
    x = y * m - s        ->  continue from y = (x + s) / m     2 instructions (s held)

with m being ANY value in the digit table (2..half, R, starting values), not only R.
It keeps the BEAM smallest numbers reached for each instruction count, stops when it reaches
a value already in a box, and keeps the result only if it beats v0's digit-by-digit chain.
Everything else (reusing earlier targets, the settings sweep, pruning the sample) is as in v0.

Folder layout:  elf.py, data/puzzles.csv, data/practice_puzzles.csv, data/sample_submission.csv
Usage:
    python elf.py            # practice report, then build submission_v1.csv
    python elf.py practice   # only the practice report
    python elf.py submit     # only build submission_v1.csv
submission_v1.csv also takes the best program per puzzle from any earlier submission*.csv
in this folder, so it is never worse than what you already have.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Optional

csv.field_size_limit(2**31 - 1)

VERSION = "v1"
DATA_DIR = Path("data")
OUT_PATH = Path(f"submission_{VERSION}.csv")
MAX_INSTR, MAX_BYTES = 4096, 131072
BEAM = 12                                       # bigger = slightly better programs, slower
HALVES = (1, 2, 3, 4, 8, 12, 16, 24, 32, 36, 40, 44)  # digit-table sizes (bigger usually wins)
RINGS = (0, 2, 4, 8, 16, 24)                    # how many recent targets to keep in boxes
DELTA_TRIES = 2                                 # nearest earlier targets to try "t = earlier +/- diff"

Puzzle = dict[str, Any]
Program = list[list[Any]]
Step = tuple[str, int, int]     # (op, other value, side): side 0 = cur op other, 1 = other op cur
Chain = tuple[int, list[Step]]  # (start value already in a box, forward steps)


# ---------------------------------------------------------------- I/O
def load_puzzles(path: Path) -> list[Puzzle]:
    out: list[Puzzle] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p: Puzzle = dict(row)
            p["start"] = json.loads(row["starting_values"])
            p["targets"] = json.loads(row["targets"])
            p["limit"] = int(row["value_limit"])
            p["regs"] = int(row["registers"])
            out.append(p)
    return out


def load_programs(path: Path) -> dict[str, Program]:
    with open(path, newline="", encoding="utf-8") as f:
        return {row["id"]: json.loads(row["program"]) for row in csv.DictReader(f)}


def dumps(prog: Program) -> str:
    return json.dumps(prog, separators=(",", ":"))


def write_programs(path: Path, programs: dict[str, Program]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "program"])
        for pid, prog in programs.items():
            w.writerow([pid, dumps(prog)])


# ---------------------------------------------------------------- verifier = your exact score
def apply_op(op: str, a: int, b: int) -> int:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        if a % b:
            raise ValueError("inexact division")
        return a // b
    raise ValueError(f"bad op {op!r}")


def verify(p: Puzzle, prog: Program) -> int:
    """Return the instruction count if the program is valid, else raise ValueError."""
    if len(prog) > MAX_INSTR or len(dumps(prog).encode()) > MAX_BYTES:
        raise ValueError("program too long")
    n: int = p["regs"]
    R: list[Optional[int]] = [None] * n
    for r, v in enumerate(p["start"]):
        R[r] = v
    need = set(p["targets"]) - set(p["start"])
    for k, ins in enumerate(prog):
        if len(ins) != 4:
            raise ValueError(f"step {k}: malformed")
        op, i, j, d = ins
        if not all(type(x) is int and 0 <= x < n for x in (i, j, d)):
            raise ValueError(f"step {k}: bad register")
        a, b = R[i], R[j]
        if a is None or b is None:
            raise ValueError(f"step {k}: undefined register")
        v = apply_op(op, a, b)
        if not 1 <= v <= p["limit"]:
            raise ValueError(f"step {k}: value {v} out of range")
        R[d] = v
        need.discard(v)
    if need:
        raise ValueError(f"missing target {min(need)}")
    return len(prog)


def prune(p: Puzzle, prog: Program) -> Program:
    """Keep the first instruction producing each target, plus everything it depends on."""
    n: int = p["regs"]
    R: list[Optional[int]] = [None] * n
    src: list[Optional[int]] = [None] * n
    for r, v in enumerate(p["start"]):
        R[r] = v
    need = set(p["targets"]) - set(p["start"])
    deps: list[tuple[Optional[int], Optional[int]]] = []
    first: dict[int, int] = {}
    for k, (op, i, j, d) in enumerate(prog):
        a, b = R[i], R[j]
        if a is None or b is None:
            raise ValueError("undefined register")
        v = apply_op(op, a, b)
        deps.append((src[i], src[j]))
        R[d], src[d] = v, k
        if v in need and v not in first:
            first[v] = k
    keep: set[int] = set()
    stack: list[Optional[int]] = list(first.values())
    while stack:
        k = stack.pop()
        if k is not None and k not in keep:
            keep.add(k)
            stack.extend(deps[k])
    return [ins for k, ins in enumerate(prog) if k in keep]


# ---------------------------------------------------------------- chains: ways to build one number
def signed_digits(n: int, radix: int) -> tuple[int, list[int]]:
    half = radix // 2
    ds: list[int] = []
    while n > half:
        e = n % radix
        if e > half:
            e -= radix
        n = (n - e) // radix
        ds.append(e)
    return n, ds[::-1]


def horner_cost(n: int, radix: int) -> int:
    if n <= radix // 2:
        return 0
    _, ds = signed_digits(n, radix)
    return len(ds) + sum(1 for e in ds if e)


def horner_chain(n: int, half: int) -> Chain:
    """v0's method: digit by digit in base 2*half."""
    radix = 2 * half
    top, ds = signed_digits(n, radix)
    steps: list[Step] = []
    for e in ds:
        steps.append(("*", radix, 0))
        if e:
            steps.append(("+" if e > 0 else "-", abs(e), 0))
    return top, steps


def one_op_table(held: dict[int, int], lim: int) -> dict[int, tuple[int, Step]]:
    """Every value reachable with ONE instruction from two held values."""
    out: dict[int, tuple[int, Step]] = {}
    vals = sorted(held)
    for a in vals:
        for b in vals:
            for v, st in ((a + b, ("+", b, 0)), (a * b, ("*", b, 0)),
                          (a - b, ("-", b, 0)), (a // b if a % b == 0 else 0, ("/", b, 0))):
                if 1 <= v <= lim and v not in held and v not in out:
                    out[v] = (a, st)
    return out


def chain_search(n: int, held: dict[int, int], mults: list[int],
                 one_op: dict[int, tuple[int, Step]], offsets: list[int],
                 lim: int, beam: int) -> Optional[Chain]:
    """Backward beam search for a short recipe that builds n from values already held.

    held    : value -> register of everything we may read
    mults   : sorted values we may multiply by (all >= 2, all held)
    offsets : extra held values usable as 'x = y + a' or 'x = a - y' (v2 uses this; v1 passes [])
    """
    if n in held:
        return (n, [])
    cost_of: dict[int, int] = {n: 0}
    parent: dict[int, tuple[int, list[Step]]] = {}
    levels: dict[int, list[int]] = defaultdict(list)
    levels[0].append(n)
    best: Optional[tuple[int, int, list[Step], int]] = None   # cost, start value, first steps, attach
    c = 0
    while c < 200 and (best is None or c < best[0]):
        xs = sorted({x for x in levels.pop(c, []) if cost_of[x] == c})[:beam]
        if not xs and not levels:
            break
        for x in xs:
            if x in one_op and (best is None or c + 1 < best[0]):
                y0, st = one_op[x]
                best = (c + 1, y0, [st], x)
            moves: list[tuple[int, int, list[Step]]] = []
            for m in mults:
                if m > x:
                    break
                q, r = divmod(x, m)
                if r == 0:
                    moves.append((q, c + 1, [("*", m, 0)]))
                    continue
                if r in held:
                    moves.append((q, c + 2, [("*", m, 0), ("+", r, 0)]))
                s = m - r
                if s in held and (q + 1) * m <= lim:
                    moves.append((q + 1, c + 2, [("*", m, 0), ("-", s, 0)]))
            for a in offsets:
                if a < x:
                    moves.append((x - a, c + 1, [("+", a, 0)]))
                elif a > x:
                    moves.append((a - x, c + 1, [("-", a, 1)]))
            for y, cc, st in moves:
                if best is not None and cc >= best[0]:
                    continue
                if cost_of.get(y, 1 << 30) <= cc:
                    continue
                cost_of[y] = cc
                parent[y] = (x, st)
                if y in held:
                    best = (cc, y, [], y)
                else:
                    levels[cc].append(y)
        c += 1
    if best is None:
        return None
    _, y0, first, v = best
    steps = list(first)
    while v != n:
        px, st = parent[v]
        steps.extend(st)
        v = px
    return (y0, steps)


# ---------------------------------------------------------------- program builder
def build(p: Puzzle, half: int, n_ring: int, cache: dict[Any, Any]) -> Optional[Program]:
    regs: int = p["regs"]
    lim: int = p["limit"]
    prog: Program = []
    anchors: dict[int, int] = {}
    for r, v in enumerate(p["start"]):
        anchors.setdefault(v, r)
    nxt = len(p["start"])

    def new_reg() -> int:
        nonlocal nxt
        if nxt >= regs:
            raise IndexError
        nxt += 1
        return nxt - 1

    def emit(op: str, i: int, j: int, d: int) -> int:
        prog.append([op, i, j, d])
        return d

    try:
        if 1 not in anchors:
            anchors[1] = emit("/", 0, 0, new_reg())
        for dg in range(2, half + 1):
            if dg not in anchors:
                anchors[dg] = emit("+", anchors[dg - 1], anchors[1], new_reg())
        radix = 2 * half
        if radix not in anchors:
            anchors[radix] = emit("+", anchors[half], anchors[half], new_reg())
        acc = new_reg()
        ring = [new_reg() for _ in range(n_ring)]
    except IndexError:
        return None
    fixed = dict(anchors)
    have = set(anchors)

    if ("one", half) not in cache:
        cache[("one", half)] = one_op_table(fixed, lim)
    one_fixed: dict[int, tuple[int, Step]] = cache[("one", half)]
    mults = sorted(v for v in fixed if v >= 2)

    def chain(n: int) -> Chain:
        key = ("chain", half, n)
        if key not in cache:
            h = horner_chain(n, half)
            s = chain_search(n, fixed, mults, one_fixed, [], lim, BEAM)
            cache[key] = s if s is not None and len(s[1]) < len(h[1]) else h
        return cache[key]

    def emit_chain(ch: Chain, dest: int) -> int:
        """Emit a chain; returns the register holding the result."""
        y0, steps = ch
        cur = fixed[y0]
        value = y0
        for k, (op, other, side) in enumerate(steps):
            d = dest if k == len(steps) - 1 else acc
            ro = fixed[other]
            if side == 0:
                emit(op, cur, ro, d)
                value = apply_op(op, value, other)
            else:
                emit(op, ro, cur, d)
                value = apply_op(op, other, value)
            have.add(value)
            cur = d
        return cur

    slot = 0
    for t in p["targets"]:
        if t in have:
            continue
        dest = ring[slot % n_ring] if n_ring else acc

        # option: from scratch
        scratch = chain(t)
        best_cost = len(scratch[1])
        plan: tuple[str, int, int, int] = ("scratch", 0, 0, 0)
        # option: one instruction / v*k +/- small (as in v0)
        for v, rv in anchors.items():
            if t > v and (t - v) in anchors:
                best_cost, plan = 1, ("op", rv, anchors[t - v], 0)
            elif v > t and (v - t) in anchors:
                best_cost, plan = 1, ("op", rv, anchors[v - t], 1)
            elif t % v == 0 and (t // v) in anchors:
                best_cost, plan = 1, ("op", rv, anchors[t // v], 2)
            elif v % t == 0 and (v // t) in anchors:
                best_cost, plan = 1, ("op", rv, anchors[v // t], 3)
            if best_cost == 1:
                break
            for k in (t // v, t // v + 1):
                if k > 1 and k in anchors and v * k <= lim:
                    e = t - v * k
                    if e and abs(e) <= half and best_cost > 2:
                        best_cost, plan = 2, ("muladj", rv, anchors[k], e)
        # option: earlier target +/- difference (try the few with the cheapest-looking difference)
        if best_cost > 2:
            near = sorted(((horner_cost(abs(t - v), radix), v) for v in anchors
                           if v != t and (t > v or v - t < v)))[:DELTA_TRIES]
            for _, v in near:
                delta = abs(t - v)
                c = 1 + (0 if delta in fixed else len(chain(delta)[1]))
                if c < best_cost:
                    best_cost, plan = c, ("delta", anchors[v], delta, int(t > v))

        kind, a, b, c = plan
        if kind == "scratch":
            emit_chain(scratch, dest)
        elif kind == "op":
            emit("+-*/"[c], a, b, dest)
        elif kind == "muladj":
            emit("*", a, b, acc)
            emit("+" if c > 0 else "-", acc, fixed[abs(c)], dest)
        else:
            s = fixed[b] if b in fixed else emit_chain(chain(b), acc)
            emit("+" if c else "-", a, s, dest)

        if n_ring:
            anchors = {v: r for v, r in anchors.items() if r != dest}
            anchors[t] = dest
            slot += 1
        have.add(t)
    return prog


def solve(p: Puzzle) -> Optional[Program]:
    best: Optional[Program] = None
    cache: dict[Any, Any] = {}
    for half in HALVES:
        for n_ring in RINGS:
            prog = build(p, half, n_ring, cache)
            if prog is None or (best is not None and len(prog) >= len(best)):
                continue
            try:
                verify(p, prog)
                best = prog
            except ValueError:
                pass
    return best


# ---------------------------------------------------------------- runners (parallel)
def _solve_task(p: Puzzle) -> tuple[str, Optional[Program]]:
    return p["id"], solve(p)


def solve_all(puzzles: list[Puzzle]) -> dict[str, Optional[Program]]:
    out: dict[str, Optional[Program]] = {}
    t0 = time.time()
    with Pool(os.cpu_count() or 1) as pool:
        for k, (pid, prog) in enumerate(pool.imap_unordered(_solve_task, puzzles, chunksize=2)):
            out[pid] = prog
            if (k + 1) % 50 == 0 or k + 1 == len(puzzles):
                print(f"  {k + 1}/{len(puzzles)} puzzles  ({time.time() - t0:.0f}s)", flush=True)
    return out


def report(data_dir: Path = DATA_DIR) -> None:
    puzzles = load_puzzles(data_dir / "practice_puzzles.csv")
    solved = solve_all(puzzles)
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for p in puzzles:
        prog = solved[p["id"]]
        for key in (p["family"], "ALL"):
            agg[key][0] += 1
            agg[key][2] += int(p["lower_bound"])
            if prog is None:
                agg[key][3] += 1
            else:
                agg[key][1] += len(prog)
    print(f"{'family':15}{'n':>5}{'mine':>10}{'lower_bd':>10}{'ratio':>8}{'failed':>8}")
    for k, (n, mine, lb, failed) in sorted(agg.items()):
        print(f"{k:15}{n:>5}{mine:>10}{lb:>10}{mine / max(lb, 1):>8.2f}{failed:>8}")


def main(data_dir: Path = DATA_DIR) -> None:
    puzzles = load_puzzles(data_dir / "puzzles.csv")
    sample = load_programs(data_dir / "sample_submission.csv")
    earlier = [load_programs(f) for f in sorted(Path(".").glob("submission*.csv")) if f != OUT_PATH]
    print(f"merging with {len(earlier)} earlier submission file(s)")
    solved = solve_all(puzzles)

    out: dict[str, Program] = {}
    mine_total = 0
    for p in puzzles:
        s = sample[p["id"]]
        cands: list[Program] = [s]
        mine = solved[p["id"]]
        if mine is not None:
            cands.append(mine)
            mine_total += len(mine)
        try:
            pr = prune(p, s)
            verify(p, pr)
            cands.append(pr)
        except ValueError:
            pass
        for e in earlier:
            prog = e.get(p["id"])
            if prog is None:
                continue
            try:
                verify(p, prog)
                cands.append(prog)
            except (ValueError, TypeError):
                pass
        out[p["id"]] = min(cands, key=len)

    write_programs(OUT_PATH, out)
    written = load_programs(OUT_PATH)
    assert set(written) == {p["id"] for p in puzzles}
    final = sum(verify(p, written[p["id"]]) for p in puzzles)
    print(f"{VERSION} solver alone: {mine_total}   ->   {OUT_PATH} (best per puzzle): {final}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("all", "practice"):
        report()
    if mode in ("all", "submit"):
        main()