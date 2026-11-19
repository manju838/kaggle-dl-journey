"""
Santa 2026 - Elf Calculator: starter solution (v0).

Folder layout expected:
    elf_starter.py
    data/puzzles.csv
    data/practice_puzzles.csv
    data/sample_submission.csv

Usage:
    python elf_starter.py            # practice report, then build submission.csv
    python elf_starter.py practice   # only the practice report
    python elf_starter.py submit     # only build submission.csv

Per puzzle the solver keeps the shortest VALID program among:
    - the sample program
    - the sample program with unused instructions removed
    - its own construction (digit table + Horner, reusing earlier targets)
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

csv.field_size_limit(10**9)

DATA_DIR = Path("data")
MAX_INSTR, MAX_BYTES = 4096, 131072

Puzzle = dict[str, Any]
Program = list[list[Any]]


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


# ---------------------------------------------------------------- step 1: prune dead instructions
def prune(p: Puzzle, prog: Program) -> Program:
    """Keep the first instruction producing each target, plus everything it depends on."""
    n: int = p["regs"]
    R: list[Optional[int]] = [None] * n
    src: list[Optional[int]] = [None] * n          # index of the instruction that wrote each register
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


# ---------------------------------------------------------------- step 2: constructor
def signed_digits(n: int, radix: int) -> tuple[int, list[int]]:
    """n = top*R^k + d1*R^(k-1) + ... + dk with |d| <= R/2 and 1 <= top <= R/2."""
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


# A plan is always (kind, a, b, c) so every plan has the same shape:
#   ("scratch", 0, 0, 0)          build t from scratch with Horner
#   ("op", reg_a, reg_b, op_code) t = reg_a <op> reg_b        op_code: 0 + / 1 - / 2 * / 3 /
#   ("muladj", reg_v, reg_k, e)   t = reg_v * reg_k + e        |e| <= half, e != 0
#   ("delta", reg_v, delta, up)   t = reg_v + delta (up=1) or reg_v - delta (up=0)
Plan = tuple[str, int, int, int]
OPS = "+-*/"


def build(p: Puzzle, half: int, n_ring: int) -> Optional[Program]:
    regs: int = p["regs"]
    lim: int = p["limit"]
    prog: Program = []
    anchors: dict[int, int] = {}                   # value -> register, values we can reuse
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
        r_radix = anchors[radix]
        acc = new_reg()                            # scratch register
        ring = [new_reg() for _ in range(n_ring)]  # registers that remember recent targets
    except IndexError:
        return None
    fixed = dict(anchors)                          # digit table + starting values, never overwritten

    def horner(n: int, dest: int) -> None:
        top, ds = signed_digits(n, radix)
        cur = fixed[top]
        for e in ds:
            emit("*", cur, r_radix, acc)
            cur = acc
            if e:
                emit("+" if e > 0 else "-", acc, fixed[abs(e)], acc)
        prog[-1][3] = dest

    have = set(anchors)
    slot = 0
    for t in p["targets"]:                         # targets are given in increasing order
        if t in have:
            continue
        dest = ring[slot % n_ring] if n_ring else acc

        best_cost = horner_cost(t, radix)
        plan: Plan = ("scratch", 0, 0, 0)
        for v, rv in anchors.items():
            # 1 instruction: combine two values we already hold
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
            # 2 instructions: v * k +/- small digit
            for k in (t // v, t // v + 1):
                if k > 1 and k in anchors and v * k <= lim:
                    e = t - v * k
                    if e and abs(e) <= half and best_cost > 2:
                        best_cost, plan = 2, ("muladj", rv, anchors[k], e)
            # 1 + chain: v +/- difference
            delta = abs(t - v)
            if delta and (t > v or delta < v):
                c = 1 + horner_cost(delta, radix)
                if c < best_cost:
                    best_cost, plan = c, ("delta", rv, delta, int(t > v))

        kind, a, b, c = plan
        if kind == "scratch":
            horner(t, dest)
        elif kind == "op":
            emit(OPS[c], a, b, dest)
        elif kind == "muladj":
            emit("*", a, b, acc)
            emit("+" if c > 0 else "-", acc, fixed[abs(c)], dest)
        else:  # delta
            if b <= half:
                s = fixed[b]
            else:
                horner(b, acc)
                s = acc
            emit("+" if c else "-", a, s, dest)

        if n_ring:
            anchors = {v: r for v, r in anchors.items() if r != dest}
            anchors[t] = dest
            slot += 1
        have.add(t)
    return prog


HALVES = (1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 30)
RINGS = (0, 4, 8, 16, 24, 32)


def solve(p: Puzzle) -> Optional[Program]:
    best: Optional[Program] = None
    for n_ring in RINGS:
        for half in HALVES:
            prog = build(p, half, n_ring)
            if prog is None or (best is not None and len(prog) >= len(best)):
                continue
            try:
                verify(p, prog)
                best = prog
            except ValueError:
                pass
    return best


# ---------------------------------------------------------------- runners
def report(data_dir: Path = DATA_DIR) -> None:
    """Practice set: your instruction count vs lower_bound, per family. Attack the biggest ratio."""
    puzzles = load_puzzles(data_dir / "practice_puzzles.csv")
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])   # n, mine, lower_bound, failed
    for p in puzzles:
        prog = solve(p)
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


def main(data_dir: Path = DATA_DIR, out_path: Path = Path("submission.csv")) -> None:
    puzzles = load_puzzles(data_dir / "puzzles.csv")
    sample = load_programs(data_dir / "sample_submission.csv")
    out: dict[str, Program] = {}
    base = 0
    for n, p in enumerate(puzzles):
        s = sample[p["id"]]
        base += verify(p, s)
        cands: list[Program] = [s]
        try:
            pr = prune(p, s)
            verify(p, pr)
            cands.append(pr)
        except ValueError:
            pass
        mine = solve(p)
        if mine is not None:
            cands.append(mine)
        out[p["id"]] = min(cands, key=len)
        if n % 100 == 0:
            print(f"{n}/{len(puzzles)}", flush=True)

    write_programs(out_path, out)
    written = load_programs(out_path)              # re-read from disk and re-score
    assert set(written) == {p["id"] for p in puzzles}
    final = sum(verify(p, written[p["id"]]) for p in puzzles)
    print(f"sample: {base}   ->   {out_path}: {final}")
    worst = sorted(puzzles, key=lambda p: -len(out[p["id"]]))[:10]
    print("longest programs (attack these):", [(p["id"], len(out[p["id"]])) for p in worst])


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("all", "practice"):
        report()
    if mode in ("all", "submit"):
        main()