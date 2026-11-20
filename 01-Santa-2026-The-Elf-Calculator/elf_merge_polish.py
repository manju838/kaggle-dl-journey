"""
Santa 2026 - Elf Calculator: merge every submission you have, then polish it.

What the top public notebooks do, in one script:
  1. MERGE.  Read every CSV with the header `id,program` under kaggle-submissions/ (any depth)
     plus submission*.csv in this folder. Replay every program with an exact referee; invalid
     ones are ignored. Keep the shortest valid program per puzzle (puzzles score independently).
  2. RE-ALLOCATE.  Rebuild each winner from its value graph and re-assign the 48 registers with
     Belady's rule (evict the value needed furthest in the future). Removes re-computations.
  3. RECOMBINE (needs `pip install ortools`).  Pool the values of several near-shortest programs
     of a puzzle (from different files), list every way a pooled value can be made from two
     others, and let CP-SAT pick the smallest set of values that still builds every target.
     Port of Roman Rozen's "Santa 2026 | IceFire Strong Start" notebook (Apache 2.0).
  A program is replaced only if the new one replays correctly and is strictly shorter.

Folder layout:
    elf_merge_polish.py
    data/puzzles.csv
    kaggle-submissions/...          <- downloaded notebook outputs, your own submissions, anything
Usage:
    python elf_merge_polish.py              # merge + re-allocate + recombine -> submission_best.csv
    python elf_merge_polish.py --merge-only # just the merge (seconds)
Output: submission_best.csv in this folder, re-read and re-scored before the script ends.
"""
from __future__ import annotations

import csv
import heapq
import json
import os
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Optional

csv.field_size_limit(2**31 - 1)

DATA_DIR = Path("data")
BANK_DIR = Path("kaggle-submissions")
OUT_PATH = Path("submission_best.csv")
MAX_INSTR, MAX_BYTES = 4096, 131072
POOL_SLACK = 2        # recombine programs up to this many instructions longer than the best
MAX_PROGRAMS = 8      # at most this many programs pooled per puzzle
CPSAT_TIME = 5.0      # deterministic CP-SAT time per puzzle (roughly seconds)
INF = 1 << 60

Puzzle = dict[str, Any]
Program = list[list[Any]]
Recipe = tuple[str, int, int]


# ---------------------------------------------------------------- I/O + referee
def load_puzzles(path: Path) -> dict[str, Puzzle]:
    out: dict[str, Puzzle] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            out[row["id"]] = dict(start=json.loads(row["starting_values"]), targets=json.loads(row["targets"]),
                                  limit=int(row["value_limit"]), regs=int(row["registers"]))
    return out


def read_bank(path: Path) -> Optional[dict[str, str]]:
    """{id: program cell} if the file is an id,program CSV, else None."""
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            rd = csv.reader(f)
            if next(rd, None) != ["id", "program"]:
                return None
            return {row[0]: row[1] for row in rd if len(row) == 2}
    except (OSError, UnicodeDecodeError, csv.Error):
        return None


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


def dumps(prog: Program) -> str:
    return json.dumps(prog, separators=(",", ":"))


def verify(p: Puzzle, prog: Program) -> int:
    """Instruction count of a valid program; raises ValueError otherwise (official rules)."""
    if not isinstance(prog, list) or len(prog) > MAX_INSTR or len(dumps(prog).encode()) > MAX_BYTES:
        raise ValueError("program too long")
    n: int = p["regs"]
    R: list[Optional[int]] = [None] * n
    for r, v in enumerate(p["start"]):
        R[r] = v
    need = set(p["targets"]) - set(p["start"])
    for k, ins in enumerate(prog):
        if not isinstance(ins, list) or len(ins) != 4:
            raise ValueError(f"step {k}: malformed")
        op, i, j, d = ins
        if not all(type(x) is int and 0 <= x < n for x in (i, j, d)):
            raise ValueError(f"step {k}: bad register")
        a, b = R[i], R[j]
        if a is None or b is None:
            raise ValueError(f"step {k}: undefined register")
        v = apply_op(op, a, b)
        if not 1 <= v <= p["limit"]:
            raise ValueError(f"step {k}: value out of range")
        R[d] = v
        need.discard(v)
    if need:
        raise ValueError("missing targets")
    return len(prog)


def write_programs(path: Path, programs: dict[str, Program]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "program"])
        for pid in sorted(programs):
            w.writerow([pid, dumps(programs[pid])])


# ---------------------------------------------------------------- value graph + Belady compiler
def recipes_of(starts: list[int], prog: Program) -> dict[int, Recipe]:
    """Values in order of first appearance -> (op, a, b)."""
    regs: list[Optional[int]] = list(starts) + [None] * (48 - len(starts))
    rec: dict[int, Recipe] = {}
    sset = set(starts)
    for op, i, j, d in prog:
        a, b = regs[i], regs[j]
        assert a is not None and b is not None
        v = apply_op(op, a, b)
        regs[d] = v
        if v not in rec and v not in sset:
            if op in "+*" and a < b:
                a, b = b, a
            rec[v] = (op, a, b)
    return rec


def dag_prune(starts: list[int], targets: list[int], rec: dict[int, Recipe]) -> dict[int, Recipe]:
    sset, need, stack = set(starts), set(), [t for t in targets if t not in starts]
    while stack:
        v = stack.pop()
        if v in need or v in sset:
            continue
        need.add(v)
        _, a, b = rec[v]
        stack += [a, b]
    return {v: rec[v] for v in need}


def dag_schedule(starts: list[int], rec: dict[int, Recipe], hint_pos: dict[int, int], pull: bool) -> list[int]:
    sset = set(starts)
    consumers: dict[int, list[int]] = defaultdict(list)
    for v, (_, a, b) in rec.items():
        consumers[a].append(v)
        if b != a:
            consumers[b].append(v)
    topo: list[int] = []
    seen = set(sset)
    for root in rec:
        stack: list[tuple[int, bool]] = [(root, False)]
        while stack:
            v, done = stack.pop()
            if done:
                topo.append(v)
                continue
            if v in seen:
                continue
            seen.add(v)
            stack.append((v, True))
            _, a, b = rec[v]
            stack += [(u, False) for u in (a, b) if u not in seen]
    prio: dict[int, float] = {}
    for v in reversed(topo):
        cons = [prio[c] for c in consumers.get(v, ()) if c in prio]
        jit = min(cons) - 0.5 if cons else 1e18
        if v in hint_pos:
            prio[v] = min(float(hint_pos[v]), jit) if pull else float(hint_pos[v])
        else:
            prio[v] = jit
    indeg = {v: len({u for u in (a, b) if u not in sset}) for v, (_, a, b) in rec.items()}
    heap = [(prio[v], v) for v, k in indeg.items() if k == 0]
    heapq.heapify(heap)
    seq: list[int] = []
    while heap:
        _, v = heapq.heappop(heap)
        seq.append(v)
        for c in consumers.get(v, ()):
            indeg[c] -= 1
            if indeg[c] == 0:
                heapq.heappush(heap, (prio[c], c))
    return seq


def compile_plan(p: Puzzle, rec: dict[int, Recipe], hint_pos: dict[int, int], pull: bool) -> Program:
    starts: list[int] = p["start"]
    nreg: int = p["regs"]
    sset = set(starts)
    rec = dag_prune(starts, p["targets"], rec)
    seq = dag_schedule(starts, rec, hint_pos, pull)
    uses: dict[int, list[int]] = defaultdict(list)
    for pos, v in enumerate(seq):
        _, a, b = rec[v]
        uses[a].append(pos)
        if b != a:
            uses[b].append(pos)
    ptr: dict[int, int] = defaultdict(int)

    def next_use(v: int, pos: int) -> int:
        lst = uses.get(v, [])
        i = ptr[v]
        while i < len(lst) and lst[i] <= pos:
            i += 1
        ptr[v] = i
        return lst[i] if i < len(lst) else INF

    bank: list[Optional[int]] = [None] * nreg
    where: dict[int, int] = {}
    prog: Program = []
    for k, s in enumerate(starts):
        bank[k] = s
        where.setdefault(s, k)

    def dest(pos: int, hard: set[int], soft: set[int]) -> int:
        best_r, best_key = -1, -1
        for r in range(nreg):
            if r in hard:
                continue
            u = bank[r]
            if u is None:
                return r
            nu = next_use(u, pos)
            if (r in soft and nu != INF) or u in sset:
                continue
            if nu > best_key:
                best_r, best_key = r, nu
                if nu == INF:
                    return r
        return best_r

    def emit(v: int, pos: int, hard: set[int], depth: int = 0) -> int:
        if v in where:
            return where[v]
        if v in sset or depth > 64:
            raise ValueError("cannot rebuild")
        op, a, b = rec[v]
        pre = hard | {where[u] for u in (a, b) if u in where}
        ra = emit(a, pos, pre, depth + 1)
        rb = emit(b, pos, pre | {ra}, depth + 1)
        rd = dest(pos, hard, {ra, rb})
        if rd < 0:
            raise ValueError("register allocation failed")
        old = bank[rd]
        if old is not None and where.get(old) == rd:
            del where[old]
        bank[rd], where[v] = v, rd
        prog.append([op, ra, rb, rd])
        return rd

    for pos, v in enumerate(seq):
        emit(v, pos, set())
    return prog


# ---------------------------------------------------------------- CP-SAT recombination (IceFire port)
def all_recipes(values: set[int], starts: set[int], cap: int) -> list[tuple[int, str, int, int]]:
    vl, out = sorted(values), []
    for i, a in enumerate(vl):
        for b in vl[: i + 1]:
            s = a + b
            if s in values and s not in starts and s != a and s != b:
                out.append((s, "+", a, b))
            if a == b:
                if 1 in values and 1 not in starts and a != 1:
                    out.append((1, "/", a, a))
            else:
                d = a - b
                if d in values and d not in starts and d != b:
                    out.append((d, "-", a, b))
                if a % b == 0:
                    q = a // b
                    if q in values and q not in starts and q != a and q != b:
                        out.append((q, "/", a, b))
            pr = a * b
            if pr <= cap and pr in values and pr not in starts and pr != a and pr != b:
                out.append((pr, "*", a, b))
    return out


def cpsat_plan(starts: list[int], targets: list[int], values: list[int], recipes: list[tuple[int, str, int, int]],
               hint: dict[int, Recipe], det_time: float) -> Optional[dict[int, Recipe]]:
    from ortools.sat.python import cp_model
    sset = set(starts)
    nonstart = [v for v in values if v not in sset]
    order = {v: k + 1 for k, v in enumerate(hint)}
    m = cp_model.CpModel()
    x = {v: m.new_bool_var("") for v in nonstart}
    lvl = {v: m.new_int_var(0, len(nonstart), "") for v in nonstart}
    by_v: dict[int, list[Any]] = defaultdict(list)
    ys: list[tuple[Any, int, str, int, int]] = []
    penalty: list[Any] = []
    for v, op, a, b in recipes:
        y = m.new_bool_var("")
        by_v[v].append(y)
        ys.append((y, v, op, a, b))
        if hint.get(v) != (op, a, b):
            late = v in order and any(u in order and order[u] > order[v] for u in (a, b))
            penalty.append((3 if late else 1) * y)
        for u in (a, b):
            if u not in sset:
                m.add_implication(y, x[u])
                m.add(lvl[v] >= lvl[u] + 1).only_enforce_if(y)
        m.add_hint(y, int(hint.get(v) == (op, a, b)))
    for v in nonstart:
        if by_v[v]:
            m.add_bool_or(by_v[v]).only_enforce_if(x[v])
        else:
            m.add(x[v] == 0)
        m.add_hint(x[v], int(v in hint))
        m.add_hint(lvl[v], order.get(v, 0))
    for t in targets:
        if t not in sset:
            if t not in x:
                return None
            m.add(x[t] == 1)
    m.minimize(1000 * sum(x.values()) + sum(penalty))
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_deterministic_time = det_time
    solver.parameters.max_time_in_seconds = 30 * det_time
    status = solver.solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    plan: dict[int, Recipe] = {}
    for y, v, op, a, b in ys:
        if v not in plan and solver.value(x[v]) and solver.value(y):
            plan[v] = (op, a, b)
    return plan


def improve(task: tuple[str, Puzzle, list[Program], bool]) -> tuple[str, Optional[Program], str]:
    """Try re-allocation, then CP-SAT recombination. Returns a strictly shorter valid program or None."""
    pid, p, progs, use_cpsat = task
    starts, targets, cap = p["start"], p["targets"], p["limit"]
    best = progs[0]
    best_len = len(best)
    result: Optional[Program] = None
    how = ""
    hint = dag_prune(starts, targets, recipes_of(starts, best))
    hint_pos = {v: k for k, v in enumerate(recipes_of(starts, best))}

    def try_plan(plan: dict[int, Recipe], label: str) -> None:
        nonlocal result, best_len, how
        for pull in (False, True):
            try:
                prog = compile_plan(p, plan, hint_pos, pull)
                n = verify(p, prog)
            except (ValueError, KeyError, RecursionError, AssertionError):
                continue
            if n < best_len:
                result, best_len, how = prog, n, label

    try_plan(hint, "re-allocate")
    if use_cpsat:
        values = set(starts)
        for prog in progs[:MAX_PROGRAMS]:
            if len(prog) <= len(best) + POOL_SLACK:
                values |= set(recipes_of(starts, prog))
        try:
            plan = cpsat_plan(starts, targets, sorted(values), all_recipes(values, set(starts), cap), hint, CPSAT_TIME)
        except Exception:
            plan = None
        if plan and len(plan) < best_len:
            try_plan(plan, "recombine")
    return pid, result, how


# ---------------------------------------------------------------- main
def find_banks() -> list[Path]:
    files = sorted(BANK_DIR.rglob("*.csv")) if BANK_DIR.is_dir() else []
    files += sorted(Path(".").glob("submission*.csv"))
    seen, out = set(), []
    for f in files:
        key = f.resolve()
        if key not in seen and f.resolve() != OUT_PATH.resolve():
            seen.add(key)
            out.append(f)
    if OUT_PATH.exists():
        out.append(OUT_PATH)           # the previous best is a bank too
    return out


def main() -> None:
    merge_only = "--merge-only" in sys.argv[1:]
    puzzles = load_puzzles(DATA_DIR / "puzzles.csv")
    banks = find_banks()
    if not banks:
        raise SystemExit(f"no id,program CSVs found in {BANK_DIR}/ or ./submission*.csv")

    # 1. merge -------------------------------------------------------------------------------
    t0 = time.time()
    cands: dict[str, dict[str, Program]] = defaultdict(dict)   # pid -> {cell: program}
    first_file: dict[str, str] = {}                              # cell -> first file it was seen in
    print(f"{'file':60}{'valid':>7}{'total':>10}")
    for path in banks:
        rows = read_bank(path)
        if rows is None:
            continue
        ok = tot = 0
        for pid, p in puzzles.items():
            cell = rows.get(pid)
            if cell is None or cell in cands[pid]:
                if cell is not None:
                    ok += 1
                    tot += len(cands[pid][cell])
                continue
            try:
                prog = json.loads(cell)
                n = verify(p, prog)
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
            ok += 1
            tot += n
            cands[pid][cell] = prog
            first_file.setdefault(cell, str(path))
        name = str(path)
        print(f"{name[-58:]:60}{ok:>7}{(f'{tot:,}' if ok == len(puzzles) else 'partial'):>10}")
    missing = [pid for pid in puzzles if not cands[pid]]
    if missing:
        raise SystemExit(f"no valid program for {len(missing)} puzzles, e.g. {missing[:3]}")
    best: dict[str, Program] = {}
    owner: Counter[str] = Counter()
    for pid in puzzles:
        cell, prog = min(cands[pid].items(), key=lambda kv: len(kv[1]))
        best[pid] = prog
        owner[first_file[cell]] += 1
    merged = sum(len(v) for v in best.values())
    print(f"\nmerged (best per puzzle over {len(banks)} files): {merged:,}   [{time.time() - t0:.0f}s]")
    print("files holding the shortest program (first file wins ties):")
    for f, k in owner.most_common(8):
        print(f"  {k:5d}  {f}")

    # 2 + 3. polish --------------------------------------------------------------------------
    if not merge_only:
        try:
            import ortools  # noqa: F401
            use_cpsat = True
        except ImportError:
            use_cpsat = False
            print("ortools not installed: skipping recombination (pip install ortools to enable it)")
        tasks = []
        for pid, p in puzzles.items():
            progs = sorted(cands[pid].values(), key=len)
            tasks.append((pid, p, progs, use_cpsat))
        tasks.sort(key=lambda t: -len(t[2][0]))          # long puzzles first for load balance
        t0 = time.time()
        gains: Counter[str] = Counter()
        with Pool(os.cpu_count() or 1) as pool:
            for k, (pid, prog, how) in enumerate(pool.imap_unordered(improve, tasks, chunksize=2), 1):
                if prog is not None and len(prog) < len(best[pid]):
                    gains[how] += len(best[pid]) - len(prog)
                    best[pid] = prog
                if k % 100 == 0 or k == len(tasks):
                    print(f"  polished {k}/{len(tasks)}  saved {sum(gains.values())}  {dict(gains)}  [{time.time() - t0:.0f}s]",
                          flush=True)

    # write + re-check ------------------------------------------------------------------------
    write_programs(OUT_PATH, best)
    back = read_bank(OUT_PATH)
    assert back is not None and set(back) == set(puzzles)
    final = sum(verify(puzzles[pid], json.loads(back[pid])) for pid in puzzles)
    print(f"\n{OUT_PATH}: {final:,} instructions (merged {merged:,})  -> submit this file")


if __name__ == "__main__":
    main()