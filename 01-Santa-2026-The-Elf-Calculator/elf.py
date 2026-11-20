"""
Santa 2026 - Elf Calculator: v4  (value-graph builder + Belady register scheduler)

What changed vs v3 (ideas from the top public notebooks):
  1. VALUE-GRAPH BUILDER. Each puzzle is planned as a graph of values (each value has one recipe
     v = a op b); registers are assigned only at the end. While planning, the 160 most recently
     used values are free to reuse, not just the 48 that fit in registers.
  2. BELADY SCHEDULER. The plan is fitted into 48 registers by evicting the value needed furthest
     in the future, re-computing a value only if it is needed again. Ported from Roman Rozen's
     "Santa 2026 | IceFire Strong Start" notebook (Apache 2.0).
  3. NEAR-MISS PRODUCT FINISH. A number can be finished as X*Y +/- R, with Y the dictionary value
     closest to x/X (from the "LNS + CP-SAT Recombine" notebook).
  4. PORTFOLIO. Six graph-builder settings plus v3's two best register-builder settings; the
     shortest valid program per puzzle wins.

Folder layout:
    elf.py
    data/puzzles.csv, data/practice_puzzles.csv, data/sample_submission.csv
    kaggle-submissions/...      <- any id,program CSVs (public outputs, older submissions), any depth
Usage:
    python elf.py            # practice report, then build submission_v4.csv
    python elf.py practice   # only the practice report
    python elf.py submit     # only build submission_v4.csv
submission_v4.csv also keeps the best program per puzzle from every CSV in kaggle-submissions/
and from submission*.csv in this folder, so it is never worse than what you already have.
"""
from __future__ import annotations

import csv
import heapq
import json
import math
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Optional

import numpy as np

csv.field_size_limit(2**31 - 1)

VERSION = "v4"
DATA_DIR = Path("data")
OUT_PATH = Path(f"submission_{VERSION}.csv")
BANK_DIR = Path("kaggle-submissions")      # downloaded notebook outputs / older submissions
MAX_INSTR, MAX_BYTES = 4096, 131072
SETTINGS: list[tuple[Any, ...]] = [
    # value-graph builder: ("dag", beam, digit table, dictionary size, recency window; 0 = no window)
    ("dag", 8, 12, 46, 160), ("dag", 8, 0, 46, 160), ("dag", 8, 6, 46, 160), ("dag", 8, 20, 46, 160),
    ("dag", 8, 12, 46, 90), ("dag", 8, 0, 46, 0),
    # register builder from v3: ("reg", beam, pin starting values, digit table)
    ("reg", 8, True, 12), ("reg", 8, True, 0),
]
MAX_COST = 60

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


# ---------------------------------------------------------------- the dictionary
class Catalog:
    """Register values (cost 0) + every value one instruction away from them (cost 1)."""

    def __init__(self, held: dict[int, int], lim: int, gen: Optional[list[int]] = None):
        """held: every value usable for free. gen: the values whose pairs form the dictionary
        (default: all of held). Keeping gen small keeps the dictionary fast when held is large."""
        hv_all = np.array(sorted(held), dtype=np.int64)
        hv = hv_all if gen is None else np.array(sorted(set(gen)), dtype=np.int64)
        n = len(hv)
        A = np.repeat(hv, n)
        B = np.tile(hv, n)
        lo_le = A <= B                                     # each unordered pair once for + and *
        parts_v, parts_op, parts_a, parts_b = [], [], [], []

        def add(mask: np.ndarray, vals: np.ndarray, op: int) -> None:
            parts_v.append(vals[mask]); parts_op.append(np.full(int(mask.sum()), op, dtype=np.int8))
            parts_a.append(A[mask]); parts_b.append(B[mask])

        s = A + B
        add(lo_le & (s <= lim), s, 0)
        safe = (A.astype(np.float64) * B.astype(np.float64)) < 4.0e18
        pr = np.where(safe, A * B, 0)
        add(lo_le & safe & (pr <= lim) & (A > 1) & (B > 1), pr, 2)
        d = A - B
        add(d > 0, d, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            exact = (B > 1) & (A % np.where(B == 0, 1, B) == 0) & (A >= B)
        add(exact, np.where(exact, A // np.where(B == 0, 1, B), 0), 3)

        dv = np.concatenate(parts_v) if parts_v else np.zeros(0, dtype=np.int64)
        dop = np.concatenate(parts_op) if parts_op else np.zeros(0, dtype=np.int8)
        da = np.concatenate(parts_a) if parts_a else np.zeros(0, dtype=np.int64)
        db = np.concatenate(parts_b) if parts_b else np.zeros(0, dtype=np.int64)
        keep = ~np.isin(dv, hv_all)
        dv, dop, da, db = dv[keep], dop[keep], da[keep], db[keep]
        dv, first = np.unique(dv, return_index=True)
        self.recipe_op, self.recipe_a, self.recipe_b = dop[first], da[first], db[first]
        self.dict_vals = dv

        allv = np.concatenate([hv_all, dv])
        cost = np.concatenate([np.zeros(len(hv_all), dtype=np.int64), np.ones(len(dv), dtype=np.int64)])
        order = np.argsort(allv, kind="stable")
        self.vals = allv[order]
        self.cost = cost[order]
        self.held = held

    def lookup(self, x: np.ndarray) -> np.ndarray:
        """Cost of each value in x: 0 held, 1 dictionary, -1 unknown."""
        idx = np.searchsorted(self.vals, x)
        idx = np.minimum(idx, len(self.vals) - 1)
        hit = self.vals[idx] == x
        return np.where(hit, self.cost[idx], -1)

    def cost_of(self, v: int) -> int:
        if v in self.held:
            return 0
        i = int(np.searchsorted(self.dict_vals, v))
        return 1 if i < len(self.dict_vals) and int(self.dict_vals[i]) == v else -1

    def recipe(self, v: int) -> tuple[str, int, int]:
        i = int(np.searchsorted(self.dict_vals, v))
        return "+-*/"[int(self.recipe_op[i])], int(self.recipe_a[i]), int(self.recipe_b[i])


# A forward step applied to the current value `cur`:
#   ("*", m, 0)  cur*m          ("*+", m, r) cur*m + r     ("*-", m, s) cur*m - s
#   ("+", a, 0)  cur + a        ("r-", a, 0) a - cur
#   ("sq", 0, 0) cur*cur        ("sq+", 0, r) cur*cur + r  ("sq-", 0, s) cur*cur - s
Step = tuple[str, int, int]


def search(n: int, cat: Catalog, lim: int, beam: int, near_miss: bool = True) -> Optional[tuple[int, list[Step]]]:
    """Backward beam search. Returns (start value, forward steps) or None.

    A finish is stored as (total cost, start value y0, state x it attaches to, step y0 -> x),
    then the chain is followed from x up to n through `parent`.
    """
    if cat.cost_of(n) >= 0:
        return (n, [])
    vals, vcost = cat.vals, cat.cost
    nv = len(vals)
    best_cost: dict[int, int] = {n: 0}
    parent: dict[int, tuple[int, Step]] = {}
    levels: dict[int, dict[int, None]] = defaultdict(dict)
    levels[0][n] = None
    found: Optional[tuple[int, int, int, Step]] = None

    def finish(total: int, y0: int, x: int, st: Step) -> None:
        nonlocal found
        if found is None or total < found[0]:
            found = (total, y0, x, st)

    c = 0
    while c < MAX_COST and (found is None or c < found[0]):
        xs = sorted(x for x in levels.pop(c, {}) if best_cost.get(x) == c)[:beam]
        if not xs and not levels:
            break
        for x in xs:
            cand_y: list[np.ndarray] = []
            cand_c: list[np.ndarray] = []
            cand_k: list[np.ndarray] = []
            cand_m: list[np.ndarray] = []
            cand_r: list[np.ndarray] = []
            hi = int(np.searchsorted(vals, x, side="right"))
            lo = int(np.searchsorted(vals, 2))
            m = vals[lo:hi]
            mc = vcost[lo:hi]
            if len(m):
                q = x // m
                r = x - q * m
                z = r == 0
                cand_y.append(q[z]); cand_c.append(c + 1 + mc[z]); cand_k.append(np.zeros(int(z.sum()), np.int8))
                cand_m.append(m[z]); cand_r.append(np.zeros(int(z.sum()), np.int64))
                nz = ~z
                rc = cat.lookup(r)
                ok = nz & (rc >= 0) & (q >= 1)
                cand_y.append(q[ok]); cand_c.append(c + 2 + mc[ok] + rc[ok]); cand_k.append(np.ones(int(ok.sum()), np.int8))
                cand_m.append(m[ok]); cand_r.append(r[ok])
                s = m - r
                sc = cat.lookup(s)
                ok2 = nz & (sc >= 0) & ((q + 1) * m <= lim)
                cand_y.append(q[ok2] + 1); cand_c.append(c + 2 + mc[ok2] + sc[ok2]); cand_k.append(np.full(int(ok2.sum()), 2, np.int8))
                cand_m.append(m[ok2]); cand_r.append(s[ok2])
                # NEAR-MISS PRODUCT (idea from the "LNS + CP-SAT Recombine" notebook):
                # x = m * Y +/- R with Y the catalog values just below / above x/m and R in the catalog.
                if near_miss:
                    idx = np.searchsorted(vals, q)
                    for j in (idx - 1, idx):
                        okj = (j >= 0) & (j < nv)
                        jj = np.clip(j, 0, nv - 1)
                        Yv = vals[jj]
                        Yc = vcost[jj]
                        okj &= (Yv >= 1) & (Yv != q)              # Y == q is the plain case above
                        okj &= (m.astype(np.float64) * Yv.astype(np.float64)) < 4.0e18
                        prod = np.where(okj, m * Yv, 0)
                        R = x - prod
                        aR = np.abs(R)
                        okj &= (aR >= 1) & (prod <= lim)
                        Rc = cat.lookup(aR)
                        okj &= Rc >= 0
                        if okj.any():
                            tot = c + 2 + mc + Yc + Rc
                            tot = np.where(okj, tot, 1 << 30)
                            k = int(np.argmin(tot))
                            if found is None or int(tot[k]) < found[0]:
                                kind = "*+" if int(R[k]) > 0 else "*-"
                                finish(int(tot[k]), int(Yv[k]), x, (kind, int(m[k]), int(aR[k])))
            # offsets: x = y + a  /  x = a - y
            hi_l = int(np.searchsorted(vals, x, side="left"))
            a_lo = vals[:hi_l]
            if len(a_lo):
                ac = vcost[:hi_l]
                cand_y.append(x - a_lo); cand_c.append(c + 1 + ac); cand_k.append(np.full(len(a_lo), 3, np.int8))
                cand_m.append(a_lo); cand_r.append(np.zeros(len(a_lo), np.int64))
            a_hi = vals[hi:]
            if len(a_hi):
                ah = vcost[hi:]
                cand_y.append(a_hi - x); cand_c.append(c + 1 + ah); cand_k.append(np.full(len(a_hi), 4, np.int8))
                cand_m.append(a_hi); cand_r.append(np.zeros(len(a_hi), np.int64))
            if cand_y:
                Y = np.concatenate(cand_y)
                C = np.concatenate(cand_c)
                K = np.concatenate(cand_k)
                M = np.concatenate(cand_m)
                Rr = np.concatenate(cand_r)
                good = Y >= 1
                if found is not None:
                    good &= C < found[0]
                Y, C, K, M, Rr = Y[good], C[good], K[good], M[good], Rr[good]
                if len(Y):
                    tc = cat.lookup(Y)
                    term = tc >= 0
                    if term.any():
                        tot = C[term] + tc[term]
                        k = int(np.argmin(tot))
                        finish(int(tot[k]), int(Y[term][k]), x,
                               _step(int(K[term][k]), int(M[term][k]), int(Rr[term][k])))
                    rest = ~term
                    Y, C, K, M, Rr = Y[rest], C[rest], K[rest], M[rest], Rr[rest]
                    for cc in np.unique(C):
                        sel = np.nonzero(C == cc)[0]
                        if len(sel) > beam:
                            sel = sel[np.argpartition(Y[sel], beam)[:beam]]
                        for i in sel:
                            yy = int(Y[i])
                            ci = int(cc)
                            if best_cost.get(yy, 1 << 30) <= ci:
                                continue
                            best_cost[yy] = ci
                            parent[yy] = (x, _step(int(K[i]), int(M[i]), int(Rr[i])))
                            levels[ci][yy] = None
            # squares
            if x >= 4:
                yq = math.isqrt(x)
                for yy, cc, st in _square_moves(x, yq, c, cat, lim):
                    tcy = cat.cost_of(yy)
                    if tcy >= 0:
                        finish(cc + tcy, yy, x, st)
                    elif best_cost.get(yy, 1 << 30) > cc:
                        best_cost[yy] = cc
                        parent[yy] = (x, st)
                        levels[cc][yy] = None
        c += 1
    if found is None:
        return None
    _, y0, v, st0 = found
    steps: list[Step] = [st0]
    while v != n:
        px, st = parent[v]
        steps.append(st)
        v = px
    return (y0, steps)


def _step(kind: int, m: int, r: int) -> Step:
    return (("*", "*+", "*-", "+", "r-")[kind], m, r)


def _square_moves(x: int, y: int, c: int, cat: Catalog, lim: int) -> list[tuple[int, int, Step]]:
    out: list[tuple[int, int, Step]] = []
    r = x - y * y
    if r == 0:
        out.append((y, c + 1, ("sq", 0, 0)))
        return out
    rc = cat.cost_of(r)
    if rc >= 0:
        out.append((y, c + 2 + rc, ("sq+", 0, r)))
    s = (y + 1) * (y + 1) - x
    sc = cat.cost_of(s)
    if sc >= 0 and (y + 1) * (y + 1) <= lim:
        out.append((y + 1, c + 2 + sc, ("sq-", 0, s)))
    return out


# ---------------------------------------------------------------- program builder
class Machine:
    """Registers with least-recently-used eviction."""

    def __init__(self, p: Puzzle, pin_starts: bool):
        self.n: int = p["regs"]
        self.lim: int = p["limit"]
        self.reg: list[Optional[int]] = [None] * self.n
        self.where: dict[int, int] = {}
        self.stamp = [0] * self.n
        self.clock = 0
        self.prog: Program = []
        self.seen: set[int] = set(p["start"])
        self.pinned: set[int] = set()
        for r, v in enumerate(p["start"]):
            self.reg[r] = v
            self.where.setdefault(v, r)
            if pin_starts:
                self.pinned.add(r)

    def touch(self, r: int) -> None:
        self.clock += 1
        self.stamp[r] = self.clock

    def alloc(self, protect: set[int]) -> int:
        for r in range(self.n):
            if self.reg[r] is None:
                return r
        cands = [r for r in range(self.n) if r not in self.pinned and self.reg[r] not in protect]
        if not cands:
            raise IndexError("no free register")
        return min(cands, key=lambda r: self.stamp[r])

    def emit(self, op: str, a: int, b: int, protect: set[int]) -> int:
        """Compute value(a) op value(b) into a register; returns the new value."""
        ra, rb = self.where[a], self.where[b]
        v = apply_op(op, a, b)
        if v in self.where:                       # already held: nothing to do
            self.touch(self.where[v])
            return v
        d = self.alloc(protect | {a, b})
        old = self.reg[d]
        if old is not None and self.where.get(old) == d:
            del self.where[old]
            for r2 in range(self.n):              # another register may hold the same value
                if r2 != d and self.reg[r2] == old:
                    self.where[old] = r2
                    break
        self.prog.append([op, ra, rb, d])
        self.reg[d] = v
        self.where[v] = d
        self.touch(ra); self.touch(rb); self.touch(d)
        self.seen.add(v)
        return v

    def ensure(self, v: int, cat: Catalog, protect: set[int]) -> None:
        """Make sure value v is in a register (build it from its dictionary recipe if needed)."""
        if v in self.where:
            self.touch(self.where[v])
            return
        op, a, b = cat.recipe(v)
        self.emit(op, a, b, protect)


def build(p: Puzzle, beam: int, pin_starts: bool, table: int = 0, order: str = "asc") -> Optional[Program]:
    mach = Machine(p, pin_starts)
    lim = mach.lim
    if table:                                        # digits 1..table, kept in registers for good
        try:
            if 1 not in mach.where:
                mach.emit("/", p["start"][0], p["start"][0], set())
            for d in range(2, table + 1):
                if d not in mach.where:
                    mach.emit("+", d - 1, 1, set())
        except (IndexError, KeyError, ValueError):
            return None
        mach.pinned |= {mach.where[d] for d in range(1, table + 1) if d in mach.where}
    targets = sorted(set(p["targets"]), reverse=(order == "desc"))
    for t in targets:
        if t in mach.seen:
            continue
        held = {v: r for v, r in mach.where.items()}
        cat = Catalog(held, lim)
        res = search(t, cat, lim, beam)
        if res is None:
            return None
        y0, steps = res
        # values the chain needs: start, every m / r / a
        needed = {y0} | {m for _, m, _ in steps if m} | {r for _, _, r in steps if r}
        dict_needed = [v for v in needed if v not in held]
        recipe_operands = set()
        for v in dict_needed:
            _, a, b = cat.recipe(v)
            recipe_operands |= {a, b}
        protect = set(needed) | recipe_operands
        try:
            for v in dict_needed:
                mach.ensure(v, cat, protect)
            cur = y0
            for k, (kind, m, r) in enumerate(steps):
                live = protect | {cur}
                if kind == "*":
                    cur = mach.emit("*", cur, m, live)
                elif kind == "*+":
                    mid = mach.emit("*", cur, m, live)
                    cur = mach.emit("+", mid, r, live | {mid})
                elif kind == "*-":
                    mid = mach.emit("*", cur, m, live)
                    cur = mach.emit("-", mid, r, live | {mid})
                elif kind == "+":
                    cur = mach.emit("+", cur, m, live)
                elif kind == "r-":
                    cur = mach.emit("-", m, cur, live)
                elif kind == "sq":
                    cur = mach.emit("*", cur, cur, live)
                elif kind == "sq+":
                    mid = mach.emit("*", cur, cur, live)
                    cur = mach.emit("+", mid, r, live | {mid})
                else:  # "sq-"
                    mid = mach.emit("*", cur, cur, live)
                    cur = mach.emit("-", mid, r, live | {mid})
        except (IndexError, KeyError, ValueError):
            return None
        if cur != t or len(mach.prog) > MAX_INSTR:
            return None
    return mach.prog


# ---------------------------------------------------------------- value-graph builder (new in v4)
# Idea from the top public notebooks: think of a program as a graph of VALUES (each value has one
# recipe v = a op b) and assign registers only at the end. While planning, every value already
# computed stays usable for free, not just the 48 that would fit in registers. A Belady scheduler
# then fits the plan into 48 registers, re-computing a value only if it was evicted and is needed
# again. The scheduler below (dag_prune / dag_schedule / compile_plan) is a port of the one in
# Roman Rozen's "Santa 2026 | IceFire Strong Start" notebook (Apache 2.0).
Recipe = tuple[str, int, int]
INF = 1 << 60


def dag_prune(starts: list[int], targets: list[int], rec: dict[int, Recipe]) -> dict[int, Recipe]:
    """Keep only the values some target depends on."""
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
    """Emit a value plan into the 48 registers (Belady eviction, re-computation if needed)."""
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


def build_dag(p: Puzzle, beam: int, table: int = 0, gen_size: int = 46, window: int = 0) -> Optional[Program]:
    """Plan every target as values in a graph, then compile into registers.
    window > 0: the planner may only use the `window` most recently used values (plus starts and the
    digit table) for free; this keeps the plan close to what fits in 48 registers."""
    starts: list[int] = p["start"]
    lim: int = p["limit"]
    rec: dict[int, Recipe] = {}
    avail: dict[int, int] = {v: 0 for v in starts}
    order: list[int] = []
    last_use: dict[int, int] = {v: 0 for v in starts}
    clock = 0
    pinned: set[int] = set(starts)

    def touch(v: int) -> None:
        nonlocal clock
        clock += 1
        last_use[v] = clock

    def add(v: int, op: str, a: int, b: int) -> int:
        if v not in avail:
            if not 1 <= v <= lim:
                raise ValueError("value out of range")
            avail[v] = 0
            rec[v] = (op, a, b)
            order.append(v)
        touch(v); touch(a); touch(b)
        return v

    try:
        if table:
            s0 = starts[0]
            if 1 not in avail:
                add(1, "/", s0, s0)
            for d in range(2, table + 1):
                if d not in avail:
                    add(d, "+", d - 1, 1)
            pinned |= set(range(1, table + 1))
        for t in sorted(set(p["targets"])):
            if t in avail:
                continue
            by_recency = sorted(last_use, key=lambda v: -last_use[v])
            recent = by_recency[:gen_size]
            free = avail if not window else {v: 0 for v in pinned | set(by_recency[:window])}
            cat = Catalog(free, lim, list(pinned | set(recent)))
            res = search(t, cat, lim, beam)
            if res is None:
                return None
            y0, steps = res

            def ensure(v: int) -> None:
                if v not in avail and v not in free:
                    op, a, b = cat.recipe(v)
                    add(v, op, a, b)
                else:
                    touch(v)

            ensure(y0)
            for _, m, r in steps:
                if m:
                    ensure(m)
                if r:
                    ensure(r)
            cur = y0
            for kind, m, r in steps:
                if kind == "*":
                    cur = add(cur * m, "*", cur, m)
                elif kind == "*+":
                    mid = add(cur * m, "*", cur, m)
                    cur = add(mid + r, "+", mid, r)
                elif kind == "*-":
                    mid = add(cur * m, "*", cur, m)
                    cur = add(mid - r, "-", mid, r)
                elif kind == "+":
                    cur = add(cur + m, "+", cur, m)
                elif kind == "r-":
                    cur = add(m - cur, "-", m, cur)
                elif kind == "sq":
                    cur = add(cur * cur, "*", cur, cur)
                elif kind == "sq+":
                    mid = add(cur * cur, "*", cur, cur)
                    cur = add(mid + r, "+", mid, r)
                else:
                    mid = add(cur * cur, "*", cur, cur)
                    cur = add(mid - r, "-", mid, r)
            if cur != t:
                return None
    except (ValueError, KeyError, IndexError):
        return None
    hint_pos = {v: k for k, v in enumerate(order)}
    best: Optional[Program] = None
    for pull in (False, True):
        try:
            prog = compile_plan(p, rec, hint_pos, pull)
        except (ValueError, KeyError, RecursionError):
            continue
        if best is None or len(prog) < len(best):
            best = prog
    return best


def solve(p: Puzzle) -> Optional[Program]:
    best: Optional[Program] = None
    for st in SETTINGS:
        prog = build_dag(p, *st[1:]) if st[0] == "dag" else build(p, *st[1:])
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
        for k, (pid, prog) in enumerate(pool.imap_unordered(_solve_task, puzzles, chunksize=1)):
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
    files = [f for f in sorted(Path(".").glob("submission*.csv")) if f != OUT_PATH]
    if BANK_DIR.is_dir():
        files += sorted(BANK_DIR.rglob("*.csv"))
    earlier: list[dict[str, Program]] = []
    for f in files:
        try:
            with open(f, newline="", encoding="utf-8-sig") as fh:
                if next(csv.reader(fh), None) != ["id", "program"]:
                    continue
            earlier.append(load_programs(f))
        except (OSError, UnicodeDecodeError, csv.Error, json.JSONDecodeError, KeyError):
            continue
    print(f"merging with {len(earlier)} earlier submission file(s) (./submission*.csv and {BANK_DIR}/)")
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