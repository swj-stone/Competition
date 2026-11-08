"""Plan-slot ("last step") run features.

Hidden generator structure exploited:
  * Each run has a hidden execution plan of L steps.  Step k is owned by worker
    W[k % |W|] (W = a2..aN, plus a1 appended for pipeline topologies).  Final
    artifacts (art_01, art_02, ...) are listed in PLAN-STEP order, not in time order.
  * A fault instance occupies one plan step (its "slot"): conflict -> the setter's
    step, deadlock -> the first waiter's step, loop/drift -> the first sender's
    step, dropped -> the dropped step, dup -> the step of the duplicated subtask.
  * The labelled fault is never placed on the LAST plan step; an unlabelled
    secondary anomaly can be.

For each fault class the top-scoring stage-1 candidate of the run is mapped to
its plan slot, giving: offset from the last slot, last-slot flag, absolute slot
index, "inside the aligned artifact span" flag and the number of subtasks
assigned to the slot agent; plus run-level plan length / gap counts.

Port of work/agents/streams/slotfeat.py::slot_frame(ext=True) ("slot2", 34 columns).
Depends only on numpy / pandas / stdlib and parse.py.

    features(runs, C, thr=0.5) -> DataFrame (index 0..len(runs)-1, 34 float columns)
    features_run(run, top, thr=0.5) -> dict  (top = {cls: (t, score)} of one run)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from parse import ASSIGN, DRIFT_HARD, DRIFT_SOFT, ITER

FAULTS = ["dropped_handoff", "duplicated_work", "deadlock", "conflict", "goal_drift",
          "runaway_loop"]
IN_PLAN = {"dropped_handoff", "conflict", "duplicated_work"}

COLUMNS: List[str] = []
for _c in FAULTS:
    COLUMNS += [f"slot_off_{_c}", f"slot_last_{_c}"]
COLUMNS += ["slot_L", "slot_npl", "slot_n_internal_gaps", "slot_last_aligned"]
for _c in FAULTS:
    COLUMNS += [f"slot_k_{_c}", f"slot_internal_{_c}", f"slot_Xnasg_{_c}"]
BASE_COLUMNS = [c for c in COLUMNS if c.startswith(("slot_off_", "slot_last_"))
                and c != "slot_last_aligned"] + ["slot_L", "slot_npl"]


# ----------------------------------------------------------------- helpers
def _tgt(m):
    return m.get("tagent") or (m.get("to") if m.get("to") not in ("user", "all", None) else "")


def _senders(ms, t, cls):
    m = ms[t]
    if cls == "runaway_loop":
        L = [x for x in ms if x["cat"] == ITER and x["sub"] == m["sub"]]
    else:
        L = [x for x in ms if x["cat"] in (DRIFT_SOFT, DRIFT_HARD)
             and x.get("intent") not in ("", "status", None)]
    L = L or [m]
    return list(dict.fromkeys(x["from"] for x in L))


def _slot_agent(run, t, cls):
    ms = run["messages"]
    m = ms[t]
    if cls in ("dropped_handoff", "duplicated_work"):
        return _tgt(m)
    if cls == "conflict":
        state = run.get("shared_state") or []
        st = [s for s in state if s.get("t") == t]
        if not st:
            return m["from"]
        k = st[0]["key"]
        blk = sorted([s for s in state if s.get("key") == k], key=lambda s: s.get("t", 0))
        return blk[0]["agent"]
    if cls == "deadlock":
        return m["from"]
    return _senders(ms, t, cls)[0]


def _next_slot(W, x, start):
    """Smallest k >= start with W[k % |W|] == x, or None.  Bounded to one cycle so that
    ids equal by identity but not by value (e.g. json NaN) can never loop forever."""
    n = len(W)
    for kk in range(start, start + n):
        if W[kk % n] == x:
            return kk
    return None


def _worker_cycle(run):
    agents = run.get("agents") or []
    ag = [a["id"] for a in agents if isinstance(a, dict) and "id" in a]
    W = ag[1:]
    topo = run.get("topology")
    if isinstance(topo, dict) and topo.get("type") == "pipeline" and ag:
        W = W + [ag[0]]
    return W


def _align(run, W):
    arts = [a for a in (run.get("artifacts") or []) if isinstance(a, dict)
            and a.get("status") == "final" and a.get("subtask") != "final_delivery"]
    seen = set()
    seq = []
    for a in arts:
        s = a.get("subtask")
        if s and s != "off_topic" and s in seen:
            continue
        if s:
            seen.add(s)
        seq.append(a)
    steps = []
    k = 0
    n = len(W)
    for a in seq:
        x = a.get("by")
        if x not in W:
            continue
        kk = _next_slot(W, x, k)
        if kk is None:
            continue
        for j in range(k, kk):
            steps.append(dict(k=j, agent=W[j % n], art=None))
        steps.append(dict(k=kk, agent=x, art=a))
        k = kk + 1
    return steps


def _inst_step(run, t, cls, steps, W):
    m = run["messages"][t]
    S = m["sub"]
    if cls == "duplicated_work":
        for st in steps:
            if st["art"] is not None and st["art"].get("subtask") == S:
                return st["k"]
        return None
    if cls == "goal_drift":
        for st in steps:
            if st["art"] is not None and st["art"].get("subtask") == "off_topic":
                return st["k"]
    X = _slot_agent(run, t, cls)
    if not X or X not in W:
        return None
    gaps = [st["k"] for st in steps if st["art"] is None and st["agent"] == X]
    if gaps:
        return gaps[0]
    k = (steps[-1]["k"] + 1) if steps else 0
    return _next_slot(W, X, k)


def _as_t(t) -> Optional[int]:
    if t is None:
        return None
    try:
        f = float(t)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return int(f)


def _empty_row() -> Dict[str, float]:
    return {c: np.nan for c in COLUMNS}


# ----------------------------------------------------------------- per run
def features_run(run: Dict[str, Any], top: Dict[str, Tuple[Any, float]],
                 thr: float = 0.5) -> Dict[str, float]:
    """top: {cls: (t, score)} = the run's top-scoring stage-1 candidate per class
    (missing class -> no candidate)."""
    try:
        return _features_run(run, top or {}, thr)
    except Exception:
        return _empty_row()


def _features_run(run, top, thr):
    ms = run.get("messages") or []
    nmsg = len(ms)
    try:
        W = _worker_cycle(run)
    except Exception:
        W = []
    try:
        steps = _align(run, W)
    except Exception:
        steps = []
    npl = len({m["sub"] for m in ms if m.get("cat") == ASSIGN and m.get("sub")})

    cand = {}
    for c in FAULTS:
        v = top.get(c)
        if v is None:
            cand[c] = (None, False)
            continue
        try:
            t, s = v
        except (TypeError, ValueError):
            cand[c] = (None, False)
            continue
        try:
            pres = bool(float(s) > thr)
        except (TypeError, ValueError):
            pres = False
        cand[c] = (_as_t(t), pres)
    present = {c: cand[c][1] for c in FAULTS}
    nextra = sum(1 for c in FAULTS if c not in IN_PLAN and present[c])
    L = npl + nextra if any(present[c] for c in IN_PLAN) else npl + max(1, nextra)
    lastk = steps[-1]["k"] if steps else -1

    d = {}
    ext = {}
    for c in FAULTS:
        t = cand[c][0]
        k = None
        X = None
        if t is not None and 0 <= t < nmsg:
            try:
                k = _inst_step(run, t, c, steps, W)
                X = _slot_agent(run, t, c)
            except Exception:
                pass
        if k is None:
            d[f"slot_off_{c}"] = np.nan
            d[f"slot_last_{c}"] = np.nan
        else:
            off = k - (L - 1)
            d[f"slot_off_{c}"] = float(np.clip(off, -5, 3))
            d[f"slot_last_{c}"] = float(off == 0)
        ext[f"slot_k_{c}"] = np.nan if k is None else float(k)
        ext[f"slot_internal_{c}"] = np.nan if k is None else float(k <= lastk)
        if X is None:
            ext[f"slot_Xnasg_{c}"] = np.nan
        else:
            try:
                ext[f"slot_Xnasg_{c}"] = float(len({m["sub"] for m in ms if m.get("cat") == ASSIGN
                                                    and m.get("sub") and _tgt(m) == X}))
            except Exception:
                ext[f"slot_Xnasg_{c}"] = np.nan
    d["slot_L"] = float(L)
    d["slot_npl"] = float(npl)
    d["slot_n_internal_gaps"] = float(sum(1 for s in steps if s["art"] is None))
    d["slot_last_aligned"] = float(lastk)
    d.update(ext)
    return d


# ----------------------------------------------------------------- frame
def top_candidates(C: Dict[str, pd.DataFrame]) -> Dict[str, Dict[int, Tuple[int, float]]]:
    """{cls: {run: (t, score)}} of each run's highest-scoring candidate (first on ties)."""
    top: Dict[str, Dict[int, Tuple[int, float]]] = {}
    for c in FAULTS:
        D = (C or {}).get(c)
        if D is None or len(D) == 0 or not {"run", "t", "score"} <= set(D.columns):
            top[c] = {}
            continue
        D = D[["run", "t", "score"]].reset_index(drop=True)
        D = D.assign(run=pd.to_numeric(D["run"], errors="coerce"),
                     score=pd.to_numeric(D["score"], errors="coerce"))
        D = D[D["score"].notna() & np.isfinite(D["run"].astype(float))]
        if len(D) == 0:
            top[c] = {}
            continue
        D = D.loc[D.groupby("run", sort=True)["score"].idxmax()]
        top[c] = {int(r): (t, float(s)) for r, t, s in zip(D["run"].values, D["t"].values,
                                                            D["score"].values)}
    return top


def features(runs: List[Dict[str, Any]], C: Dict[str, pd.DataFrame],
             thr: float = 0.5) -> pd.DataFrame:
    top = top_candidates(C)
    rows = []
    for i, run in enumerate(runs):
        tp = {c: top[c][i] for c in FAULTS if i in top[c]}
        rows.append(features_run(run, tp, thr))
    return pd.DataFrame(rows, index=pd.RangeIndex(len(runs)), columns=COLUMNS, dtype=float)


def slot_frame(parsed_runs, C, thr=0.5, ext=True):
    """Research-compatible alias (ext=False -> the 14 base columns)."""
    F = features(parsed_runs, C, thr)
    return F if ext else F[BASE_COLUMNS]
