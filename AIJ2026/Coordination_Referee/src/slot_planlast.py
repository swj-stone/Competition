"""Hidden-plan "last slot" features (plan reconstructions v4 and v6).

Generator structure (recovered on train): each run executes a hidden plan of K
slots owned round-robin by ``cyc`` = [a2..aN] (or [a2..aN, a1]).  Artifacts are
listed in plan order; every fault instance occupies one slot; the LABELLED
fault is never on the last slot K-1, while an unlabelled secondary anomaly can
be.  For every fault class we take the best stage-1 candidate of the run and
ask whether, given the plan reconstruction, that instance must sit on the last
slot (=> it is probably the secondary anomaly).

* v4: greedy alignment of the de-duplicated artifact deliverers on the cycle;
  plan = assigned in-domain subtasks + fault-instance subtasks.
  ``l4c_<cls>`` = all candidate slots of the instance are K-1,
  ``l4p_<cls>`` = K-1 is one of its candidate slots.
* v6: plan-owner based artifact selection, owner-set alignment and an exact
  count over the feasible assignments of undelivered steps to free slots.
  ``l6_<cls>`` = fraction of feasible assignments putting the instance at K-1.
  Every step has exactly one owner, so the feasible assignments factorise over
  owners and the count is done in closed form: no enumeration, O(#steps^2) per
  run.  The research code stopped a DFS after ``max_perm + 1`` = 50001
  assignments; that cap is kept (the prefix of the DFS order is counted
  analytically, never enumerated), so the per-run cost is bounded whatever the
  number of combinations.
* pair level: ``*_top1_last`` / ``*_top2_last`` = value for the two classes with
  the highest stage-1 max score, ``*_diff`` = top1 - top2 (NaN -> 0),
  ``l46_both_top1/2`` = v4 certain AND v6 == 1 for the top1 / top2 class.

Port of work/agents/microstructure/{plan3,plan4,plan5,plan6,slots,runfeat_v46,
ev_v46}.py.  Uses only the parsed run and the stage-1 candidates (run, t,
score) - no labels.

    features(runs, C) -> DataFrame (index 0..len(runs)-1)
    top_candidates(C, n_runs) -> list[dict cls -> (t, max_score)]
    features_run(run, top) -> dict      (per-run, for parallel callers)
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from parse import ACK, ASSIGN, DELIVER, SUB2DOM

FAULTS = ["dropped_handoff", "duplicated_work", "deadlock", "conflict", "goal_drift",
          "runaway_loop"]
MAX_PERM = 50000          # research DFS cap (counts the first MAX_PERM + 1 assignments)
NAN = float("nan")

COLUMNS_V4 = []
for _c in FAULTS:
    COLUMNS_V4 += ["l4c_" + _c, "l4p_" + _c]
COLUMNS_V4 += ["l4c_top1_last", "l4c_top2_last", "l4c_diff"]
COLUMNS_V6 = ["l6_" + _c for _c in FAULTS] + ["l6_top1_last", "l6_top2_last", "l6_diff"]
COLUMNS = COLUMNS_V4 + COLUMNS_V6 + ["l46_both_top1", "l46_both_top2"]


# ----------------------------------------------------------------- run helpers
def _msgs(run):
    return run.get("messages") or []


def _is_int(x) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


def _msg(run, t):
    ms = _msgs(run)
    if _is_int(t) and 0 <= t < len(ms):
        return ms[t]
    return None


def _agent_ids(run) -> List[Any]:
    return [a.get("id") for a in (run.get("agents") or []) if isinstance(a, dict)]


def _arts(run) -> List[Dict]:
    return [a for a in (run.get("artifacts") or []) if isinstance(a, dict)]


def _run_domain(run) -> str:
    c = Counter(SUB2DOM[m.get("sub")] for m in _msgs(run)
                if m.get("sub") in SUB2DOM and m.get("cat") in (ASSIGN, DELIVER, ACK))
    c.update(SUB2DOM[a.get("subtask")] for a in _arts(run) if a.get("subtask") in SUB2DOM)
    return c.most_common(1)[0][0] if c else run.get("domain", "unknown")


def _fault_owner(run, t, cls):
    m = _msg(run, t)
    if m is None:
        return None
    if cls in ("dropped_handoff", "conflict"):
        return m.get("to")
    return m.get("from")


def _inst_sub(run, t, cls) -> str:
    m = _msg(run, t)
    if m is None:
        return ""
    if cls in ("deadlock", "runaway_loop", "dropped_handoff", "duplicated_work"):
        return m.get("sub", "")
    if cls == "conflict":
        state = [s for s in (run.get("shared_state") or []) if isinstance(s, dict)]
        st = [s for s in state if s.get("t") == t]
        if st:
            k = st[0].get("key")
            first = [s for s in state if s.get("key") == k][0]
            ft = first.get("t")
            fm = _msg(run, ft) if _is_int(ft) and ft >= 0 else None
            if fm is not None:
                return fm.get("sub", "")
        return m.get("sub", "")
    return ""


# ------------------------------------------------------------------- plan v4
def _align(seq, cyc):
    L = len(cyc)
    p = 0
    slots = []
    skips = 0
    for b in seq:
        if b not in cyc:
            return None, 999
        k = 0
        while cyc[(p + k) % L] != b:
            k += 1
        skips += k
        slots.append(p + k)
        p = p + k + 1
    return slots, skips


def _v4_base(run, ags, dom):
    """Everything of the v4 reconstruction that does not depend on the fault instances."""
    seen = set()
    seq = []
    for a in _arts(run):
        sub = a.get("subtask")
        if a.get("status") == "copy" or sub == "final_delivery":
            continue
        if sub == "off_topic":
            seq.append(a)
            continue
        if sub in seen:
            continue
        seen.add(sub)
        seq.append(a)
    B = [a.get("by") for a in seq]
    if not B:
        return None
    best = None
    for cyc in (ags[1:], ags[1:] + ags[:1]):
        s, k = _align(B, cyc)
        if best is None or k < best[1]:
            best = (s, k, cyc)
    s, k, cyc = best
    if s is None:
        return None
    delivered = {a.get("subtask") for a in seq}
    plan = []
    for m in _msgs(run):
        x = m.get("sub")
        if m.get("cat") == ASSIGN and x and SUB2DOM.get(x) == dom and x not in plan:
            plan.append(x)
    last = max(s)
    ss = set(s)
    return dict(cyc=cyc, L=len(cyc), slots=s, seq=seq, delivered=delivered, plan=plan,
                interior=[p for p in range(last + 1) if p not in ss], last=last)


def _v4_K(B4, extra_subs):
    plan = list(B4["plan"])
    for x in extra_subs:
        if x and x not in plan:
            plan.append(x)
    U = [x for x in plan if x not in B4["delivered"]]
    return len(B4["seq"]) + len(U)


def _v4_last(run, t, cls, B4, K):
    """(certain, possible) that instance (t, cls) sits on the last plan slot (v4)."""
    if B4 is None:
        return NAN, NAN
    if cls == "goal_drift":
        idx = [n for n, a in enumerate(B4["seq"]) if a.get("subtask") == "off_topic"]
        c = [B4["slots"][idx[0]]] if idx else []
    elif cls == "duplicated_work":
        m = _msg(run, t)
        S = m.get("sub") if m is not None else None
        idx = [n for n, a in enumerate(B4["seq"]) if a.get("subtask") == S]
        c = [B4["slots"][idx[0]]] if idx else []
    else:
        o = _fault_owner(run, t, cls)
        cyc, L = B4["cyc"], B4["L"]
        c = [p for p in B4["interior"] if cyc[p % L] == o]
        c += [p for p in range(B4["last"] + 1, K) if cyc[p % L] == o]
    if not c:
        return NAN, NAN
    return float(all(x == K - 1 for x in c)), float(K - 1 in c)


# ------------------------------------------------------------------- plan v6
def _plan_owner_map(run, ags) -> Dict[str, List[Any]]:
    """subtask -> assignees, orchestrator (a1) assignments first, then by time.
    Keys in first-assignment order (research iterated a set here)."""
    a1 = ags[0] if ags else "a1"
    orch, other = {}, {}
    for m in _msgs(run):
        if m.get("cat") == ASSIGN and m.get("sub"):
            d = orch if m.get("from") == a1 else other
            lst = d.setdefault(m["sub"], [])
            if m.get("to") not in lst:
                lst.append(m.get("to"))
    out = {}
    for s in list(orch) + [x for x in other if x not in orch]:
        o = orch.get(s, [])
        out[s] = o + [x for x in other.get(s, []) if x not in o]
    return out


def _v6_base(run, ags, dom, own):
    ms = _msgs(run)
    arts = []
    for j, a in enumerate(_arts(run)):
        if a.get("status") == "copy" or a.get("subtask") == "final_delivery":
            continue
        s = a.get("subtask") or ""
        at = a.get("t")
        if not s and _is_int(at) and at < len(ms):
            s = ms[at].get("sub", "") if -len(ms) <= at else ""
        if a.get("subtask") == "off_topic" or (s and SUB2DOM.get(s) not in (None, dom)):
            s = "off_topic#%d" % j
        arts.append(dict(j=j, sub=s, by=a.get("by")))
    chosen = {}
    for e in arts:
        s = e["sub"]
        if not s or s.startswith("off_topic"):
            continue
        po = own.get(s, [])
        if s not in chosen:
            chosen[s] = e
        elif po and chosen[s]["by"] != po[0] and e["by"] == po[0]:
            chosen[s] = e
    seq = [e for e in arts if (not e["sub"]) or e["sub"].startswith("off_topic")
           or chosen.get(e["sub"]) is e]
    for e in seq:
        e["owners"] = {e["by"]} | ({own[e["sub"]][0]} if own.get(e["sub"]) else set())
    if not seq:
        return None
    best = None
    for cyc in (ags[1:], ags[1:] + ags[:1]):
        L = len(cyc)
        p = 0
        slots = []
        skips = 0
        ok = True
        for e in seq:
            k = 0
            while k < 2 * L and cyc[(p + k) % L] not in e["owners"]:
                k += 1
            if k >= 2 * L:
                ok = False
                break
            skips += k
            slots.append(p + k)
            p = p + k + 1
        if ok and (best is None or skips < best[1]):
            best = (slots, skips, cyc)
    if best is None:
        return None
    slots, skips, cyc = best
    last = max(slots)
    ss = set(slots)
    return dict(cyc=cyc, L=len(cyc), seq=seq, slots=slots, last=last,
                delivered={e["sub"] for e in seq},
                interior=[p for p in range(last + 1) if p not in ss])


def _v6_steps(run, B6, dom, own, faults):
    fsubs = {}
    for (t, c) in faults:
        if c in ("goal_drift", "duplicated_work"):
            continue
        fsubs[(t, c)] = _inst_sub(run, t, c)
    fset = set(fsubs.values())
    steps = []
    for s, lst in own.items():
        if SUB2DOM.get(s) != dom or s in B6["delivered"] or s in fset:
            continue
        steps.append((s, lst[0]))
    for (t, c) in fsubs:
        steps.append((("F", t, c), _fault_owner(run, t, c)))
    return steps


def _v6_last_fracs(B6, steps, max_perm=MAX_PERM):
    """Fraction of feasible step->free-slot bijections that put each step on slot K-1.

    Equivalent to the research DFS (plan5.enumerate_last) which counts the first
    ``max_perm + 1`` leaves of a lexicographic DFS (steps ordered by number of
    candidate slots, slots ascending).  Each step has exactly one owner, so a
    step's candidates are the free slots of its owner; the tree is walked
    analytically: sibling subtrees all have the same leaf count and their stats
    are closed-form, only one path is descended.  Returns (dict or None, n)."""
    n_st = len(steps)
    n_tr = n_st - len(B6["interior"])
    if n_tr < 0:
        return None, 0
    K = B6["last"] + 1 + n_tr
    cyc, L = B6["cyc"], B6["L"]
    free = B6["interior"] + list(range(B6["last"] + 1, K))
    if n_st == 0:
        return {}, 1
    slot_owner = {p: cyc[p % L] for p in free}
    by_owner: Dict[Any, List[int]] = {}
    for p in free:
        by_owner.setdefault(slot_owner[p], []).append(p)
    so = [o for _, o in steps]
    if Counter(so) != Counter(slot_owner.values()):
        return None, 0                   # no feasible bijection (research: nfeas == 0)
    cand_len = [len(by_owner[o]) for o in so]
    order = sorted(range(n_st), key=lambda i: cand_len[i])
    total = 1
    for o, ps in by_owner.items():
        total *= math.factorial(len(ps))
    budget = total if total <= max_perm else max_perm + 1
    last_slot = K - 1
    last_free = last_slot in slot_owner  # False when K-1 is a delivered slot (n_tr == 0)
    o_last = slot_owner.get(last_slot)
    cnt = [0] * n_st
    rem = {o: len(ps) for o, ps in by_owner.items()}   # unassigned steps per owner
    used = set()
    assigned: List[Tuple[int, int]] = []

    def sub_size():
        s = 1
        for r in rem.values():
            s *= math.factorial(r)
        return s

    def add_full(S, k_next):
        # stats of a complete subtree of S leaves below the current partial assignment
        for i, p in assigned:
            if p == last_slot:
                cnt[i] += S
        if last_free and last_slot not in used:
            r = rem[o_last]
            share = S // r
            for kk in range(k_next, n_st):
                i = order[kk]
                if so[i] == o_last:
                    cnt[i] += share

    k = 0
    b = budget
    while True:
        if k == n_st:                    # single leaf (b == 1)
            add_full(1, n_st)
            break
        i = order[k]
        o = so[i]
        rem[o] -= 1
        S = sub_size()                   # leaves below each child at depth k
        descend = None
        for p in by_owner[o]:
            if p in used:
                continue
            if b <= 0:
                break
            if b >= S:
                used.add(p)
                assigned.append((i, p))
                add_full(S, k + 1)
                assigned.pop()
                used.discard(p)
                b -= S
            else:
                descend = p
                break
        if descend is None or b <= 0:
            break
        used.add(descend)
        assigned.append((i, descend))
        k += 1
    nfeas = budget
    return {steps[i][0]: cnt[i] / nfeas for i in range(n_st)}, nfeas


def _v6_plast(run, t, cls, B6, K, E):
    if B6 is None:
        return NAN
    if cls == "goal_drift":
        idx = [n for n, e in enumerate(B6["seq"]) if e["sub"].startswith("off_topic")]
        return float(B6["slots"][idx[0]] == K - 1) if idx else NAN
    if cls == "duplicated_work":
        m = _msg(run, t)
        S = m.get("sub") if m is not None else None
        idx = [n for n, e in enumerate(B6["seq"]) if e["sub"] == S]
        return float(B6["slots"][idx[0]] == K - 1) if idx else NAN
    if E is None:
        return NAN
    v = E.get(("F", t, cls))
    return NAN if v is None else float(v)


# ------------------------------------------------------------------ features
def _empty_row() -> Dict[str, float]:
    d = {c: NAN for c in COLUMNS}
    d["l4c_diff"] = 0.0
    d["l6_diff"] = 0.0
    d["l46_both_top1"] = 0.0
    d["l46_both_top2"] = 0.0
    return d


def _nz(x):
    return x if x == x else 0.0


def _features_run(run, top) -> Dict[str, float]:
    out = _empty_row()
    ms = _msgs(run)
    top = {c: v for c, v in top.items()
           if c in FAULTS and _is_int(v[0]) and 0 <= v[0] < len(ms)}
    M = np.array([float(top[c][1]) if c in top else 0.0 for c in FAULTS])
    o = np.argsort(-M, kind="stable")
    a, b = FAULTS[o[0]], FAULTS[o[1]]
    base = [(int(top[x][0]), x) for x in (a, b) if x in top]
    if top:
        ags = _agent_ids(run)
        dom = _run_domain(run)
        own = _plan_owner_map(run, ags)
        B4 = _v4_base(run, ags, dom)
        B6 = _v6_base(run, ags, dom, own)
        cache = {}
        for c in FAULTS:
            if c not in top:
                continue
            t = int(top[c][0])
            fl = base + ([(t, c)] if c not in (a, b) else [])
            key = tuple(fl)
            if key not in cache:
                K4 = _v4_K(B4, [_inst_sub(run, tt, cc) for tt, cc in fl]) if B4 else None
                if B6 is not None:
                    steps = _v6_steps(run, B6, dom, own, fl)
                    K6 = B6["last"] + 1 + max(len(steps) - len(B6["interior"]), 0)
                    E, _ = _v6_last_fracs(B6, steps)
                else:
                    K6, E = None, None
                cache[key] = (K4, K6, E)
            K4, K6, E = cache[key]
            out["l4c_" + c], out["l4p_" + c] = _v4_last(run, t, c, B4, K4)
            out["l6_" + c] = _v6_plast(run, t, c, B6, K6, E)
    for pre in ("l4c_", "l6_"):
        va = out.get(pre + a, NAN) if a in top else NAN
        vb = out.get(pre + b, NAN) if b in top else NAN
        out[pre + "top1_last"] = va
        out[pre + "top2_last"] = vb
        out[pre + "diff"] = _nz(va) - _nz(vb)
    out["l46_both_top1"] = float(out["l4c_top1_last"] == 1 and out["l6_top1_last"] == 1)
    out["l46_both_top2"] = float(out["l4c_top2_last"] == 1 and out["l6_top2_last"] == 1)
    return out


def features_run(run, top) -> Dict[str, float]:
    """Features of one parsed run.  ``top`` = {cls: (t, max_score)} of the best stage-1
    candidate per class (see ``top_candidates``); a value may also be a DataFrame of this
    run's candidate rows for that class (columns t, score)."""
    try:
        tt = {}
        for c, v in (top or {}).items():
            if isinstance(v, pd.DataFrame):
                v = v[v["score"].notna()]
                if len(v) == 0:
                    continue
                v = v.reset_index(drop=True)
                j = v["score"].idxmax()
                v = (v["t"].iloc[j], v["score"].iloc[j])
            tt[c] = (int(v[0]) if _is_int(v[0]) or float(v[0]).is_integer() else v[0], float(v[1]))
        return _features_run(run, tt)
    except Exception:                    # never crash on an odd run
        return _empty_row()


def top_candidates(C, n_runs: int) -> List[Dict[str, Tuple[int, float]]]:
    """Per run: {cls: (t of the highest-scoring candidate, max score)} (first row on ties)."""
    out: List[Dict[str, Tuple[int, float]]] = [dict() for _ in range(n_runs)]
    for c in FAULTS:
        D = (C or {}).get(c)
        if D is None or len(D) == 0 or not {"run", "t", "score"} <= set(D.columns):
            continue
        D = D[["run", "t", "score"]].reset_index(drop=True)
        D = D[D["score"].notna() & D["run"].notna() & D["t"].notna()]
        D = D[(D["run"] >= 0) & (D["run"] < n_runs)]
        if len(D) == 0:
            continue
        Dm = D.loc[D.groupby("run")["score"].idxmax()]
        for r, t, s in zip(Dm["run"].values, Dm["t"].values, Dm["score"].values):
            out[int(r)][c] = (int(t), float(s))
    return out


def features(runs, C) -> pd.DataFrame:
    n = len(runs)
    top = top_candidates(C, n)
    rows = [features_run(runs[i], top[i]) for i in range(n)]
    return pd.DataFrame(rows, index=range(n), columns=COLUMNS).astype(float)
