"""Last-plan-slot run features (port of work/agents/artifacts_state_ids: lastslot_final.py + runfeats_top2.py).

Generator structure: each run has a hidden plan of N subtask slots; slot k is owned by ``cyc[k % W]`` with
``cyc = agents[1:]`` (C1) or ``agents[1:] + [a1]`` (C2).  Plan deliveries get artifact ids in slot order.  Every fault
instance occupies one slot:

    conflict        -> slot of the agent that made the first 'set' of the conflicting key
    dropped_handoff -> slot of the assignee of the dropped handoff
    duplicated_work -> slot of the original delivery
    goal_drift      -> slot of the off_topic artifact (else sender of the drift message)
    runaway_loop    -> slot of the sender of the first loop message (phantom slot unless the loop subtask is its own)
    deadlock        -> slot of the sender of the first wait message (phantom slot)

The LABELLED fault is never on the last slot, while an unlabelled secondary anomaly can be.  For the top stage-1
candidate of every class the plan is reconstructed jointly with the other strong candidates (score > 0.3) and the
position of the candidate's slot is reported:

    lj_{c}_{last,after,cost,N}   greedy reconstruction
    l5_{c}_{last,after,cost,N}   DP reconstruction
    lj2_{c}_{last,score}         greedy, for the SECOND best candidate of class c
    n_strong, n_strong_last, max_score_nonlast   joint reconstruction of all strong top candidates

API
    features(runs, C) -> DataFrame (index 0..len(runs)-1)
    features_run(run, cand_rows) -> dict     cand_rows: {class: [(t, score), ...]} (candidates of this run)
    candidates_by_run(C, n_runs) -> list of cand_rows (helper for callers that parallelise features_run)

Deterministic port: the research code iterated the plan subtasks in ``set`` order (hash-seed dependent tie breaks);
here undelivered subtasks are ordered by first assignment, which only matters for exact ties.
Depends only on numpy / pandas / stdlib / parse.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from parse import ASSIGN, RECOVER, SUB2DOM

FAULTS = ["dropped_handoff", "duplicated_work", "deadlock", "conflict", "goal_drift", "runaway_loop"]
TH = 0.3
_KEYS = ("last", "after", "cost", "N")
DP_PAR = dict(mis=2.5, skip_has=0.4, skip_none=1.5, maxskip=12)
_BIG_T = 10 ** 9

COLUMNS: List[str] = []
for _c in FAULTS:
    for _v in ("lj", "l5"):
        COLUMNS += [f"{_v}_{_c}_{k}" for k in _KEYS]
for _c in FAULTS:
    COLUMNS += [f"lj2_{_c}_last", f"lj2_{_c}_score"]
COLUMNS += ["n_strong", "n_strong_last", "max_score_nonlast"]


# --------------------------------------------------------------------------------------------- helpers
def _num_t(v):
    try:
        if v is None:
            return _BIG_T
        f = float(v)
        return v if f == f else _BIG_T
    except Exception:
        return _BIG_T


def _art_num(aid: str) -> int:
    s = aid[4:]
    return int(s) if s.isdigit() else 999


class _Run:
    """Per-run cache of everything the slot reconstruction needs."""

    def __init__(self, run: Dict[str, Any]):
        ms = [m for m in (run.get("messages") or []) if isinstance(m, dict)]
        self.ms = ms
        self.arts_all = [a for a in (run.get("artifacts") or []) if isinstance(a, dict)]
        self.state = [s for s in (run.get("shared_state") or []) if isinstance(s, dict)]
        self.ids = [a.get("id") for a in (run.get("agents") or []) if isinstance(a, dict)]
        asg: Dict[str, list] = {}
        nmsg: Counter = Counter()
        recov_t: Dict[str, list] = defaultdict(list)
        ids = self.ids
        for m in ms:
            s = m.get("sub") or ""
            if s in SUB2DOM:
                nmsg[s] += 1
            cat = m.get("cat")
            if cat == ASSIGN and s in SUB2DOM and m.get("tagent") in ids:
                asg.setdefault(s, []).append((m.get("t"), m.get("tagent")))
            if cat == RECOVER and s in SUB2DOM:
                recov_t[s].append(m.get("t"))
        # artifacts: (subtask, key, (id_num, by, t)) in list order
        arts = []
        for a in self.arts_all:
            s = a.get("subtask")
            if a.get("status") == "copy" or s == "final_delivery":
                continue
            if s and s != "off_topic" and s not in SUB2DOM:
                continue
            aid = str(a.get("id", "") or "")
            key = s if s else ("?" + aid)
            by = a.get("by")
            by = "" if by is None else by
            arts.append((s, key, (_art_num(aid), by, _num_t(a.get("t")))))
        self.asg, self.nmsg, self.recov_t, self.arts = asg, nmsg, recov_t, arts
        self._plan = {}
        self._rec = {}
        self._item = {}
        self._dpc = {}
        self._grc = {}

    # ---------------------------------------------------------------- plan items (lastslot3.plan_items)
    def plan_items(self, skip: frozenset):
        r = self._plan.get(skip)
        if r is not None:
            return r
        arts: Dict[str, list] = {}
        for s, key, tup in self.arts:
            if s in skip:
                continue
            arts.setdefault(key, []).append(tup)
        asg, recov_t = self.asg, self.recov_t
        keys = list(asg) + [k for k in arts if k not in asg]
        Q = []
        und = {}
        for s in keys:
            A = asg.get(s, [])
            D = sorted(arts.get(s, []))
            fa = A[0][1] if A else None
            plan_art = None
            if D:
                idn, by, t = D[0]
                if fa is None or by == fa:
                    plan_art = (idn, by)
                else:
                    later_asg = [x for x in A if x[1] == by]
                    rec = any(_num_t(rt) < t for rt in recov_t.get(s, []))
                    if not rec and not (later_asg and later_asg[0][0] > A[0][0] and len(A) >= 2
                                        and any(x[1] == fa for x in A[1:])):
                        plan_art = (idn, by)
                    else:
                        own_art = [x for x in D if x[1] == fa]
                        if own_art and not rec:
                            plan_art = (own_art[0][0], fa)
            if plan_art:
                Q.append((plan_art[0], s, plan_art[1]))
            elif fa is not None:
                und[s] = fa
        Q.sort(key=lambda x: (x[0], x[1], str(x[2])))
        r = ([(s, by) for _, s, by in Q], und)
        self._plan[skip] = r
        return r

    # ---------------------------------------------------------------- instance -> plan item (lastslot3.inst_item)
    def first_assignee(self, S):
        for m in self.ms:
            if m.get("cat") == ASSIGN and m.get("sub") == S and m.get("tagent"):
                return m.get("tagent")
        return None

    def conflict_owner(self, t):
        L = [s for s in self.state if s.get("key") != "run_status"]
        idx = [j for j, s in enumerate(L) if s.get("t") == t]
        if not idx:
            return None
        j = idx[0]
        k = L[j].get("key")
        while j - 1 >= 0 and L[j - 1].get("key") == k and L[j - 1].get("op") == "override":
            j -= 1
        if j - 1 >= 0 and L[j - 1].get("key") == k:
            j -= 1
        return L[j].get("agent")

    def inst_item(self, t, cls):
        ck = (t, cls)
        if ck in self._item:
            return self._item[ck]
        r = self._inst_item(t, cls)
        self._item[ck] = r
        return r

    def _inst_item(self, t, cls):
        ms = self.ms
        if t is None or t < 0 or t >= len(ms):
            return None
        m = ms[t]
        sub = m.get("sub") or ""
        frm = m.get("from")
        if cls == "conflict":
            o = self.conflict_owner(t)
            if o is None:
                return None
            S = sub or m.get("intent") or None
            S = S or "__cf"
            return (S, o), frozenset([S])
        if cls == "dropped_handoff":
            o = m.get("tagent") or None
            S = sub or "__dr"
            return ((S, o), frozenset([S])) if o else None
        if cls == "duplicated_work":
            S = sub
            if not S:
                return None
            fin = [a for a in self.arts_all if a.get("subtask") == S and a.get("status") != "copy"]
            o = fin[0].get("by") if fin else self.first_assignee(S)
            if o is None:
                return None
            return (S, o), frozenset()
        if cls == "goal_drift":
            off = [a for a in self.arts_all if a.get("subtask") == "off_topic"]
            if off:
                return ("off_topic", off[0].get("by")), frozenset()
            return ("off_topic", frm), frozenset()
        if cls == "runaway_loop":
            S = sub
            if S:
                asg = [x.get("tagent") for x in ms if x.get("cat") == ASSIGN and x.get("sub") == S]
                if (not asg) or asg[0] == frm:
                    return (S, frm), frozenset([S])
            return ("__loop", frm), frozenset()
        if cls == "deadlock":
            return ("__dl", frm), frozenset()
        return None

    # ---------------------------------------------------------------- joint reconstruction
    def reconstruct(self, items, skip: frozenset, dp: bool):
        ck = (dp, tuple(items), skip)
        r = self._rec.get(ck)
        if r is not None:
            return r
        Q, und0 = self.plan_items(skip)
        und = dict(und0)
        qk = {q[0] for q in Q}
        forced = set()
        for key, o in items:
            if o is None or key in qk:
                continue
            und[key] = o
            forced.add(key)
        nmsg = self.nmsg
        ids = self.ids
        best = None
        regimes = []
        if len(ids) >= 2:
            regimes.append(("C1", ids[1:]))
        if len(ids) >= 1:
            regimes.append(("C2", ids[1:] + [ids[0]]))
        pri = lambda s: (0 if s in forced else 1, -nmsg[s])  # noqa: E731
        for regime, cyc in regimes:
            W = len(cyc)
            if dp:
                has = {o for o, n in Counter(und.values()).items() if n > 0}
                # the alignment is a pure function of (Q <- skip, cyc <- regime, has): memoised per run
                dk = (skip, regime, frozenset(has))
                al = self._dpc.get(dk)
                if al is None:
                    al = _dp_align(Q, cyc, has)
                    self._dpc[dk] = al
                slots, p, acost = al
            else:
                gk = (skip, regime)
                al = self._grc.get(gk)
                if al is None:
                    al = _greedy_align(Q, cyc)
                    self._grc[gk] = al
                slots, p = al
                acost = 0.0
            used = set(s for s in slots if s is not None)
            gaps = [k for k in range(p) if k not in used]
            U = dict(und)
            slot_of = {}
            for (key, by), s in zip(Q, slots):
                if s is not None:
                    slot_of[key] = s
            cost = acost
            unfilled = 0
            for g in gaps:
                o = cyc[g % W]
                c = sorted([s for s in U if U[s] == o], key=pri)
                if not c:
                    if not dp:
                        cost += 1
                    unfilled += 1
                    continue
                slot_of[c[0]] = g
                del U[c[0]]
            k = p
            orphans = 0
            while U:
                o = cyc[k % W]
                c = sorted([s for s in U if U[s] == o], key=pri)
                if not c:
                    s = min(U, key=lambda s: (1 if s in forced else 0, nmsg[s]))
                    del U[s]
                    orphans += 1
                    cost += 0.5
                    continue
                slot_of[c[0]] = k
                del U[c[0]]
                k += 1
            N = max(slot_of.values()) + 1 if slot_of else 0
            if dp:
                cost += 0.25 * (regime == "C2") + 2 * max(0, N - 8)
            else:
                cost += 0.25 * (regime == "C2") + sum(1 for s in slots if s is None) + 2 * max(0, N - 8)
            cand = (cost, regime, N, slot_of)
            if best is None or cand[0] < best[0]:
                best = cand
        r = best
        self._rec[ck] = r
        return r

    def inst_last_multi(self, insts, dp: bool):
        its = []
        skip = set()
        for t, cls in insts:
            it = self.inst_item(t, cls)
            if it is None:
                its.append(None)
                continue
            (S, o), sk = it
            if S.startswith("__"):
                S = S + "_" + cls
            its.append((S, o))
            skip |= sk
        R = self.reconstruct([x for x in its if x], frozenset(skip), dp)
        out = []
        for x in its:
            if R is None or x is None or x[0] not in R[3]:
                out.append(None)
                continue
            k = R[3][x[0]]
            N = R[2]
            out.append(dict(last=int(k == N - 1), after=N - 1 - k, cost=R[0], N=N))
        return out


def _greedy_align(Q, cyc):
    W = len(cyc)
    p = 0
    slots = []
    for key, by in Q:
        if by not in cyc:
            slots.append(None)
            continue
        k = p
        while cyc[k % W] != by:
            k += 1
        slots.append(k)
        p = k + 1
    return slots, p


def _dp_align(Q, cyc, has):
    """DP alignment of plan deliveries to the owner cycle (lastslot4.dp_align, PAR mis=2.5 skip 0.4/1.5)."""
    W = len(cyc)
    n = len(Q)
    if n == 0:
        return [], 0, 0.0
    INF = 1e9
    mis, s_has, s_none, maxskip = DP_PAR["mis"], DP_PAR["skip_has"], DP_PAR["skip_none"], DP_PAR["maxskip"]
    skc = [s_has if cyc[j] in has else s_none for j in range(W)]
    maxS = n + maxskip + 1
    dp = [dict() for _ in range(n + 1)]
    bk = [dict() for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(n):
        by = Q[i][1]
        mc_k = [0.0 if cyc[j] == by else mis for j in range(W)]
        nxt = dp[i + 1]
        nbk = bk[i + 1]
        for p, c0 in dp[i].items():
            skipc = 0.0
            for k in range(p, min(p + maxskip + 1, maxS)):
                if k > p:
                    skipc += skc[(k - 1) % W]
                c = c0 + skipc + mc_k[k % W]
                if c < nxt.get(k + 1, INF):
                    nxt[k + 1] = c
                    nbk[k + 1] = (p, k)
    last = dp[n]
    pend = min(last, key=lambda p: last[p])
    cost = last[pend]
    slots = [None] * n
    p = pend
    for i in range(n, 0, -1):
        pp, k = bk[i][p]
        slots[i - 1] = k
        p = pp
    return slots, pend, cost


# --------------------------------------------------------------------------------------------- per-run features
def _top2(rows: Sequence[Tuple[Any, Any]]):
    """stable top-2 by descending score (ties: first in given order), NaN/invalid rows dropped."""
    good = []
    for i, (t, s) in enumerate(rows):
        try:
            tf = float(t)
            sf = float(s)
        except Exception:
            continue
        if tf != tf or sf != sf:
            continue
        good.append((-sf, i, int(tf), sf))
    good.sort(key=lambda x: (x[0], x[1]))
    return [(g[2], g[3]) for g in good[:2]]


def features_run(run: Dict[str, Any], cand_rows: Dict[str, Sequence[Tuple[Any, Any]]]) -> Dict[str, float]:
    """cand_rows: {class: [(t, score), ...]} stage-1 candidates of this run (any order; ties keep list order)."""
    d: Dict[str, float] = {}
    try:
        R = _Run(run)
        top, sec = {}, {}
        for c in FAULTS:
            rows = cand_rows.get(c) if cand_rows else None
            if rows is None or len(rows) == 0:
                continue
            tt = _top2(rows)
            if tt:
                top[c] = tt[0]
            if len(tt) > 1:
                sec[c] = tt[1]
        strong = [(t, c) for c, (t, s) in top.items() if s > TH]
        # --- lastslot_final.run_features
        for c, (t, s) in top.items():
            insts = [(int(t), c)] + [x for x in strong if x[1] != c]
            f = R.inst_last_multi(insts, dp=False)[0]
            if f:
                for k in _KEYS:
                    d[f"lj_{c}_{k}"] = f[k]
            f = R.inst_last_multi(insts, dp=True)[0]
            if f:
                for k in _KEYS:
                    d[f"l5_{c}_{k}"] = f[k]
        # --- runfeats_top2
        for c, (t, s) in sec.items():
            f = R.inst_last_multi([(t, c)] + [x for x in strong if x[1] != c], dp=False)[0]
            if f:
                d[f"lj2_{c}_last"] = f["last"]
                d[f"lj2_{c}_score"] = s
        fl = R.inst_last_multi(strong, dp=False) if strong else []
        d["n_strong"] = len(strong)
        d["n_strong_last"] = sum(1 for f in fl if f and f["last"])
        d["max_score_nonlast"] = max([top[c][1] for (t, c), f in zip(strong, fl) if f and not f["last"]],
                                     default=0)
    except Exception:
        d = {}
    return d


def candidates_by_run(C: Dict[str, pd.DataFrame], n_runs: int) -> List[Dict[str, list]]:
    out: List[Dict[str, list]] = [dict() for _ in range(n_runs)]
    for c in FAULTS:
        D = C.get(c) if C else None
        if D is None or len(D) == 0 or not {"run", "t", "score"}.issubset(D.columns):
            continue
        r = pd.to_numeric(D["run"], errors="coerce").to_numpy(dtype=float)
        t = pd.to_numeric(D["t"], errors="coerce").to_numpy(dtype=float)
        s = pd.to_numeric(D["score"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(r) & np.isfinite(t) & np.isfinite(s) & (r >= 0) & (r < n_runs)
        idx = np.nonzero(ok)[0]
        if len(idx) == 0:
            continue
        # stable: by run, then descending score, ties keep row order
        order = idx[np.lexsort((np.arange(len(idx)), -s[idx], r[idx]))]
        ri = r[order].astype(np.int64)
        ti = t[order].astype(np.int64)
        so = s[order]
        # keep first two rows per run
        first = np.r_[True, ri[1:] != ri[:-1]]
        second = np.r_[False, first[:-1]] & ~first
        keep = first | second
        for i, tt, ss in zip(ri[keep].tolist(), ti[keep].tolist(), so[keep].tolist()):
            out[i].setdefault(c, []).append((tt, ss))
    return out


def features(runs: List[Dict[str, Any]], C: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    n = len(runs)
    cands = candidates_by_run(C, n)
    rows = [features_run(runs[i], cands[i]) for i in range(n)]
    F = pd.DataFrame.from_records(rows, index=range(n), columns=COLUMNS) if n else pd.DataFrame(columns=COLUMNS)
    return F.astype(np.float64)
