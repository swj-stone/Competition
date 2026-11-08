"""Probabilistic last-slot estimate P(slot of a fault instance = last plan slot).

Generator structure (recovered on train, see solution/README.md v4):
  * hidden plan of N slots, slot k owned by W[k % |W|], W = [a2..aN] (+ a1 in pipeline topology);
  * plan deliveries are listed in slot order in ``artifacts``; single 'set' entries of the slot owner on the
    slot subtask are listed in slot order in ``shared_state``;
  * plan subtasks are handed out by the planner (a1, or the previous agent of the cycle in pipelines);
    deliveries of subtasks handed out by somebody else (delegation noise) and deliveries after a
    "re-assigning X to recover" message are extras (recovery deliveries are appended after the plan);
  * every fault instance occupies one slot; the labelled fault is never on the last one.

For a set of fault instances [(t, cls)] the module builds the plan items
  - anchored in the artifact list: delivered plan subtasks, the off-topic artifact (goal drift);
  - anchored in the state list: conflict blocks, single 'set' entries of plan subtasks;
  - free: undelivered plan subtasks, dropped handoffs, deadlocks, loops;
and sums over every plan hypothesis -- choice of the plan artifact among several candidates of a subtask,
spurious undelivered assignments (prior SIGMA each), h hidden slots (prior RHO**h) -- the number of
arrangements in which the anchored items keep both list orders and every item sits on a slot of its owner
(exact counting: DP over slots for the two anchor chains, closed form for the free items), normalised by
the number of all owner-consistent arrangements.  Output: posterior P(instance occupies slot N-1).

    Run(run).plast(insts) -> list of P(last) (NaN when the instance cannot be mapped)
    features(runs, C) -> DataFrame   (per class top candidate + top-2 pair features)
"""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from parse import ASSIGN, FINAL, ITER, RECOVER, SUB2DOM, WAIT

FAULTS = ["dropped_handoff", "duplicated_work", "deadlock", "conflict", "goal_drift", "runaway_loop"]
NAN = float("nan")
RHO = 0.05          # prior weight per hidden slot
SIGMA = 0.05        # prior weight per spurious (dropped) undelivered assignment
HMAX = 3
MAXDROP = 2
MAXCOMB = 16
NORM = True         # normalise each hypothesis by the number of all owner-consistent arrangements
FALLBACK = 1        # infeasible plan -> retry with up to |W| more hidden slots and one more spurious assignment
TAU = 0.3           # weight per undelivered plan subtask placed before the last delivered slot (prefix rule:
                    # in clean runs undelivered plan subtasks always form the tail of the plan)
BLANKOFF = 1        # artifact without any recoverable subtask = off-topic (drift) artifact
OFFDOM = True       # artifact of another domain's subtask = off-topic (drift) artifact
BLANKPL = True      # planner assignment with unparsable subtask: that agent's unassigned deliveries are plan items
# research toggles, off (both measured worse on the train pairs):
EARLIEST = 0        # owner of deadlock / loop = sender of the earliest message of the instance
NONOWNER = 0        # 0 off, 1 non-owner artifact of an undelivered subtask as optional anchor, 2 forced
TH = 0.3            # strong candidate threshold for the joint instance set


def _is_int(x) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


class Run:
    def __init__(self, run: Dict[str, Any]):
        ms = [m for m in (run.get("messages") or []) if isinstance(m, dict)]
        self.ms = ms
        ags = [a.get("id") for a in (run.get("agents") or []) if isinstance(a, dict)]
        self.ags = ags
        self.a1 = ags[0] if ags else "a1"
        topo = run.get("topology")
        self.pipe = isinstance(topo, dict) and topo.get("type") == "pipeline"
        self.W = ags[1:] + ([ags[0]] if (self.pipe and ags) else [])
        self.L = len(self.W)
        self.Wset = set(self.W)
        # ---- assignments, recoveries
        rec_pending = {}
        self.planner: Dict[str, List[Any]] = {}
        self.rec_agents = defaultdict(set)
        self.blank_planner = set()           # agents that got a planner assignment with unknown subtask
        for m in ms:
            s = m.get("sub") or ""
            c = m.get("cat")
            if c == RECOVER and s:
                ta = m.get("tagent") or m.get("to")
                rec_pending.setdefault(s, set()).add(ta)
                self.rec_agents[s].add(ta)
            elif c == ASSIGN:
                to = m.get("tagent") or m.get("to")
                if s and to in rec_pending.get(s, ()):
                    rec_pending[s].discard(to)
                    continue
                if self._planner_ok(m.get("from"), to):
                    if s:
                        lst = self.planner.setdefault(s, [])
                        if to not in lst:
                            lst.append(to)
                    else:
                        self.blank_planner.add(to)
        dc = Counter(SUB2DOM[s] for s in self.planner if s in SUB2DOM)
        self.dom = dc.most_common(1)[0][0] if dc else run.get("domain", "unknown")
        # ---- artifacts
        arts = []
        for j, a in enumerate(run.get("artifacts") or []):
            if not isinstance(a, dict):
                continue
            s0 = a.get("subtask") or ""
            if a.get("status") == "copy" or s0 == "final_delivery":
                continue
            at = a.get("t")
            m = ms[at] if _is_int(at) and 0 <= at < len(ms) else None
            if not s0 and m is not None and m.get("cat") == FINAL:
                continue
            s = s0 or (m.get("sub", "") if m is not None else "")
            off = s0 == "off_topic" or (OFFDOM and s in SUB2DOM and SUB2DOM[s] != self.dom) or (BLANKOFF and not s)
            arts.append(dict(pos=j, sub=s, by=a.get("by"), hash=a.get("hash"), t=at, off=off))
        self.arts = arts
        # subtasks delivered by blank-planner agents without any planner assignment -> planner subtasks
        for a in arts:
            if a["off"] or not a["sub"] or a["sub"] in self.planner:
                continue
            if BLANKPL and a["by"] in self.blank_planner and a["by"] not in self.rec_agents.get(a["sub"], ()):
                self.planner[a["sub"]] = [a["by"]]
        # ---- state blocks (run_status excluded): a 'set' followed by overrides of the same key
        L = [s for s in (run.get("shared_state") or []) if isinstance(s, dict) and s.get("key") != "run_status"]
        blocks = []
        j = 0
        while j < len(L):
            k = j + 1
            while k < len(L) and L[k].get("key") == L[j].get("key") and L[k].get("op") == "override":
                k += 1
            B = L[j:k]
            t0 = B[0].get("t")
            m = ms[t0] if _is_int(t0) and 0 <= t0 < len(ms) else None
            blocks.append(dict(agent=B[0].get("agent"), sub=(m.get("sub", "") if m is not None else ""),
                               op=B[0].get("op"), n=len(B), ts={s.get("t") for s in B}, t0=t0))
            j = k
        self.blocks = blocks

    def _planner_ok(self, frm, to) -> bool:
        if not self.pipe:
            return frm == self.a1
        W = self.W
        if frm not in self.Wset or to not in self.Wset:
            return False
        return W.index(frm) == (W.index(to) - 1) % len(W)

    # ------------------------------------------------------------------ instances
    def inst_spec(self, t, cls):
        ms = self.ms
        if not _is_int(t) or not 0 <= t < len(ms):
            return None
        m = ms[t]
        S = m.get("sub") or ""
        if cls == "goal_drift":
            offs = [a for a in self.arts if a["off"]]
            own = [a for a in offs if a["by"] == m.get("from")]
            a = (own or offs or [None])[0]
            if a is not None:
                return dict(kind="art", pos=a["pos"], owner=a["by"], cls=cls)
            return dict(kind="free", owner=m.get("from"), cls=cls)
        if cls == "duplicated_work":
            if not S:
                return None
            return dict(kind="sub", sub=S, cls=cls)
        if cls == "conflict":
            for bi, b in enumerate(self.blocks):
                if t in b["ts"]:
                    o = b["agent"]
                    cons = b["sub"] if (b["sub"] and o in self.planner.get(b["sub"], ())) else None
                    return dict(kind="blk", blk=bi, owner=o, consumes=cons, extra_sub=b["sub"], cls=cls)
            return None
        if cls == "dropped_handoff":
            o = m.get("tagent") or m.get("to")
            return dict(kind="free", owner=o, consumes=S if S else None, extra_sub=S, cls=cls)
        if cls == "deadlock":
            o = m.get("from")
            if EARLIEST:
                pair = {o, m.get("tagent") or m.get("to")}
                for x in ms:
                    if x.get("cat") == WAIT and {x.get("from"), x.get("tagent") or x.get("to")} == pair:
                        o = x.get("from")
                        break
            return dict(kind="free", owner=o, cls=cls)
        if cls == "runaway_loop":
            o = m.get("from")
            if EARLIEST and S:
                for x in ms:
                    if x.get("cat") == ITER and x.get("sub") == S:
                        o = x.get("from")
                        break
            cons = None
            if S and o in self.planner.get(S, ()) and not any(a["sub"] == S and a["by"] == o for a in self.arts):
                cons = S
            return dict(kind="free", owner=o, consumes=cons, cls=cls)
        return None

    # ------------------------------------------------------------------ items
    def build(self, specs):
        """Returns (items, inst_item, alts, droppable).
        items: dict(owner, art, blk, sub); alts: {item: [(owner, pos), ...]} alternative plan artifacts;
        droppable: items that may be spurious (undelivered planner assignments)."""
        consumed = {}
        extra_subs = set()
        for n, sp in enumerate(specs):
            if sp is None:
                continue
            if sp.get("consumes"):
                consumed.setdefault(sp["consumes"], n)
            if sp.get("extra_sub"):
                extra_subs.add(sp["extra_sub"])
        items = []
        inst_item = [None] * len(specs)
        sub_item = {}
        used_pos = set()
        alts = {}
        droppable = []
        for n, sp in enumerate(specs):
            if sp is None or sp["kind"] == "sub":
                continue
            if sp["kind"] == "art":
                ex = [k for k, it in enumerate(items) if it["art"] == sp["pos"]]
                if ex:
                    inst_item[n] = ex[0]
                    continue
                used_pos.add(sp["pos"])
                items.append(dict(owner=sp["owner"], art=sp["pos"], blk=None, sub="off_topic"))
            elif sp["kind"] == "blk":
                ex = [k for k, it in enumerate(items) if it["blk"] == sp["blk"]]
                if ex:
                    inst_item[n] = ex[0]
                    continue
                items.append(dict(owner=sp["owner"], art=None, blk=sp["blk"], sub=sp.get("consumes")))
            else:
                items.append(dict(owner=sp["owner"], art=None, blk=None, sub=sp.get("consumes")))
            inst_item[n] = len(items) - 1
            if sp.get("consumes"):
                sub_item.setdefault(sp["consumes"], len(items) - 1)
        for S, tos in self.planner.items():
            if S in consumed:
                continue
            cand = [] if S in extra_subs else [a for a in self.arts if a["sub"] == S and a["by"] in tos
                                                 and not a["off"]]
            if cand:
                a = cand[0]
                used_pos.update(c["pos"] for c in cand)
                items.append(dict(owner=a["by"], art=a["pos"], blk=None, sub=S, normal=True))
                if len(cand) > 1:
                    alts[len(items) - 1] = [(c["by"], c["pos"]) for c in cand]
            else:
                others = [] if (S in extra_subs or not NONOWNER) else [
                    a for a in self.arts if a["sub"] == S and not a["off"]
                    and a["by"] not in self.rec_agents.get(S, ())]
                items.append(dict(owner=tos[0], art=None, blk=None, sub=S, normal=True))
                droppable.append(len(items) - 1)
                pos_opts = ([] if NONOWNER == 2 and others else [None]) + [a["pos"] for a in others[:1]]
                if len(tos) > 1 or len(pos_opts) > 1:
                    alts[len(items) - 1] = [(o, p) for o in tos for p in pos_opts]
                elif pos_opts[0] is not None:
                    items[-1]["art"] = pos_opts[0]
            sub_item[S] = len(items) - 1
        offs = [a for a in self.arts if a["off"]]
        if offs and not any(a["pos"] in used_pos for a in offs):
            items.append(dict(owner=offs[0]["by"], art=offs[0]["pos"], blk=None, sub="off_topic"))
            used_pos.add(offs[0]["pos"])
        seen_h = set()
        for a in self.arts:
            if a["sub"] or a["pos"] in used_pos or a["by"] not in self.Wset:
                continue
            if a["hash"] in seen_h:
                continue
            seen_h.add(a["hash"])
            items.append(dict(owner=a["by"], art=a["pos"], blk=None, sub=""))
        for n, sp in enumerate(specs):
            if sp is not None and sp["kind"] == "sub":
                inst_item[n] = sub_item.get(sp["sub"])
        for bi, b in enumerate(self.blocks):
            if b["n"] != 1 or b["op"] != "set" or not b["sub"]:
                continue
            k = sub_item.get(b["sub"])
            if k is None or items[k]["blk"] is not None:
                continue
            if b["agent"] in self.rec_agents.get(b["sub"], ()):
                continue
            if items[k]["owner"] != b["agent"] and not any(o == b["agent"] for o, _ in alts.get(k, ())):
                continue
            items[k]["blk"] = bi
            items[k]["blk_agent"] = b["agent"]
        return items, inst_item, alts, droppable

    # ------------------------------------------------------------------ counting
    def weigh(self, items, rho, hmax):
        """Sum over N of the (normalised, prefix-rule weighted) number of consistent arrangements.
        Returns (total, per-item weight on the last slot) or (0, None)."""
        W, L = self.W, self.L
        n = len(items)
        if n == 0:
            return 0.0, None
        A = sorted([k for k, it in enumerate(items) if it["art"] is not None], key=lambda k: items[k]["art"])
        S = sorted([k for k, it in enumerate(items) if it["blk"] is not None
                    and it.get("blk_agent", it["owner"]) == it["owner"]], key=lambda k: items[k]["blk"])
        A, S = _make_consistent(A, S)
        chain = set(A) | set(S)
        free = [k for k in range(n) if k not in chain]
        chain_o = Counter(items[k]["owner"] for k in chain)
        fn_o = Counter(items[k]["owner"] for k in free if items[k].get("normal"))
        fi_o = Counter(items[k]["owner"] for k in free if not items[k].get("normal"))
        owner_items = Counter(it["owner"] for it in items)
        Aown = Counter(items[k]["owner"] for k in A)
        sonly = [k for k in S if k not in set(A)]
        tot = 0.0
        acc = np.zeros(n)
        for h in range(hmax + 1):
            N = n + h
            slots_o = Counter(W[k % L] for k in range(N))
            if any(slots_o[o] < c for o, c in owner_items.items()):
                continue
            Fo = {o: slots_o[o] - chain_o.get(o, 0) for o in slots_o}
            if any(Fo.get(o, 0) < fn_o.get(o, 0) + fi_o.get(o, 0) for o in set(fn_o) | set(fi_o)):
                continue
            finals, lasts = _chain_count2(A, S, items, W, N, TAU)
            if not finals:
                continue
            norm = 1.0
            if NORM:
                for o, c in owner_items.items():
                    norm *= _perm(slots_o[o], c)
            base = rho ** h / norm
            o_last = W[(N - 1) % L]
            cum = {}
            for (K, jK), c in finals.items():
                Io = _interior(W, L, K, Aown, [items[k]["owner"] for k in S[:jK] if k in set(sonly)])
                wf = 1.0
                for o in set(fn_o) | set(fi_o):
                    wf *= _free_w(Fo[o], Io.get(o, 0), fn_o.get(o, 0), fi_o.get(o, 0), TAU)
                    if wf == 0.0:
                        break
                cum[(K, jK)] = (wf, Io)
                tot += base * c * wf
            for (K, jK, who), c in lasts.items():
                wf, Io = cum.get((K, jK), (0.0, None))
                if who is not None:
                    if wf > 0:
                        acc[who] += base * c * wf
                    continue
                # slot N-1 free: distribute over the free items of its owner
                if Io is None:
                    Io = _interior(W, L, K, Aown, [items[k]["owner"] for k in S[:jK] if k in set(sonly)])
                other = 1.0
                for o in set(fn_o) | set(fi_o):
                    if o != o_last:
                        other *= _free_w(Fo[o], Io.get(o, 0), fn_o.get(o, 0), fi_o.get(o, 0), TAU)
                if other == 0.0:
                    continue
                F, I = Fo.get(o_last, 0), Io.get(o_last, 0)
                nn, mm = fn_o.get(o_last, 0), fi_o.get(o_last, 0)
                if F - I <= 0:
                    continue
                w_inst = _free_w_last(F, I, nn, mm, TAU, normal=False)
                w_norm = _free_w_last(F, I, nn, mm, TAU, normal=True)
                for k in free:
                    if items[k]["owner"] != o_last:
                        continue
                    ww = w_norm if items[k].get("normal") else w_inst
                    acc[k] += base * c * other * ww
        if tot <= 0:
            return 0.0, None
        return tot, acc

    def plast(self, insts, rho=RHO, hmax=HMAX, sigma=SIGMA):
        specs = [self.inst_spec(t, c) for t, c in insts]
        items, inst_item, alts, droppable = self.build(specs)
        bad = {k for k, it in enumerate(items) if it["owner"] not in self.Wset}
        akeys = [k for k in alts if k not in bad]
        choices = [alts[k] for k in akeys]
        combos = list(itertools.islice(itertools.product(*choices), MAXCOMB)) if choices else [()]
        drops = [d for d in droppable if d not in bad]
        tot, acc = self._sum(items, bad, akeys, combos, drops, rho, hmax, sigma, MAXDROP)
        if tot <= 0 and FALLBACK:
            tot, acc = self._sum(items, bad, akeys, combos, drops, rho, hmax + self.L, sigma, MAXDROP + 1)
        out = []
        for k in inst_item:
            if k is None or k in bad or tot <= 0:
                out.append(NAN)
            else:
                out.append(float(acc[k] / tot))
        return out

    def _sum(self, items, bad, akeys, combos, drops, rho, hmax, sigma, maxdrop):
        tot = 0.0
        acc = np.zeros(len(items))
        for nd in range(0, maxdrop + 1):
            if nd > len(drops):
                break
            for D in itertools.combinations(drops, nd):
                Dset = set(D) | bad
                for combo in combos:
                    its = [dict(it) for it in items]
                    for k, (o, pos) in zip(akeys, combo):
                        its[k]["owner"] = o
                        if pos is not None:
                            its[k]["art"] = pos
                    keep = [k for k in range(len(its)) if k not in Dset and its[k]["owner"] in self.Wset]
                    w, a = self.weigh([its[k] for k in keep], rho, hmax)
                    if w <= 0:
                        continue
                    f = sigma ** nd
                    tot += f * w
                    for i, k in enumerate(keep):
                        acc[k] += f * a[i]
        return tot, acc


def _perm(n, k):
    if k > n:
        return 0.0
    r = 1.0
    for x in range(k):
        r *= (n - x)
    return r


def _interior(W, L, K, Aown, sown):
    """free slots per owner at slots <= K (K = slot of the last artifact-anchored item)."""
    I = Counter(W[k % L] for k in range(K + 1))
    for o, c in Aown.items():
        I[o] -= c
    for o in sown:
        I[o] -= 1
    return I


def _free_w(F, I, n, m, tau):
    """weighted # of placements of n normal (tau per interior one) + m instance items into F free slots,
    I of them interior."""
    T = F - I
    s = 0.0
    for j in range(0, min(n, I) + 1):
        if n - j > T:
            continue
        s += math.comb(n, j) * _perm(I, j) * _perm(T, n - j) * (tau ** j)
    return s * _perm(F - n, m)


def _free_w_last(F, I, n, m, tau, normal):
    """same with one given item on the (tail) last slot."""
    T = F - I
    if normal:
        s = 0.0
        for j in range(0, min(n - 1, I) + 1):
            if n - 1 - j > T - 1:
                continue
            s += math.comb(n - 1, j) * _perm(I, j) * _perm(T - 1, n - 1 - j) * (tau ** j)
        return s * _perm(F - n, m)
    s = 0.0
    for j in range(0, min(n, I) + 1):
        if n - j > T - 1:
            continue
        s += math.comb(n, j) * _perm(I, j) * _perm(T - 1, n - j) * (tau ** j)
    return s * _perm(F - n - 1, m - 1)


def _chain_count2(A, S, items, W, N, tau):
    """DP over slots for the anchor chains.  State (i, j, K, jK): i / j chain items placed, K = slot of the
    last A item (-1 before), jK = S items placed up to K.  S-only normal items placed before the last A item
    get weight tau.  Returns (finals {(K, jK): w}, lasts {(K, jK, item or None): w} for slot N-1)."""
    L = len(W)
    Aset = set(A)
    shared = Aset & set(S)
    nA, nS = len(A), len(S)
    cur = {(0, 0, -1, 0): 1.0}
    lasts = defaultdict(float)
    for slot in range(N):
        o = W[slot % L]
        nxt = defaultdict(float)
        is_last = slot == N - 1
        for (i, j, K, jK), c in cur.items():
            nxt[(i, j, K, jK)] += c
            if is_last and i == nA and j == nS:
                lasts[(K, jK, None)] += c
            if i < nA:
                k = A[i]
                if items[k]["owner"] == o:
                    if k not in shared:
                        st = (i + 1, j, slot if i + 1 == nA else K, j if i + 1 == nA else jK)
                        nxt[st] += c
                        if is_last and i + 1 == nA and j == nS:
                            lasts[(st[2], st[3], k)] += c
                    elif j < nS and S[j] == k:
                        st = (i + 1, j + 1, slot if i + 1 == nA else K, j + 1 if i + 1 == nA else jK)
                        nxt[st] += c
                        if is_last and i + 1 == nA and j + 1 == nS:
                            lasts[(st[2], st[3], k)] += c
            if j < nS:
                k = S[j]
                if k not in shared and items[k]["owner"] == o:
                    w = c * (tau if (items[k].get("normal") and i < nA) else 1.0)
                    st = (i, j + 1, K, jK)
                    nxt[st] += w
                    if is_last and i == nA and j + 1 == nS:
                        lasts[(K, jK, k)] += w
        cur = nxt
    finals = {(K, jK): c for (i, j, K, jK), c in cur.items() if i == nA and j == nS and c > 0}
    return finals, dict(lasts)


def _make_consistent(A, S):
    posA = {k: i for i, k in enumerate(A)}
    S2 = []
    last = -1
    for k in S:
        if k in posA:
            if posA[k] < last:
                continue
            last = posA[k]
        S2.append(k)
    return A, S2


def plast_run(run, insts, **kw):
    try:
        return Run(run).plast(insts, **kw)
    except Exception:
        return [NAN] * len(insts)


# ============================================================ run-level features
COLUMNS = (["pl_" + c for c in FAULTS]
           + ["pl_top1", "pl_top2", "pl_lo12", "pl_sum_strong", "pl_n_strong"])


def _lo(p):
    p = min(max(float(p), 0.0), 0.995)
    return math.log(1.0 - p)


def top_candidates(C, n_runs: int, col: str = "score") -> List[Dict[str, Tuple[int, float]]]:
    """Per run: {cls: (t of the highest-scoring candidate, max score)} (first row on ties)."""
    out: List[Dict[str, Tuple[int, float]]] = [dict() for _ in range(n_runs)]
    for c in FAULTS:
        D = (C or {}).get(c)
        if D is None or len(D) == 0 or not {"run", "t", col} <= set(D.columns):
            continue
        D = D[["run", "t", col]].reset_index(drop=True)
        D = D[D[col].notna() & D["run"].notna() & D["t"].notna()]
        D = D[(D["run"] >= 0) & (D["run"] < n_runs)]
        if len(D) == 0:
            continue
        Dm = D.loc[D.groupby("run")[col].idxmax()]
        for r, t, s in zip(Dm["run"].values, Dm["t"].values, Dm[col].values):
            out[int(r)][c] = (int(t), float(s))
    return out


def features_run(run, top: Dict[str, Tuple[int, float]], th: float = TH) -> Dict[str, float]:
    d = {c: NAN for c in COLUMNS}
    d["pl_lo12"] = 0.0
    d["pl_sum_strong"] = 0.0
    d["pl_n_strong"] = 0.0
    try:
        R = Run(run)
        strong = [(t, c) for c, (t, s) in top.items() if s > th]
        pv = {}
        if strong:
            for (t, c), p in zip(strong, R.plast(strong)):
                pv[c] = p
        for c, (t, s) in top.items():
            if c in pv:
                continue
            pv[c] = R.plast([(t, c)] + strong)[0]
        for c, p in pv.items():
            d["pl_" + c] = p
        order = sorted(top, key=lambda c: (-top[c][1], FAULTS.index(c)))
        if order:
            d["pl_top1"] = pv.get(order[0], NAN)
        if len(order) > 1:
            d["pl_top2"] = pv.get(order[1], NAN)
        p1 = d["pl_top1"] if d["pl_top1"] == d["pl_top1"] else 0.0
        p2 = d["pl_top2"] if d["pl_top2"] == d["pl_top2"] else 0.0
        d["pl_lo12"] = _lo(p2) - _lo(p1)
        d["pl_sum_strong"] = float(sum(pv[c] for t, c in strong if pv.get(c) == pv.get(c)))
        d["pl_n_strong"] = float(len(strong))
    except Exception:
        pass
    return d


def features(runs, C) -> pd.DataFrame:
    n = len(runs)
    top = top_candidates(C, n)
    rows = [features_run(runs[i], top[i]) for i in range(n)]
    return pd.DataFrame(rows, index=range(n), columns=COLUMNS).astype(float)
