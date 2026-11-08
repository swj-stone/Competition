"""resid_post: final-pass residual rules for the v65 pipeline (exp60 resid_synth).  Label-free, nothing fitted.

Hook (pipeline.fit_predict_features, inside the arbiter branch, right after _cset_addon(...) has returned, i.e. after
the pair arbiter, the clean-vs-top-fault gate, the optional none-pool (CR_NONE_POOL) and the cset pointer turns):

    pred, turns = resid_post.final_pass(pred, turns, P, T_nn, Cte, runs_te, arb_info, gate="all", turn_rules=True,
                                        log=log)

Inputs
  pred      list of str (n test runs): current labels
  turns     list of int (n): current fault turns (-1 for clean)
  P         (n, 7) stage-2 blend probabilities in LABELS order (the P handed to arbiter.fit_apply)
  T_nn      (n, 6) cset pointer turn per fault (third output of csetnn.predict; -1 = none) or None
  Cte       dict fault -> DataFrame with columns run, t, score (stage-1 test candidates after scoring)
  runs      list (n) of parsed runs (pipeline runs_te = parse.parse_row of the test rows)
  arb_info  dict returned by arbiter.fit_apply; only D_test (columns run, kind 'ff' / 'cf', a) is read
  gate      'off' | 'cf' (clean decisions of clean-vs-top-fault gate rows only) | 'all' (every clean decision)
  turn_rules  bool

Steps (exp60 resid_gate and resid_turns, verbatim logic; see exp60/resid_synth/rule.py for the OOF evaluation)
  GATE   for runs whose current label is clean (and that are gate rows when gate == 'cf'): if a never-in-clean
         fault signature fires, the label becomes the fired fault with the highest P (recovery message only: the
         argmax of P over the six faults); turn = documented turn of the evidence, else T_nn of that fault, else the
         top stage-1 candidate of that fault (the last fallback never occurs on the 20k OOF runs).
         Gate rows whose current label is duplicated_work (gated fault duplicated_work) without any duplicate
         evidence become clean.
  TURNS  for runs labelled conflict / goal_drift / duplicated_work / deadlock: ontology re-pointing R1-R4.
Every rule reads documented fields only (messages t / from / to / type / intent / text / refs, shared_state t / agent /
key / op / value sorted by t, artifacts by / t / subtask / hash (equality only) / status); never run_id, list order
inside artifacts / shared_state, numeric parts of ids or the data source.  A per-run exception leaves that run
unchanged.  Success is not touched.
"""
from __future__ import annotations

import collections
import os
import re

import numpy as np

import parse as PZ

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock", "conflict", "goal_drift", "runaway_loop"]
FAULTS = LABELS[1:]

# =====================================================================================================================
# GATE: never-in-clean fault signatures (verbatim from exp60/resid_gate/rule.py, md5 70aa38603c0f4073a9d29e67b582e15e)
# =====================================================================================================================
FINAL_SUBS = ("final_delivery", "off_topic")
ACTIVE = (PZ.WORK, PZ.ITER, PZ.SET, PZ.DISAGREE, PZ.DELIVER, PZ.WAIT)
WORD_END = re.compile(r"([a-z][a-z_]{4,})$")


def _parse(row):
    if isinstance(row, dict) and isinstance(row.get("messages"), list) and row["messages"] and "cat" in row["messages"][0]:
        return row
    get = (lambda k: row.get(k)) if isinstance(row, dict) else (lambda k: getattr(row, k))
    return PZ.parse_row(dict(goal=get("goal"), agents=get("agents"), topology=get("topology"), messages=get("messages"),
                             shared_state=get("shared_state"), artifacts=get("artifacts")))


def _arts(run):
    return [a for a in run["artifacts"] if isinstance(a, dict)]


def _assigns(run):
    return [m for m in run["messages"] if m["type"] == "handoff" and not m["refs"] and m["cat"] == PZ.ASSIGN]


def _delivered_subs(run):
    subs = {a["subtask"] for a in _arts(run) if a.get("subtask") in PZ.SUB2DOM}
    subs |= {m["sub"] for m in run["messages"] if m["type"] == "handoff" and m["refs"] and m["sub"]}
    return subs


def _state_sorted(ss):
    return sorted([s for s in ss if isinstance(s, dict) and s.get("key") != "run_status"],
                  key=lambda s: (s.get("t", 0) if isinstance(s.get("t"), (int, float)) else 0, str(s.get("agent", "")),
                                 str(s.get("op", "")), str(s.get("value"))))


def _msg_subtasks(run):
    ms, arts = run["messages"], _arts(run)
    aid = {a.get("id"): a for a in arts}
    at = {}
    for a in arts:
        at.setdefault((a.get("t"), a.get("by")), []).append(a)
    dom = PZ.DOMAIN_SUBTASKS.get(run.get("domain"), sorted(PZ.SUB2DOM))
    out = []
    for m in ms:
        s = m["sub"]
        if not s:
            for r in m["refs"]:
                x = aid.get(r)
                if x and x.get("subtask") in PZ.SUB2DOM:
                    s = x["subtask"]
                    break
        if not s:
            w = WORD_END.search((m["text"] or "").rstrip())
            if w:
                c = [z for z in dom if z.startswith(w.group(1))]
                s = c[0] if len(c) == 1 else ""
        if not s and m["type"] == "handoff":
            cs = {x.get("subtask") for x in at.get((m["t"], m["from"]), []) if x.get("subtask") in PZ.SUB2DOM}
            if len(cs) == 1:            # order-free: only an unambiguous same-turn artifact names the subtask
                s = cs.pop()
        out.append(s)
    return out


def sig_conflict(run):
    by = {}
    for s in _state_sorted(run["shared_state"]):
        by.setdefault(s.get("key"), []).append(s)
    n, turn = 0, -1
    for k, L in by.items():
        vals = [str(s.get("value")) for s in L]
        kn = 0
        for i in range(1, len(L)):
            v, ag = vals[i], L[i].get("agent")
            if (L[i].get("op") == "override" and v != vals[i - 1] and ag != L[i - 1].get("agent")
                    and any(vals[j] == v and L[j].get("agent") == ag for j in range(i - 1))):
                kn += 1
        if kn:
            n += kn
            t1 = min(s.get("t", 0) for s in L if s.get("op") == "override")
            turn = t1 if turn < 0 else min(turn, t1)
    return n, turn


def sig_deadlock(run):
    ms = run["messages"]
    pairs = {}
    for m in ms:
        if m["type"] == "status" and m.get("from") and m.get("to") and m["from"] != m["to"]:
            pairs.setdefault(frozenset((m["from"], m["to"])), []).append(m)
    n, turn = 0, -1
    for k, L in pairs.items():
        a, b = tuple(k)
        answers = [m["t"] for m in ms if m["type"] == "handoff" and {m.get("from"), m.get("to")} == {a, b}
                   and (m["refs"] or m["cat"] not in (PZ.ASSIGN, PZ.RECOVER))]
        last = max(answers) if answers else -1
        tail = [m for m in L if m["t"] > last]
        if any(m["from"] == a for m in tail) and any(m["from"] == b for m in tail):
            n += 1
            t0 = min(m["t"] for m in tail)
            turn = t0 if turn < 0 else min(turn, t0)
    return n, turn


def sig_duplicate(run):
    arts = [a for a in _arts(run) if a.get("status") != "copy" and a.get("subtask") not in FINAL_SUBS]
    byh = {}
    for a in arts:
        if a.get("hash"):
            byh.setdefault(a["hash"], []).append(a)
    groups = [L for L in byh.values() if len(L) >= 2]
    if not groups:
        return 0, -1
    subs_of = {}
    for m, s in zip(run["messages"], _msg_subtasks(run)):
        if m["type"] == "handoff" and s:
            subs_of.setdefault((m["t"], m["from"]), s)
            for r in m["refs"]:
                subs_of.setdefault(("ref", r), s)
    A = _assigns(run)
    turn = -1
    for L in groups:
        subs = set()
        for a in L:
            s = a.get("subtask") if a.get("subtask") in PZ.SUB2DOM else ""
            s = s or subs_of.get(("ref", a.get("id")), "") or subs_of.get((a.get("t"), a.get("by")), "")
            if s:
                subs.add(s)
        for s in subs:
            ts = sorted(m["t"] for m in A if m["sub"] == s)
            if len(ts) >= 2:
                turn = ts[1] if turn < 0 else min(turn, ts[1])
    return len(groups), turn


def sig_dropped(run):
    ms = run["messages"]
    dl = _delivered_subs(run)
    n_sil = n_wait = 0
    turn = -1
    for a in _assigns(run):
        s, t, to = a["sub"], a["t"], a.get("to")
        if not s or s in dl:
            continue
        after = [m for m in ms if m["t"] > t and m.get("from") == to]
        fired = False
        if not after:
            n_sil += 1
            fired = True
        elif not any(m["sub"] == s and m["cat"] in ACTIVE for m in after):
            waits = [m["t"] for m in ms if m["t"] > t and m["type"] == "status" and m["sub"] == s and m.get("from") != to]
            claims = [m["t"] for m in ms if m["type"] == "handoff" and m["sub"] == s and m["cat"] != PZ.ASSIGN]
            if waits and (not claims or max(waits) > max(claims)):
                n_wait += 1
                fired = True
        if fired and turn < 0:
            turn = t
    return n_sil, n_wait, turn


def multi_deliverer(run):
    ms, arts = run["messages"], _arts(run)
    aid = {a.get("id"): a for a in arts}
    at = {}
    for a in arts:
        at.setdefault((a.get("t"), a.get("by")), []).append(a)
    deliv, seen = [], set()
    for m, s in zip(ms, _msg_subtasks(run)):
        if m["type"] != "handoff":
            continue
        for x in ([aid[r] for r in m["refs"] if r in aid] or at.get((m["t"], m["from"]), [])):
            if x.get("status") == "copy" or x.get("subtask") in FINAL_SUBS:
                continue
            deliv.append((m["from"], s or (x.get("subtask") if x.get("subtask") in PZ.SUB2DOM else ""), x.get("id")))
            seen.add(x.get("id"))
    for x in arts:
        if x.get("id") not in seen and x.get("status") != "copy" and x.get("subtask") not in FINAL_SUBS:
            deliv.append((x.get("by"), x.get("subtask") if x.get("subtask") in PZ.SUB2DOM else "", x.get("id")))
    by = {}
    for ag, s, i in deliv:
        if s:
            by.setdefault(s, {}).setdefault(ag, set()).add(i)
    n_multi = sum(1 for v in by.values() if len(v) >= 2 and len(set().union(*v.values())) >= 2)
    return n_multi, sum(1 for d in deliv if d[1])


def signatures(run):
    run = _parse(run)
    c, tc = sig_conflict(run)
    d, td = sig_deadlock(run)
    u, tu = sig_duplicate(run)
    sil, wait, th = sig_dropped(run)
    md, nk = multi_deliverer(run)
    return dict(conflict=c, t_conflict=tc, deadlock=d, t_deadlock=td, duplicated_work=u, t_duplicated_work=tu,
                dh_silent=sil, dh_wait=wait, t_dropped_handoff=th,
                off_topic=sum(1 for a in _arts(run) if a.get("subtask") == "off_topic"),
                recovery=sum(1 for m in run["messages"] if m["cat"] == PZ.RECOVER),
                multi_deliv=md, n_known_deliv=nk)


SIG_CLASS = {"conflict": "conflict", "deadlock": "deadlock", "duplicated_work": "duplicated_work",
             "dh_silent": "dropped_handoff", "dh_wait": "dropped_handoff", "off_topic": "goal_drift"}
ALL_SIGS = ("conflict", "deadlock", "duplicated_work", "dh_silent", "dh_wait", "off_topic")

# =====================================================================================================================
# TURNS: ontology re-pointing (verbatim from exp60/resid_turns/rule.py, md5 4e6a1272466ae0717e8e4f6086ca06c7)
# =====================================================================================================================
HIT = 2
SCORE_MIN = 0.01
_RECOV_PREFIXES = ("Recovery path taken for", "Detected an issue, re-assigning", "Repairing the plan around",
                   "Mitigation applied for", "Fallback engaged on", "Escalating and retrying")
_WSUB = re.compile(r"(I need (\w+) first|On hold: (\w+) from|for a\d+ on (\w+)\.|to deliver (\w+) before|closing (\w+)\.|finishes (\w+)\.)")


def _is_recov_text(txt, minlen=8):
    for p in _RECOV_PREFIXES:
        if txt.startswith(p) or (len(txt) >= minlen and p.startswith(txt.rstrip())):
            return True
    return False


def _cands(row, c):
    return [(int(t), float(s)) for t, s in (row["cand_" + c] if isinstance(row["cand_" + c], (list, tuple)) else [])]


def _known_subtasks(r):
    S = set()
    for m in r["messages"]:
        if m.get("intent") and m["intent"] != "status":
            S.add(m["intent"])
    for a in r["artifacts"]:
        if a.get("subtask"):
            S.add(a["subtask"])
    return S


def _resolve(s, K):
    if not s or s in K:
        return s
    c = [k for k in K if k.startswith(s)]
    return c[0] if len(c) == 1 else s


def blink_overrides(r):
    W = collections.defaultdict(list)
    for s in r["shared_state"]:
        if s["key"] != "run_status":
            W[s["key"]].append(s)
    out = []
    for k, ws in W.items():
        ws = sorted(ws, key=lambda s: s["t"])
        for j, wj in enumerate(ws):
            if wj["op"] != "override":
                continue
            Y, v2 = wj["agent"], wj["value"]
            prev = [(w["agent"], w["value"]) for w in ws[:j] if w["agent"] != Y and w["value"] != v2]
            if any(w["agent"] == X and w["value"] == v1 for (X, v1) in prev for w in ws[j + 1:]):
                out.append((wj["t"], k))
    return sorted(out)


def rule_conflict(row, r):
    pt = int(row["pturn"])
    b = blink_overrides(r)
    if not b:
        return None
    if not any(abs(pt - t) <= HIT for t, _ in b):
        return b[0][0], "R1a"
    ws = sorted(r["shared_state"], key=lambda s: s["t"])
    at = [w for w in ws if w["t"] == pt and w["key"] != "run_status"]
    if at and at[0]["op"] == "override":
        w = at[0]
        if any(x["key"] == w["key"] and x["t"] < pt and x["agent"] == w["agent"] and x["value"] == w["value"] for x in ws):
            e = [t for t, k in b if k == w["key"] and t < pt - HIT]
            if e:
                return min(e), "R1b"
    return None


def rule_goal_drift(row, r):
    pt = int(row["pturn"])
    D = {a["by"] for a in r["artifacts"] if a.get("subtask") == "off_topic"}
    if not D:
        return None
    bym = {m["t"]: m for m in r["messages"]}
    dc = sorted(t for t, s in _cands(row, "goal_drift") if s >= SCORE_MIN and t in bym and bym[t]["from"] in D)
    dc_all = [t for t, s in _cands(row, "goal_drift") if t in bym and bym[t]["from"] in D]
    if (pt in bym and bym[pt]["from"] in D) or any(abs(t - pt) <= HIT for t in dc_all):
        return None
    if not dc:
        return None
    return dc[0], "R2"


def recovery_turns(msgs):
    by = {m["t"]: m for m in msgs}
    notes, assigns = set(), set()
    for m in msgs:
        if m["type"] != "handoff" or m.get("refs"):
            continue
        if _is_recov_text(m["text"]):
            notes.add(m["t"])
            continue
        p = by.get(m["t"] - 1)
        if p is not None and p["type"] == "handoff" and not p.get("refs") and p["from"] == m["from"] and p["to"] == m["to"] \
                and _is_recov_text(p["text"]):
            assigns.add(m["t"])
    return notes | assigns


def rule_duplicated(row, r):
    pt = int(row["pturn"])
    rec = recovery_turns(r["messages"])
    if pt not in rec:
        return None
    cc = [(t, s) for t, s in _cands(row, "duplicated_work") if t not in rec and abs(t - pt) > HIT and s >= SCORE_MIN]
    if not cc:
        return None
    return max(cc, key=lambda z: z[1])[0], "R3"


def _waited(m):
    mm = _WSUB.search(m["text"])
    if mm:
        for g in mm.groups()[1:]:
            if g:
                return g
    return None


def wait_pairs(r):
    K = _known_subtasks(r)
    info = []
    for m in r["messages"]:
        if m["type"] != "status":
            continue
        w = _waited(m)
        info.append((m["t"], m["from"], m["to"], m.get("intent") or None, _resolve(w, K) if w else None))
    out = []
    for i, (t1, a, b, i1, w1) in enumerate(info):
        for (t2, c, d, i2, w2) in info[i + 1:]:
            if c != b or d != a:
                continue
            ok, known = True, 0
            for x, y in ((w1, i2), (w2, i1)):
                if x and y:
                    known += 1
                    ok &= (x == y)
            if i1 and i2:
                ok &= (i1 != i2)
            if w1 and w2:
                ok &= (w1 != w2)
            if ok and known >= 1:
                out.append((t1, t2))
    return out


def rule_deadlock(row, r):
    pt = int(row["pturn"])
    P = wait_pairs(r)
    if not P:
        return None
    mine = [p for p in P if abs(p[0] - pt) <= HIT or abs(p[1] - pt) <= HIT]
    if mine:
        firsts = [p[0] for p in mine]
        if any(abs(f - pt) <= HIT for f in firsts):
            return None
        return min(firsts), "R4a"
    return min(p[0] for p in P), "R4b"


TURN_RULES = {"conflict": rule_conflict, "goal_drift": rule_goal_drift, "duplicated_work": rule_duplicated,
              "deadlock": rule_deadlock}


# =====================================================================================================================
# wiring
# =====================================================================================================================
def pair_groups(arb_info, n):
    """-> (grp, pair_a): 'ff' / 'cf' / 'none' per test run and the gated fault of cf rows (from arb_info['D_test'])."""
    grp, pair_a = ["none"] * n, [""] * n
    D = arb_info.get("D_test") if isinstance(arb_info, dict) else None
    if D is None or len(D) == 0:
        return grp, pair_a
    kind = D["kind"].fillna("ff").astype(str).values if "kind" in D.columns else np.full(len(D), "ff")
    for r, k, a in zip(D["run"].values.astype(int), kind, D["a"].values.astype(int)):
        if k == "cf":
            grp[r], pair_a[r] = "cf", FAULTS[a]
        else:
            grp[r] = "ff"
    return grp, pair_a


def top_candidates(Cte, n, k=5):
    """{fault: {run: [(t, score), ...]}}: the k best stage-1 candidates per run, best first, scores rounded to 4
    decimals (exactly the cand_<fault> lists of the exp60 residual tables)."""
    out = {}
    for c in FAULTS:
        D = Cte.get(c) if isinstance(Cte, dict) else None
        if D is None or len(D) == 0 or not {"run", "t", "score"} <= set(D.columns):
            out[c] = {}
            continue
        D = D[["run", "t", "score"]].sort_values(["run", "score"], ascending=[True, False])
        out[c] = {int(run): list(zip(g.t.astype(int).values[:k], g.score.values[:k].round(4)))
                  for run, g in D.groupby("run")}
    return out


def final_pass(pred, turns, P, T_nn, Cte, runs, arb_info, gate="all", turn_rules=True, log=None):
    """-> (pred, turns) after the GATE and TURNS steps (see module doc)."""
    pred = [str(x) for x in pred]
    turns = [int(x) for x in turns]
    pred_in, turns_in = list(pred), list(turns)
    n = len(pred)
    P = np.asarray(P, dtype=float).reshape(n, len(LABELS))
    T_nn = None if T_nn is None else np.asarray(T_nn).reshape(n, len(FAULTS))
    grp, pair_a = pair_groups(arb_info, n)
    cands = top_candidates(Cte, n)
    cnt = collections.Counter()
    if gate in ("cf", "all"):
        for i in range(n):
            try:
                if pred[i] == "clean" and (gate == "all" or grp[i] == "cf"):
                    sg = signatures(runs[i])
                    fired = sorted({SIG_CLASS[k] for k in ALL_SIGS if sg[k] > 0})
                    recovery = False
                    if not fired and sg["recovery"] > 0:
                        fired, recovery = list(FAULTS), True
                    if not fired:
                        continue
                    c = max(fired, key=lambda k: P[i, LABELS.index(k)])
                    t = int(T_nn[i, FAULTS.index(c)]) if T_nn is not None else -1
                    if not recovery and sg.get("t_" + c, -1) >= 0:
                        t = int(sg["t_" + c])
                    if t < 0:
                        cl = cands[c].get(i, [])
                        t = int(cl[0][0]) if cl else 0
                    pred[i], turns[i] = c, t
                    cnt["gate:%s%s" % ("recovery->" if recovery else "", c)] += 1
                elif grp[i] == "cf" and pred[i] == "duplicated_work" and pair_a[i] == "duplicated_work":
                    sg = signatures(runs[i])
                    if sg["multi_deliv"] == 0 and sg["duplicated_work"] == 0 and sg["n_known_deliv"] >= 2:
                        pred[i], turns[i] = "clean", -1
                        cnt["gate:rev_dup"] += 1
            except Exception:
                cnt["gate:error"] += 1
    if turn_rules:
        for i in range(n):
            c = pred[i]
            if c not in TURN_RULES:
                continue
            try:
                res = TURN_RULES[c]({"pturn": turns[i], "cand_" + c: cands[c].get(i, [])}, runs[i])
                if res is not None and int(res[0]) != turns[i]:
                    turns[i] = int(res[0])
                    cnt["turn:" + res[1]] += 1
            except Exception:
                cnt["turn:error"] += 1
    if log is not None:
        log("resid final pass (gate=%s, turn rules=%s): %s" % (gate, turn_rules, dict(sorted(cnt.items())) or "no change"))
        if os.environ.get("CR_RESID_DEBUG") == "1":
            for i in range(n):
                if (pred[i], turns[i]) != (pred_in[i], turns_in[i]):
                    log("   resid change run %d: %s@%d -> %s@%d" % (i, pred_in[i], turns_in[i], pred[i], turns[i]))
    return pred, turns
