"""Per-candidate hidden-plan slot features (prefix ``sl_``).

Where they are used (v5): NOT in the stage-1 candidate scorers nor in the run-level
models (pipeline._lgb ignores ``sl_`` columns) -- feeding them there was measured to be
neutral (fasteval +arbiter 0.9262 / 0.9266 vs 0.9263 / 0.9267): the run-level plan-slot
features of slot_* already carry the information, and the single-instance last-slot flags
are biased in two-anomaly runs.  Only the subset arbiter.GATE_SLOT_COLS (does the key agent
own a free plan slot, plan size) feeds the slot-aware genuineness detector g2s of the
arbiter's clean-vs-top-fault gate, where it removes ~10 of ~65 detection errors.

The plan reconstructions of slot_lastslot / slot_planlast do not depend on the
stage-1 scores, so they can be evaluated for EVERY candidate of every class at
featurization time (single-instance reconstruction: the candidate is the only
fault instance placed on the plan).  Generator facts used:

* plan of K slots owned round-robin by cyc = [a2..aN] (or [a2..aN, a1]);
* every fault instance occupies one slot of its key agent;
* the labelled fault never sits on the last slot K-1, a secondary anomaly can;
* a clean run often ends with an assigned-but-undelivered task on the last slot.

Features (prefix ``sl_``):
  lj_* / l5_*   greedy / DP joint reconstruction (slot_lastslot): last, after, cost, N
  v4_*          v4 reconstruction (slot_planlast): number of free slots of the key
                agent inside the delivered span (nint) and after it (ntrail), all
                candidate slots on K-1 (certain), K-1 among them (possible),
                distance of the latest / earliest candidate slot from K-1
  v6_*          v6 reconstruction: feasible, fraction of feasible assignments that
                put the instance on K-1 (last), #steps, #gaps, K
Uses only the parsed run (no labels, no scores).
"""

from __future__ import annotations

from typing import Any, Dict, List

NAN = float("nan")
VERSION = 1


def cand_slot_features(run: Dict[str, Any], cands: Dict[str, List[Dict]]) -> None:
    """Add ``sl_*`` features in place to every candidate dict of ``cands``
    ({cls: [cand, ...]}, each cand has key ``t`` = message index)."""
    import slot_lastslot as LS
    import slot_planlast as PL
    try:
        ms = run.get("messages") or []
        R = LS._Run(run)
        ags = PL._agent_ids(run)
        dom = PL._run_domain(run)
        own = PL._plan_owner_map(run, ags)
        B4 = PL._v4_base(run, ags, dom)
        B6 = PL._v6_base(run, ags, dom, own)
    except Exception:
        return
    for c, lst in cands.items():
        for cand in lst:
            t = cand.get("t")
            if not isinstance(t, int) or not (0 <= t < len(ms)):
                continue
            f = {}
            try:
                for dp, pre in ((False, "lj"), (True, "l5")):
                    r = R.inst_last_multi([(t, c)], dp)[0]
                    if r:
                        f[pre + "_last"] = r["last"]
                        f[pre + "_after"] = r["after"]
                        f[pre + "_cost"] = r["cost"]
                        f[pre + "_N"] = r["N"]
            except Exception:
                pass
            try:
                if B4 is not None:
                    K4 = PL._v4_K(B4, [PL._inst_sub(run, t, c)])
                    if c == "goal_drift":
                        idx = [n for n, a in enumerate(B4["seq"]) if a.get("subtask") == "off_topic"]
                        cs = [B4["slots"][idx[0]]] if idx else []
                    elif c == "duplicated_work":
                        S = ms[t].get("sub")
                        idx = [n for n, a in enumerate(B4["seq"]) if a.get("subtask") == S]
                        cs = [B4["slots"][idx[0]]] if idx else []
                    else:
                        o = PL._fault_owner(run, t, c)
                        cyc, L = B4["cyc"], B4["L"]
                        ci = [p for p in B4["interior"] if cyc[p % L] == o]
                        ct = [p for p in range(B4["last"] + 1, K4) if cyc[p % L] == o]
                        cs = ci + ct
                        f["v4_nint"] = len(ci)
                        f["v4_ntrail"] = len(ct)
                        f["v4_incyc"] = float(o in cyc)
                    f["v4_ncand"] = len(cs)
                    f["v4_K"] = K4
                    f["v4_ngap"] = len(B4["interior"])
                    f["v4_ntrail_tot"] = K4 - B4["last"] - 1
                    if cs:
                        f["v4_certain"] = float(all(x == K4 - 1 for x in cs))
                        f["v4_possible"] = float(K4 - 1 in cs)
                        f["v4_fromlast"] = K4 - 1 - max(cs)
                        f["v4_minfromlast"] = K4 - 1 - min(cs)
            except Exception:
                pass
            try:
                if B6 is not None:
                    steps = PL._v6_steps(run, B6, dom, own, [(t, c)])
                    K6 = B6["last"] + 1 + max(len(steps) - len(B6["interior"]), 0)
                    E, _ = PL._v6_last_fracs(B6, steps)
                    f["v6_feas"] = float(E is not None)
                    f["v6_K"] = K6
                    f["v6_nsteps"] = len(steps)
                    f["v6_ngap"] = len(B6["interior"])
                    v = PL._v6_plast(run, t, c, B6, K6, E)
                    if v == v:
                        f["v6_last"] = v
            except Exception:
                pass
            for k, v in f.items():
                cand["sl_" + k] = float(v)
