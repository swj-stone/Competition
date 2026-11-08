"""Run-level features + per-class fault candidates.

``run_features(run)`` returns ``(features: dict, cands: dict[label -> list[dict]])``.
Each candidate is a message index plus a small feature vector; the per-class
localiser ranks them to predict ``fault_turn``.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List

from parse import (ACK, ART_DOMAIN, ASSIGN, CAT_NAMES, CHAT, DELIVER, DISAGREE,
                   DRIFT_HARD, DRIFT_SOFT, FINAL, ITER, OTHER, RECOVER, SET, STOP,
                   SUB2DOM, WAIT, WORK, DOMAINS, drift_topic)

TOPOS = ["star", "pipeline", "mesh", "hierarchical", "blackboard"]
# per-candidate hidden-plan slot features (cand_slots.py, columns "sl_*"; used only by the
# arbiter's slot-aware genuineness detector g2s, see arbiter.py); bump CAND_SLOTS_TAG when
# cand_slots.py changes (the fast evaluator caches featurization per feats.py content)
USE_CAND_SLOTS = True
CAND_SLOTS_TAG = "v1"
NOVEL_ROLES = {"coordinator", "evaluator", "auditor", "drafter", "scout", "checker",
               "operator"}


def _sd(a, b):
    return float(a) / b if b else 0.0


def _entropy(counts):
    tot = sum(counts)
    if tot <= 0:
        return 0.0
    return -sum(c / tot * math.log(c / tot) for c in counts if c > 0)


def run_features(run: Dict[str, Any]):
    msgs: List[Dict] = run["messages"]
    arts = [a for a in run["artifacts"] if isinstance(a, dict)]
    state = [s for s in run["shared_state"] if isinstance(s, dict)]
    agents = [a for a in run["agents"] if isinstance(a, dict)]
    n = len(msgs)
    N = max(n, 1)
    dom = run["domain"]
    f: Dict[str, float] = {}
    cands: Dict[str, List[Dict]] = {}

    # ------------------------------------------------------------ context
    f["n_msgs"] = n
    f["n_agents"] = len(agents)
    f["n_arts"] = len(arts)
    f["n_state"] = len(state)
    topo = run["topology"].get("type", "")
    for t in TOPOS:
        f["topo_" + t] = float(topo == t)
    roles = [a.get("role", "") for a in agents]
    f["novel_role_share"] = _sd(sum(r in NOVEL_ROLES for r in roles), len(roles))
    for d in DOMAINS:
        f["dom_" + d] = float(dom == d)
    f["intent_blank_share"] = _sd(sum(1 for m in msgs if not m.get("intent")), N)
    f["refs_share_deliver"] = _sd(sum(1 for m in msgs if m["cat"] == DELIVER and m["refs"]),
                                  sum(1 for m in msgs if m["cat"] == DELIVER))
    f["short_text_share"] = _sd(sum(1 for m in msgs if len(m["text"]) <= 30), N)
    f["art_blank_sub_share"] = _sd(sum(1 for a in arts if not a.get("subtask")), len(arts))

    cc = Counter(m["cat"] for m in msgs)
    for i, name in enumerate(CAT_NAMES):
        f["c_" + name] = cc.get(i, 0)
        f["s_" + name] = _sd(cc.get(i, 0), N)

    # position helpers
    orch = agents[0]["id"] if agents else "a1"
    for a in agents:
        if a.get("role") in ("orchestrator", "coordinator"):
            orch = a["id"]
            break

    art_by_id = {a.get("id"): a for a in arts}
    # deliveries: (t, agent, sub, real)
    deliveries = []
    for m in msgs:
        if m["cat"] == DELIVER:
            sub = m["sub"]
            real = bool(m["refs"]) or ("art_x" not in m["text"])
            if not sub:
                for r in m["refs"]:
                    a = art_by_id.get(r)
                    if a and a.get("subtask"):
                        sub = a["subtask"]
            deliveries.append((m["t"], m.get("from"), sub, real, "art_x" in m["text"]))
    art_deliv = [(a.get("t", 0), a.get("by"), a.get("subtask", ""), a.get("status"))
                 for a in arts if a.get("subtask") not in ("final_delivery", "off_topic")]

    def delivered_by(agent, sub, after_t, fake_ok=True):
        for (t, ag, s, real, fake) in deliveries:
            if t > after_t and ag == agent and s == sub and (fake_ok or not fake):
                return t
        for (t, ag, s, st) in art_deliv:
            if t > after_t and ag == agent and s == sub:
                return t
        return None

    def delivered_any(sub, after_t):
        for (t, ag, s, real, fake) in deliveries:
            if t > after_t and s == sub:
                return t
        for (t, ag, s, st) in art_deliv:
            if t > after_t and s == sub:
                return t
        return None

    # =========================================================== DROPPED
    assigns = [m for m in msgs if m["cat"] == ASSIGN]
    f["n_assign"] = len(assigns)
    f["n_deliver"] = len(deliveries)
    f["n_deliver_fake"] = sum(1 for d in deliveries if d[4])
    und_to = und_any = und_acked = und_silent = und_unknown = 0
    first_und = -1
    dcands = []
    for m in assigns:
        t, to, sub = m["t"], m.get("to"), m["sub"]
        later_from_to = [x for x in msgs if x["t"] > t and x.get("from") == to]
        acked = any(x["cat"] in (ACK, WORK) and (x["sub"] == sub or not x["sub"]) for x in later_from_to)
        worked = sum(1 for x in later_from_to if x["cat"] == WORK and x["sub"] == sub)
        d_to = delivered_by(to, sub, t) if sub else None
        d_to_real = delivered_by(to, sub, t, fake_ok=False) if sub else None
        d_any = delivered_any(sub, t) if sub else None
        d_to_anysub = any(dd[0] > t and dd[1] == to for dd in deliveries) or \
            any(a[0] > t and a[1] == to for a in art_deliv)
        prior_del = any(dd[0] < t and dd[2] == sub for dd in deliveries) if sub else False
        reassigned_later = any(x["t"] > t and x["sub"] == sub for x in assigns) if sub else False
        n_assign_sub = sum(1 for x in assigns if x["sub"] == sub) if sub else 0
        recov = any(x["cat"] == RECOVER and x["sub"] == sub for x in msgs) if sub else False
        silent = len(later_from_to) == 0
        unknown = not sub
        if unknown:
            und_unknown += int(not d_to_anysub)
        else:
            if d_to is None:
                und_to += 1
                if first_und < 0:
                    first_und = t
                if acked:
                    und_acked += 1
                if silent:
                    und_silent += 1
            if d_any is None:
                und_any += 1
        # chain view: next assignment of the same subtask, responses before it
        same_to_later = [x for x in assigns if x["t"] > t and sub and x["sub"] == sub
                         and x.get("to") == to]
        same_to_earlier = [x for x in assigns if x["t"] < t and sub and x["sub"] == sub
                           and x.get("to") == to]
        other_to_later = [x for x in assigns if x["t"] > t and sub and x["sub"] == sub
                          and x.get("to") != to]
        other_to_earlier = [x for x in assigns if x["t"] < t and sub and x["sub"] == sub
                            and x.get("to") != to]
        t_next = min([x["t"] for x in same_to_later + other_to_later], default=10 ** 6)
        resp = [x for x in later_from_to if x["t"] < t_next and (x["sub"] == sub or not x["sub"])]
        first_resp = next((x for x in later_from_to if x["sub"] == sub), None)
        real_del_before_next = any(dd[0] > t and dd[0] < t_next and dd[1] == to and dd[2] == sub
                                   and not dd[4] for dd in deliveries) or \
            any(a[0] > t and a[0] < t_next and a[1] == to and a[2] == sub for a in art_deliv)
        fake_del = any(dd[0] > t and dd[1] == to and dd[2] == sub and dd[4] for dd in deliveries)
        art_by_to = any(a[0] > t and a[1] == to and a[2] == sub for a in art_deliv)
        art_any = any(a[0] > t and a[2] == sub for a in art_deliv)
        dcands.append({
            "same_to_later": len(same_to_later), "same_to_earlier": len(same_to_earlier),
            "other_to_later": len(other_to_later), "other_to_earlier": len(other_to_earlier),
            "gap_next_assign": (t_next - t) if t_next < 10 ** 6 else -1,
            "resp_before_next": len(resp),
            "resp_cats_before_next": sum(1 << x["cat"] for x in {x["cat"]: x for x in resp}.values()),
            "first_resp_cat": first_resp["cat"] if first_resp else -1,
            "first_resp_delay": (first_resp["t"] - t) if first_resp else -1,
            "real_del_before_next": float(real_del_before_next), "fake_del": float(fake_del),
            "art_by_to": float(art_by_to), "art_any": float(art_any),
            "t": t, "rel_t": _sd(t, N), "unknown": float(unknown),
            "undeliv_to": float(d_to is None), "undeliv_to_real": float(d_to_real is None),
            "undeliv_any": float(d_any is None), "acked": float(acked), "worked": worked,
            "silent": float(silent), "n_later_from_to": len(later_from_to),
            "to_deliv_anysub": float(d_to_anysub), "prior_del": float(prior_del),
            "reassigned_later": float(reassigned_later), "n_assign_sub": n_assign_sub,
            "recov": float(recov), "from_orch": float(m.get("from") == orch),
            "sub_in_domain": float(SUB2DOM.get(sub, "") == dom),
        })
    f["undeliv_to"] = und_to
    f["undeliv_any"] = und_any
    f["undeliv_acked"] = und_acked
    f["undeliv_silent"] = und_silent
    f["undeliv_unknown"] = und_unknown
    f["undeliv_ratio"] = _sd(und_to, len(assigns))
    f["undeliv_first_rel"] = _sd(first_und, N) if first_und >= 0 else -1
    f["undeliv_to_orch"] = sum(1 for c in dcands if c["undeliv_to"] and c["from_orch"]
                               and not c["unknown"])
    f["undeliv_nonrecov"] = sum(1 for c in dcands if c["undeliv_to"] and not c["recov"]
                                and not c["unknown"] and not c["reassigned_later"])
    f["undeliv_any_nonrecov"] = sum(1 for c in dcands if c["undeliv_any"] and not c["recov"]
                                    and not c["unknown"])
    cands["dropped_handoff"] = dcands

    # =========================================================== DUPLICATED
    by_sub = defaultdict(list)
    for m in assigns:
        if m["sub"]:
            by_sub[m["sub"]].append(m)
    dup_c = []
    n_dup_assign = n_dup_both = n_dup_diff = 0
    for sub, ms in by_sub.items():
        for k, m in enumerate(ms):
            if k == 0:
                continue
            prev = ms[:k]
            t, to = m["t"], m.get("to")
            prev_to = {p.get("to") for p in prev}
            prev_delivered = any(delivered_by(p.get("to"), sub, p["t"]) is not None for p in prev)
            this_delivered = delivered_by(to, sub, t) is not None
            prev_deliv_before = any(dd[0] < t and dd[2] == sub for dd in deliveries)
            recov_before = any(x["cat"] == RECOVER and x["sub"] == sub and x["t"] <= t + 1
                               for x in msgs)
            recov_any = any(x["cat"] == RECOVER and x["sub"] == sub for x in msgs)
            n_dup_assign += 1
            n_dup_both += int(prev_delivered and this_delivered)
            n_dup_diff += int(to not in prev_to)
            dup_c.append({
                "t": t, "rel_t": _sd(t, N), "k": k, "same_to": float(to in prev_to),
                "prev_delivered": float(prev_delivered), "this_delivered": float(this_delivered),
                "prev_deliv_before": float(prev_deliv_before), "recov_before": float(recov_before),
                "recov_any": float(recov_any), "from_orch": float(m.get("from") == orch),
                "prev_from_orch": float(prev[0].get("from") == orch),
                "gap": t - prev[-1]["t"],
            })
    cands["duplicated_work"] = dup_c
    f["dup_assign"] = n_dup_assign
    f["dup_assign_both"] = n_dup_both
    f["dup_assign_diff"] = n_dup_diff
    f["dup_assign_orch"] = sum(1 for c in dup_c if c["from_orch"] and c["prev_from_orch"])
    f["dup_assign_both_norecov"] = sum(1 for c in dup_c if c["prev_delivered"] and
                                       c["this_delivered"] and not c["recov_before"])
    # delivery side
    deliverers = defaultdict(set)
    for (t, ag, s, real, fake) in deliveries:
        if s:
            deliverers[s].add(ag)
    for (t, ag, s, st) in art_deliv:
        if s and st != "copy":
            deliverers[s].add(ag)
    f["dup_sub_multi_deliverer"] = sum(1 for v in deliverers.values() if len(v) > 1)
    f["dup_max_deliverers"] = max((len(v) for v in deliverers.values()), default=0)
    noncopy = [a for a in arts if a.get("status") != "copy"
               and a.get("subtask") not in ("final_delivery", "off_topic")]
    hc = Counter(a.get("hash") for a in noncopy)
    f["dup_hash_noncopy"] = sum(v - 1 for v in hc.values() if v > 1)
    hc_all = Counter(a.get("hash") for a in arts)
    f["dup_hash_all"] = sum(v - 1 for v in hc_all.values() if v > 1)
    sc = Counter(a.get("subtask") for a in noncopy if a.get("subtask"))
    f["dup_sub_2arts"] = sum(1 for v in sc.values() if v > 1)
    by_sub_by = defaultdict(set)
    for a in noncopy:
        if a.get("subtask"):
            by_sub_by[a["subtask"]].add(a.get("by"))
    f["dup_sub_2arts_diffby"] = sum(1 for v in by_sub_by.values() if len(v) > 1)
    f["n_copy"] = sum(1 for a in arts if a.get("status") == "copy")
    f["n_partial"] = sum(1 for a in arts if a.get("status") == "partial")
    f["n_recover"] = cc.get(RECOVER, 0)
    f["recover_first_rel"] = next((_sd(m["t"], N) for m in msgs if m["cat"] == RECOVER), -1)
    msg_deliv_cnt = Counter(d[2] for d in deliveries if d[2] and not d[4])
    f["dup_msg_deliv_max"] = max(msg_deliv_cnt.values(), default=0)

    # =========================================================== DEADLOCK
    waits = [m for m in msgs if m["cat"] == WAIT]
    wedges = defaultdict(list)       # (waiter, target) -> msgs
    for m in waits:
        tgt = m["tagent"] or m.get("to")
        m["_tgt"] = tgt
        wedges[(m.get("from"), tgt)].append(m)
    f["wait_intent_ne_text"] = sum(1 for m in waits if m.get("intent") and m["sub"]
                                   and m.get("intent") != m["sub"])
    # text-sub of wait msg: sub extracted from text
    from parse import text_subtask
    for m in waits:
        m["_tsub"] = text_subtask(m["text"])
        m["_own"] = m.get("intent") if m.get("intent") in SUB2DOM else ""
    pairs = []
    seen = set()
    for (a, b), ms in wedges.items():
        if (b, a) in wedges and (b, a) not in seen and a != b:
            seen.add((a, b))
            ms2 = wedges[(b, a)]
            allm = sorted(ms + ms2, key=lambda x: x["t"])
            subs_ab = {x["_tsub"] for x in ms if x["_tsub"]}
            subs_ba = {x["_tsub"] for x in ms2 if x["_tsub"]}
            diff = bool(subs_ab and subs_ba and not (subs_ab & subs_ba))
            same = bool(subs_ab & subs_ba)
            own_ne = sum(1 for x in allm if x["_own"] and x["_tsub"] and x["_own"] != x["_tsub"])
            t0 = allm[0]["t"]
            tl = allm[-1]["t"]
            # resolution: delivery by a or b of awaited subs after t0
            awaited = subs_ab | subs_ba
            res = [d for d in deliveries if d[0] > t0 and d[1] in (a, b)
                   and (d[2] in awaited or not d[2])]
            res_real = [d for d in res if not d[4]]
            waits_after_res = sum(1 for x in allm if res and x["t"] > res[0][0])
            arts_after = sum(1 for x in arts if x.get("t", 0) > t0
                             and x.get("subtask") not in ("final_delivery",))
            pairs.append({"t": t0, "n": len(allm), "diff": diff, "same": same, "own_ne": own_ne,
                          "span": tl - t0, "resolved": len(res), "resolved_real": len(res_real),
                          "waits_after_res": waits_after_res, "arts_after": arts_after,
                          "last_rel": _sd(tl, N), "first_rel": _sd(t0, N),
                          "msgs": allm})
    f["wait_n"] = len(waits)
    f["wait_recip_pairs"] = len(pairs)
    f["wait_recip_diff"] = sum(p["diff"] for p in pairs)
    f["wait_recip_same"] = sum(p["same"] for p in pairs)
    f["wait_recip_ownne"] = sum(p["own_ne"] for p in pairs)
    bp = max(pairs, key=lambda p: (p["diff"], p["own_ne"], p["n"]), default=None)
    for k in ("n", "span", "resolved", "resolved_real", "waits_after_res", "arts_after",
              "last_rel", "first_rel", "own_ne"):
        f["dl_best_" + k] = bp[k] if bp else -1
    f["dl_best_diff"] = float(bp["diff"]) if bp else -1
    f["wait_tail_share"] = _sd(sum(1 for m in msgs[-max(1, n // 5):] if m["cat"] == WAIT),
                               max(1, n // 5))
    f["wait_distinct_edges"] = len(wedges)
    last_art_t = max((a.get("t", 0) for a in arts if a.get("subtask") != "final_delivery"),
                     default=-1)
    f["last_art_rel"] = _sd(last_art_t, N)
    wc = []
    for m in waits:
        p = next((p for p in pairs if m in p["msgs"]), None)
        wc.append({"t": m["t"], "rel_t": _sd(m["t"], N), "in_pair": float(p is not None),
                   "pair_diff": float(p["diff"]) if p else 0.0,
                   "pair_first": float(p is not None and p["t"] == m["t"]),
                   "pair_n": p["n"] if p else 0, "own_ne": float(bool(m["_own"] and m["_tsub"]
                                                                 and m["_own"] != m["_tsub"])),
                   "pair_resolved": p["resolved"] if p else -1,
                   "pair_rank": (sorted(x["t"] for x in p["msgs"]).index(m["t"]) if p else -1)})
    cands["deadlock"] = wc

    # =========================================================== CONFLICT
    per_key = defaultdict(list)
    for s in sorted(state, key=lambda s: s.get("t", 0)):
        if s.get("key") != "run_status":
            per_key[s.get("key")].append(s)
    kstats = []
    for k, ss in per_key.items():
        ov = [s for s in ss if s.get("op") == "override"]
        agents_k = [s.get("agent") for s in ss]
        vals = [s.get("value") for s in ss]
        alt = sum(1 for i in range(2, len(vals)) if vals[i] == vals[i - 2] and vals[i] != vals[i - 1])
        ov_agents = {s.get("agent") for s in ov}
        agent_switch = sum(1 for i in range(1, len(agents_k)) if agents_k[i] != agents_k[i - 1])
        kstats.append({"key": k, "n": len(ss), "ov": len(ov), "ov_agents": len(ov_agents),
                       "agents": len(set(agents_k)), "vals": len(set(vals)), "alt": alt,
                       "switch": agent_switch, "t_first_ov": ov[0].get("t") if ov else -1})
    kb = max(kstats, key=lambda k: (k["ov"], k["alt"], k["n"]), default=None)
    for k in ("n", "ov", "ov_agents", "agents", "vals", "alt", "switch"):
        f["cf_best_" + k] = kb[k] if kb else 0
    f["cf_best_first_ov_rel"] = _sd(kb["t_first_ov"], N) if kb and kb["t_first_ov"] >= 0 else -1
    f["cf_keys_2ov"] = sum(1 for k in kstats if k["ov"] >= 2)
    f["cf_keys_1ov"] = sum(1 for k in kstats if k["ov"] >= 1)
    f["cf_total_ov"] = sum(k["ov"] for k in kstats)
    f["cf_alt_total"] = sum(k["alt"] for k in kstats)
    f["n_state_keys"] = len(kstats)
    # disagreement messages per key (map by t)
    st_by_t = defaultdict(list)
    for s in state:
        st_by_t[s.get("t")].append(s)
    dis_key = Counter()
    cc_ = []
    for m in msgs:
        if m["cat"] in (DISAGREE, SET):
            key = None
            for s in st_by_t.get(m["t"], []):
                key = s.get("key")
            m["_key"] = key
            if m["cat"] == DISAGREE:
                dis_key[key] += 1
    f["cf_dis_max_key"] = max(dis_key.values(), default=0)
    f["cf_dis_keys_2"] = sum(1 for v in dis_key.values() if v >= 2)
    kmap = {k["key"]: k for k in kstats}
    for m in msgs:
        if m["cat"] == DISAGREE:
            ks = kmap.get(m.get("_key"))
            prior_dis = sum(1 for x in msgs if x["cat"] == DISAGREE and x["t"] < m["t"]
                            and x.get("_key") == m.get("_key"))
            cc_.append({"t": m["t"], "rel_t": _sd(m["t"], N),
                        "key_ov": ks["ov"] if ks else -1, "key_n": ks["n"] if ks else -1,
                        "key_alt": ks["alt"] if ks else -1,
                        "key_dis": dis_key.get(m.get("_key"), 0),
                        "prior_dis": prior_dis,
                        "is_first_ov": float(bool(ks) and ks["t_first_ov"] == m["t"]),
                        "has_key": float(ks is not None)})
    cands["conflict"] = cc_

    # =========================================================== DRIFT
    drifts = [m for m in msgs if m["cat"] in (DRIFT_SOFT, DRIFT_HARD)]
    def topic_intent(m):
        it = m.get("intent") or ""
        return it and it != "status" and it not in SUB2DOM
    f["drift_n"] = len(drifts)
    f["drift_hard"] = sum(1 for m in drifts if m["cat"] == DRIFT_HARD)
    f["drift_topic_intent"] = sum(1 for m in drifts if topic_intent(m))
    f["drift_status_intent"] = sum(1 for m in drifts if m.get("intent") == "status")
    topics = Counter(drift_topic(m["text"]) for m in drifts)
    f["drift_topics"] = len([t for t in topics if t])
    f["drift_topic_max"] = max((v for k, v in topics.items() if k), default=0)
    offdom = [m for m in msgs if m["sub"] and SUB2DOM.get(m["sub"]) and SUB2DOM[m["sub"]] != dom]
    f["offdom_msgs"] = len(offdom)
    f["offdom_share"] = _sd(len(offdom), N)
    f["offdom_first_rel"] = _sd(offdom[0]["t"], N) if offdom else -1
    f["offdom_subs"] = len({m["sub"] for m in offdom})
    f["offdom_deliver"] = sum(1 for m in offdom if m["cat"] == DELIVER)
    f["offdom_assign"] = sum(1 for m in offdom if m["cat"] == ASSIGN)
    f["offdom_work"] = sum(1 for m in offdom if m["cat"] in (WORK, ACK, ITER))
    f["offtopic_arts"] = sum(1 for a in arts if a.get("subtask") == "off_topic")
    f["art_offdom"] = sum(1 for a in arts if ART_DOMAIN.get(a.get("type"), dom) != dom)
    fin = [a for a in arts if a.get("subtask") == "final_delivery"]
    f["final_art_offdom"] = float(any(ART_DOMAIN.get(a.get("type"), dom) != dom for a in fin))
    f["final_art_partial"] = float(any(a.get("status") == "partial" for a in fin))
    f["final_art_n"] = len(fin)
    # messages after first topic-intent drift
    dc = []
    for m in drifts:
        t = m["t"]
        after = [x for x in msgs if x["t"] > t]
        off_after = sum(1 for x in after if x in offdom)
        same_topic_after = sum(1 for x in drifts if x["t"] > t and
                               drift_topic(x["text"]) == drift_topic(m["text"]))
        dc.append({"t": t, "rel_t": _sd(t, N), "hard": float(m["cat"] == DRIFT_HARD),
                   "topic_intent": float(bool(topic_intent(m))),
                   "status_intent": float(m.get("intent") == "status"),
                   "blank_intent": float(not m.get("intent")),
                   "off_after": off_after, "off_before": len(offdom) - off_after,
                   "same_topic_after": same_topic_after,
                   "drift_before": sum(1 for x in drifts if x["t"] < t),
                   "topic_intent_before": sum(1 for x in drifts if x["t"] < t and topic_intent(x))})
    cands["goal_drift"] = dc
    f["drift_first_topic_rel"] = next((_sd(m["t"], N) for m in drifts if topic_intent(m)), -1)

    # =========================================================== LOOP
    iters = [m for m in msgs if m["cat"] == ITER]
    it_sub = defaultdict(list)
    for m in iters:
        it_sub[m["sub"] or "?"].append(m)
    assigned_subs = {m["sub"] for m in assigns}
    lstats = []
    for s, ms in it_sub.items():
        senders = {m.get("from") for m in ms}
        prs = {(m.get("from"), m.get("to")) for m in ms}
        t0, t1 = ms[0]["t"], ms[-1]["t"]
        between = [x for x in msgs if t0 <= x["t"] <= t1]
        deliv_after = s != "?" and delivered_any(s, t1) is not None
        deliv_during = s != "?" and any(t0 <= d[0] <= t1 and d[2] == s for d in deliveries)
        lstats.append({"sub": s, "n": len(ms), "senders": len(senders), "pairs": len(prs),
                       "t0": t0, "span": t1 - t0, "density": _sd(len(ms), len(between)),
                       "assigned": float(s in assigned_subs), "deliv_after": float(deliv_after),
                       "deliv_during": float(deliv_during),
                       "sub_same_dom": float(SUB2DOM.get(s, "") == dom)})
    lb = max(lstats, key=lambda x: (x["n"], x["senders"]), default=None)
    for k in ("n", "senders", "pairs", "span", "density", "assigned", "deliv_after",
              "deliv_during"):
        f["lp_best_" + k] = lb[k] if lb else -1
    f["lp_best_t0_rel"] = _sd(lb["t0"], N) if lb else -1
    f["lp_subs"] = len(lstats)
    f["lp_subs_multi_sender"] = sum(1 for x in lstats if x["senders"] >= 2)
    f["lp_multi_sender_n"] = max((x["n"] for x in lstats if x["senders"] >= 2), default=0)
    f["lp_multi_sender_t0_rel"] = min((_sd(x["t0"], N) for x in lstats if x["senders"] >= 2),
                                      default=-1)
    f["lp_unknown"] = len(it_sub.get("?", []))
    db_by_sub = Counter(m["sub"] for m in msgs if m["cat"] == WORK and m.get("tool") == "db_query" and m["sub"])
    f["lp_db_work_max"] = max((db_by_sub.get(s, 0) for s in it_sub), default=0)
    f["lp_db_work_any"] = max(db_by_sub.values(), default=0)
    f["work_notool"] = sum(1 for m in msgs if m["cat"] == WORK and not m.get("tool"))
    f["lp_total"] = len(iters)
    # exact normalised text repetition
    import re as _re
    norms = [_re.sub(r"\d+", "#", m["text"]) for m in msgs]
    nc = Counter(norms)
    f["norm_dup_max"] = max(nc.values(), default=0)
    streak = best = 0
    prev = None
    for m in msgs:
        key = (m["cat"], m["sub"]) if m["cat"] == ITER else None
        streak = streak + 1 if key is not None and key == prev else (1 if key else 0)
        best = max(best, streak)
        prev = key
    f["lp_consec"] = best
    lc = []
    lstat_by = {x["sub"]: x for x in lstats}
    for s, ms in it_sub.items():
        st = lstat_by[s]
        for k, m in enumerate(ms):
            lc.append({"t": m["t"], "rel_t": _sd(m["t"], N), "k": k, "sub_n": st["n"],
                       "remaining": st["n"] - k, "senders": st["senders"],
                       "is_first": float(k == 0), "gap_prev": (m["t"] - ms[k - 1]["t"]) if k else -1,
                       "gap_next": (ms[k + 1]["t"] - m["t"]) if k + 1 < len(ms) else -1,
                       "sender_changes_after": len({x.get("from") for x in ms[k:]}),
                       "is_best": float(lb is not None and s == lb["sub"]),
                       "unknown": float(s == "?")})
    cands["runaway_loop"] = lc

    # =========================================================== SUCCESS / END
    f["has_final"] = float(cc.get(FINAL, 0) > 0)
    f["has_stop"] = float(cc.get(STOP, 0) > 0)
    rs = [s.get("value") for s in state if s.get("key") == "run_status"]
    for v in ("completed", "timeout", "aborted", "incomplete"):
        f["rs_" + v] = float(v in rs)
    plan = {m["sub"] for m in assigns if m["sub"] and SUB2DOM.get(m["sub"]) == dom}
    art_subs = {a.get("subtask") for a in arts}
    del_subs = {d[2] for d in deliveries if not d[4]}
    f["plan_n"] = len(plan)
    f["plan_cov_art"] = _sd(len(plan & art_subs), len(plan))
    f["plan_cov_del"] = _sd(len(plan & (art_subs | del_subs)), len(plan))
    f["plan_missing"] = len(plan - (art_subs | del_subs))
    tail = msgs[-max(1, n // 5):]
    tc = Counter(m["cat"] for m in tail)
    for c in (WAIT, ITER, DELIVER, DRIFT_HARD, DRIFT_SOFT, DISAGREE):
        f["tail_" + CAT_NAMES[c]] = _sd(tc.get(c, 0), len(tail))
    f["last_cat"] = msgs[-1]["cat"] if msgs else -1
    f["n_senders"] = len({m.get("from") for m in msgs})
    active = {m.get("from") for m in msgs} | {m.get("to") for m in msgs}
    f["silent_agents"] = sum(1 for a in agents if a.get("id") not in active)
    f["sender_entropy"] = _entropy(list(Counter(m.get("from") for m in msgs).values()))
    cands = (candidate_context_fast if FAST_CTX else candidate_context)(run, cands)
    if USE_CAND_SLOTS:
        try:
            from cand_slots import cand_slot_features
            cand_slot_features(run, cands)
        except Exception:
            pass
    return f, cands


# ============================================================================
# generic per-candidate context features
# ============================================================================
CAND_CATS = {
    "dropped_handoff": (ASSIGN,),
    "duplicated_work": (ASSIGN,),
    "deadlock": (WAIT,),
    "conflict": (DISAGREE,),
    "goal_drift": (DRIFT_SOFT, DRIFT_HARD),
    "runaway_loop": (ITER,),
}
_CTX_CATS = [ASSIGN, DELIVER, ACK, WORK, ITER, WAIT, RECOVER, DISAGREE, SET, CHAT]


def candidate_context(run: Dict[str, Any], cands: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """Merge class-specific candidate dicts with generic context features."""
    msgs = run["messages"]
    n = len(msgs)
    N = max(n, 1)
    arts = [a for a in run["artifacts"] if isinstance(a, dict)]
    dom = run["domain"]
    by_t = {m["t"]: i for i, m in enumerate(msgs)}
    # prefix structures
    out: Dict[str, List[Dict]] = {}
    for lab, cats in CAND_CATS.items():
        spec = {c["t"]: c for c in cands.get(lab, [])}
        lst = []
        idxs = [i for i, m in enumerate(msgs) if m["cat"] in cats]
        same_cat_pos = {i: k for k, i in enumerate(idxs)}
        for i in idxs:
            m = msgs[i]
            t = m["t"]
            A = m.get("from")
            B = m.get("_tgt") or (m.get("to") if m["cat"] == ASSIGN else (m["tagent"] or m.get("to")))
            S = m["sub"]
            g = {"t": t, "rel_t": _sd(t, N), "to_end": n - 1 - i, "n": n,
                 "cat": m["cat"], "intent_blank": float(not m.get("intent")),
                 "intent_status": float(m.get("intent") == "status"),
                 "intent_topic": float(bool(m.get("intent")) and m.get("intent") != "status"
                                       and m.get("intent") not in SUB2DOM),
                 "sub_known": float(bool(S)), "sub_dom": float(SUB2DOM.get(S, "") == dom),
                 "from_a1": float(A == "a1"), "short": float(len(m["text"]) <= 30),
                 "same_cat_before": same_cat_pos[i], "same_cat_after": len(idxs) - same_cat_pos[i] - 1}
            before, after = msgs[:i], msgs[i + 1:]
            for c in _CTX_CATS:
                nm = CAT_NAMES[c]
                g["S_b_" + nm] = sum(1 for x in before if x["cat"] == c and S and x["sub"] == S)
                g["S_a_" + nm] = sum(1 for x in after if x["cat"] == c and S and x["sub"] == S)
                g["SB_a_" + nm] = sum(1 for x in after if x["cat"] == c and S and x["sub"] == S
                                      and x.get("from") == B)
                g["B_a_" + nm] = sum(1 for x in after if x["cat"] == c and x.get("from") == B)
                g["A_a_" + nm] = sum(1 for x in after if x["cat"] == c and x.get("from") == A)
            g["B_a_any"] = sum(1 for x in after if x.get("from") == B)
            g["to_a1"] = float(m.get("to") == "a1")
            g["SB_a_work_notool"] = sum(1 for x in after if x["cat"] == WORK and S and x["sub"] == S
                                        and x.get("from") == B and not x.get("tool"))
            g["S_a_work_notool"] = sum(1 for x in after if x["cat"] == WORK and S and x["sub"] == S
                                       and not x.get("tool"))
            g["S_work_db"] = sum(1 for x in msgs if x["cat"] == WORK and S and x["sub"] == S
                                 and x.get("tool") == "db_query")
            g["S_work_tool"] = sum(1 for x in msgs if x["cat"] == WORK and S and x["sub"] == S
                                   and x.get("tool"))
            g["S_work_to_a1"] = sum(1 for x in msgs if x["cat"] == WORK and S and x["sub"] == S
                                    and x.get("to") == "a1")
            g["S_iter_to_a1"] = sum(1 for x in msgs if x["cat"] == ITER and S and x["sub"] == S
                                    and x.get("to") == "a1")
            g["a1_wait_S_after"] = sum(1 for x in after if x["cat"] == WAIT and x.get("from") == "a1"
                                       and S and x["sub"] == S)
            g["B_a_substantive"] = sum(1 for x in after if x.get("from") == B
                                       and x["cat"] in (WORK, DELIVER, ITER, SET, DISAGREE))
            g["B_a_toA"] = sum(1 for x in after if x.get("from") == B and x.get("to") == A)
            g["S_a_fake"] = sum(1 for x in after if x["cat"] == DELIVER and S and x["sub"] == S
                                and "art_x" in x["text"])
            g["pair_same_b"] = sum(1 for x in before if x["cat"] == m["cat"] and x.get("from") == A
                                   and x.get("to") == m.get("to"))
            g["pair_same_a"] = sum(1 for x in after if x["cat"] == m["cat"] and x.get("from") == A
                                   and x.get("to") == m.get("to"))
            g["pair_rev_b"] = sum(1 for x in before if x["cat"] == m["cat"] and x.get("from") == m.get("to")
                                  and x.get("to") == A)
            g["pair_rev_a"] = sum(1 for x in after if x["cat"] == m["cat"] and x.get("from") == m.get("to")
                                  and x.get("to") == A)
            g["S_same_b"] = sum(1 for x in before if x["cat"] == m["cat"] and S and x["sub"] == S)
            g["S_same_a"] = sum(1 for x in after if x["cat"] == m["cat"] and S and x["sub"] == S)
            sa = [a for a in arts if S and a.get("subtask") == S]
            g["S_arts"] = len(sa)
            g["S_arts_after"] = sum(1 for a in sa if a.get("t", 0) > t)
            g["S_arts_B_after"] = sum(1 for a in sa if a.get("t", 0) > t and a.get("by") == B)
            g["S_arts_noncopy"] = sum(1 for a in sa if a.get("status") != "copy")
            g["S_arts_by"] = len({a.get("by") for a in sa if a.get("status") != "copy"})
            hs = Counter(a.get("hash") for a in sa)
            g["S_arts_samehash"] = max(hs.values(), default=0)
            g["arts_after"] = sum(1 for a in arts if a.get("t", 0) > t
                                  and a.get("subtask") not in ("final_delivery", "off_topic"))
            g["arts_before"] = sum(1 for a in arts if a.get("t", 0) <= t
                                   and a.get("subtask") not in ("final_delivery", "off_topic"))
            g["deliv_after"] = sum(1 for x in after if x["cat"] == DELIVER)
            g["offdom_after"] = sum(1 for x in after if x["sub"] and SUB2DOM.get(x["sub"], dom) != dom)
            g["offdom_before"] = sum(1 for x in before if x["sub"] and SUB2DOM.get(x["sub"], dom) != dom)
            # local window densities
            win = msgs[i + 1:i + 11]
            g["win_same_cat"] = sum(1 for x in win if x["cat"] == m["cat"])
            g["win_same_S"] = sum(1 for x in win if S and x["sub"] == S and x["cat"] == m["cat"])
            sp = spec.get(t, {})
            for k, v in sp.items():
                if k not in ("t", "rel_t"):
                    g["x_" + k] = v
            lst.append(g)
        out[lab] = lst
    return out


# Exact re-implementation of candidate_context with prefix counters (one pass over the messages instead
# of ~70 list scans per candidate); verified bit-identical on all train + test runs.
FAST_CTX = True
_SUBST = (WORK, DELIVER, ITER, SET, DISAGREE)


def _msg_keys(x, dom):
    cat, sub, frm, to = x["cat"], x["sub"], x.get("from"), x.get("to")
    tool = x.get("tool")
    ks = [("cs", cat, sub), ("csf", cat, sub, frm), ("cf", cat, frm), ("f", frm), ("ft", frm, to),
          ("cft", cat, frm, to)]
    if cat == WORK:
        if not tool:
            ks += [("wsfn", sub, frm), ("wsn", sub)]
        if tool == "db_query":
            ks.append(("wsdb", sub))
        if tool:
            ks.append(("wst", sub))
        if to == "a1":
            ks.append(("wsa1", sub))
    if cat == ITER and to == "a1":
        ks.append(("isa1", sub))
    if cat == WAIT and frm == "a1":
        ks.append(("a1w", sub))
    if cat in _SUBST:
        ks.append(("fsub", frm))
    if cat == DELIVER:
        ks.append(("d",))
        if "art_x" in x["text"]:
            ks.append(("fake", sub))
    if sub and SUB2DOM.get(sub, dom) != dom:
        ks.append(("od",))
    return ks


def candidate_context_fast(run: Dict[str, Any], cands: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    msgs = run["messages"]
    n = len(msgs)
    N = max(n, 1)
    arts = [a for a in run["artifacts"] if isinstance(a, dict)]
    dom = run["domain"]
    keys = [_msg_keys(x, dom) for x in msgs]
    tot = Counter()
    for ks in keys:
        tot.update(ks)
    want = set()
    idx_by_lab = {}
    for lab, cats in CAND_CATS.items():
        idxs = [i for i, m in enumerate(msgs) if m["cat"] in cats]
        idx_by_lab[lab] = idxs
        want.update(idxs)
    base = {}
    pre = Counter()
    for i in range(n):
        if i in want:
            base[i] = _ctx_base(msgs, i, pre, tot, keys[i], arts, dom)
        pre.update(keys[i])
    out: Dict[str, List[Dict]] = {}
    for lab, cats in CAND_CATS.items():
        spec = {c["t"]: c for c in cands.get(lab, [])}
        idxs = idx_by_lab[lab]
        lst = []
        for k, i in enumerate(idxs):
            m = msgs[i]
            t = m["t"]
            b0, b1 = base[i]
            g = {"t": t, "rel_t": _sd(t, N), "to_end": n - 1 - i, "n": n,
                 "cat": m["cat"], "intent_blank": float(not m.get("intent")),
                 "intent_status": float(m.get("intent") == "status"),
                 "intent_topic": float(bool(m.get("intent")) and m.get("intent") != "status"
                                       and m.get("intent") not in SUB2DOM),
                 "sub_known": float(bool(m["sub"])), "sub_dom": float(SUB2DOM.get(m["sub"], "") == dom),
                 "from_a1": float(m.get("from") == "a1"), "short": float(len(m["text"]) <= 30),
                 "same_cat_before": k, "same_cat_after": len(idxs) - k - 1}
            g.update(b0)
            g.update(b1)
            sp = spec.get(t, {})
            for kk, v in sp.items():
                if kk not in ("t", "rel_t"):
                    g["x_" + kk] = v
            lst.append(g)
        out[lab] = lst
    return out


def _ctx_base(msgs, i, pre, tot, own_keys, arts, dom):
    """Generic context counts of message i: (block before the window features, window block)."""
    m = msgs[i]
    t = m["t"]
    A = m.get("from")
    B = m.get("_tgt") or (m.get("to") if m["cat"] == ASSIGN else (m["tagent"] or m.get("to")))
    S = m["sub"]
    own = Counter(own_keys)

    def aft(key):
        return tot[key] - pre[key] - own[key]

    g = {}
    for c in _CTX_CATS:
        nm = CAT_NAMES[c]
        g["S_b_" + nm] = pre[("cs", c, S)] if S else 0
        g["S_a_" + nm] = aft(("cs", c, S)) if S else 0
        g["SB_a_" + nm] = aft(("csf", c, S, B)) if S else 0
        g["B_a_" + nm] = aft(("cf", c, B))
        g["A_a_" + nm] = aft(("cf", c, A))
    g["B_a_any"] = aft(("f", B))
    g["to_a1"] = float(m.get("to") == "a1")
    g["SB_a_work_notool"] = aft(("wsfn", S, B)) if S else 0
    g["S_a_work_notool"] = aft(("wsn", S)) if S else 0
    g["S_work_db"] = tot[("wsdb", S)] if S else 0
    g["S_work_tool"] = tot[("wst", S)] if S else 0
    g["S_work_to_a1"] = tot[("wsa1", S)] if S else 0
    g["S_iter_to_a1"] = tot[("isa1", S)] if S else 0
    g["a1_wait_S_after"] = aft(("a1w", S)) if S else 0
    g["B_a_substantive"] = aft(("fsub", B))
    g["B_a_toA"] = aft(("ft", B, A))
    g["S_a_fake"] = aft(("fake", S)) if S else 0
    mc, mt = m["cat"], m.get("to")
    g["pair_same_b"] = pre[("cft", mc, A, mt)]
    g["pair_same_a"] = aft(("cft", mc, A, mt))
    g["pair_rev_b"] = pre[("cft", mc, mt, A)]
    g["pair_rev_a"] = aft(("cft", mc, mt, A))
    g["S_same_b"] = pre[("cs", mc, S)] if S else 0
    g["S_same_a"] = aft(("cs", mc, S)) if S else 0
    sa = [a for a in arts if S and a.get("subtask") == S]
    g["S_arts"] = len(sa)
    g["S_arts_after"] = sum(1 for a in sa if a.get("t", 0) > t)
    g["S_arts_B_after"] = sum(1 for a in sa if a.get("t", 0) > t and a.get("by") == B)
    g["S_arts_noncopy"] = sum(1 for a in sa if a.get("status") != "copy")
    g["S_arts_by"] = len({a.get("by") for a in sa if a.get("status") != "copy"})
    hs = Counter(a.get("hash") for a in sa)
    g["S_arts_samehash"] = max(hs.values(), default=0)
    g["arts_after"] = sum(1 for a in arts if a.get("t", 0) > t
                          and a.get("subtask") not in ("final_delivery", "off_topic"))
    g["arts_before"] = sum(1 for a in arts if a.get("t", 0) <= t
                           and a.get("subtask") not in ("final_delivery", "off_topic"))
    g["deliv_after"] = aft(("d",))
    g["offdom_after"] = aft(("od",))
    g["offdom_before"] = pre[("od",)]
    win = msgs[i + 1:i + 11]
    w = {"win_same_cat": sum(1 for x in win if x["cat"] == mc),
         "win_same_S": sum(1 for x in win if S and x["sub"] == S and x["cat"] == mc)}
    return g, w
