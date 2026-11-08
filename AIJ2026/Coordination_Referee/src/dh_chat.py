"""Dropped-handoff "chat to the assigner" signature.

Generator finding (detection_side investigation, confirmed on both run halves):
after a real dropped handoff A -> B at turn t, the injector makes agents other than
A (mostly not B) send 1-3 CHAT messages addressed to A ("Understood.", "Sounds
good.", ...) at random later turns.  Chats to the orchestrator a1 per run are ~2.3
in DH runs versus 0.5-0.8 in every other class.  Background chats pick a recipient
roughly uniformly among the agents, so the background share addressed to any one
agent falls like 1 / n_agents (0.30 with 3-4 agents -> 0.11 with 8-11), while the
DH excess is roughly constant.  The ``*_xs`` / ``*_ratio`` columns therefore
normalise by agent count (expected background = chats / n_agents); the raw
research counts are kept alongside.

Two entry points:

``cand_features(run, m)``  -> dict, prefix ``dhc_``
    for one dropped_handoff candidate (``m`` = the assign message).  Add it to
    every DH candidate dict *before* stage-1 training so the DH scorer is retrained
    with it.

``features(runs, C)``      -> DataFrame, prefix ``dhchat_``, index 0..len(runs)-1
    run-level chat features plus features anchored at the best-scoring
    dropped_handoff candidate of ``C['dropped_handoff']`` (columns run, t, score).
    Anchored columns are NaN when the run has no DH candidate / the turn is invalid.

``features_run(run, best_t)`` -> dict, the per-run worker (for parallel callers).

Depends only on numpy / pandas / stdlib and ``parse``.  No labels are used.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from parse import CHAT

CAND_PREFIX = "dhc_"
RUN_PREFIX = "dhchat_"
_NAN = float("nan")


# ----------------------------------------------------------------------------- helpers
def _msgs(run) -> List[Dict[str, Any]]:
    ms = run.get("messages") if isinstance(run, dict) else None
    return [x for x in ms if isinstance(x, dict)] if isinstance(ms, list) else []


def _n_agents(run, ms) -> int:
    """Agent count: declared agents, else distinct speakers/recipients in the messages."""
    ag = run.get("agents") if isinstance(run, dict) else None
    n = len([a for a in ag if isinstance(a, dict)]) if isinstance(ag, list) else 0
    if n <= 0:
        seen = set()
        for x in ms:
            for k in ("from", "to"):
                v = x.get(k)
                if isinstance(v, str) and v:
                    seen.add(v)
        n = len(seen)
    return n


def _t(x, default=-1) -> int:
    try:
        return int(x.get("t", default))
    except (TypeError, ValueError):
        return default


def _pos(ms, t: int, m=None) -> Optional[int]:
    """List position of the message with turn t (normally t itself)."""
    if 0 <= t < len(ms) and (ms[t] is m or _t(ms[t]) == t):
        return t
    for i, x in enumerate(ms):
        if x is m or _t(x) == t:
            return i
    return None


def _is_chat(x) -> bool:
    return x.get("cat") == CHAT


def _h(v):
    """Hashable key for an agent id (ids are str on real data; lists etc. -> repr)."""
    return v if v is None or isinstance(v, (str, int, float)) else repr(v)


def _split(ms, t: int, m=None):
    """(before, after) message lists around turn t (list order, research semantics)."""
    p = _pos(ms, t, m)
    if p is not None:
        return ms[:p], ms[p + 1:], p
    return [x for x in ms if _t(x) < t], [x for x in ms if _t(x) > t], None


# ----------------------------------------------------------------------------- candidate
def cand_features(run: Dict[str, Any], m: Dict[str, Any]) -> Dict[str, float]:
    """Chat-to-assigner features for one dropped_handoff candidate (m = assign message).

    Raw columns reproduce work/agents/detection_side/dh_chat_feats.dh_chat_cand exactly;
    ``*_xs`` subtract the uniform-recipient expectation (chats / n_agents)."""
    P = CAND_PREFIX
    try:
        ms = _msgs(run)
        t = _t(m) if isinstance(m, dict) else -1
        A = m.get("from") if isinstance(m, dict) else None
        B = m.get("to") if isinstance(m, dict) else None
        before, after_all, p = _split(ms, t, m)
        hasA = A is not None
        after = [x for x in after_all if hasA and _is_chat(x) and x.get("to") == A]
        bef = [x for x in before if hasA and _is_chat(x) and x.get("to") == A]
        # unprompted = A did not address the chat's sender in the 2 preceding messages
        unpr = 0
        for x in after:
            j = _pos(ms, _t(x), x)
            win = ms[max(0, j - 2):j] if j is not None else [y for y in ms if _t(x) - 2 <= _t(y) < _t(x)]
            if not any(y.get("from") == A and y.get("to") == x.get("from") for y in win):
                unpr += 1
        ch_all = [x for x in ms if _is_chat(x)]
        run_toa1 = sum(1 for x in ch_all if x.get("to") == "a1")
        n_after_chat = sum(1 for x in after_all if _is_chat(x))
        n_bef_chat = sum(1 for x in before if _is_chat(x))
        after_nB = sum(1 for x in after if x.get("from") != B)
        na = _n_agents(run, ms)
        inv = 1.0 / na if na > 0 else _NAN
        # background chats that could reach A: chats not sent by A / non-B chats not sent by A
        n_after_notA = sum(1 for x in after_all if _is_chat(x) and x.get("from") != A)
        n_after_notAB = sum(1 for x in after_all if _is_chat(x) and x.get("from") not in (A, B))
        inv1 = 1.0 / (na - 1) if na > 1 else _NAN
        return {
            P + "chA_after": len(after),
            P + "chA_before": len(bef),
            P + "chA_after_B": sum(1 for x in after if x.get("from") == B),
            P + "chA_after_nB": after_nB,
            P + "chA_after_senders": len({_h(x.get("from")) for x in after}),
            P + "chA_first_dt": (_t(after[0]) - t) if after else -1,
            P + "chA_after_unpr": unpr,
            P + "run_toa1": run_toa1,
            # agent-count normalised
            P + "chA_after_xs": len(after) - n_after_chat * inv,
            P + "chA_after_nB_xs": after_nB - n_after_notAB * inv1,
            P + "chA_after_ratio": len(after) / (n_after_notA * inv1) if n_after_notA and na > 1 else 0.0,
            P + "chA_before_xs": len(bef) - n_bef_chat * inv,
            P + "run_toa1_xs": run_toa1 - len(ch_all) * inv,
        }
    except Exception:                                   # never crash on an odd run
        return {k: _NAN for k in CAND_COLUMNS}


CAND_COLUMNS = [CAND_PREFIX + k for k in (
    "chA_after", "chA_before", "chA_after_B", "chA_after_nB", "chA_after_senders",
    "chA_first_dt", "chA_after_unpr", "run_toa1", "chA_after_xs", "chA_after_nB_xs",
    "chA_after_ratio", "chA_before_xs", "run_toa1_xs")]


# ----------------------------------------------------------------------------- run level
RUN_COLUMNS = [RUN_PREFIX + k for k in (
    "ch_to_a1", "ch_to_a1_share", "ch_not_a1", "ch_to_max",
    "dhb_chA_after", "dhb_chA_before", "dhb_chBA_after", "dhb_chA_rate_after",
    "ch_to_a1_xs", "ch_to_a1_ratio", "ch_to_max_ratio",
    "dhb_chA_after_nB", "dhb_chA_after_xs", "dhb_chA_after_nB_xs")]


def features_run(run: Dict[str, Any], best_t=None) -> Dict[str, float]:
    """Run-level chat features; ``best_t`` = turn of the best DH candidate (or None).

    The first eight columns reproduce work/agents/detection_side/chatfeat.chat_feats."""
    P = RUN_PREFIX
    out = {k: _NAN for k in RUN_COLUMNS}
    try:
        ms = _msgs(run)
        ch = [x for x in ms if _is_chat(x)]
        na = _n_agents(run, ms)
        inv = 1.0 / na if na > 0 else _NAN
        inv1 = 1.0 / (na - 1) if na > 1 else _NAN
        to_a1 = sum(1 for x in ch if x.get("to") == "a1")
        c = Counter(_h(x.get("to")) for x in ch)
        out[P + "ch_to_a1"] = to_a1
        out[P + "ch_to_a1_share"] = to_a1 / max(len(ch), 1)
        out[P + "ch_not_a1"] = len(ch) - to_a1
        out[P + "ch_to_max"] = max(c.values(), default=0)
        out[P + "ch_to_a1_xs"] = to_a1 - len(ch) * inv
        out[P + "ch_to_a1_ratio"] = to_a1 / max(len(ch), 1) * na if na > 0 else _NAN
        out[P + "ch_to_max_ratio"] = (max(c.values(), default=0) / max(len(ch), 1)) * na if na > 0 else _NAN
        try:                                            # None / NaN / inf / junk -> no anchor
            t = int(best_t)
        except (TypeError, ValueError, OverflowError):
            return out
        if t < 0:
            return out
        p = _pos(ms, t)                                 # message whose 't' field is t (normally ms[t])
        if p is None:
            if t >= len(ms):
                return out
            p = t
        m = ms[p]
        A, B = m.get("from"), m.get("to")
        if A is None:                                   # no assigner id -> nothing to anchor on
            return out
        after = [x for x in ch if _t(x) > t and x.get("to") == A]
        nB = sum(1 for x in after if x.get("from") != B)
        out[P + "dhb_chA_after"] = len(after)
        out[P + "dhb_chA_before"] = sum(1 for x in ch if 0 <= _t(x) < t and x.get("to") == A)
        out[P + "dhb_chBA_after"] = sum(1 for x in after if x.get("from") == B)
        out[P + "dhb_chA_rate_after"] = len(after) / max(len(ms) - t - 1, 1)
        out[P + "dhb_chA_after_nB"] = nB
        n_after = sum(1 for x in ch if _t(x) > t)
        n_after_notAB = sum(1 for x in ch if _t(x) > t and x.get("from") not in (A, B))
        out[P + "dhb_chA_after_xs"] = len(after) - n_after * inv
        out[P + "dhb_chA_after_nB_xs"] = nB - n_after_notAB * inv1
    except Exception:
        pass
    return out


def best_dh_turns(C, n: Optional[int] = None) -> Dict[int, int]:
    """run -> t of the highest-scoring dropped_handoff candidate (first on ties)."""
    D = C.get("dropped_handoff") if isinstance(C, dict) else None
    if D is None or len(D) == 0 or not {"run", "t", "score"} <= set(D.columns):
        return {}
    D = D[["run", "t", "score"]].dropna(subset=["run", "t"])
    D = D.assign(score=pd.to_numeric(D.score, errors="coerce").astype(float).fillna(-np.inf)).reset_index(drop=True)
    if len(D) == 0:
        return {}
    idx = D.groupby("run", sort=False).score.idxmax()
    b = D.loc[idx.values]
    out = {}
    for r, t in zip(b.run.values, b.t.values):
        try:
            out[int(r)] = int(t)
        except (TypeError, ValueError):
            continue
    return out


def features(runs, C) -> pd.DataFrame:
    """Run-level chat features for every run (index 0..len(runs)-1)."""
    best = best_dh_turns(C)
    rows = [features_run(r, best.get(i)) for i, r in enumerate(runs)]
    return pd.DataFrame(rows, index=pd.RangeIndex(len(runs)), columns=RUN_COLUMNS).astype(float)
