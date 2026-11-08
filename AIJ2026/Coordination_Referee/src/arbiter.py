"""Pair arbiter: decide between two genuine fault instances in one run.

~23 % of runs contain two strong anomalies of different classes (the labelled
fault plus an unlabelled secondary one).  The stack is ~61 % right on them when
the run failed.  The arbiter re-decides those "two-strong" runs with a small
logistic model on:

  * class-pair priors x predicted success (antisymmetric one-hot of the two
    classes, and the same one-hot times S),
  * log-odds differences of the two classes: stage-2 probabilities (mlo),
    no-context genuineness detectors (glo), stage-1 stack scores (slo), each
    also times S,
  * last-slot difference (dl = la - lb): the labelled fault is never on the
    last plan slot, a secondary anomaly can be;
  * probabilistic last-slot log-odds (dlp = log(1-pa) - log(1-pb), slot_plast
    posterior P(instance on the last slot), USE_PLAST).

"Genuineness" detectors are per-class candidate LightGBMs trained only on runs
labelled with that class or clean, WITHOUT run-context features (position,
counts before/after ...), so a secondary instance of class c scores as high as a
primary one.  A run is "two-strong" when >= 2 classes have genuineness > thr;
the two with the highest genuineness + P are arbitrated.

Port of work/agents/contrarian/{arbiter3.py, noctx.py} as applied in
work/agents/synthesis/combo.py (research: +0.0122 / +0.0148 macro-F1 on the base
stack, seeds 3 / 11).

v5 additions (track stage1_slots, flags USE_G2_PAIR / USE_DET_GATE):

  * contextual genuineness detectors g2n / g2s: same training population as the
    no-context detectors (runs labelled c or clean) but WITH the run-context
    candidate features; g2s also gets the per-candidate plan-slot features of
    cand_slots.py that say whether the key agent owns a free plan slot
    (GATE_SLOT_COLS; single-instance last-slot flags are left out on purpose);
  * pair feature s2lo = logit g2n(a) - logit g2n(b) (+ times S) in the ff arbiter;
  * clean-vs-top-fault gate: every run outside the pair table whose stage-2
    decision is between clean and its top fault f (P(clean) + P(f) > GATE_REGION)
    gets a 'cf' row; a logistic model on class one-hot x [logit P(clean) vs P(f),
    max g2s(f), max stage-1 score(f), max gscore(f)] re-decides clean vs f.
    It mainly fixes clean <-> dropped_handoff confusions (the slot features
    separate a dropped handoff whose assignee owns a free plan slot from a clean
    run's truncated last assignment).
  fasteval (5-fold OOF on train, +arbiter): macro 0.9263 -> 0.9285 (seed 3),
  0.9267 -> 0.9292 (seed 11); s2lo alone ~+0.0013, gate alone ~+0.0009 / +0.0014.

v5b addition (flag USE_JOINT_LAST): the JOINT last-slot reconstructions of the plan-slot
modules (slot_lastslot DP l5_{c}_last, slot_planlast v6 l6_{c}; the class's top stage-1
candidate placed on the plan together with the other strong candidates) enter the ff
arbiter as differences jl5 / jl6 (+ times S) and the clean-vs-top-fault gate as gz_l5.
fasteval +arbiter: macro 0.9294 / 0.9306, composite 0.9308 / 0.9300 (seeds 3 / 11;
v4 0.9263 / 0.9267, 0.9276 / 0.9256); on unused seeds 5 / 7 (same protocol) macro
+0.0015 / +0.0034, composite +0.0031 / +0.0035 over v4.

Pipeline use (all cross-fitted, no label of an evaluated fold is used):

    G_oof, gm = fit_noctx(Ctr, lab, ft, fold_of)          # stage-1 fold ids
    G_te = score_noctx(gm, Cte)
    Dtr = pair_table(runs_tr, Ctr, G_oof, P_oof, S_oof)   # stage-2 OOF P / S on train
    Dte = pair_table(runs_te, Cte, G_te, P_te, S_te)
    am = fit_arbiter(Dtr, lab)
    pred_te = apply_arbiter(pred_te, Dte, predict_arbiter(am, Dte))

or ``fit_apply(...)`` which does exactly that.  For a train-side OOF evaluation:
``crossfit_arbiter(Dtr, lab, fold_of)`` -> OOF pair probabilities.

Generic interface: ``features(runs, C)`` -> per-run last-slot flags of each
class's top candidate (prefix ``arb_``), ranked by ``gscore`` when present.

Depends on numpy / pandas / stdlib and parse.py; lightgbm and scikit-learn are
imported lazily by the fitting functions (both are in the platform image).
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from parse import ASSIGN, DRIFT_HARD, DRIFT_SOFT, ITER, RECOVER, WAIT

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock", "conflict",
          "goal_drift", "runaway_loop"]
FAULTS = LABELS[1:]
IN_PLAN = {"dropped_handoff", "conflict", "duplicated_work"}
PREFIX = "arb_"
THR = 0.5

# run-context candidate features excluded from the genuineness detectors
CONTEXT = frozenset({"offdom_after", "offdom_before", "arts_after", "arts_before", "deliv_after",
                     "n", "to_end", "same_cat_before", "same_cat_after", "t"})
NOT_FEATURES = frozenset({"run", "score", "gscore"})
NOCTX_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8,
                    subsample_freq=1, colsample_bytree=0.7, verbose=-1, random_state=0)
PAIR_FEATS = ("mlo", "glo", "slo")
SEED = 0                  # LightGBM seed of the genuineness detectors (set by the pipeline; 0 = v8)
USE_PLAST = True          # add the probabilistic last-slot log-odds (slot_plast) to the arbiter
PLAST_CLIP = 5.0


# ============================================================ last-slot flags
# Port of work/agents/streams/slotfeat.py::slot_features (identical logic).  This
# module uses its own copy by default (self-contained); it gives the same
# slot_last_* / slot_off_* / slot_L as solution/slot_streams.features_run on train
# and test (verified), which can still be passed explicitly via lastslot_fn.
def _tgt(m):
    return m.get("tagent") or (m.get("to") if m.get("to") not in ("user", "all", None) else "")


def _slot_agent(run, t, cls):
    ms = run["messages"]
    m = ms[t]
    if cls in ("dropped_handoff", "duplicated_work"):
        return _tgt(m)
    if cls == "conflict":
        state = run.get("shared_state") or []
        st = [s for s in state if isinstance(s, dict) and s.get("t") == t]
        if not st:
            return m["from"]
        k = st[0]["key"]
        blk = sorted([s for s in state if isinstance(s, dict) and s.get("key") == k],
                     key=lambda s: s.get("t", 0))
        return blk[0]["agent"]
    if cls == "deadlock":
        return m["from"]
    if cls == "runaway_loop":
        L = [x for x in ms if x["cat"] == ITER and x["sub"] == m["sub"]]
    else:
        L = [x for x in ms if x["cat"] in (DRIFT_SOFT, DRIFT_HARD)
             and x.get("intent") not in ("", "status", None)]
    L = L or [m]
    return list(dict.fromkeys(x["from"] for x in L))[0]


def _worker_cycle(run):
    ag = [a["id"] for a in (run.get("agents") or []) if isinstance(a, dict) and "id" in a]
    W = ag[1:]
    topo = run.get("topology")
    if isinstance(topo, dict) and topo.get("type") == "pipeline" and ag:
        W = W + [ag[0]]
    return W


def _align(run, W):
    arts = [a for a in (run.get("artifacts") or []) if isinstance(a, dict)
            and a.get("status") == "final" and a.get("subtask") != "final_delivery"]
    seen, seq = set(), []
    for a in arts:
        s = a.get("subtask")
        if s and s != "off_topic" and s in seen:
            continue
        if s:
            seen.add(s)
        seq.append(a)
    steps, k, n = [], 0, len(W)
    for a in seq:
        x = a.get("by")
        if x not in W:
            continue
        kk = k
        while W[kk % n] != x:
            kk += 1
        for j in range(k, kk):
            steps.append(dict(k=j, agent=W[j % n], art=None))
        steps.append(dict(k=kk, agent=x, art=a))
        k = kk + 1
    return steps


def _inst_step(run, t, cls, steps, W):
    S = run["messages"][t]["sub"]
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
    n = len(W)
    k = (steps[-1]["k"] + 1) if steps else 0
    while W[k % n] != X:
        k += 1
    return k


def _as_t(t) -> Optional[int]:
    try:
        f = float(t)
    except (TypeError, ValueError):
        return None
    return int(f) if math.isfinite(f) else None


def slot_last_run(run: Dict[str, Any], top: Dict[str, Tuple[Any, float]],
                  thr: float = THR) -> Dict[str, float]:
    """top = {cls: (t, score)} (missing class = no candidate).  Returns
    slot_last_{c} / slot_off_{c} (NaN when unmappable), slot_L, slot_npl."""
    out = {f"slot_{k}_{c}": np.nan for c in FAULTS for k in ("off", "last")}
    out.update(slot_L=np.nan, slot_npl=np.nan)
    try:
        ms = run.get("messages") or []
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
                pres = bool(float(v[1]) > thr)
            except (TypeError, ValueError):
                pres = False
            cand[c] = (_as_t(v[0]), pres)
        nextra = sum(1 for c in FAULTS if c not in IN_PLAN and cand[c][1])
        L = npl + nextra if any(cand[c][1] for c in IN_PLAN) else npl + max(1, nextra)
        for c in FAULTS:
            t = cand[c][0]
            if t is None or not 0 <= t < len(ms):
                continue
            try:
                k = _inst_step(run, t, c, steps, W)
            except Exception:
                k = None
            if k is not None:
                off = k - (L - 1)
                out[f"slot_off_{c}"] = float(np.clip(off, -5, 3))
                out[f"slot_last_{c}"] = float(off == 0)
        out["slot_L"] = float(L)
        out["slot_npl"] = float(npl)
    except Exception:
        pass
    return out


def default_lastslot_fn() -> Callable:
    """This module's own last-slot implementation.  It deliberately does not import
    solution/slot_streams: the output must not depend on whether, or in which
    version, another module is present (a changed signature there would otherwise
    be swallowed by the try/except in features()/pair_table and silently give
    NaN / 0 features).  Pass slot_streams.features_run as lastslot_fn to share it."""
    return slot_last_run


# ============================================================ small helpers
def _labels(labels) -> np.ndarray:
    return np.array(["" if (x is None or (isinstance(x, float) and x != x)) else str(x)
                     for x in labels], dtype=object)


def _turns(fault_turn) -> np.ndarray:
    return pd.to_numeric(pd.Series(list(fault_turn)), errors="coerce").fillna(-1).astype(int).values


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def top_table(C: Dict[str, pd.DataFrame], n: int, col: str = "score"):
    """(n, 6) max `col` per run and class (0 when no candidate) and
    {cls: {run: t}} of the arg-max candidate (first on ties)."""
    M = np.zeros((n, len(FAULTS)))
    top: Dict[str, Dict[int, int]] = {}
    for j, c in enumerate(FAULTS):
        D = (C or {}).get(c)
        top[c] = {}
        if D is None or len(D) == 0 or not {"run", "t", col} <= set(D.columns):
            continue
        D = D[["run", "t", col]].reset_index(drop=True)
        D = D.assign(run=pd.to_numeric(D.run, errors="coerce"),
                     **{col: pd.to_numeric(D[col], errors="coerce")})
        D = D[D[col].notna() & D.run.notna()]
        D = D[(D.run >= 0) & (D.run < n)]
        if len(D) == 0:
            continue
        B = D.loc[D.groupby("run", sort=True)[col].idxmax()]
        r = B.run.values.astype(int)
        M[r, j] = B[col].values.astype(float)
        top[c] = {int(i): t for i, t in zip(r, B.t.values)}
    return M, top


# ============================================================ (a) genuineness
# candidate-feature prefixes excluded from the no-context detectors: the per-candidate plan-slot
# features of cand_slots.py ("sl_") are run context (they only feed the g2s detector below)
SLOT_PREFIX = "sl_"
NOCTX_EXCLUDE_PREFIXES: Tuple[str, ...] = (SLOT_PREFIX,)

# Contextual own-class genuineness detectors ("g2"), trained like the no-context ones (only on
# runs labelled with the class or clean, so a secondary instance is never a negative) but WITH the
# run-context candidate features:
#   g2n  no plan-slot columns                    -> extra pair feature s2lo (USE_G2_PAIR)
#   g2s  + the plan-slot columns in GATE_SLOT_COLS -> clean-vs-top-fault detection gate (USE_DET_GATE)
# GATE_SLOT_COLS = the per-candidate slot features that describe whether the key agent owns a free
# plan slot / the plan size; the single-instance last-slot flags (lj_last, v4_certain, v6_last ...)
# are left out: they are biased in two-anomaly runs (the other instance's slot is ignored).
USE_G2_PAIR = True
USE_DET_GATE = True
GATE_SLOT_COLS = ("lj_cost", "lj_N", "l5_cost", "l5_N", "v4_nint", "v4_ntrail", "v4_ncand", "v4_K",
                  "v4_ngap", "v4_ntrail_tot", "v6_feas", "v6_K", "v6_nsteps", "v6_ngap")
G2_KINDS = ("g2n", "g2s")


def noctx_columns(D: pd.DataFrame) -> List[str]:
    return [x for x in D.columns if x not in NOT_FEATURES and x not in CONTEXT
            and x not in G2_KINDS
            and not str(x).startswith("score")
            and not (NOCTX_EXCLUDE_PREFIXES and str(x).startswith(NOCTX_EXCLUDE_PREFIXES))]


def g2_columns(D: pd.DataFrame, kind: str) -> List[str]:
    """g2n: every candidate feature except plan-slot columns (context included);
    g2s: g2n + the GATE_SLOT_COLS plan-slot columns present in D."""
    cols = [x for x in D.columns if x not in NOT_FEATURES and x not in G2_KINDS
            and not str(x).startswith("score") and not str(x).startswith(SLOT_PREFIX)]
    if kind == "g2s":
        keep = {SLOT_PREFIX + k for k in GATE_SLOT_COLS}
        cols += [x for x in D.columns if x in keep]
    return cols


def _g2_kinds() -> List[str]:
    out = []
    if USE_G2_PAIR:
        out.append("g2n")
    if USE_DET_GATE:
        out.append("g2s")
    return out


def _new_lgb(n_jobs, params):
    import lightgbm as lgb
    p = dict(NOCTX_PARAMS)
    p.update(params or {})
    p["n_jobs"] = n_jobs
    return lgb.LGBMClassifier(**p)


def _fit_detector(X, tgt, own, fo, n_jobs, params, fit_full, Xte=None):
    """OOF scores over the folds fo (None -> zeros) + optional model on all own rows.
    Xte given (and fo) -> also the test scores = mean of the fold models (cross-fit bagging)."""
    sc = np.zeros(len(X))
    te, n_te = (np.zeros(len(Xte)) if Xte is not None else None), 0
    if fo is not None:
        for k in np.unique(fo):
            m = fo != k
            mt = m & own
            if tgt[mt].sum() == 0 or tgt[mt].sum() == mt.sum():
                continue
            mdl_k = _new_lgb(n_jobs, params).fit(X[mt], tgt[mt])
            sc[~m] = mdl_k.predict_proba(X[~m])[:, 1]
            if te is not None and len(Xte):
                te += mdl_k.predict_proba(Xte)[:, 1]
                n_te += 1
    mdl = None
    if fit_full and 0 < tgt[own].sum() < own.sum():
        mdl = _new_lgb(n_jobs, params).fit(X[own], tgt[own])
    if te is not None:
        te = te / n_te if n_te else np.zeros(len(Xte))
        return sc, mdl, te
    return sc, mdl


def fit_noctx(C: Dict[str, pd.DataFrame], labels: Sequence, fault_turn: Sequence,
              fold_of: Optional[np.ndarray] = None, n_jobs: int = 2,
              params: Optional[dict] = None,
              feats: Optional[Dict[str, List[str]]] = None,
              fit_full: bool = True):
    """No-context own-class genuineness detectors (+ the contextual g2 detectors).

    C: {cls: candidate DataFrame with run, t and candidate features} (train rows,
    run = index into labels).  fold_of: fold id per run (stage-1 folds) -> OOF
    scores; None -> no OOF (models only).  Returns (G, models) with
    G = {cls: DataFrame(run, t, gscore[, g2n, g2s])} aligned to C[cls]'s rows (OOF) and
    models = {cls: (model or None, feature list)} fitted on all train rows
    (+ models["__g2__"] = {cls: {kind: (model or None, feature list)}})."""
    lab, ft = _labels(labels), _turns(fault_turn)
    G, models = {}, {}
    g2m: Dict[str, Dict[str, Tuple[Any, List[str]]]] = {}
    kinds = _g2_kinds()
    for c in FAULTS:
        D = (C or {}).get(c)
        if D is None or len(D) == 0:
            G[c] = pd.DataFrame({"run": [], "t": [], "gscore": [], **{k: [] for k in kinds}})
            models[c] = (None, [])
            g2m[c] = {k: (None, []) for k in kinds}
            continue
        cols = list(feats[c]) if feats and c in feats else noctx_columns(D)
        X = D.reindex(columns=cols).astype(float).fillna(-1.0)
        r = D.run.values.astype(int)
        tgt = ((lab[r] == c) & (D.t.values == ft[r])).astype(int)
        own = np.isin(lab[r], [c, "clean"])
        fo = np.asarray(fold_of)[r] if fold_of is not None else None
        sc, mdl = _fit_detector(X, tgt, own, fo, n_jobs, params, fit_full)
        G[c] = pd.DataFrame({"run": r, "t": D.t.values, "gscore": sc}, index=D.index)
        models[c] = (mdl, cols)
        g2m[c] = {}
        for kind in kinds:
            cols2 = g2_columns(D, kind)
            X2 = D.reindex(columns=cols2).astype(float).fillna(-1.0)
            sc2, mdl2 = _fit_detector(X2, tgt, own, fo, n_jobs, params, fit_full)
            G[c][kind] = sc2
            g2m[c][kind] = (mdl2, cols2)
    models["__g2__"] = g2m
    return G, models


PAR_WORKERS = 1           # concurrent detector fits (threads, 1 OpenMP thread each); set by the pipeline


def _pmap(fn, items, workers):
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(min(workers, len(items))) as ex:
        return list(ex.map(fn, items))


def fit_score_noctx(C: Dict[str, pd.DataFrame], labels: Sequence, fault_turn: Sequence,
                    fold_of: np.ndarray, Cte: Dict[str, pd.DataFrame], n_jobs: int = 2,
                    params: Optional[dict] = None,
                    feats: Optional[Dict[str, List[str]]] = None,
                    workers: Optional[int] = None, full: bool = False):
    """fit_noctx(fit_full=False) + score_noctx in one go, the test scores being the mean of the
    fold models (cross-fit bagging instead of an extra fit on all rows).  With workers > 1 the
    (class, detector, fold) fits run concurrently, one OpenMP thread each.  full=True: test scores from
    one model on all own rows instead (= fit_noctx + score_noctx, the v6 behaviour).  Returns (G_oof, G_te)."""
    workers = PAR_WORKERS if workers is None else workers
    lab, ft = _labels(labels), _turns(fault_turn)
    kinds = _g2_kinds()
    allk = ["gscore"] + kinds
    nj = n_jobs if workers <= 1 else 1
    G, Gt = {}, {}
    for c in FAULTS:          # one class at a time (memory); its (detector, fold) fits run concurrently
        D = (C or {}).get(c)
        T = (Cte or {}).get(c)
        nt = 0 if T is None else len(T)
        if D is None or len(D) == 0:
            G[c] = pd.DataFrame({"run": [], "t": [], "gscore": [], **{k: [] for k in kinds}})
            Gt[c] = pd.DataFrame({"run": [] if T is None else T.run.values.astype(int),
                                  "t": [] if T is None else T.t.values,
                                  "gscore": np.zeros(nt), **{k: np.zeros(nt) for k in kinds}},
                                 index=None if T is None else T.index)
            continue
        r = D.run.values.astype(int)
        tgt = ((lab[r] == c) & (D.t.values == ft[r])).astype(int)
        own = np.isin(lab[r], [c, "clean"])
        fo = np.asarray(fold_of)[r]
        # one matrix over the union of the detectors' columns (memory); each detector takes its
        # own columns in its own order -> the same data as a per-detector matrix
        kcols = {kind: ((list(feats[c]) if feats and c in feats else noctx_columns(D)) if kind == "gscore"
                        else g2_columns(D, kind)) for kind in allk}
        ucols = list(dict.fromkeys(x for kind in allk for x in kcols[kind]))
        pos = {x: j for j, x in enumerate(ucols)}
        kidx = {kind: np.array([pos[x] for x in kcols[kind]], dtype=int) for kind in allk}
        XU = D.reindex(columns=ucols).astype(float).fillna(-1.0).to_numpy(np.float64)
        XUt = (T.reindex(columns=ucols).astype(float).fillna(-1.0).to_numpy(np.float64) if nt
               else XU[:0])
        rows_all = np.arange(len(XU))
        folds = [int(k) for k in np.unique(fo)]
        tasks = [(kind, k) for kind in allk for k in folds]
        if full:
            tasks += [(kind, -1) for kind in allk]

        def _fit(task):
            kind, k = task
            ci = kidx[kind]
            if k < 0:
                if nt and 0 < tgt[own].sum() < own.sum():
                    return _new_lgb(nj, params).fit(XU[np.ix_(rows_all[own], ci)], tgt[own]).predict_proba(
                        XUt[:, ci])[:, 1]
                return None
            m = fo != k
            mt = m & own
            if tgt[mt].sum() == 0 or tgt[mt].sum() == mt.sum():
                return None
            mdl = _new_lgb(nj, params).fit(XU[np.ix_(rows_all[mt], ci)], tgt[mt])
            return (mdl.predict_proba(XU[np.ix_(rows_all[~m], ci)])[:, 1],
                    (mdl.predict_proba(XUt[:, ci])[:, 1] if nt else None))

        res = dict(zip(tasks, _pmap(_fit, tasks, workers)))
        G[c] = pd.DataFrame({"run": r, "t": D.t.values}, index=D.index)
        if nt:
            Gt[c] = pd.DataFrame({"run": T.run.values.astype(int), "t": T.t.values}, index=T.index)
        else:
            Gt[c] = pd.DataFrame({"run": [], "t": []})
        for kind in allk:
            sc = np.zeros(len(D))
            te = np.zeros(nt)
            n_te = 0
            for k in folds:
                rk = res.get((kind, k))
                if rk is None:
                    continue
                sc[fo == k] = rk[0]
                if rk[1] is not None and not full:
                    te += rk[1]
                    n_te += 1
            if full:
                rf = res.get((kind, -1))
                te, n_te = (rf, 1) if rf is not None else (np.zeros(nt), 1)
            G[c][kind] = sc
            Gt[c][kind] = te / n_te if n_te else np.zeros(nt)
        XU = XUt = res = None
    return G, Gt


DET_TEST_BAG = True       # genuineness detectors: test scores from the fold models (no full fits)


def score_noctx(models, C: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Score (test) candidates with fit_noctx models; missing feature columns -> -1."""
    G = {}
    g2m = models.get("__g2__", {}) if isinstance(models, dict) else {}
    for c in FAULTS:
        D = (C or {}).get(c)
        mdl, cols = models.get(c, (None, []))
        kinds = list(g2m.get(c, {}))
        if D is None or len(D) == 0:
            G[c] = pd.DataFrame({"run": [], "t": [], "gscore": [], **{k: [] for k in kinds}})
            continue
        G[c] = pd.DataFrame({"run": D.run.values.astype(int), "t": D.t.values,
                             "gscore": _score_with(mdl, cols, D, c, "noctx")}, index=D.index)
        for kind in kinds:
            m2, cols2 = g2m[c][kind]
            G[c][kind] = _score_with(m2, cols2, D, c, kind)
    return G


def _score_with(mdl, cols, D, c, what):
    if mdl is None:
        return np.zeros(len(D))
    miss = [x for x in cols if x not in D.columns]
    if miss:
        warnings.warn("arbiter.score_noctx[%s/%s]: %d train feature columns missing in the "
                      "scored candidates, filled with -1 (train/test mismatch): %s"
                      % (c, what, len(miss), miss[:10]))
    X = D.reindex(columns=cols).astype(float).fillna(-1.0)
    return mdl.predict_proba(X)[:, 1]


# ============================================================ (b) pair table
GATE_REGION = 0.95   # gate rows: runs outside the pair table with P(clean) + P(top fault) > this
GATE_C = 1.0
GATE_FEATS = ("gz_p", "gz_g2s", "gz_s", "gz_g")

# v5b: JOINT last-slot reconstructions of the plan-slot modules as extra arbiter inputs.
# The arbiter's own flag (la / lb, slot_last_run) is a single simple reconstruction; the
# slot modules place each class's top stage-1 candidate on the plan JOINTLY with the other
# strong candidates (slot_lastslot DP -> l5_{c}_last; slot_planlast v6 exact count over the
# feasible slot assignments -> l6_{c} = fraction putting the instance on the last slot).
# They are recomputed here from (runs, C) -- identical to the stage-2 columns of the same
# name -- so pair_table keeps its signature.
#   ff rows:  jl5 = l5(a) - l5(b), jl6 = l6(a) - l6(b)   (NaN -> 0; each also times S)
#   cf rows:  gz_l5 = l5 of the gated fault's top candidate (NaN -> -1): a genuine
#             dropped handoff is never on the last slot, a clean run's truncated
#             assignment often is.
# fasteval (+arbiter macro, seeds 3 / 11): v5 0.9285 / 0.9292 -> 0.9294 / 0.9306.
USE_JOINT_LAST = True

JOINT_FF = ("jl5", "jl6")
JOINT_GATE = ("gz_l5",)

# v7 (track residual_hunt): recovery-names-instance pair feature.  In a successful faulty run the
# "re-assigning X to recover" message names the labelled fault's SLOT subtask:
#   * dropped_handoff / duplicated_work / runaway_loop / deadlock: the subtask of the fault message;
#   * conflict: the planner subtask of the block owner that the conflict slot consumed (never delivered)
#     -- the disagree message itself often has no subtask;
#   * goal_drift: the drift slot's own subtask, which is never handed out: the recovery names a subtask
#     nobody was assigned / waited on / iterated on before it ("fresh": 77 % of recovered drift runs,
#     0 % of the recovered runs of every other class, train).
#   drec = rec(a) - rec(b) enters the ff arbiter as one more column.
# fasteval-equivalent OOF arbiter refit (seeds 3 / 11 / 23 / 37): macro +0.0021 / +0.0016 / +0.0016 /
# +0.0011, errors -18 / -13 / -13 / -10.
USE_REC_PAIR = True
REC_CONFLICT_SLOT = True    # conflict: also the consumed (undelivered) planner subtask of the block owner
REC_DEADLOCK_PAIR = False   # research switch (measured neutral), kept off
REC_DRIFT_FRESH = True      # goal_drift: recovery names a subtask never handed out before


def _conflict_subs(run, t, ms) -> set:
    """Subtasks of a conflict instance: the override block's messages, plus the planner subtasks the
    block owner (agent of the block's 'set' entry, list order) was assigned but never delivered -- the
    conflict slot consumes that subtask, and the recovery re-assigns it."""
    out = set()
    L = [x for x in (run.get("shared_state") or []) if isinstance(x, dict) and x.get("key") != "run_status"]
    j = next((k for k, x in enumerate(L) if x.get("t") == t), None)
    if j is None:
        return out
    key = L[j].get("key")
    k = j
    while k > 0 and L[k].get("op") == "override" and L[k - 1].get("key") == key:
        k -= 1
    owner = L[k].get("agent")
    e = j
    while e + 1 < len(L) and L[e + 1].get("key") == key and L[e + 1].get("op") == "override":
        e += 1
    for x in L[k:e + 1]:
        tt = _as_t(x.get("t"))
        if tt is not None and 0 <= tt < len(ms) and ms[tt].get("sub"):
            out.add(ms[tt]["sub"])
    delivered = {a.get("subtask") for a in (run.get("artifacts") or []) if isinstance(a, dict)
                 and a.get("by") == owner and a.get("status") != "copy"}
    for m in ms:
        if m.get("cat") == ASSIGN and (m.get("tagent") or m.get("to")) == owner and m.get("sub") \
                and m["sub"] not in delivered:
            out.add(m["sub"])
    return out


def _rec_names(run, t, cls) -> float:
    try:
        ms = run.get("messages") or []
        t = _as_t(t)
        if t is None or not 0 <= t < len(ms):
            return 0.0
        rec = {m.get("sub") for m in ms if m.get("cat") == RECOVER and m.get("sub")}
        if not rec:
            return 0.0
        m = ms[t]
        subs = {m.get("sub")} if m.get("sub") else set()
        if cls == "conflict" and REC_CONFLICT_SLOT:
            subs |= _conflict_subs(run, t, ms)
        if cls == "deadlock" and REC_DEADLOCK_PAIR:
            pair = {m.get("from"), m.get("tagent") or m.get("to")}
            subs |= {x.get("sub") for x in ms if x.get("cat") == WAIT and x.get("sub")
                     and {x.get("from"), x.get("tagent") or x.get("to")} == pair}
        if cls == "goal_drift":
            if not REC_DRIFT_FRESH:
                return 0.0
            # the drift slot's own subtask is never handed out: a recovery naming a subtask nobody
            # was assigned / waited on / iterated on before it
            t0 = min(x.get("t", 0) for x in ms if x.get("cat") == RECOVER)
            seen = {x.get("sub") for x in ms if x.get("t", 0) < t0 and x.get("sub")
                    and x.get("cat") in (ASSIGN, WAIT, ITER)}
            return float(bool(rec - seen))
        return float(bool(rec & subs))
    except Exception:
        return 0.0


REC_COLUMNS = ["rec_fresh", "rec_confl"]


def rec_features_run(run) -> Dict[str, float]:
    """Run-level recovery-target features for stage 2 (v7, track residual_hunt):
    rec_fresh -- the recovery names a subtask nobody was assigned / waited on / iterated on before it (the
                 goal-drift slot's own subtask is never handed out: 8.8 % of drift runs, 0 % of the others);
    rec_confl -- the recovery names a subtask of a conflict block (its messages or the block owner's consumed,
                 undelivered planner subtask): 30.6 % of conflict runs, <= 7 % of the others."""
    d = {c: 0.0 for c in REC_COLUMNS}
    try:
        ms = run.get("messages") or []
        recs = [m for m in ms if m.get("cat") == RECOVER]
        if not recs:
            return d
        rs = {m.get("sub") for m in recs if m.get("sub")}
        t0 = min(m.get("t", 0) for m in recs)
        seen = {x.get("sub") for x in ms if x.get("t", 0) < t0 and x.get("sub")
                and x.get("cat") in (ASSIGN, WAIT, ITER)}
        confl = set()
        L = [x for x in (run.get("shared_state") or []) if isinstance(x, dict) and x.get("key") != "run_status"]
        for j, x in enumerate(L):
            if x.get("op") == "override" and j > 0:
                confl |= _conflict_subs(run, x.get("t"), ms)
        d["rec_fresh"] = float(bool(rs - seen))
        d["rec_confl"] = float(bool(rs & confl))
    except Exception:
        pass
    return d


def rec_features(runs) -> pd.DataFrame:
    return pd.DataFrame([rec_features_run(r) for r in runs], index=range(len(runs)),
                        columns=REC_COLUMNS).astype(float)


def joint_last_table(runs, C: Dict[str, pd.DataFrame], n: int) -> Dict[str, np.ndarray]:
    """{'l5': (n, 6), 'l6': (n, 6)} joint last-slot values of each class's top stage-1
    candidate (NaN when unavailable / on failure)."""
    out = {k: np.full((n, len(FAULTS)), np.nan) for k in ("l5", "l6")}
    empty = pd.DataFrame({"run": pd.Series(dtype=int), "t": pd.Series(dtype=int),
                          "score": pd.Series(dtype=float)})
    Cs = {}
    for c in FAULTS:
        D = (C or {}).get(c)
        ok = D is not None and len(D) > 0 and {"run", "t", "score"} <= set(D.columns)
        Cs[c] = D[["run", "t", "score"]] if ok else empty
    for key, modname, tmpl in (("l5", "slot_lastslot", "l5_%s_last"), ("l6", "slot_planlast", "l6_%s")):
        try:
            mod = __import__(modname)
            F = mod.features(list(runs), Cs).reindex(range(n))
            for j, c in enumerate(FAULTS):
                col = tmpl % c
                if col in F.columns:
                    out[key][:, j] = pd.to_numeric(F[col], errors="coerce").values.astype(float)
        except Exception as exc:                                  # pragma: no cover
            warnings.warn("arbiter.joint_last_table: %s failed (%s) -> NaN" % (modname, exc))
    return out


def _logit4(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _has(G, col) -> bool:
    return all(G.get(c) is not None and col in G[c].columns for c in FAULTS)


def pair_table(runs: List[Dict[str, Any]], C: Dict[str, pd.DataFrame],
               G: Dict[str, pd.DataFrame], P: np.ndarray, S: np.ndarray,
               thr: float = THR, lastslot_fn: Optional[Callable] = None,
               J: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
    """One row per two-strong run (>= 2 classes with genuineness > thr), kind 'ff'
    [+ one 'cf' gate row per other run whose stage-2 decision is clean-vs-top-fault].

    C: stage-1 candidates with 'score' (OOF on train); G: genuineness candidates
    with 'gscore' [and 'g2n' / 'g2s'] (fit_noctx / score_noctx); P: (n, 7) stage-2 class
    probabilities in LABELS order; S: (n,) success probability.
    ff columns: run, a, b (class indices into FAULTS, a < b), la, lb (last-slot flags
    of the two genuineness-top candidates), n_strong, ps, mlo, glo, slo, dl [, s2lo].
    cf columns: run, a (top stage-2 fault), b = -1, gz_p (logit P(clean) vs P(a)), gz_g2s /
    gz_s / gz_g (logits of the max g2s / stage-1 score / gscore of class a)."""
    n = len(runs)
    P = np.asarray(P, float).reshape(n, len(LABELS))
    S = np.asarray(S, float).reshape(n)
    M, _ = top_table(C, n, "score")
    Gm, topg = top_table(G, n, "gscore")
    fn = lastslot_fn or default_lastslot_fn()
    rows = []
    for i in range(n):
        s = np.where(Gm[i] > thr)[0]
        if len(s) < 2:
            continue
        a, b = sorted(sorted(s, key=lambda c: -(Gm[i, c] + P[i, 1 + c]))[:2])
        top = {FAULTS[j]: (topg[FAULTS[j]][i], Gm[i, j]) for j in range(len(FAULTS))
               if i in topg[FAULTS[j]]}
        try:
            f = fn(runs[i], top, thr)
        except Exception:
            f = {}
        la = f.get("slot_last_%s" % FAULTS[a], np.nan)
        lb = f.get("slot_last_%s" % FAULTS[b], np.nan)
        pa = pb = 0.0
        if USE_PLAST:
            pa, pb = _plast_pair(runs[i], top, FAULTS[a], FAULTS[b], thr)
        ra = rb = 0.0
        if USE_REC_PAIR:
            ra = _rec_names(runs[i], topg[FAULTS[a]].get(i), FAULTS[a])
            rb = _rec_names(runs[i], topg[FAULTS[b]].get(i), FAULTS[b])
        rows.append(dict(run=i, a=int(a), b=int(b), la=0.0 if la != la else float(la),
                         lb=0.0 if lb != lb else float(lb), n_strong=len(s), pa=pa, pb=pb,
                         drec=ra - rb))
    cols = ["run", "a", "b", "la", "lb", "n_strong", "ps", "mlo", "glo", "slo", "dl", "pa", "pb", "dlp"]
    use_s2 = USE_G2_PAIR and _has(G, "g2n")
    if use_s2:
        cols.append("s2lo")
    D = pd.DataFrame(rows, columns=["run", "a", "b", "la", "lb", "n_strong", "pa", "pb", "drec"])
    if USE_REC_PAIR:
        cols.append("drec")
    r, a, b = D.run.values.astype(int), D.a.values.astype(int), D.b.values.astype(int)
    D["ps"] = S[r]
    D["mlo"] = np.log(P[r, 1 + a] + 1e-6) - np.log(P[r, 1 + b] + 1e-6)
    D["glo"] = _logit(Gm[r, a]) - _logit(Gm[r, b])
    D["slo"] = _logit(M[r, a]) - _logit(M[r, b])
    D["dl"] = D.la - D.lb
    D["dlp"] = np.clip(np.log(1.0 - np.clip(D.pa.values.astype(float), 0, 0.995))
                       - np.log(1.0 - np.clip(D.pb.values.astype(float), 0, 0.995)),
                       -PLAST_CLIP, PLAST_CLIP)
    if use_s2:
        M2n, _ = top_table(G, n, "g2n")
        D["s2lo"] = _logit4(M2n[r, a]) - _logit4(M2n[r, b])
    if USE_JOINT_LAST:
        J = J if J is not None else joint_last_table(runs, C, n)
    else:
        J = None
    if J is not None:
        for f, key in zip(JOINT_FF, ("l5", "l6")):
            V = np.clip(np.nan_to_num(J[key], nan=0.0), 0.0, 1.0)
            D[f] = V[r, a] - V[r, b]
            cols.append(f)
    D = D[cols].assign(kind="ff")
    if not (USE_DET_GATE and _has(G, "g2s")):
        return D.reset_index(drop=True)
    # ---- clean-vs-top-fault gate rows
    M2s, _ = top_table(G, n, "g2s")
    inpair = np.zeros(n, bool)
    inpair[r] = True
    fi = np.argmax(P[:, 1:], axis=1)
    pc, pf = P[:, 0], P[np.arange(n), 1 + fi]
    sel = np.where(~inpair & (pc + pf > GATE_REGION))[0]
    fs = fi[sel]
    E = pd.DataFrame({"run": sel, "a": fs, "b": -1, "kind": "cf",
                      "gz_p": _logit4(pc[sel] / (pc[sel] + pf[sel] + 1e-9)),
                      "gz_g2s": _logit4(M2s[sel, fs]), "gz_s": _logit4(M[sel, fs]),
                      "gz_g": _logit4(Gm[sel, fs])})
    if J is not None:
        E["gz_l5"] = np.nan_to_num(J["l5"], nan=-1.0)[sel, fs]
    return pd.concat([D, E], ignore_index=True)


def _plast_pair(run, top, ca, cb, thr):
    """P(last) of the two arbitrated instances, jointly with the other strong instances (slot_plast)."""
    try:
        import slot_plast
        ta, tb = _as_t(top[ca][0]), _as_t(top[cb][0])
        insts = [(ta, ca), (tb, cb)] + [(_as_t(v[0]), c) for c, v in top.items()
                                         if c not in (ca, cb) and _gt(v[1], thr)]
        p = slot_plast.Run(run).plast(insts)
        pa, pb = p[0], p[1]
        return (0.0 if pa != pa else float(pa)), (0.0 if pb != pb else float(pb))
    except Exception:
        return 0.0, 0.0


# ============================================================ (c) arbiter
def _kind(D: pd.DataFrame) -> np.ndarray:
    if "kind" in D.columns:
        return D["kind"].fillna("ff").values.astype(str)
    return np.full(len(D), "ff")


def arb_design(D: pd.DataFrame) -> np.ndarray:
    """Antisymmetric class one-hot (+ times S), log-odds diffs (+ times S), dl ('ff' rows)."""
    n = len(D)
    a, b, ps = D.a.values.astype(int), D.b.values.astype(int), D.ps.values.astype(float)
    Z = np.zeros((n, 12))
    Z[np.arange(n), a] += 1
    Z[np.arange(n), b] -= 1
    Z[np.arange(n), 6 + a] += ps
    Z[np.arange(n), 6 + b] -= ps
    ex = []
    feats = PAIR_FEATS + (("s2lo",) if (USE_G2_PAIR and "s2lo" in D.columns) else ())
    feats += tuple(f for f in JOINT_FF if USE_JOINT_LAST and f in D.columns)
    for f in feats:
        v = D[f].values.astype(float)
        ex += [v, v * ps]
    ex.append(D.dl.values.astype(float))
    if USE_PLAST and "dlp" in D.columns:
        ex.append(D.dlp.values.astype(float))
    if USE_REC_PAIR and "drec" in D.columns:
        ex.append(D.drec.fillna(0.0).values.astype(float))
    return np.column_stack([Z] + ex)


def gate_design(D: pd.DataFrame) -> np.ndarray:
    """Class one-hot of the top fault + each gate logit times that one-hot ('cf' rows)."""
    n = len(D)
    oh = np.zeros((n, len(FAULTS)))
    oh[np.arange(n), D.a.values.astype(int)] = 1
    feats = GATE_FEATS + tuple(f for f in JOINT_GATE if USE_JOINT_LAST and f in D.columns)
    return np.column_stack([oh] + [D[f].values.astype(float)[:, None] * oh for f in feats])


def pair_targets(D: pd.DataFrame, labels: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    """ff: y = 1 if the true label is class a; valid = true label is a or b.
    cf: y = 1 if the true label is fault a; valid = true label is a or clean."""
    lab = _labels(labels)[D.run.values.astype(int)]
    fa = np.array(FAULTS, dtype=object)[D.a.values.astype(int)]
    fb = np.array(FAULTS + ["clean"], dtype=object)[D.b.values.astype(int)]   # b = -1 -> clean
    y = (lab == fa).astype(int)
    return y, (lab == fa) | (lab == fb)


def fit_arbiter(D: pd.DataFrame, labels: Sequence, C: float = 1.0):
    """ff: logistic model (no intercept) on the valid train pairs; None if unusable.
    With gate rows: {'ff': that model, 'cf': gate logistic (intercept, C=GATE_C)}."""
    from sklearn.linear_model import LogisticRegression
    if len(D) == 0:
        return None
    y, valid = pair_targets(D, labels)
    kind = _kind(D)
    ff = kind == "ff"
    m_ff = None
    if (valid & ff).sum() >= 10 and len(set(y[valid & ff])) == 2:
        m_ff = LogisticRegression(C=C, fit_intercept=False, max_iter=3000).fit(
            arb_design(D[valid & ff]), y[valid & ff])
    if not (kind == "cf").any():
        return m_ff
    cf = valid & (kind == "cf")
    m_cf = None
    if cf.sum() >= 10 and len(set(y[cf])) == 2:
        m_cf = LogisticRegression(C=GATE_C, max_iter=3000).fit(gate_design(D[cf]), y[cf])
    return {"ff": m_ff, "cf": m_cf}


def predict_arbiter(model, D: pd.DataFrame) -> np.ndarray:
    """ff rows: P(true label = class a); cf rows: P(true label = fault a, not clean);
    NaN when there is no model."""
    out = np.full(len(D), np.nan)
    if model is None or len(D) == 0:
        return out
    models = model if isinstance(model, dict) else {"ff": model}
    kind = _kind(D)
    for k, design in (("ff", arb_design), ("cf", gate_design)):
        m = models.get(k)
        sel = kind == k
        if m is not None and sel.any():
            out[sel] = m.predict_proba(design(D[sel]))[:, 1]
    return out


def crossfit_arbiter(D: pd.DataFrame, labels: Sequence, fold_of: np.ndarray,
                     C: float = 1.0) -> np.ndarray:
    """Out-of-fold pair probabilities for train evaluation (fold of the run)."""
    pA = np.full(len(D), np.nan)
    if len(D) == 0:
        return pA
    fo = np.asarray(fold_of)[D.run.values.astype(int)]
    for k in np.unique(fo):
        te = fo == k
        m = fit_arbiter(D[~te], labels, C)
        pA[te] = predict_arbiter(m, D[te])
    return pA


def apply_arbiter(pred, D: pd.DataFrame, pA: np.ndarray) -> np.ndarray:
    """ff rows: override the label of every two-strong run with a (pA > .5) or b.
    cf rows: fault a (pA > .5) or clean.
    pred: labels as strings or LABELS indices (same type is returned)."""
    pred = np.asarray(pred)
    out = pred.copy()
    if len(D) == 0:
        return out
    pA = np.asarray(pA, float)
    ok = ~np.isnan(pA)
    r = D.run.values.astype(int)[ok]
    a, b = D.a.values.astype(int)[ok], D.b.values.astype(int)[ok]
    cf = (_kind(D) == "cf")[ok]
    k = np.where(pA[ok] > 0.5, a, np.where(cf, -1, b))        # -1 -> clean
    if np.issubdtype(pred.dtype, np.integer):
        out[r] = 1 + k                                          # clean -> 0
    else:
        out = out.astype(object)
        out[r] = np.array(FAULTS + ["clean"], dtype=object)[k]
    return out


# ============================================================ end-to-end
def fit_apply(runs_tr, Ctr, labels, fault_turn, fold_of, P_oof, S_oof,
              runs_te, Cte, P_te, S_te, pred_te=None, n_jobs: int = 2,
              thr: float = THR, lastslot_fn: Optional[Callable] = None,
              feats: Optional[Dict[str, List[str]]] = None,
              J_tr: Optional[Dict[str, np.ndarray]] = None,
              J_te: Optional[Dict[str, np.ndarray]] = None):
    """Train everything on train (OOF pieces), apply to test.
    J_tr / J_te: precomputed joint last-slot tables (joint_last_table) or None.
    Returns (new test labels or None, info dict with G/pair tables/models)."""
    prm = {"random_state": SEED} if SEED else None
    if DET_TEST_BAG or PAR_WORKERS > 1:
        G_oof, G_te = fit_score_noctx(Ctr, labels, fault_turn, fold_of, Cte, n_jobs=n_jobs,
                                      params=prm, feats=feats, full=not DET_TEST_BAG)
        gm = None
    else:
        G_oof, gm = fit_noctx(Ctr, labels, fault_turn, fold_of, n_jobs=n_jobs, params=prm, feats=feats)
        G_te = score_noctx(gm, Cte)
    Dtr = pair_table(runs_tr, Ctr, G_oof, P_oof, S_oof, thr, lastslot_fn, J=J_tr)
    Dte = pair_table(runs_te, Cte, G_te, P_te, S_te, thr, lastslot_fn, J=J_te)
    am = fit_arbiter(Dtr, labels)
    pA = predict_arbiter(am, Dte)
    if pred_te is None:
        pred_te = np.array(LABELS, dtype=object)[np.asarray(P_te).argmax(1)]
    new = apply_arbiter(pred_te, Dte, pA)
    return new, dict(G_oof=G_oof, G_te=G_te, noctx_models=gm, D_train=Dtr, D_test=Dte,
                     arbiter=am, pA_test=pA)


# ============================================================ generic features
def features_run(run: Dict[str, Any], top: Dict[str, Tuple[Any, float]],
                 thr: float = THR, lastslot_fn: Optional[Callable] = None) -> Dict[str, float]:
    """top = {cls: (t, score)} of the run's top candidate per class."""
    f = (lastslot_fn or default_lastslot_fn())(run, top, thr)
    d = {}
    for c in FAULTS:
        d[PREFIX + "last_" + c] = float(f.get("slot_last_" + c, np.nan))
        d[PREFIX + "off_" + c] = float(f.get("slot_off_" + c, np.nan))
    d[PREFIX + "L"] = float(f.get("slot_L", np.nan))
    d[PREFIX + "n_strong"] = float(sum(1 for c in FAULTS if c in top and _gt(top[c][1], thr)))
    return d


def _gt(v, thr) -> bool:
    try:
        return bool(float(v) > thr)
    except (TypeError, ValueError):
        return False


FEATURE_COLUMNS = ([PREFIX + k + "_" + c for c in FAULTS for k in ("last", "off")]
                   + [PREFIX + "L", PREFIX + "n_strong"])


def features(runs: List[Dict[str, Any]], C: Dict[str, pd.DataFrame],
             thr: float = THR, lastslot_fn: Optional[Callable] = None) -> pd.DataFrame:
    """Last-slot flag / slot offset of each class's top candidate (ranked by
    'gscore' when C[c] carries it, else 'score' -- decided per class), plan
    length, #strong classes."""
    n = len(runs)
    C = C or {}
    Mx = np.zeros((n, len(FAULTS)))
    top: Dict[str, Dict[int, Any]] = {}
    for j, c in enumerate(FAULTS):
        D = C.get(c)
        col = "gscore" if D is not None and "gscore" in getattr(D, "columns", ()) else "score"
        Mj, tj = top_table({c: D} if D is not None else {}, n, col)
        Mx[:, j] = Mj[:, j]
        top[c] = tj[c]
    fn = lastslot_fn or default_lastslot_fn()
    rows = []
    for i, run in enumerate(runs):
        tp = {c: (top[c][i], Mx[i, j]) for j, c in enumerate(FAULTS) if i in top[c]}
        try:
            rows.append(features_run(run, tp, thr, fn))
        except Exception:
            rows.append({})
    return pd.DataFrame(rows, index=pd.RangeIndex(n), columns=FEATURE_COLUMNS, dtype=float)
