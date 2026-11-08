"""Training / inference pipeline: candidate models -> run model -> localiser."""
from __future__ import annotations
import os
import sys
import time
from typing import Dict, List
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore', message='X does not have valid feature names')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from feats import run_features
from parse import parse_row
LABELS = ['clean', 'dropped_handoff', 'duplicated_work', 'deadlock', 'conflict', 'goal_drift', 'runaway_loop']
FAULTS = LABELS[1:]
_T0 = time.time()

def _elapsed() -> float:
    return time.time() - _T0
RUN_PARAMS = dict(n_estimators=800, learning_rate=0.03, num_leaves=31, colsample_bytree=0.3,
    min_child_samples=40, reg_lambda=5.0)
PSEUDO_THR = 0.9
MC_FAST_PARAMS = dict(RUN_PARAMS, max_bin=63, learning_rate=0.06, n_estimators=400)

def _mc_inputs(Xtr2, A_tr, A_te):
    """Drop training-constant columns only for final multiclass tree fits."""
    keep = (Xtr2.nunique(dropna=False) > 1).to_numpy()
    if not keep.any():
        print('v18 multiclass features: %d -> %d' % (A_tr.shape[1], A_tr.shape[1]), flush=True)
        return (A_tr, A_te)
    print('v18 multiclass features: %d -> %d' % (A_tr.shape[1], int(keep.sum())), flush=True)
    return (A_tr[:, keep], A_te[:, keep])

def _mc_params():
    return MC_FAST_PARAMS
STAGE2_REPEATS = 1
PAR_EXTRA = True
N_JOBS = int(os.environ.get('CR_N_JOBS', min(8, os.cpu_count() or 1)))
SEED = int(os.environ.get('CR_SEED', 0))

def _pmap(fn, items):
    """map fn over items with N_JOBS worker threads (LightGBM releases the GIL); results in order."""
    items = list(items)
    if N_JOBS <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(min(N_JOBS, len(items))) as ex:
        return list(ex.map(fn, items))

def _task_threads(n_tasks: int) -> int:
    """OpenMP threads per LightGBM model when n_tasks models are fitted concurrently: always 1 with
    PAR_FITS -- multi-threaded LightGBM on these small matrices collapses under CPU contention (a lone
    2-thread multiclass fit took 126 s vs 50 s single-threaded on the loaded dev machine)."""
    return 1
CSET_MODE = os.environ.get('CR_CSET_MODE', 'auto')
CSET_W = float(os.environ.get('CR_CSET_W', 0.5))
RESID_GATE = os.environ.get('CR_RESID_GATE', 'all')   # exp60 never-in-clean signatures: off | cf | all
RESID_TURNS = os.environ.get('CR_RESID_TURNS', '1') == '1'   # exp60 ontology fault-turn re-pointing
_RESID = {}
NONE_POOL = os.environ.get('CR_NONE_POOL', '1') == '1'   # log-pool stage-2 P with the cset class head on runs outside the arbiter table
LM_PROC = None                                            # (Popen, out.pkl) of lm/lm_infer.py, set by solution.py
LM_W = float(os.environ.get('CR_LM_W', '0.2'))             # log-linear weight of the LM class head on stage-2 P
LM_A = float(os.environ.get('CR_LM_A', '0.0'))             # weight of the LM success head in the success average
CSET_WEIGHTS = os.path.join(HERE, 'cset_weights.npz')
import nnblend
import catboost_member

def log(msg: str) -> None:
    print('[%7.1fs] %s' % (time.time() - _T0, msg), flush=True)

def _proc(rec):
    run = parse_row(rec)
    f, c = run_features(run)
    if c.get('dropped_handoff'):
        import dh_chat
        by_t = {m['t']: m for m in run['messages']}
        for cand in c['dropped_handoff']:
            m = by_t.get(cand.get('t'))
            if m is not None:
                cand.update(dh_chat.cand_features(run, m))
    return (f, c, run)

def featurize(df: pd.DataFrame, n_jobs: int=8):
    recs = df.to_dict('records')
    out = None
    if n_jobs > 1 and len(recs) > 200:
        try:
            from multiprocessing import get_context
            with get_context('fork').Pool(n_jobs) as p:
                out = p.map(_proc, recs, chunksize=64)
        except Exception as exc:
            log('multiprocessing failed (%s), running serially' % exc)
            out = None
    if out is None:
        out = [_proc(r) for r in recs]
    X = pd.DataFrame([o[0] for o in out])
    C = {}
    for lab in FAULTS:
        rows = [dict(c, run=i) for i, o in enumerate(out) for c in o[1].get(lab, [])]
        C[lab] = pd.DataFrame(rows)
    runs = [o[2] for o in out]
    return (X, C, runs)
SLOT_PREFIX = 'sl_'

def _no_slot(X):
    if isinstance(X, pd.DataFrame):
        drop = [c for c in X.columns if str(c).startswith(SLOT_PREFIX)]
        if drop:
            return X.drop(columns=drop)
    return X
_LGB_CLS = None

def _lgb_class():
    """LGBMClassifier subclass that ignores the plan-slot candidate columns (module level,
    created on first use so that importing pipeline does not import lightgbm)."""
    global _LGB_CLS
    if _LGB_CLS is None:
        import lightgbm as lgb

        class LGBNoSlot(lgb.LGBMClassifier):

            def fit(self, X, y, *a, **k):
                return super().fit(_no_slot(X), y, *a, **k)

            def predict_proba(self, X, *a, **k):
                return super().predict_proba(_no_slot(X), *a, **k)
        LGBNoSlot.__qualname__ = 'LGBNoSlot'
        globals()['LGBNoSlot'] = LGBNoSlot
        _LGB_CLS = LGBNoSlot
    return _LGB_CLS

def _lgb(**kw):
    params = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=N_JOBS, random_state=SEED)
    params.update(kw)
    return _lgb_class()(**params)

def _cand_target(D, lab, ft, c):
    r = D.run.values
    return ((lab[r] == c) & (D.t.values == ft[r])).astype(int)

def _aggregate(X: pd.DataFrame, C: Dict[str, pd.DataFrame], key: str='score') -> pd.DataFrame:
    X = X.copy()
    for c in FAULTS:
        D = C[c]
        cols = ['%s_%s_max' % (key, c), '%s_%s_sum' % (key, c), '%s_%s_n05' % (key, c),
            '%s_%s_argt' % (key, c), '%s_%s_gap' % (key, c)]
        if len(D) == 0:
            for col in cols:
                X[col] = 0.0 if 'argt' not in col else -1.0
            continue
        g = D.groupby('run')[key]
        agg = pd.DataFrame({cols[0]: g.max(), cols[1]: g.sum(), cols[2]: g.apply(lambda s: float((s > 0.5).sum()))})
        top2 = g.apply(lambda s: np.sort(s.values)[-2] if len(s) > 1 else 0.0)
        agg[cols[4]] = agg[cols[0]] - top2
        idx = D.loc[g.idxmax()]
        agg[cols[3]] = pd.Series(idx.rel_t.values, index=idx.run.values)
        X = X.join(agg, how='left')
        X[cols[3]] = X[cols[3]].fillna(-1.0)
        X[[cols[0], cols[1], cols[2], cols[4]]] = X[[cols[0], cols[1], cols[2], cols[4]]].fillna(0.0)
    return X

def fit_predict(train: pd.DataFrame, test: pd.DataFrame, n_folds: int=5, seeds=(0, 1,
    2), n_jobs: int=8, return_debug: bool=False, pseudo_thr=PSEUDO_THR):
    global N_JOBS
    N_JOBS = max(1, int(n_jobs))
    log('featurizing train (%d)' % len(train))
    Xtr, Ctr, runs_tr = featurize(train, n_jobs)
    log('featurizing test (%d)' % len(test))
    Xte, Cte, runs_te = featurize(test, n_jobs)
    return fit_predict_features(train, test, Xtr, Ctr, Xte, Cte, n_folds, seeds, return_debug,
        pseudo_thr, runs_tr=runs_tr, runs_te=runs_te)

def _extra_modules():
    mods = []
    mods += ['slot_streams', 'slot_lastslot', 'slot_planlast']
    mods += ['slot_plast']
    mods += ['dh_chat']
    return mods

def _extra_serial(runs, C, n, mods):
    parts = []
    for name in mods:
        mod = __import__(name)
        F = mod.features(runs, C)
        F = F.reindex(range(n))
        F.columns = [str(c) for c in F.columns]
        parts.append(F.astype(float))
    import arbiter
    parts.append(arbiter.rec_features(runs).reindex(range(n)).astype(float))
    if not parts:
        return pd.DataFrame(index=range(n))
    out = pd.concat(parts, axis=1)
    return out.loc[:, ~out.columns.duplicated()]
_EXTRA_CTX = None

def _extra_chunk(bounds):
    """Fork-pool worker: extra features of runs lo..hi-1 (candidate frames re-indexed to the chunk).
    Every module is per-run and only reads the run / t / score columns of the candidates."""
    lo, hi = bounds
    runs, C, n, mods = _EXTRA_CTX
    Cs = {}
    for c, D in C.items():
        if D is None or len(D) == 0 or (not {'run', 't', 'score'} <= set(D.columns)):
            Cs[c] = D
            continue
        r = D['run'].values
        m = (r >= lo) & (r < hi)
        E = D.loc[m, ['run', 't', 'score']]
        Cs[c] = E.assign(run=E['run'].values - lo)
    F = _extra_serial(runs[lo:hi], Cs, hi - lo, mods)
    F.index = range(lo, hi)
    return F

def _extra_features(runs, C, n, n_jobs=None):
    """Run-level features that need the parsed runs and the stage-1 candidate scores."""
    global _EXTRA_CTX
    mods = _extra_modules()
    runs = list(runs)
    n_jobs = N_JOBS if n_jobs is None else n_jobs
    if PAR_EXTRA and n_jobs > 1 and (n > 400) and mods:
        k = max(2, 4 * n_jobs)
        edges = np.linspace(0, n, k + 1).astype(int)
        chunks = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]
        try:
            from multiprocessing import get_context
            _EXTRA_CTX = (runs, C, n, mods)
            with get_context('fork').Pool(n_jobs) as pool:
                parts = pool.map_async(_extra_chunk, chunks, chunksize=1).get()
            out = pd.concat(parts, axis=0).reindex(range(n))
            return out.loc[:, ~out.columns.duplicated()]
        except Exception as exc:
            log('parallel extra features failed (%s), running serially' % exc)
        finally:
            _EXTRA_CTX = None
    return _extra_serial(runs, C, n, mods)

def _concat(Xa, Ca, Xb, Cb, sel):
    """Append rows ``sel`` of (Xb, Cb) to (Xa, Ca), re-indexing candidate runs."""
    sel = np.asarray(sel, dtype=int)
    remap = {int(o): len(Xa) + i for i, o in enumerate(sel)}
    X = pd.concat([Xa, Xb.iloc[sel]], ignore_index=True)
    C = {}
    for c in FAULTS:
        B = Cb[c]
        if len(B):
            B = B[B.run.isin(remap)].copy()
            B['run'] = B.run.map(remap)
        C[c] = pd.concat([Ca[c], B], ignore_index=True)
    return (X, C)

def fit_predict_features(train, test, Xtr, Ctr, Xte, Cte, n_folds=5, seeds=(0, 1, 2),
    return_debug=False, pseudo_thr=None, runs_tr=None, runs_te=None, use_arbiter=None,
    n_real=None):
    if use_arbiter is None:
        use_arbiter = True
    if pseudo_thr:
        global HIT_WINDOW
        _hw, HIT_WINDOW = (HIT_WINDOW, False)
        try:
            out0, dbg0 = fit_predict_features(train, test, Xtr, Ctr, Xte, Cte, n_folds,
                seeds[:1], True, None, runs_tr=runs_tr, runs_te=runs_te, use_arbiter=False)
        finally:
            HIT_WINDOW = _hw
        conf = dbg0['P'].max(1)
        sel = np.where(conf >= pseudo_thr)[0].astype(int)
        log('pseudo-labelling: %d / %d test runs with p >= %.2f' % (len(sel), len(test), pseudo_thr))
        if len(sel) < 20:
            return (out0, dbg0) if return_debug else out0
        ps = out0.iloc[sel]
        train2 = pd.concat([train[['label', 'success', 'fault_turn']].reset_index(drop=True),
            pd.DataFrame({'label': ps.label.values, 'success': ps.success.values, 'fault_turn': ps.fault_turn.values})],
            ignore_index=True)
        Xtr2, Ctr2 = _concat(Xtr.reindex(columns=Xtr.columns), Ctr, Xte.reindex(columns=Xtr.columns),
            Cte, sel)
        runs_tr2 = list(runs_tr) + [runs_te[j] for j in sel] if runs_tr is not None and runs_te is not None else None
        return fit_predict_features(train2, test, Xtr2, Ctr2, Xte, Cte, n_folds, seeds,
            return_debug, None, runs_tr=runs_tr2, runs_te=runs_te, use_arbiter=use_arbiter,
            n_real=len(train))
    from sklearn.model_selection import StratifiedKFold
    Ctr, Cte = (dict(Ctr), dict(Cte))
    Xtr = Xtr.fillna(0.0)
    Xte = Xte.reindex(columns=Xtr.columns).fillna(0.0)
    lab = train['label'].values
    ft = train['fault_turn'].fillna(-1).astype(int).values
    y = np.array([LABELS.index(l) for l in lab])
    folds = list(StratifiedKFold(n_folds, shuffle=True, random_state=42 + SEED).split(Xtr, y))
    fold_of = np.zeros(len(Xtr), int)
    for k, (_, b) in enumerate(folds):
        fold_of[b] = k
    log('candidate models')
    prep = {}
    for c in FAULTS:
        D, T = (Ctr[c], Cte[c])
        if len(D) == 0:
            Ctr[c] = D.assign(score=[])
            Cte[c] = T.assign(score=np.zeros(len(T)))
            continue
        feats = [x for x in D.columns if x != 'run' and (not str(x).startswith(SLOT_PREFIX))]
        XD = D[feats].fillna(-1.0).to_numpy(np.float64)
        XT = T.reindex(columns=feats).fillna(-1.0).to_numpy(np.float64) if len(T) else None
        prep[c] = (len(D), len(T), XD, XT, _cand_target(D, lab, ft, c), fold_of[D.run.values])
    order = sorted(prep, key=lambda c: -prep[c][0])
    tasks = [(c, k) for c in order for k in range(n_folds)]
    nj1 = _task_threads(len(tasks))

    def _fit1(task):
        c, k = task
        _, _, XD, XT, tgt, fo = prep[c]
        m = fo != k
        if tgt[m].sum() == 0:
            return None
        mdl = _lgb(n_jobs=nj1, random_state=SEED).fit(XD[m], tgt[m])
        oof = mdl.predict_proba(XD[~m])[:, 1]
        te = mdl.predict_proba(XT)[:, 1] if XT is not None else None
        return (oof, te)
    res1 = dict(zip(tasks, _pmap(_fit1, tasks)))
    for c in order:
        nD, nT, _, _, _, fo = prep[c]
        prep[c] = None
        D = Ctr[c].fillna(-1.0)
        T = Cte[c].reindex(columns=D.columns).fillna(-1.0) if nT else Cte[c]
        sc = np.zeros(nD)
        te = np.zeros(nT)
        n_te = 0
        for k in range(n_folds):
            rk = res1.get((c, k))
            if rk is None:
                continue
            sc[fo == k] = rk[0]
            if rk[1] is not None:
                te += rk[1]
                n_te += 1
        Ctr[c] = D.assign(score=sc)
        if not len(T):
            Cte[c] = T.assign(score=[])
        else:
            Cte[c] = T.assign(score=te / n_te if n_te else 0.0)
    prep = res1 = None
    Xtr2 = _aggregate(Xtr, Ctr)
    Xte2 = _aggregate(Xte, Cte).reindex(columns=Xtr2.columns).fillna(0.0)
    have_runs = runs_tr is not None and runs_te is not None
    Etr = Ete = None
    if have_runs:
        log('plan-slot / chat features')
        Etr = _extra_features(runs_tr, Ctr, len(Xtr2))
        Ete = _extra_features(runs_te, Cte, len(Xte2)).reindex(columns=Etr.columns)
        Xtr2 = pd.concat([Xtr2.reset_index(drop=True), Etr.reset_index(drop=True)], axis=1)
        Xte2 = pd.concat([Xte2.reset_index(drop=True), Ete.reset_index(drop=True)], axis=1)
    log('run-level models (%d features)' % Xtr2.shape[1])
    suc = train['success'].astype(int).values
    want_arb = use_arbiter and have_runs
    P = np.zeros((len(Xte2), len(LABELS)))
    S = np.zeros(len(Xte2))
    P_oof = S_oof = None
    A_tr = Xtr2.to_numpy(np.float64)
    A_te = Xte2.to_numpy(np.float64)
    if want_arb:
        M_tr, M_te = _mc_inputs(Xtr2, A_tr, A_te)
        P_oof = np.zeros((len(Xtr2), len(LABELS)))
        S_oof = np.zeros(len(Xtr2))
        reps = max(1, int(STAGE2_REPEATS))
        fos = [fold_of]
        for rep in range(1, reps):
            fo = np.zeros(len(Xtr2), int)
            for k, (_, bb) in enumerate(StratifiedKFold(n_folds, shuffle=True, random_state=42 + SEED + 1000 * rep).split(Xtr2,
                y)):
                fo[bb] = k
            fos.append(fo)
        tasks = [('m', rep, k) for rep in range(reps) for k in range(n_folds)] + [('s',
            rep, k) for rep in range(reps) for k in range(n_folds)]
        nj2 = _task_threads(len(tasks))

        def _fit2(task):
            kind, rep, k = task
            a = fos[rep] != k
            rs = seeds[0] + k + 100 * rep + 10007 * SEED
            if kind == 'm':
                from mean3 import predict
                return predict(_lgb, dict(objective='multiclass', random_state=rs, n_jobs=nj2,
                    extra_trees=True, **_mc_params()), M_tr[a], y[a], M_tr[~a], M_te)
            m = _lgb(random_state=rs, n_jobs=nj2, **RUN_PARAMS).fit(A_tr[a], suc[a])
            return (m.predict_proba(A_tr[~a])[:, 1], m.predict_proba(A_te)[:, 1])
        for (kind, rep, k), (o, t) in zip(tasks, _pmap(_fit2, tasks)):
            b = fos[rep] == k
            if kind == 'm':
                P_oof[b] += o / reps
                P += t / (reps * n_folds)
            else:
                S_oof[b] += o / reps
                S += t / (reps * n_folds)
    else:
        tasks = [('m', s) for s in seeds] + [('s', s) for s in seeds]
        nj2 = _task_threads(len(tasks))
        njm = njs = nj2

        def _fit2f(task):
            kind, s = task
            if kind == 'm':
                return _lgb(objective='multiclass', random_state=s + 10007 * SEED, n_jobs=njm,
                    **_mc_params()).fit(A_tr, y).predict_proba(A_te)
            return _lgb(random_state=s + 10007 * SEED, n_jobs=njs, **RUN_PARAMS).fit(A_tr,
                suc).predict_proba(A_te)[:, 1]
        for (kind, s), t in zip(tasks, _pmap(_fit2f, tasks)):
            if kind == 'm':
                P += t / len(seeds)
            else:
                S += t / len(seeds)
    if P_oof is not None and nnblend.NN_W > 0:
        t_nn = _elapsed()
        Z_tr, Z_te = nnblend.prep(A_tr, A_te)
        tasks = [(kind, k) for k in range(n_folds) for kind in (('m', 's') if nnblend.NN_SUCCESS else ('m',))]

        def _fitnn(task):
            kind, k = task
            a = fold_of != k
            tgt, nc = (y, len(LABELS)) if kind == 'm' else (suc, 2)
            return nnblend.fit_predict(Z_tr[a], tgt[a], [Z_tr[~a], Z_te], nc, seed=seeds[0] + 31 * k + 10007 * SEED)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            res_nn = _pmap(_fitnn, tasks)
        Pn_oof = np.zeros_like(P_oof)
        Pn = np.zeros_like(P)
        Sn_oof = np.zeros_like(S_oof)
        Sn = np.zeros_like(S)
        for (kind, k), (o, t) in zip(tasks, res_nn):
            b = fold_of == k
            if kind == 'm':
                Pn_oof[b] = o
                Pn += t / n_folds
            else:
                Sn_oof[b] = o
                Sn += t / n_folds
        tree_oof, tree_test = (P_oof.copy(), P.copy())
        P_oof = nnblend.mix(P_oof, Pn_oof)
        P = nnblend.mix(P, Pn)
        if want_arb and have_runs and (n_folds == 5) and (n_real is not None):
            sequence_oof, sequence_test = nnblend._v27_runtime.run_sequence(nnblend.__file__,
                runs_tr, runs_te, y, suc, fold_of, n_real, log)
            blended_oof = nnblend.mix(tree_oof, sequence_oof)
            blended_test = nnblend.mix(tree_test, sequence_test)
            P_oof, P = (blended_oof, blended_test)
            log('v27 selected: sequence_class')
        if nnblend.NN_SUCCESS:
            S_oof = nnblend.mix_s(S_oof, Sn_oof)
            S = nnblend.mix_s(S, Sn)
        log('MLP member blended (w=%.2f, %s) in %.0fs' % (nnblend.NN_W, nnblend.NN_MODE, _elapsed() - t_nn))
    if want_arb and P_oof is not None and (n_real is not None):
        cb_oof, cb_test, cb_metadata = catboost_member.run_worker(__file__, A_tr, A_te,
            y, fold_of, n_real, log)
        P_oof = catboost_member.combine(P_oof, cb_oof)
        P = catboost_member.combine(P, cb_test)
        log('v45_v46 catboost arm=%s real_rows=%d folds=5' % ('v46_ensemble', n_real))
    pred = [LABELS[i] for i in P.argmax(1)]
    arb_info = None
    if want_arb:
        import arbiter
        log('pair arbiter: genuineness detectors + arbitration')
        J_tr = J_te = None
        if Etr is not None:
            J_tr = _joint_from_extra(Etr, len(Xtr2))
            J_te = _joint_from_extra(Ete, len(Xte2))
        arbiter.PAR_WORKERS = N_JOBS
        arbiter.SEED = SEED
        new, arb_info = arbiter.fit_apply(runs_tr, Ctr, lab, ft, fold_of, P_oof, S_oof,
            runs_te, Cte, P, S, np.array(pred, dtype=object), n_jobs=N_JOBS, J_tr=J_tr,
            J_te=J_te)
        if new is not None:
            changed = int(sum((1 for x, y_ in zip(pred, new) if x != y_)))
            log('pair arbiter changed %d / %d test labels' % (changed, len(pred)))
            pred = [str(x) for x in new]
    turns = localize(pred, Cte, len(Xte2))
    if want_arb:
        S, turns, pred = _cset_addon(Ctr, Cte, lab, ft, suc, n_real or len(train), len(Xte2), pred, S, turns,
                                     P=P, arb_info=arb_info)
    if want_arb and LM_PROC is not None:
        pred, turns, S = _lm_blend(pred, turns, S, P, test, Cte, len(Xte2))
    if want_arb and (RESID_GATE != 'off' or RESID_TURNS):
        import resid_post
        pred, turns = resid_post.final_pass(pred, turns, P, _RESID.get('T_nn'), Cte, runs_te, arb_info,
                                            gate=RESID_GATE, turn_rules=RESID_TURNS, log=log)
    out = pd.DataFrame({'run_id': test['run_id'].values, 'label': pred, 'success': (S > 0.5).astype(int),
        'fault_turn': turns})
    if return_debug:
        return (out, {'P': P, 'S': S, 'Cte': Cte, 'Xte': Xte2, 'arbiter': arb_info})
    return out

def _lm_blend(pred, turns, S, P, test, Cte, n_te):
    """Join the LM subprocess and blend its class / success heads (see module docstring of patch_lm)."""
    t0 = _elapsed()
    proc, path = LM_PROC
    rc = proc.wait() if proc is not None else 0
    if rc != 0:
        raise RuntimeError('LM member failed with exit code %d' % rc)
    L = pd.read_pickle(path).reindex(test['run_id'].values)
    Pl = np.clip(L[['P_' + c for c in LABELS]].values.astype(float), 1e-6, 1.0)
    Sl = L['S'].values.astype(float)
    assert Pl.shape == (n_te, len(LABELS)) and np.isfinite(Pl).all() and np.isfinite(Sl).all()
    P0 = np.clip(np.asarray(P, dtype=float), 1e-6, 1.0)
    l0 = np.log(P0 / P0.sum(1, keepdims=True))
    z = l0 + LM_W * np.log(Pl / Pl.sum(1, keepdims=True))
    moved = np.where(z.argmax(1) != l0.argmax(1))[0]
    T_nn = _RESID.get('T_nn')
    pred = list(pred)
    turns = list(turns)
    changed = []
    for i in moved:
        c = LABELS[int(z[i].argmax())]
        if c != pred[i]:
            pred[i] = c
            changed.append(int(i))
    if changed:
        loc = localize(pred, Cte, n_te)
        for i in changed:
            if pred[i] == 'clean':
                turns[i] = -1
                continue
            t = int(T_nn[i, LABELS.index(pred[i]) - 1]) if T_nn is not None else -1
            turns[i] = t if t >= 0 else int(loc[i])
    S = (1 - LM_A) * np.asarray(S, dtype=float) + LM_A * Sl
    log('LM member (w=%.2f, a=%.2f): argmax moved on %d runs, %d labels changed, waited %.0fs'
        % (LM_W, LM_A, len(moved), len(changed), _elapsed() - t0))
    return pred, turns, S


def _cset_addon(Ctr, Cte, lab, ft, suc, n_real, n_te, pred, S, turns, P=None, arb_info=None):
    """-> (S, turns) with the candidate-set transformer's success probability averaged in and its pointer
    turn for the predicted class.  Trained here on the real (non-pseudo) train rows when a CUDA device is
    available, else run from the shipped weights (numpy inference). Errors propagate to the CLI."""
    import csetnn
    t_c = _elapsed()
    mode = CSET_MODE
    if mode == 'auto':
        import torch
        mode = 'train' if torch.cuda.is_available() else 'ship'
    if mode == 'train':
        Cr = {c: Ctr[c][Ctr[c].run.values < n_real] for c in csetnn.FAULTS}
        st = csetnn.fit_stats(Cr)
        cand, mask, turn = csetnn.build(Cr, n_real, st)
        y = np.array([LABELS.index(l) for l in lab[:n_real]])
        ws = csetnn.train(cand, mask, turn, y, suc[:n_real], ft[:n_real], log=log)
    else:
        ws, st = csetnn.load(CSET_WEIGHTS)
    cand, mask, turn = csetnn.build(Cte, n_te, st)
    P_cs, S_nn, T_nn = csetnn.predict(ws, cand, mask, turn)
    _RESID['T_nn'] = np.asarray(T_nn)
    pred = list(pred)
    n_pool = 0
    if NONE_POOL and P is not None and isinstance(arb_info, dict) and P_cs is not None:
        D_te = arb_info.get('D_test')
        in_table = set(D_te.run.values.astype(int).tolist()) if D_te is not None and len(D_te) else set()
        Pq = np.asarray(P_cs, dtype=float)
        if Pq.shape == (n_te, len(LABELS)) and np.isfinite(Pq).all():
            z = np.log(np.clip(np.asarray(P, dtype=float), 1e-6, 1.0)) + np.log(np.clip(Pq, 1e-6, 1.0))
            changed = []
            for i in range(n_te):
                if i in in_table:
                    continue
                c = LABELS[int(z[i].argmax())]
                if c != pred[i]:
                    pred[i] = c
                    changed.append(i)
            if changed:
                loc = localize(pred, Cte, n_te)
                turns = list(turns)
                for i in changed:
                    turns[i] = -1 if pred[i] == 'clean' else int(loc[i])
            n_pool = len(changed)
    S = (1 - CSET_W) * S + CSET_W * S_nn
    turns = list(turns)
    n_ptr = 0
    for i, p in enumerate(pred):
        if p != 'clean':
            t = int(T_nn[i, LABELS.index(p) - 1])
            if t >= 0:
                n_ptr += int(t != turns[i])
                turns[i] = t
    log('candidate-set add-on (%s): success blended (w=%.2f), %d fault turns moved, %d labels re-pooled outside the arbiter table, %.0fs' % (mode,
        CSET_W, n_ptr, n_pool, _elapsed() - t_c))
    return (S, turns, pred)

def _joint_from_extra(E, n):
    """{'l5': (n, 6), 'l6': (n, 6)} from the stage-2 extra columns l5_{c}_last / l6_{c} (the arbiter's
    joint_last_table recomputes exactly these from the same runs and stage-1 scores); None if missing."""
    out = {}
    for key, tmpl in (('l5', 'l5_%s_last'), ('l6', 'l6_%s')):
        cols = [tmpl % c for c in FAULTS]
        if not all((c in E.columns for c in cols)):
            return None
        out[key] = E[cols].reindex(range(n)).values.astype(float)
    return out
HIT_WINDOW = True

def window_turn(tt, p, default=None):
    """Integer turn maximising sum(p[|tt - o| <= 2]); ties -> nearest to the argmax candidate."""
    tt = np.asarray(tt, float)
    p = np.clip(np.asarray(p, float), 0, None)
    ok = np.isfinite(tt) & (tt >= 0)
    if not ok.any() or p[ok].sum() <= 0:
        return default
    tt, p = (tt[ok].astype(int), p[ok])
    am = tt[p.argmax()]
    opts = np.arange(tt.min() - 2, tt.max() + 3)
    mass = (np.abs(opts[:, None] - tt[None, :]) <= 2).astype(float) @ p
    cands = opts[mass >= mass.max() - 1e-09 * max(1.0, mass.max())]
    return int(cands[np.argmin(np.abs(cands - am))])

def localize(pred: List[str], C: Dict[str, pd.DataFrame], n: int) -> List[int]:
    best = {}
    need = {c: set((i for i, p in enumerate(pred) if p == c)) for c in FAULTS}
    for c in FAULTS:
        D = C[c]
        if len(D) == 0:
            best[c] = {}
            continue
        idx = D.groupby('run')['score'].idxmax()
        best[c] = dict(zip(D.loc[idx, 'run'].values, D.loc[idx, 't'].values))
        if HIT_WINDOW and need[c]:
            Dn = D[D.run.isin(need[c])]
            for r, g in Dn.groupby('run'):
                w = window_turn(g.t.values, g.score.values)
                if w is not None:
                    best[c][r] = w
    turns = []
    for i, p in enumerate(pred):
        if p == 'clean':
            turns.append(-1)
        else:
            turns.append(int(best[p].get(i, 0)))
    return turns
if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--v45-catboost-worker':
        raise SystemExit('Invalid v45_v46 worker arguments')
    catboost_member.worker(sys.argv[2], sys.argv[3])
