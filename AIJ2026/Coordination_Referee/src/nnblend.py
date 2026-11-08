"""Stage-2 MLP member (CPU, scikit-learn) blended into the LightGBM stage-2 probabilities.

Track nn (work/nn): a plain MLP on the stage-2 feature matrix is a complementary member -- blended
geometrically with the LightGBM class probabilities (and linearly for success) it lowered the
length-weighted composite error on the fasteval protocol (nested blend-weight selection, 2 seeds,
+0.0012).  Inputs: signed-log1p + logit transforms of the stage-2 columns, standardised on train+test
(unsupervised), clipped, NaN -> 0, plus de-duplicated missingness indicators.
"""

from __future__ import annotations

import os

import numpy as np

NN_W = float(os.environ.get("CR_NN_W", 0.35))          # blend weight of the MLP (0 disables the member)
NN_MODE = os.environ.get("CR_NN_MODE", "log")           # 'log' = weighted geometric mean, 'lin' = arithmetic
NN_REPS = int(os.environ.get("CR_NN_REPS", 1))          # MLPs per fold (different inits), averaged
NN_EPOCHS = int(os.environ.get("CR_NN_EPOCHS", 30))
NN_H = tuple(int(x) for x in os.environ.get("CR_NN_H", "256,256").split(","))
NN_ALPHA = float(os.environ.get("CR_NN_ALPHA", 1e-3))
NN_SUCCESS = os.environ.get("CR_NN_SUCCESS", "1") == "1"


def _std(Z):
    return (Z - np.nanmean(Z, 0)) / (np.nanstd(Z, 0) + 1e-6)


def prep(A_tr: np.ndarray, A_te: np.ndarray, clip: float = 5.0):
    """(Z_tr, Z_te) float32 network inputs from the raw stage-2 matrices."""
    V = np.vstack([A_tr, A_te]).astype(np.float64)
    n = len(A_tr)
    nan = np.isnan(V)
    M = nan[:, nan.any(0)].astype(np.float32)
    if M.shape[1]:
        _, ix = np.unique(M, axis=1, return_index=True)
        M = M[:, np.sort(ix)]
    with np.errstate(all="ignore"):
        sd = np.nanstd(V, 0)
    V = V[:, np.isfinite(sd) & (sd > 0)]
    with np.errstate(all="ignore"):
        blocks = [_std(np.sign(V) * np.log1p(np.abs(V)))]
        mn, mx = np.nanmin(V, 0), np.nanmax(V, 0)
        nonbin = np.array([len(np.unique(V[~np.isnan(V[:, j]), j])) > 2 for j in range(V.shape[1])])
        sel = (mn >= 0) & (mx <= 1) & nonbin
        if sel.any():
            p = np.clip(V[:, sel], 1e-4, 1 - 1e-4)
            blocks.append(_std(np.log(p / (1 - p))))
    Z = np.concatenate([np.clip(np.nan_to_num(b, nan=0.0), -clip, clip) for b in blocks], 1)
    Z = np.concatenate([Z.astype(np.float32), M], 1)
    return Z[:n], Z[n:]


def fit_predict(Z_tr, target, Z_outs, n_classes, seed):
    """Average of NN_REPS MLPs fitted on (Z_tr, target); class probabilities (n, n_classes) for every
    matrix in Z_outs (binary target -> P(1) vector)."""
    import warnings
    from sklearn.neural_network import MLPClassifier
    outs = [np.zeros((len(Z), n_classes)) for Z in Z_outs]
    for r in range(NN_REPS):
        m = MLPClassifier(hidden_layer_sizes=NN_H, alpha=NN_ALPHA, batch_size=256, learning_rate_init=1e-3,
                          max_iter=NN_EPOCHS, early_stopping=False, n_iter_no_change=10 ** 6,
                          random_state=seed + 7919 * r)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(Z_tr, target)
        cls = m.classes_.astype(int)
        for o, Z in zip(outs, Z_outs):
            o[:, cls] += m.predict_proba(Z) / NN_REPS
    if n_classes == 2:
        return [o[:, 1] for o in outs]
    return [o / np.maximum(o.sum(1, keepdims=True), 1e-12) for o in outs]


def mix(P0, Pn, w=None, mode=None):
    w = NN_W if w is None else w
    mode = NN_MODE if mode is None else mode
    if mode == "log":
        L = (1 - w) * np.log(np.clip(P0, 1e-6, 1)) + w * np.log(np.clip(Pn, 1e-6, 1))
        P = np.exp(L - L.max(1, keepdims=True))
        return P / P.sum(1, keepdims=True)
    return (1 - w) * P0 + w * Pn


def mix_s(S0, Sn, w=None):
    w = NN_W if w is None else w
    return (1 - w) * S0 + w * Sn

# Readable frozen full-message model and worker protocol.
import sequence_model as _v27_module
import sequence_runtime as _v27_runtime
import sys as _v27_sys

if __name__ == '__main__':
    if len(_v27_sys.argv) != 4 or _v27_sys.argv[1] != '--v27-sequence-worker':
        raise SystemExit('Invalid v27 sequence worker arguments')
    _v27_runtime.worker(_v27_module, _v27_sys.argv[2], _v27_sys.argv[3])
