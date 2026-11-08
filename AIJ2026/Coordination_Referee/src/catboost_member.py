"""Frozen v45/v46 CatBoost leaf-wise multiclass member and worker.

Training receives only the original v27 numeric stage-two matrices, training-side
targets and fold IDs. No evaluation targets, tuning, early stopping or extra rows.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

CFG = dict(iterations=400, learning_rate=.06, depth=8, max_leaves=31,
    min_data_in_leaf=40, l2_leaf_reg=5, rsm=.5, border_count=64,
    boosting_type='Plain', grow_policy='Lossguide', loss_function='MultiClass',
    task_type='CPU', thread_count=1, allow_writing_files=False, verbose=False)
SEED0 = 45000
N_FOLDS = 5
N_CLASSES = 7
PAYLOAD_KEYS = ('Xtr', 'Xte', 'y', 'fold_of', 'n_real')


def array_sha256(values):
    values = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(json.dumps({'dtype': values.dtype.str, 'shape': values.shape},
                             sort_keys=True).encode('ascii'))
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def matrix(value, name):
    values = np.asarray(value)
    if values.ndim != 2 or not values.shape[1] or values.dtype.kind not in 'buif':
        raise ValueError(name + ': expected a real numeric feature matrix')
    if np.isinf(values).any():
        raise ValueError(name + ': infinite feature values are forbidden')
    return np.ascontiguousarray(values, dtype=np.float64)


def ids(value, name, size, count):
    values = np.asarray(value)
    if (values.shape != (size,) or values.dtype.kind not in 'buif'
            or not np.isfinite(values).all() or not np.isin(values, np.arange(count)).all()):
        raise ValueError(name + ': invalid integer IDs')
    return values.astype(np.int64)


def align_probabilities(raw, classes):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError('Class probabilities must be a matrix')
    classes = ids(classes, 'classes', raw.shape[1], N_CLASSES)
    if (not len(classes) or len(np.unique(classes)) != len(classes)
            or not np.isfinite(raw).all() or (raw < 0).any() or (raw > 1).any()
            or (raw.sum(1) <= 0).any()):
        raise ValueError('Invalid native CatBoost class probabilities')
    result = np.zeros((len(raw), N_CLASSES), np.float64)
    result[:, classes] = raw / raw.sum(1, keepdims=True)
    return result


def probabilities(value, rows):
    values = np.asarray(value, dtype=np.float64)
    if (values.shape != (rows, N_CLASSES) or not np.isfinite(values).all()
            or (values < 0).any() or (values > 1).any()
            or not np.allclose(values.sum(1), 1., rtol=0, atol=1e-6)):
        raise ValueError('Expected finite normalized seven-class probabilities')
    return values


def fit_native(x, y, held, test, seed, params, checkpoint=None):
    """Fit and replay a native CBM; explicit parameters support small synthetic tests.

    The production crossfit entry point always supplies the frozen CFG unchanged.
    """
    from catboost import CatBoostClassifier

    path = Path(checkpoint) if checkpoint is not None else None
    if path is not None and path.exists():
        raise FileExistsError('Checkpoint must not be overwritten: ' + str(path))
    classes = np.unique(y)
    if len(classes) == 1:
        payload = json.dumps({'class_id': int(classes[0])}).encode()
        if path is not None:
            with path.open('xb') as stream:
                stream.write(payload)
            payload = path.read_bytes()
        class_id = json.loads(payload)['class_id']
        outputs = []
        for features in (held, test):
            p = np.zeros((len(features), N_CLASSES)); p[:, class_id] = 1.
            outputs.append(p)
        return *outputs, dict(iterations=0, constant_class_id=class_id,
            checkpoint_replayed=True, checkpoint_predictions_exact=True,
            checkpoint_sha256=hashlib.sha256(payload).hexdigest())
    model = CatBoostClassifier(**dict(params, random_seed=int(seed)))
    model.fit(x, y)
    parent = path.parent if path is not None else Path.cwd()
    with tempfile.TemporaryDirectory(prefix='v45_cbm_', dir=parent) as temporary:
        native = Path(temporary) / 'model.cbm'
        model.save_model(str(native), format='cbm')
        payload = native.read_bytes()
    if path is not None:
        with path.open('xb') as stream:
            stream.write(payload)
        payload = path.read_bytes()
    loaded = CatBoostClassifier()
    loaded.load_model(blob=payload, format='cbm')
    if (not np.array_equal(model.classes_, loaded.classes_)
            or model.tree_count_ != params['iterations']
            or loaded.tree_count_ != params['iterations']):
        raise ValueError('Native checkpoint changed classes or iteration count')
    outputs = []
    for features in (held, test):
        if len(features):
            before = np.asarray(model.predict_proba(features, thread_count=1))
            after = np.asarray(loaded.predict_proba(features, thread_count=1))
        else:
            before = after = np.empty((0, len(loaded.classes_)))
        if not np.array_equal(before, after):
            raise ValueError('Native checkpoint probability replay differs')
        outputs.append(align_probabilities(after, loaded.classes_))
    return *outputs, dict(iterations=int(loaded.tree_count_), classes=loaded.classes_.astype(int).tolist(),
        checkpoint_replayed=True, checkpoint_predictions_exact=True,
        checkpoint_sha256=hashlib.sha256(payload).hexdigest())


def fit_fold(x, y, held, test, fold, directory):
    path = directory / ('fold_%d.cbm' % fold) if directory is not None else None
    po, pt, record = fit_native(x, y, held, test, SEED0 + fold, dict(CFG), path)
    return po, pt, dict(record, fold=fold, seed=SEED0 + fold,
        training_rows=len(x), held_rows=len(held), test_rows=len(test))


def crossfit(Xtr, Xte, y, fold_of, n_real, save_dir=None):
    train, test = matrix(Xtr, 'Xtr'), matrix(Xte, 'Xte')
    if not len(train) or train.shape[1] != test.shape[1]:
        raise ValueError('Nonempty training and matching feature widths are required')
    labels = ids(y, 'y', len(train), N_CLASSES)
    folds = ids(fold_of, 'fold_of', len(train), N_FOLDS)
    if set(folds.tolist()) != set(range(N_FOLDS)):
        raise ValueError('Exactly all five original folds must occur')
    if (not isinstance(n_real, (int, np.integer)) or isinstance(n_real, (bool, np.bool_))
            or not 0 < n_real <= len(train)):
        raise ValueError('n_real must identify the original real-row prefix')
    directory = Path(save_dir).resolve() if save_dir is not None else None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
        if any((directory / ('fold_%d.cbm' % k)).exists() for k in range(N_FOLDS)):
            raise FileExistsError('A fold checkpoint already exists')

    def one(k):
        fit, held = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        po, pt, record = fit_fold(train[fit], labels[fit], train[held], test, k, directory)
        record.update(training_indices=fit.tolist(), held_indices=held.tolist())
        return held, probabilities(po, len(held)), probabilities(pt, len(test)), record

    po = np.zeros((len(train), N_CLASSES), np.float64)
    pt = np.zeros((len(test), N_CLASSES), np.float64)
    coverage = np.zeros(len(train), np.int8)
    records = []
    with ThreadPoolExecutor(max_workers=N_FOLDS) as executor:
        for held, hp, tp, record in executor.map(one, range(N_FOLDS)):
            po[held] = hp; pt += tp / N_FOLDS; coverage[held] += 1
            records.append(record)
    if not np.all(coverage == 1):
        raise ValueError('Original OOF rows were not predicted exactly once')
    if len(pt): pt /= pt.sum(1, keepdims=True)
    import catboost
    meta = dict(complete=True, n_real=int(n_real), n_augmented=len(train), test_rows=len(test),
        feature_count=train.shape[1], config=dict(CFG), seed0=SEED0, n_folds=N_FOLDS,
        catboost_version=catboost.__version__, max_workers=N_FOLDS, folds=records,
        all_checkpoints_replayed=all(r['checkpoint_replayed'] for r in records),
        heldout_truth_loaded=False, evaluation_labels_used=False, early_stopping=False)
    meta['input_sha256'] = {key: array_sha256(value) for key, value in
                           (('Xtr', Xtr), ('Xte', Xte), ('y', y), ('fold_of', fold_of))}
    meta['probability_sha256'] = dict(po=array_sha256(po), pt=array_sha256(pt))
    meta['fold_summaries'] = [{key: record[key] for key in (
        'fold', 'seed', 'training_rows', 'held_rows', 'test_rows', 'iterations',
        'checkpoint_sha256', 'checkpoint_replayed', 'checkpoint_predictions_exact') if key in record}
        for record in records]
    return probabilities(po, len(train)), probabilities(pt, len(test)), meta


def combine(original, catboost):
    """The retained v51 classification blend, fixed at 50/50."""
    original = probabilities(original, len(original))
    catboost = probabilities(catboost, len(original))
    return .5 * original + .5 * catboost


def run_worker(worker_path, Xtr, Xte, y, fold_of, n_real, log):
    started = time.perf_counter()
    environment = os.environ.copy()
    environment.update({k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_THREAD_LIMIT')})
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    # The active run workspace is writable on both the local sandbox and server.
    # A Windows sandbox may allow creation but deny writes in the default TEMP.
    with tempfile.TemporaryDirectory(prefix='v45_worker_', dir=Path.cwd()) as temporary:
        directory = Path(temporary)
        source, output, logs = directory / 'input.npz', directory / 'output.npz', directory / 'worker.log'
        np.savez_compressed(source, Xtr=Xtr, Xte=Xte, y=y, fold_of=fold_of, n_real=np.array(n_real))
        command = [sys.executable, '-B', str(Path(worker_path).resolve()), '--v45-catboost-worker',
                   str(source), str(output)]
        with logs.open('xb') as stream:
            result = subprocess.run(command, env=environment, stdout=stream,
                stderr=subprocess.STDOUT, check=False, shell=False)
        tail = logs.read_text(encoding='utf-8', errors='replace')[-4000:]
        if result.returncode != 0 or not output.is_file():
            raise RuntimeError('CatBoost worker failed: ' + tail)
        with np.load(output, allow_pickle=False) as saved:
            if set(saved.files) != {'po', 'pt', 'metadata'}:
                raise ValueError('CatBoost output archive has unexpected members')
            po = probabilities(saved['po'], len(Xtr))
            pt = probabilities(saved['pt'], len(Xte))
            meta = json.loads(str(saved['metadata'].item()))
        if not meta.get('complete') or not meta.get('all_checkpoints_replayed'):
            raise ValueError('CatBoost did not complete native checkpoint verification')
        expected_inputs = {key: array_sha256(value) for key, value in
                           (('Xtr', Xtr), ('Xte', Xte), ('y', y), ('fold_of', fold_of))}
        if (meta.get('input_sha256') != expected_inputs or
                meta.get('probability_sha256') != dict(po=array_sha256(po), pt=array_sha256(pt))):
            raise ValueError('CatBoost worker input or probability digest mismatch')
        meta['worker_seconds'] = time.perf_counter() - started
        log('v45_v46 CatBoost worker validated: ' + json.dumps({k: meta[k] for k in
            ('feature_count', 'n_augmented', 'n_real', 'n_folds', 'catboost_version', 'config',
             'input_sha256', 'probability_sha256', 'fold_summaries', 'worker_seconds')}))
        return po, pt, meta


def worker(input_path, output_path):
    with np.load(input_path, allow_pickle=False) as saved:
        if set(saved.files) != set(PAYLOAD_KEYS):
            raise ValueError('CatBoost worker payload keys mismatch')
        payload = {name: saved[name] for name in PAYLOAD_KEYS}
    if payload['n_real'].shape != () or payload['n_real'].dtype.kind not in 'iu':
        raise ValueError('Worker n_real must be a scalar integer')
    payload['n_real'] = int(payload['n_real'])
    po, pt, meta = crossfit(**payload, save_dir=Path(output_path).parent / 'models')
    with Path(output_path).open('xb') as stream:
        np.savez_compressed(stream, po=po, pt=pt, metadata=np.array(json.dumps(meta)))

