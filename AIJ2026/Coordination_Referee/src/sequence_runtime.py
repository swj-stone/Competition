"""Subprocess execution for the frozen full-message sequence model.

The outer CLI harness owns the execution limit; this stage never skips a member.
"""
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile

import numpy as np

PAYLOAD_KEYS = ('runs_tr', 'runs_te', 'y', 'suc', 'fold_of', 'n_real')


def _classes(values, count, name):
    values = np.asarray(values)
    if values.dtype.kind not in 'fiu' or values.shape != (count, 7):
        raise ValueError(f'{name}: expected numeric shape ({count}, 7)')
    checked = values.astype(np.float64)
    if (not np.isfinite(checked).all() or np.any(checked < 0) or np.any(checked > 1)
            or not np.allclose(checked.sum(1), 1., rtol=0, atol=1e-6)):
        raise ValueError(f'{name}: expected finite normalized class probabilities')
    return values if values.dtype.kind == 'f' else checked


def _tail(path):
    if not path.exists():
        return ''
    with path.open('rb') as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 2000))
        return stream.read(2000).decode('utf-8', errors='replace').strip()


def run_sequence(worker_path, runs_tr, runs_te, y, suc, fold_of, n_real, log):
    """Return complete class OOF/test arrays or raise on an invalid worker result."""
    worker_path = Path(worker_path).resolve()
    if not worker_path.is_file():
        raise ValueError('Sequence worker source does not exist')
    environment = os.environ.copy()
    environment.update({name: '4' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_THREAD_LIMIT')})
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    with tempfile.TemporaryDirectory(prefix='v27_sequence_') as directory:
        directory = Path(directory)
        source, output, child_log = directory/'input.pkl', directory/'output.npz', directory/'worker.log'
        with source.open('wb') as stream:
            pickle.dump(dict(zip(PAYLOAD_KEYS, (runs_tr, runs_te, y, suc, fold_of, n_real))),
                        stream, protocol=pickle.HIGHEST_PROTOCOL)
        command = [sys.executable, str(worker_path), '--v27-sequence-worker', str(source), str(output)]
        with child_log.open('wb') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                env=environment, check=False, shell=False)
        if result.returncode != 0:
            message = f'v27 sequence worker exit {result.returncode}; {_tail(child_log)}'.rstrip('; ')
            log(message)
            raise RuntimeError(message)
        if not output.is_file():
            raise RuntimeError('Sequence worker returned no output archive')
        try:
            with np.load(output, allow_pickle=False) as saved:
                if set(saved.files) != {'po', 'pt'}:
                    raise ValueError('Sequence output archive must contain exactly po and pt')
                po = _classes(saved['po'], len(runs_tr), 'po')
                pt = _classes(saved['pt'], len(runs_te), 'pt')
        except (OSError, EOFError, TypeError, AttributeError, KeyError) as exc:
            raise ValueError('Malformed sequence output archive') from exc
        tail = _tail(child_log)
        log(f'v27 sequence output validated; {tail}' if tail else 'v27 sequence output validated')
        return po, pt


def worker(module, input_path, output_path):
    """Consume the parent's local temporary pickle; invoked only by the explicit worker CLI."""
    with Path(input_path).open('rb') as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or set(payload) != set(PAYLOAD_KEYS):
        raise ValueError('Invalid sequence worker payload')
    po, pt, _, _, meta = module.train_predict(*(payload[key] for key in PAYLOAD_KEYS), save_dir=None)
    if not isinstance(meta, dict) or not meta.get('complete'):
        raise ValueError('Sequence training did not complete')
    po = _classes(po, len(payload['runs_tr']), 'po')
    pt = _classes(pt, len(payload['runs_te']), 'pt')
    with Path(output_path).open('xb') as stream:
        np.savez_compressed(stream, po=po, pt=pt)
    print('v27 sequence worker complete: '
          f'device={meta.get("device", "unknown")} maximum_messages={meta.get("maximum_messages", "unknown")}',
          flush=True)
