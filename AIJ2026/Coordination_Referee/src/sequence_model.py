"""Fixed full-message CNN probe; Torch training, standalone NumPy inference."""
from __future__ import annotations

import io
from pathlib import Path
import numpy as np

TYPES = ('handoff', 'inform', 'status', 'state_update', 'final', 'system')
COLUMNS = tuple([f'cat_{i}' for i in range(15)] + [f'type_{t}' for t in TYPES] + [
    'sender_in_team', 'receiver_in_team', 'receiver_user', 'self_message',
    'sender_orchestrator', 'receiver_orchestrator', 'sender_reviewer', 'receiver_reviewer',
    'sender_executor', 'receiver_executor', 'same_sender_previous', 'same_receiver_previous',
    'same_task_previous', 'has_tool', 'ref_count', 'intent_blank', 'text_length', 'position', 'gap',
    'art_count', 'art_final', 'art_partial', 'art_copy', 'art_final_delivery',
    'state_count', 'state_set', 'state_override', 'state_completed', 'state_timeout',
    'state_aborted', 'state_incomplete'])
CFG = dict(width=32, dilations=(1, 2, 4), epochs=12, bs=128, lr=.001, wd=.01, seed0=27000)


def _rows(value):
    return [v for v in value if isinstance(v, dict)] if isinstance(value, (list, tuple)) else []


def _time(value, default=0):
    try:
        t = float(value)
        return t if np.isfinite(t) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _count(value):
    return float(np.log1p(min(max(float(value), 0), 1e6)) / np.log(9.))


def encode_run(run):
    """Encode all parsed messages; arbitrary identifiers occur only in equality joins."""
    messages = sorted(enumerate(_rows(run.get('messages'))), key=lambda x: _time(x[1].get('t'), x[0]))
    roles = {a.get('id'): str(a.get('role', '')).lower() for a in _rows(run.get('agents'))}
    roles.pop(None, None)
    events = {}
    for artifact in _rows(run.get('artifacts')):
        if 't' not in artifact:
            continue
        a = events.setdefault(_time(artifact['t']), np.zeros(12, np.float32))
        status = artifact.get('status', '')
        a[:5] += [1, status == 'final', status == 'partial', status == 'copy',
                  artifact.get('subtask') == 'final_delivery']
    for state in _rows(run.get('shared_state')):
        if 't' not in state:
            continue
        a = events.setdefault(_time(state['t']), np.zeros(12, np.float32))
        terminal = state.get('value') if state.get('key') == 'run_status' else None
        a[5:] += [1, state.get('op') == 'set', state.get('op') == 'override',
                  terminal == 'completed', terminal == 'timeout', terminal == 'aborted',
                  terminal == 'incomplete']
    out = np.zeros((len(messages), len(COLUMNS)), np.float32)
    previous, previous_t = {}, 0.
    for i, (original_index, message) in enumerate(messages):
        row = out[i]
        raw_cat = _time(message.get('cat'), 14)
        cat = int(raw_cat) if raw_cat.is_integer() and 0 <= raw_cat < 15 else 14
        row[cat] = 1
        typ = message.get('type')
        if typ in TYPES:
            row[15 + TYPES.index(typ)] = 1
        sender, receiver = message.get('from'), message.get('to')
        sr, rr = roles.get(sender, ''), roles.get(receiver, '')
        task = message.get('sub') or message.get('intent') or ''
        prior_task = previous.get('sub') or previous.get('intent') or ''
        refs = message.get('refs')
        nrefs = len(refs) if isinstance(refs, (list, tuple)) else 0
        t = _time(message.get('t'), original_index)
        row[21:40] = [sender in roles, receiver in roles, receiver == 'user',
            bool(sender) and sender == receiver, sr in ('orchestrator', 'coordinator'),
            rr in ('orchestrator', 'coordinator'), sr in ('reviewer', 'validator', 'evaluator', 'checker'),
            rr in ('reviewer', 'validator', 'evaluator', 'checker'), sr == 'executor', rr == 'executor',
            bool(i and sender and sender == previous.get('from')),
            bool(i and receiver and receiver == previous.get('to')), bool(i and task and task == prior_task),
            bool(message.get('tool')), _count(nrefs), not bool(message.get('intent')),
            np.log1p(min(len(str(message.get('text') or '')), 1000000)) / np.log(257.),
            i / max(len(messages) - 1, 1), min(max(t - previous_t, 0), 32) / 32 if i else 0]
        if t in events:
            row[40:] = np.log1p(events[t]) / np.log(9.)
        previous, previous_t = message, t
    return out


def pack(sequences):
    """Zero-pad without truncation, including empty runs."""
    arrays = [np.asarray(a, np.float32) for a in sequences]
    if any(a.ndim != 2 or a.shape[1] != len(COLUMNS) or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Invalid encoded sequence')
    length = max([1] + [len(a) for a in arrays])
    values = np.zeros((len(arrays), len(COLUMNS), length), np.float32)
    mask = np.zeros((len(arrays), length), bool)
    for i, a in enumerate(arrays):
        values[i, :, :len(a)] = a.T
        mask[i, :len(a)] = True
    return values, mask


def make_net():
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList([nn.Conv1d(len(COLUMNS) if i == 0 else 32, 32, 3,
                padding=d, dilation=d) for i, d in enumerate(CFG['dilations'])])
            self.lab = nn.Linear(96, 7)
            self.suc = nn.Linear(96, 1)

        def forward(self, x, mask):
            m = mask[:, None, :]
            for layer in self.convs:
                x = torch.relu(layer(x)) * m
            lengths = mask.sum(1)
            mean = x.sum(2) / lengths.clamp_min(1)[:, None]
            maximum = x.masked_fill(~m, -1e30).amax(2)
            maximum = torch.where(lengths[:, None] > 0, maximum, 0.)
            last = x[torch.arange(len(x), device=x.device), :, (lengths - 1).clamp_min(0)]
            z = torch.cat((mean, maximum, last), 1)
            return self.lab(z), self.suc(z).squeeze(1)
    return Net()


def predict_torch(net, sequences, device='cpu', batch_size=128):
    import torch
    previous = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    try:
        # The export reference must use float32 inference, regardless of training policy.
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
        net.to(device).eval()
        classes, success = [], []
        with torch.no_grad():
            for start in range(0, len(sequences), batch_size):
                x, m = pack(sequences[start:start + batch_size])
                a, b = net(torch.from_numpy(x).to(device), torch.from_numpy(m).to(device))
                classes.append(torch.softmax(a, 1).cpu().numpy())
                success.append(torch.sigmoid(b).cpu().numpy())
        return (np.concatenate(classes) if classes else np.empty((0, 7), np.float32),
                np.concatenate(success) if success else np.empty(0, np.float32))
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = previous


def _checked_weights(weights):
    shapes = {'lab.weight': (7, 96), 'lab.bias': (7,), 'suc.weight': (1, 96), 'suc.bias': (1,)}
    for i in range(3):
        shapes[f'convs.{i}.weight'] = (32, len(COLUMNS) if i == 0 else 32, 3)
        shapes[f'convs.{i}.bias'] = (32,)
    if set(weights) != set(shapes):
        raise ValueError('Incomplete or unexpected CNN weights')
    w = {key: np.asarray(value, np.float32) for key, value in weights.items()}
    if any(w[key].shape != shape or not np.isfinite(w[key]).all() for key, shape in shapes.items()):
        raise ValueError('Invalid CNN weight shape or value')
    return w


def predict_numpy(weights, sequences, batch_size=128):
    """Pure NumPy convolution and pooling; no Torch dependency at inference."""
    w = _checked_weights(weights)
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be positive')
    classes, success = [], []
    for start in range(0, len(sequences), batch_size):
        h, mask = pack(sequences[start:start + batch_size])
        for i, d in enumerate(CFG['dilations']):
            padded = np.pad(h, ((0, 0), (0, 0), (d, d)))
            a = np.broadcast_to(w[f'convs.{i}.bias'][None, :, None], (len(h), 32, h.shape[2])).copy()
            for k in range(3):
                a += np.einsum('bct,oc->bot', padded[:, :, k*d:k*d+h.shape[2]],
                               w[f'convs.{i}.weight'][:, :, k], optimize=False)
            h = np.maximum(a, 0) * mask[:, None, :]
        z = np.zeros((len(h), 96), np.float32)
        for j, length in enumerate(mask.sum(1)):
            if length:
                valid = h[j, :, :length]
                z[j] = np.concatenate((valid.mean(1), valid.max(1), valid[:, -1]))
        a = np.einsum('bi,oi->bo', z, w['lab.weight'], optimize=False) + w['lab.bias']
        a = np.exp(a - a.max(1, keepdims=True))
        a /= a.sum(1, keepdims=True)
        b = np.einsum('bi,oi->bo', z, w['suc.weight'], optimize=False)[:, 0] + w['suc.bias'][0]
        b = 1 / (1 + np.exp(-np.clip(b, -80, 80)))
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Nonfinite CNN output')
        classes.append(a); success.append(b)
    return (np.concatenate(classes) if classes else np.empty((0, 7), np.float32),
            np.concatenate(success) if success else np.empty(0, np.float32))


def _targets(y, suc, n):
    y, suc = np.asarray(y, dtype=float), np.asarray(suc, dtype=float)
    if (y.shape != (n,) or suc.shape != (n,) or not np.isfinite(y).all()
            or not np.isfinite(suc).all() or not np.isin(y, np.arange(7)).all()
            or not np.isin(suc, (0, 1)).all() or n == 0):
        raise ValueError('Invalid class or success targets')
    return y.astype(np.int64), suc.astype(np.float32)


def train_one(sequences, y, suc, seed, device):
    import torch
    import torch.nn.functional as F
    y, suc = _targets(y, suc, len(sequences))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(previous_threads, 4))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    losses, batches = [], 0
    try:
        net = make_net().to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'])
        for _ in range(CFG['epochs']):
            net.train()
            order, loss_sum = rng.permutation(len(y)), 0.
            for start in range(0, len(order), CFG['bs']):
                idx = order[start:start + CFG['bs']]
                x, m = pack([sequences[i] for i in idx])
                a, b = net(torch.from_numpy(x).to(device), torch.from_numpy(m).to(device))
                loss = F.cross_entropy(a, torch.from_numpy(y[idx]).to(device)) + F.binary_cross_entropy_with_logits(
                    b, torch.from_numpy(suc[idx]).to(device))
                if not bool(torch.isfinite(loss)):
                    raise ValueError('Nonfinite training loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(idx)
                batches += 1
            losses.append(loss_sum / len(y))
        weights = _checked_weights({key: value.detach().cpu().numpy().copy()
                                    for key, value in net.state_dict().items()})
        audit = sequences[:min(128, len(sequences))]
        pt, pn = predict_torch(net, audit, device), predict_numpy(weights, audit)
        errors = [float(np.max(np.abs(a - b))) for a, b in zip(pt, pn)]
        for a, b in zip(pt, pn):
            np.testing.assert_allclose(a, b, atol=2e-5, rtol=2e-5)
        return weights, dict(epochs=CFG['epochs'], batches=batches, losses=losses,
                             torch_numpy_max_abs=max(errors), torch_numpy_verified=True)
    finally:
        torch.set_num_threads(previous_threads)


def train_predict(runs_train, runs_test, y, suc, fold_of, n_real, save_dir=None):
    """Five-fold fit on all augmented rows, held-fold OOF and mean-fold test outputs."""
    y, suc = _targets(y, suc, len(runs_train))
    folds = np.asarray(fold_of)
    if (folds.shape != y.shape or not np.isin(folds, np.arange(5)).all()
            or set(folds.tolist()) != set(range(5)) or not isinstance(n_real, (int, np.integer))
            or not 0 < n_real <= len(y)):
        raise ValueError('Invalid folds or real-row prefix')
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train, test = [encode_run(run) for run in runs_train], [encode_run(run) for run in runs_test]
    directory = Path(save_dir).resolve() if save_dir is not None else None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
        if any((directory / f'fold_{k}.npz').exists() for k in range(5)):
            raise FileExistsError('CNN fold checkpoint already exists')
    po, pt = np.zeros((len(train), 7), np.float32), np.zeros((len(test), 7), np.float64)
    so, st = np.zeros(len(train), np.float32), np.zeros(len(test), np.float64)
    records = []
    for k in range(5):
        fit, held = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        weights, record = train_one([train[i] for i in fit], y[fit], suc[fit],
                                    seed=CFG['seed0'] + k, device=device)
        held_sequences = [train[i] for i in held]
        before = (predict_numpy(weights, held_sequences), predict_numpy(weights, test))
        path = directory / f'fold_{k}.npz' if directory is not None else None
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **weights)
        payload = buffer.getvalue()
        if path is not None:
            with path.open('xb') as stream:
                stream.write(payload)
        source = path if path is not None else io.BytesIO(payload)
        with np.load(source, allow_pickle=False) as saved:
            loaded = {key: saved[key].copy() for key in saved.files}
        after = (predict_numpy(loaded, held_sequences), predict_numpy(loaded, test))
        for old, new in zip(before, after):
            for a, b in zip(old, new):
                np.testing.assert_array_equal(a, b)
        po[held], so[held] = after[0]
        pt += after[1][0] / 5.
        st += after[1][1] / 5.
        records.append(dict(record, fold=k, seed=CFG['seed0'] + k,
            training_indices=fit.tolist(), held_indices=held.tolist(),
            checkpoint=str(path) if path is not None else None, checkpoint_replayed=True))
    return po, pt, so, st, dict(complete=True, all_checkpoints_replayed=True, folds=records,
        config=CFG.copy(), columns=list(COLUMNS), n_real=int(n_real), n_augmented=len(y), device=device,
        maximum_messages=max([0] + [len(a) for a in train + test]))
