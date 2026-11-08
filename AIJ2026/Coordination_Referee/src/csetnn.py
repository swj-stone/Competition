"""Retained v53 CSET with fixed v55 conditional_logpool inference pooling.

Original PLR model, five-member eight-epoch training, weights, prior
layout and window decoder are unchanged. Only ensemble aggregation
uses conditional location probabilities and mean Success log odds.

Retained numerical embedding: https://arxiv.org/abs/2203.05556.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import time
from typing import Dict, List, Optional

import numpy as np

FAULTS = ['dropped_handoff', 'duplicated_work', 'deadlock', 'conflict', 'goal_drift', 'runaway_loop']
CFG = dict(K=8, d=128, L=3, H=4, drop=.1, in_drop=.1, epochs=8,
           bs=128, lr=5e-4, wd=.05, ptr_w=.5, suc_w=.5, n_ens=5, seed0=1000)
FORMAL_CFG = dict(CFG)
NUMERIC_FEATURES = ['t', 'rel_t', 'to_end', 'n', 'candidate_score',
                    'candidate_logit_div4', 'log1p_candidate_count_div2']
SCHEMA = 'v55_cset_pooling_v1'
EMBEDDING = dict(method='plr_lite', n_frequencies=16, frequency_init_scale=.01,
                 d_embedding=24, output_dim=128, zero_initialized_output=True, raw_passthrough=True)

AGGREGATION = {'success': 'mean_clipped_log_odds', 'success_clip': 1e-06, 'pointer': 'conditional_mean_logits', 'pointer_support': 'valid_candidates_only', 'null_mass': 'excluded_per_member', 'all_invalid': 'zeros', 'decoder': 'original_hit2_window', 'class': 'original_arithmetic_mean'}

# Per-process evidence for the single registered train/predict lifecycle.
# It does not contribute to the model's loss or its predictions.
_CONTEXT = {}


def _slog(a):
    return np.sign(a) * np.log1p(np.abs(a))


def fit_stats(C: Dict, runs_mask: Optional[np.ndarray] = None) -> Dict:
    """Per class: candidate feature column order and (mu, sd) of slog(features) over the given runs' candidates."""
    st = {}
    for c in FAULTS:
        D = C[c]
        cols = [k for k in D.columns if k not in ("run", "score")]
        A = D[cols].fillna(-1.0).to_numpy(np.float64)
        if runs_mask is not None:
            A = A[runs_mask[D.run.values.astype(int)]]
        A = _slog(A)
        mu = A.mean(0) if len(A) else np.zeros(len(cols))
        sd = A.std(0) if len(A) else np.ones(len(cols))
        sd[sd < 1e-6] = 1.0
        st[c] = dict(cols=cols, mu=mu.astype(np.float32), sd=sd.astype(np.float32))
    return st


def digest_arrays(arrays):
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        if array.dtype.hasobject:
            raise ValueError('object arrays cannot enter numerical evidence')
        description = json.dumps([name, array.dtype.str, list(array.shape)], separators=(',', ':')).encode()
        digest.update(len(description).to_bytes(8, 'big'))
        digest.update(description)
        digest.update(array.tobytes())
    return digest.hexdigest()


def feature_indices(st):
    result = {}
    for fault in FAULTS:
        cols = list(st[fault]['cols'])
        if len(cols) != len(set(cols)) or any(cols.count(name) != 1 for name in NUMERIC_FEATURES[:4]):
            raise ValueError('numerical candidate names are absent or ambiguous')
        n = len(cols)
        result[fault] = [cols.index(name) for name in NUMERIC_FEATURES[:4]] + [n, n + 1, n + 4]
    return result


def fit_specs(cand, mask, indices):
    specs = {}
    for fault in FAULTS:
        x = np.asarray(cand[fault])
        valid = np.asarray(mask[fault])
        ix = np.asarray(indices[fault], np.int64)
        if x.ndim != 3 or valid.shape != x.shape[:2] or valid.dtype != bool:
            raise ValueError('invalid original token/mask shape')
        if ix.shape != (7,) or len(set(ix.tolist())) != 7 or (ix < 0).any() or (ix >= x.shape[-1]).any():
            raise ValueError('invalid seven numerical feature indices')
        if not np.isfinite(x).all():
            raise ValueError('original tokens must be finite')
        specs[fault] = dict(indices=ix, input_dim=x.shape[-1], valid_tokens=int(valid.sum()))
    return specs


def _build_tensors(C: Dict, n: int, st: Dict, K: int = CFG["K"]):
    """-> cand {c: (n, K, F_c)}, mask {c: (n, K) bool}, turn {c: (n, K) int (-1 pad)} for runs 0..n-1."""
    cand, mask, turn = {}, {}, {}
    for c in FAULTS:
        D = C[c]
        cols, mu, sd = st[c]["cols"], st[c]["mu"], st[c]["sd"]
        Fdim = len(cols) + 5
        T = np.zeros((n, K, Fdim), np.float32); M = np.zeros((n, K), bool); TT = np.full((n, K), -1, np.int64)
        if len(D):
            A = D.reindex(columns=cols).fillna(-1.0).to_numpy(np.float64)
            A = np.clip((_slog(A) - mu) / sd, -6, 6)
            s = np.asarray(D["score"].values, np.float64)
            p = np.clip(s, 1e-4, 1 - 1e-4)
            lg = np.log(p / (1 - p))
            run = D.run.values.astype(int)
            order = np.lexsort((-s, run))                      # by run, then score descending (stable)
            rs = run[order]
            first = np.r_[0, np.flatnonzero(rs[1:] != rs[:-1]) + 1]
            rank = np.empty(len(D), np.int64)
            rank[order] = np.arange(len(D)) - np.repeat(first, np.diff(np.r_[first, len(D)]))
            ncand = np.bincount(run, minlength=n)[run]
            extra = np.stack([s, lg / 4, rank / K, (rank == 0).astype(float), np.log1p(ncand) / 2], 1)
            F = np.concatenate([A, extra], 1).astype(np.float32)
            keep = (rank < K) & (run >= 0) & (run < n)
            r, k = run[keep], rank[keep]
            T[r, k] = F[keep]; M[r, k] = True; TT[r, k] = D.t.values[keep].astype(np.int64)
        cand[c], mask[c], turn[c] = T, M, TT
    return cand, mask, turn


def build(C, n, st, K=CFG['K']):
    """Build original top-K tensors and retain their named numerical mapping."""
    try:
        indices = feature_indices(st)
        if 'indices' in _CONTEXT and _CONTEXT['indices'] != indices:
            raise ValueError('CSET train/test column mapping changed')
        _CONTEXT['indices'] = indices
        return _build_tensors(C, n, st, K)
    except Exception as exc:
        raise SystemExit('v51 CSET build error: ' + str(exc)) from exc


def _net(Fc: List[int], c=CFG):
    import torch
    import torch.nn as nn
    K, d = c["K"], c["d"]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.ModuleList([nn.Sequential(nn.Dropout(c["in_drop"]), nn.Linear(f, 2 * d), nn.GELU(),
                                                     nn.Linear(2 * d, d)) for f in Fc])
            self.cemb = nn.Parameter(torch.randn(6, d) * 0.02)
            self.remb = nn.Parameter(torch.randn(K + 1, d) * 0.02)
            self.null = nn.Parameter(torch.randn(6, d) * 0.02)
            self.cls = nn.Parameter(torch.randn(d) * 0.02)
            layer = nn.TransformerEncoderLayer(d, c["H"], 2 * d, dropout=c["drop"], batch_first=True,
                                               norm_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, c["L"], enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            self.lab = nn.Linear(d, 7); self.suc = nn.Linear(d, 1); self.ptr = nn.Linear(d, 1)
            self.pool_a = nn.Parameter(torch.zeros(6))

        def forward(self, cands, masks):
            B = cands[0].shape[0]
            dev = cands[0].device
            toks = [self.cls.expand(B, -1)[:, None]]
            ms = [torch.zeros(B, 1, dtype=torch.bool, device=dev)]
            for j in range(6):
                h = self.proj[j](cands[j]) + self.cemb[j] + self.remb[:K]
                nul = (self.null[j] + self.cemb[j] + self.remb[K]).expand(B, 1, -1)
                toks += [h, nul]
                ms += [~masks[j], torch.zeros(B, 1, dtype=torch.bool, device=dev)]
            h = torch.cat(toks, 1); m = torch.cat(ms, 1)
            h = self.norm(self.enc(h, src_key_padding_mask=m))
            z = h[:, 0]
            lab = self.lab(z); suc = self.suc(z).squeeze(-1)
            pl = self.ptr(h[:, 1:]).squeeze(-1).view(B, 6, K + 1)
            pl = pl.masked_fill(m[:, 1:].view(B, 6, K + 1), -1e4)
            ev = torch.logsumexp(pl[:, :, :K], -1) - pl[:, :, K]
            lab = torch.cat([lab[:, :1], lab[:, 1:] + self.pool_a * ev.clamp(-20, 20)], 1)
            return lab, suc, pl

    return Net()


def make_residual(spec, d):
    import torch
    import torch.nn as nn

    class Residual(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('indices', torch.tensor(spec['indices'], dtype=torch.float32))
            self.register_buffer('input_dim', torch.tensor(float(spec['input_dim'])))
            self.frequencies = nn.Parameter(torch.empty(7, 16))
            nn.init.trunc_normal_(self.frequencies, mean=0., std=.01, a=-.03, b=.03)
            self.outer = nn.Linear(32, 24)
            self.output = nn.Linear(168, d, bias=False)
            nn.init.zeros_(self.output.weight)

        def forward(self, x):
            numeric = x.index_select(-1, self.indices.long())
            angles = (2 * math.pi) * numeric[..., None] * self.frequencies
            embedding = self.outer(torch.cat([angles.cos(), angles.sin()], -1)).relu()
            return self.output(embedding.flatten(-2))

    return Residual()


def make_net(Fc, specs, c=CFG):
    import torch
    import torch.nn as nn
    net = _net(Fc, c)
    original = {key: value.detach().cpu().numpy().copy() for key, value in net.state_dict().items()}
    cpu_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []

    class Projection(nn.Sequential):
        def forward(self, x):
            # Named 0..3 original modules preserve every old state_dict key.
            old = self[3](self[2](self[1](self[0](x))))
            return old + self.numeric(x)

    # All new modules are made on CPU. Never manual_seed inside this block:
    # it would change CUDA RNG states that devices=[] does not save.
    with torch.random.fork_rng(devices=[]):
        for j, fault in enumerate(FAULTS):
            if specs[fault]['input_dim'] != Fc[j]:
                raise ValueError('original feature width changed')
            projection = Projection(*list(net.proj[j]))
            projection.numeric = make_residual(specs[fault], c['d'])
            net.proj[j] = projection
    net.register_buffer('v51_mode', torch.tensor(1.))
    net.register_buffer('v51_indices', torch.tensor(np.stack([specs[f]['indices'] for f in FAULTS]), dtype=torch.float32))
    net.register_buffer('v51_dims', torch.tensor(Fc, dtype=torch.float32))
    # These persisted markers identify the retained v51 residual prior.
    net.register_buffer('v51_schema_version', torch.tensor(1.))
    old_preserved = all(np.array_equal(value, net.state_dict()[key].detach().cpu().numpy()) for key, value in original.items())
    cpu_preserved = torch.equal(cpu_before, torch.get_rng_state())
    cuda_after = torch.cuda.get_rng_state_all() if cuda_before else []
    cuda_preserved = all(torch.equal(a, b) for a, b in zip(cuda_before, cuda_after))
    if not old_preserved or not cpu_preserved or not cuda_preserved:
        raise ValueError('new CSET modules disturbed original initialization or RNG')
    net.v51_initialization = dict(original_initialization_preserved=old_preserved,
        cpu_rng_preserved=cpu_preserved, cuda_rng_preserved=cuda_preserved,
        initial_backbone_sha256=digest_arrays(original))
    return net


def residual_numpy(w, x):
    x = np.asarray(x, np.float32)
    if x.shape[-1] != int(np.asarray(w['input_dim']).item()):
        raise ValueError('residual original input width mismatch')
    numeric = x[..., np.asarray(w['indices'], np.int64)]
    angles = (2 * math.pi) * numeric[..., None] * w['frequencies']
    embedding = np.concatenate([np.cos(angles), np.sin(angles)], -1) @ w['outer.weight'].T + w['outer.bias']
    embedding = np.maximum(embedding, 0.)
    return embedding.reshape(x.shape[:-1] + (168,)) @ w['output.weight'].T


def validate_weights(w):
    """Validate the retained PLR prior, including original feature mapping."""
    if 'v51_mode' not in w or 'v51_schema_version' not in w:
        raise ValueError('legacy CSET prior lacks v51 training provenance')
    if (float(np.asarray(w['v51_mode']).item()) != 1.
            or float(np.asarray(w['v51_schema_version']).item()) != 1.):
        raise ValueError('unsupported v51 residual model schema or mode')
    indices, dims = np.asarray(w['v51_indices']), np.asarray(w['v51_dims'])
    if indices.shape != (6, 7) or dims.shape != (6,) or not np.array_equal(indices, indices.astype(int)):
        raise ValueError('invalid persisted numerical mapping')
    for j in range(6):
        if len(set(indices[j].tolist())) != 7 or (indices[j] < 0).any() or (indices[j] >= dims[j]).any():
            raise ValueError('invalid persisted numerical indices')
        if w['proj.%d.1.weight' % j].shape[1] != int(dims[j]):
            raise ValueError('original projection width was not preserved')
        prefix = 'proj.%d.numeric.' % j
        if not any(key.startswith(prefix) for key in w):
            raise ValueError('saved v51 residual is missing')
        if (not np.array_equal(w[prefix + 'indices'], indices[j])
                or int(w[prefix + 'input_dim']) != dims[j]
                or w[prefix + 'frequencies'].shape != (7, 16)
                or w[prefix + 'outer.weight'].shape != (24, 32)
                or w[prefix + 'outer.bias'].shape != (24,)
                or w[prefix + 'output.weight'].shape != (w['cls'].shape[0], 168)):
            raise ValueError('residual weights have incompatible dimensions')
    if any(not np.isfinite(value).all() for value in w.values()):
        raise ValueError('nonfinite saved CSET state')
    return 'plr_residual'


def prepare_actions(mask, turn, y, ft, K):
    """Audit-only action support; this does not enter the retained training loss.

    Exact compression of the original integer action domain by breakpoints.

    Candidate k contributes on integer interval [t_k-2,t_k+2]. Cover masks
    change only at t_k-2 and t_k+3. Split additionally at ft-2 and ft+3 so
    that each segment is wholly good or bad. One representative per segment
    preserves both maxima exactly, without clipping large turn spans.
    """
    labels, truth = np.asarray(y), np.asarray(ft)
    if labels.ndim != 1 or truth.shape != labels.shape or not np.isin(labels, np.arange(7)).all():
        raise ValueError('invalid training-only decoder targets')
    n = len(labels)
    rows = []
    eligible = np.zeros(n, bool)
    heads = np.maximum(labels.astype(np.int64) - 1, 0)
    for i in range(n):
        good, bad = [], []
        if labels[i] > 0 and np.isfinite(truth[i]) and truth[i] >= 0:
            fault = FAULTS[heads[i]]
            valid = np.asarray(mask[fault][i], bool)
            turns = np.asarray(turn[fault][i])
            if valid.shape != (K,) or turns.shape != (K,):
                raise ValueError('invalid training candidate support shape')
            valid = valid & np.isfinite(turns) & (turns >= 0)
            if valid.any():
                tt = [int(t) for t in turns[valid]]
                lo, end = min(tt) - 2, max(tt) + 3
                points = {lo, end}
                points.update(t - 2 for t in tt)
                points.update(t + 3 for t in tt)
                points.update(p for p in (int(truth[i]) - 2, int(truth[i]) + 3) if lo < p < end)
                seen = (set(), set())
                for action in sorted(points)[:-1]:
                    cover = tuple(bool(v and abs(int(t) - action) <= 2) for v, t in zip(valid, turns))
                    is_good = abs(action - int(truth[i])) <= 2
                    group = good if is_good else bad
                    key = 0 if is_good else 1
                    if cover not in seen[key]:
                        group.append(cover); seen[key].add(cover)
                eligible[i] = bool(good and bad and np.asarray(good, bool).any())
        rows.append((good, bad))
    result = dict(heads=heads, eligible=eligible)
    for name, offset in (('good', 0), ('bad', 1)):
        width = max(1, max((len(row[offset]) for row in rows), default=0))
        covers = np.zeros((n, width, K), bool)
        available = np.zeros((n, width), bool)
        for i, row in enumerate(rows):
            if row[offset]:
                covers[i, :len(row[offset])] = row[offset]
                available[i, :len(row[offset])] = True
        result[name] = covers
        result[name + '_available'] = available
    return result


def _train_members(cand, mask, turn, y, succ, ft, c, device, log, specs, initializations):
    """Fit c['n_ens'] members on all given rows; returns their weights as numpy dicts (state_dict order)."""
    import torch
    import torch.nn.functional as Fn
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    K = c["K"]; N = len(y)
    X = [torch.tensor(cand[f], device=dev) for f in FAULTS]
    M = [torch.tensor(mask[f], device=dev) for f in FAULTS]
    yt = torch.tensor(np.asarray(y), device=dev, dtype=torch.long)
    st = torch.tensor(np.asarray(succ, np.float32), device=dev)
    PT = np.full((N, 6), K, np.int64)                     # pointer target: true candidate slot, K = none
    for i in np.where(np.asarray(y) > 0)[0]:
        j = int(y[i]) - 1
        hit = np.where(mask[FAULTS[j]][i] & (turn[FAULTS[j]][i] == ft[i]))[0]
        if len(hit):
            PT[i, j] = hit[0]
    PT = torch.tensor(PT, device=dev)
    out = []
    for e in range(c["n_ens"]):
        torch.manual_seed(c["seed0"] + e); np.random.seed(c["seed0"] + e)
        net = make_net([x.shape[-1] for x in X], specs, c)
        initializations.append(dict(net.v51_initialization))
        net = net.to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=c["lr"], weight_decay=c["wd"])
        spe = math.ceil(N / c["bs"]); tot = spe * c["epochs"]; warm = max(1, int(0.05 * tot))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / tot))))
        for ep in range(c["epochs"]):
            net.train()
            perm = torch.randperm(N, device=dev)
            for b in range(spe):
                ix = perm[b * c["bs"]:(b + 1) * c["bs"]]
                lab, suc, pl = net([x[ix] for x in X], [m[ix] for m in M])
                loss = Fn.cross_entropy(lab, yt[ix])
                loss = loss + c["suc_w"] * Fn.binary_cross_entropy_with_logits(suc, st[ix])
                lp = Fn.cross_entropy(pl.reshape(-1, K + 1), PT[ix].reshape(-1), reduction="none").view(-1, 6)
                loss = loss + c["ptr_w"] * lp.mean()
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step(); sched.step()
        out.append({k: v.detach().float().cpu().numpy() for k, v in net.state_dict().items()})
    return out


def train(cand, mask, turn, y, succ, ft, c=CFG, device=None, log=print):
    """Train the retained five-member residual model, then audit its replay."""
    try:
        if c != FORMAL_CFG or CFG != FORMAL_CFG:
            raise ValueError('formal v51 CSET parameters cannot be overridden')
        if _CONTEXT.get('pending') or _CONTEXT.get('complete'):
            raise ValueError('multiple CSET train calls are not registered')
        started = time.monotonic()
        specs = fit_specs(cand, mask, _CONTEXT['indices'])
        # Preserve historical evidence without constructing unused loss tensors.
        support = prepare_actions(mask, turn, y, ft, c['K'])
        hashes = {name: digest_arrays(values) for name, values in
                  [('train_cand', cand), ('train_mask', mask), ('train_turn', turn)]}
        for name, values in [('train_y', y), ('train_success', succ), ('train_fault_turn', ft)]:
            hashes[name] = digest_arrays({'values': np.asarray(values)})
        initializations = []
        weights = _train_members(cand, mask, turn, y, succ, ft, c, device, log, specs, initializations)
        if len(weights) != 5 or len(initializations) != 5:
            raise ValueError('CSET ensemble is incomplete')
        summaries = []
        for e, w in enumerate(weights):
            summary = replay_member(w, specs, cand, mask, c)
            summary.update(initializations[e], seed=1000 + e, epochs=8)
            summaries.append(summary)
        _CONTEXT.update(pending=True, real_rows=len(y), specs=specs, summaries=summaries,
            input_sha256=hashes, started=started, log=log,
            eligible_counts=dict(decoder_margin=int(support['eligible'].sum()),
                real_fault=int((np.asarray(y) > 0).sum()), success_positive=int((np.asarray(succ) > 0).sum())),
            action_support_sha256=digest_arrays(support))
        return weights
    except Exception as exc:
        raise SystemExit('v51 CSET training error: ' + str(exc)) from exc


def _erf(x):
    try:
        from scipy.special import erf
        return erf(x)
    except Exception:                                           # pragma: no cover
        # Abramowitz-Stegun 7.1.26 would lose ~1e-7; use the exact math.erf elementwise instead
        return np.vectorize(math.erf, otypes=[np.float64])(x).astype(x.dtype)


def _gelu(x):
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def _ln(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def _lin(x, W, b):
    return x @ W.T + b


def _forward_numpy(w, cand, mask, c, bs=1024):
    """Original NumPy transformer, with one added residual at the token projection."""
    lin, gelu, ln = _lin, _gelu, _ln
    K, d, H, L = c['K'], c['d'], c['H'], c['L']
    n = cand[FAULTS[0]].shape[0]
    P = np.zeros((n, 7), np.float32); S = np.zeros(n, np.float32)
    PL = np.zeros((n, 6, K + 1), np.float32)
    for b0 in range(0, n, bs):
        sl = slice(b0, min(n, b0 + bs)); B = sl.stop - sl.start
        toks = [np.broadcast_to(w['cls'], (B, d))[:, None]]
        ms = [np.zeros((B, 1), bool)]
        for j, fault in enumerate(FAULTS):
            x = cand[fault][sl].astype(np.float32)
            h = lin(gelu(lin(x, w['proj.%d.1.weight' % j], w['proj.%d.1.bias' % j])),
                    w['proj.%d.3.weight' % j], w['proj.%d.3.bias' % j])
            prefix = 'proj.%d.numeric.' % j
            branch = {key[len(prefix):]: value for key, value in w.items() if key.startswith(prefix)}
            h = h + residual_numpy(branch, x)
            h = h + w['cemb'][j] + w['remb'][:K]
            null = np.broadcast_to(w['null'][j] + w['cemb'][j] + w['remb'][K], (B, 1, d))
            toks += [h, null]
            ms += [~mask[fault][sl], np.zeros((B, 1), bool)]
        h = np.concatenate(toks, 1).astype(np.float32); masked = np.concatenate(ms, 1)
        T = h.shape[1]; dh = d // H
        negative = np.where(masked, -np.inf, 0.).astype(np.float32)[:, None, None, :]
        for layer in range(L):
            prefix = 'enc.layers.%d.' % layer
            x = ln(h, w[prefix + 'norm1.weight'], w[prefix + 'norm1.bias'])
            qkv = lin(x, w[prefix + 'self_attn.in_proj_weight'], w[prefix + 'self_attn.in_proj_bias'])
            q, key, value = np.split(qkv, 3, -1)
            q = q.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
            key = key.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
            value = value.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
            attention = q @ key.transpose(0, 1, 3, 2) / math.sqrt(dh) + negative
            attention = np.exp(attention - attention.max(-1, keepdims=True)); attention /= attention.sum(-1, keepdims=True)
            output = (attention @ value).transpose(0, 2, 1, 3).reshape(B, T, d)
            h = h + lin(output, w[prefix + 'self_attn.out_proj.weight'], w[prefix + 'self_attn.out_proj.bias'])
            x = ln(h, w[prefix + 'norm2.weight'], w[prefix + 'norm2.bias'])
            h = h + lin(gelu(lin(x, w[prefix + 'linear1.weight'], w[prefix + 'linear1.bias'])),
                        w[prefix + 'linear2.weight'], w[prefix + 'linear2.bias'])
        h = ln(h, w['norm.weight'], w['norm.bias'])
        z = h[:, 0]
        lab = lin(z, w['lab.weight'], w['lab.bias'])
        success = lin(z, w['suc.weight'], w['suc.bias'])[:, 0]
        pointer = lin(h[:, 1:], w['ptr.weight'], w['ptr.bias'])[..., 0].reshape(B, 6, K + 1)
        pointer = np.where(masked[:, 1:].reshape(B, 6, K + 1), -1e4, pointer)
        maximum = pointer[:, :, :K].max(-1, keepdims=True)
        lse = (np.log(np.exp(pointer[:, :, :K] - maximum).sum(-1, keepdims=True)) + maximum)[..., 0]
        evidence = np.clip(lse - pointer[:, :, K], -20, 20)
        lab[:, 1:] = lab[:, 1:] + w['pool_a'] * evidence
        exp = np.exp(lab - lab.max(1, keepdims=True))
        P[sl] = exp / exp.sum(1, keepdims=True)
        S[sl] = 1 / (1 + np.exp(-success)); PL[sl] = pointer
    return P, S, PL


def forward_np(w, cand, mask, c=CFG, bs=1024):
    """Evaluate one compatible member and collect optional replay evidence."""
    try:
        validate_weights(w)
        output = _forward_numpy(w, cand, mask, c, bs)
        if _CONTEXT.get('collect') is not None:
            _CONTEXT['collect'].append(digest_arrays(dict(zip(
                ('probabilities', 'success', 'pointer_logits'), output))))
        return output
    except Exception as exc:
        raise SystemExit('v51 CSET inference error: ' + str(exc)) from exc


def replay_member(w, specs, cand, mask, c):
    import torch
    validate_weights(w)
    stream = io.BytesIO()
    np.savez_compressed(stream, **w)
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as archive:
        loaded = {key: archive[key] for key in archive.files}
    if digest_arrays(loaded) != digest_arrays(w):
        raise ValueError('NPZ state replay changed model values')
    # Audit construction is outside training; nevertheless restore its CPU RNG.
    with torch.random.fork_rng(devices=[]):
        net = make_net([specs[f]['input_dim'] for f in FAULTS], specs, c).cpu().eval()
    net.load_state_dict({key: torch.tensor(value) for key, value in loaded.items()}, strict=True)
    small = {f: np.asarray(cand[f])[:16] for f in FAULTS}
    masks = {f: np.asarray(mask[f])[:16] for f in FAULTS}
    with torch.no_grad():
        p, s, pointer = net([torch.tensor(small[f]) for f in FAULTS], [torch.tensor(masks[f]) for f in FAULTS])
        expected = (p.softmax(1).numpy(), s.sigmoid().numpy(), pointer.numpy())
    actual = _forward_numpy(loaded, small, masks, c)
    error = max(float(np.max(np.abs(a - b), initial=0)) for a, b in zip(actual, expected))
    if not np.isfinite(error) or error > 1e-4:
        raise ValueError('CSET Torch/NumPy replay exceeds tolerance: %g' % error)
    again = _forward_numpy(w, small, masks, c)
    # tolerance instead of bit equality: multi-threaded BLAS may change the last bits under CPU contention
    if any(not np.allclose(a, b, rtol=0, atol=1e-6) for a, b in zip(actual, again)):
        raise ValueError('CSET reloaded predictions differ')
    return dict(state_sha256=digest_arrays(w), buffer_sha256=digest_arrays({key: loaded[key] for key, _ in net.named_buffers()}),
                train_replay_sha256=digest_arrays(dict(zip(('probabilities', 'success', 'pointer_logits'), actual))),
                numpy_replay=True, save_load_replay=True, max_abs_error=error)



def pool_success(probabilities):
    """Equal log-odds pooling with fixed clipping for finite endpoint values."""
    probabilities = np.clip(np.asarray(probabilities, np.float64), 1e-6, 1 - 1e-6)
    logits = np.log(probabilities) - np.log1p(-probabilities)
    return 1 / (1 + np.exp(-logits.mean(axis=0)))


def conditional_softmax(logits, valid):
    """Normalize valid candidates only; an empty candidate set stays zero."""
    masked = np.where(valid, logits, -np.inf)
    maximum = masked.max(axis=-1, keepdims=True)
    maximum = np.where(np.isfinite(maximum), maximum, 0.)
    mass = np.exp(masked - maximum)
    total = mass.sum(axis=-1, keepdims=True)
    return np.divide(mass, total, out=np.zeros_like(mass), where=total > 0)


def pool_pointer(logits, valid):
    """Normalize the mean candidate logits, excluding the per-member null."""
    candidates = np.asarray(logits, np.float64)[..., :valid.shape[-1]]
    return conditional_softmax(candidates.mean(axis=0), valid)


def _predict_ensemble(weights: List[Dict[str, np.ndarray]], cand, mask, turn, c=CFG):
    """Fixed pooling -> P (n,7), S (n,), T (n,6); retain the original window decoder."""
    Ps, Ss, PLs = [], [], []
    for w in weights:
        P, S, PL = forward_np(w, cand, mask, c)
        PLs.append(PL)
        Ps.append(P); Ss.append(S)
    valid = np.stack([mask[f] for f in FAULTS], axis=1)
    P, S, PL = np.mean(Ps, 0), pool_success(Ss), pool_pointer(PLs, valid)
    K = c["K"]; n = len(S)
    T = np.full((n, 6), -1, np.int64)
    for j, f in enumerate(FAULTS):
        sc = np.where(mask[f], PL[:, j, :K], -np.inf)
        a = sc.argmax(1); has = mask[f].any(1)
        T[has, j] = turn[f][np.where(has)[0], a[has]]
        for i in np.where(has)[0]:
            m = np.asarray(mask[f][i], bool)
            w = _window_turn(np.asarray(turn[f][i])[m], PL[i, j, :K][m])
            if w is not None:
                T[i, j] = w
    return P, S, T


def _window_turn(tt, p):
    tt = np.asarray(tt, float); p = np.clip(np.asarray(p, float), 0, None)
    ok = np.isfinite(tt) & (tt >= 0)
    if not ok.any() or p[ok].sum() <= 0:
        return None
    tt, p = tt[ok].astype(int), p[ok]
    am = tt[p.argmax()]
    opts = np.arange(tt.min() - 2, tt.max() + 3)
    mass = (np.abs(opts[:, None] - tt[None, :]) <= 2).astype(float) @ p
    cands = opts[mass >= mass.max() - 1e-9 * max(1.0, mass.max())]
    return int(cands[np.argmin(np.abs(cands - am))])


def predict(weights, cand, mask, turn, c=CFG):
    """Average members, decode window turns, and finish any training audit."""
    try:
        _CONTEXT['collect'] = [] if _CONTEXT.get('pending') else None
        output = _predict_ensemble(weights, cand, mask, turn, c)
        if _CONTEXT.get('pending'):
            if c != FORMAL_CFG or len(weights) != 5 or len(_CONTEXT['collect']) != 5:
                raise ValueError('final CSET inference is incomplete')
            for e, w in enumerate(weights):
                if digest_arrays(w) != _CONTEXT['summaries'][e]['state_sha256']:
                    raise ValueError('trained CSET state changed before prediction')
                _CONTEXT['summaries'][e]['test_output_sha256'] = _CONTEXT['collect'][e]
            hashes = dict(_CONTEXT['input_sha256'])
            for name, values in [('test_cand', cand), ('test_mask', mask), ('test_turn', turn)]:
                hashes[name] = digest_arrays(values)
            specs = _CONTEXT['specs']
            metadata = dict(schema=SCHEMA, arm='conditional_logpool', mode='conditional_logpool',
                scheme=dict(residual=True, target_aux=False),
                real_rows=_CONTEXT['real_rows'], test_rows=len(output[1]), members=5, epochs=8,
                seeds=list(range(1000, 1005)), config=dict(FORMAL_CFG), numeric_features=list(NUMERIC_FEATURES),
                feature_indices=_CONTEXT['indices'], feature_dims={f: specs[f]['input_dim'] for f in FAULTS},
                expanded_dims={f: specs[f]['input_dim'] for f in FAULTS},
                train_valid_tokens={f: specs[f]['valid_tokens'] for f in FAULTS},
                fit_scope='real_training_valid_top8_tokens_only', input_sha256=hashes,
                embedding=dict(EMBEDDING), aggregation=dict(AGGREGATION), auxiliary_losses=None,
                eligible_counts=_CONTEXT['eligible_counts'], action_support_sha256=_CONTEXT['action_support_sha256'],
                auxiliary_training_batches=0,
                action_support='exact_integer_decoder_domain_compressed_at_coverage_and_truth_boundaries',
                member_summaries=_CONTEXT['summaries'],
                output_sha256={name: digest_arrays({'values': value}) for name, value in
                               zip(('probabilities', 'success', 'turns'), output)},
                runtime_seconds=time.monotonic() - _CONTEXT['started'], complete=True)
            _CONTEXT['log']('v55 CSET validated: ' + json.dumps(metadata, sort_keys=True, separators=(',', ':')))
            _CONTEXT['log']('v55 cset arm=conditional_logpool real_rows=%d members=5 epochs=8' % _CONTEXT['real_rows'])
            _CONTEXT['pending'] = False
            _CONTEXT['complete'] = True
        _CONTEXT['collect'] = None
        return output
    except Exception as exc:
        raise SystemExit('v51 CSET completion error: ' + str(exc)) from exc


def save(path: str, weights, st, c=CFG):
    arr = {}
    for i, w in enumerate(weights):
        for k, v in w.items():
            arr["m%d|%s" % (i, k)] = v
    for f in FAULTS:
        arr["st|%s|mu" % f] = st[f]["mu"]; arr["st|%s|sd" % f] = st[f]["sd"]
        arr["st|%s|cols" % f] = np.array(st[f]["cols"], dtype=object)
    np.savez_compressed(path, **arr)


def _load_archive(path):
    """Read exactly the original NPZ layout and close the archive promptly."""
    ws: Dict[int, Dict] = {}
    st = {f: {} for f in FAULTS}
    with np.load(path, allow_pickle=True) as archive:
        for key in archive.files:
            if key.startswith('m'):
                i, name = key[1:].split('|', 1)
                ws.setdefault(int(i), {})[name] = archive[key]
            else:
                _, fault, what = key.split('|')
                st[fault][what] = list(archive[key]) if what == 'cols' else archive[key]
    return [ws[i] for i in sorted(ws)], st


def load(path):
    """Load a complete v51 residual prior with its original feature stats."""
    try:
        weights, st = _load_archive(path)
        if len(weights) != FORMAL_CFG['n_ens']:
            raise ValueError('saved v51 prior must contain all five members')
        indices = feature_indices(st)
        for w in weights:
            validate_weights(w)
            for j, fault in enumerate(FAULTS):
                if (not np.array_equal(w['v51_indices'][j], indices[fault])
                        or int(w['v51_dims'][j]) != len(st[fault]['cols']) + 5):
                    raise ValueError('saved CSET feature mapping differs from prior columns')
        return weights, st
    except Exception as exc:
        raise SystemExit('v51 CSET incompatible prior: ' + str(exc)) from exc
