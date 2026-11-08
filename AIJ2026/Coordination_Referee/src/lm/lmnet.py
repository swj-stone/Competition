"""ModernBERT multi-task referee head (shared by training and in-package inference).

Heads: 7-way fault class + success logit from [CLS ; mean-pool]; per-class fault-turn pointer (6 channels) on the
first token of every message line.  Encode = serialize -> tokenize -> message-start token positions."""
import numpy as np
import torch
import torch.nn as nn

from serial import serialize, LABELS

MAXLEN = 8192


def encode(tok, row):
    text, offs = serialize(row)
    enc = tok(text, return_offsets_mapping=True, truncation=True, max_length=MAXLEN)
    ids = enc["input_ids"]
    starts = np.array([o[0] if o[1] > o[0] else -1 for o in enc["offset_mapping"]], dtype=np.int64)
    starts[0] = -1
    for j in range(len(starts) - 1, 0, -1):          # trailing specials
        if starts[j] >= 0:
            break
        starts[j] = 1 << 40
    mono = np.maximum.accumulate(starts)
    ts = sorted(offs, key=lambda t: offs[t])
    pos = np.searchsorted(mono, [offs[t] for t in ts], side="left")
    keep = pos < len(ids) - 1
    return dict(ids=np.asarray(ids, np.int32), ts=np.asarray(ts, np.int64)[keep], pos=pos[keep].astype(np.int64))


class Referee(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        d = enc.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, len(LABELS) + 1))
        self.ptr = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, len(LABELS) - 1))

    def forward(self, ids, att, pos, pmask):
        h = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        m = att.unsqueeze(-1).to(h.dtype)
        mean = (h * m).sum(1) / m.sum(1).clamp_min(1)
        z = self.head(torch.cat([h[:, 0], mean], -1))
        g = torch.gather(h, 1, pos.unsqueeze(-1).expand(-1, -1, h.size(-1)))
        pl = self.ptr(g).float().masked_fill(~pmask.unsqueeze(-1), -1e4)       # (B, M, 6)
        return z[:, :len(LABELS)].float(), z[:, len(LABELS)].float(), pl


def collate(items, pad_id):
    B = len(items)
    L = max(len(it["ids"]) for it in items)
    M = max(max(len(it["pos"]), 1) for it in items)
    ids = np.full((B, L), pad_id, np.int64); att = np.zeros((B, L), np.int64)
    pos = np.zeros((B, M), np.int64); pm = np.zeros((B, M), bool)
    for i, it in enumerate(items):
        n = len(it["ids"]); ids[i, :n] = it["ids"]; att[i, :n] = 1
        k = len(it["pos"]); pos[i, :k] = it["pos"]; pm[i, :k] = True
    return torch.from_numpy(ids), torch.from_numpy(att), torch.from_numpy(pos), torch.from_numpy(pm)


def batches_by_tokens(lengths, budget, order):
    out, cur, mx = [], [], 0
    for i in order:
        l = lengths[i]
        if cur and max(mx, l) * (len(cur) + 1) > budget:
            out.append(cur); cur, mx = [], 0
        cur.append(i); mx = max(mx, l)
    if cur:
        out.append(cur)
    return out


@torch.no_grad()
def predict(model, items, pad_id, device, budget=65536):
    """-> P (N,7), S (N,), list of per-run pointer prob arrays (M_i, 6) aligned with items[i]['ts']."""
    model.eval()
    lengths = np.array([len(it["ids"]) for it in items])
    order = np.argsort(lengths)
    P = np.zeros((len(items), len(LABELS)), np.float32); S = np.zeros(len(items), np.float32); PT = [None] * len(items)
    for b in batches_by_tokens(lengths, budget, order):
        ids, att, pos, pm = (x.to(device) for x in collate([items[i] for i in b], pad_id))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lc, ls, pl = model(ids, att, pos, pm)
        P[b] = torch.softmax(lc, -1).cpu().numpy(); S[b] = torch.sigmoid(ls).cpu().numpy()
        pp = torch.softmax(pl, 1).cpu().numpy()
        for j, i in enumerate(b):
            PT[i] = pp[j, :len(items[i]["pos"])]
    return P, S, PT
