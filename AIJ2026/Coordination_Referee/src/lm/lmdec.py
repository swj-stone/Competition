"""Decoder-LM (Qwen3) referee head, shared by training and in-package inference.

Causal backbone, right padding.  Class + success heads on [last token ; mean-pool]; per-class fault-turn pointer on
the LAST token of every turn block (message line + its state/artifact lines) concatenated with the last-token state,
so each turn sees its own content, its history, and (through the summary) the whole run."""
import numpy as np
import torch
import torch.nn as nn

from serial import serialize, LABELS

MAXLEN = 8192


def encode(tok, row):
    text, offs = serialize(row)
    text = text + (tok.eos_token or "")
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False, truncation=True, max_length=MAXLEN)
    ids = enc["input_ids"]
    ends = np.array([o[1] for o in enc["offset_mapping"]], dtype=np.int64)
    ts = sorted(offs, key=lambda t: offs[t])
    starts = [offs[t] for t in ts]
    block_end = starts[1:] + [len(text) - len(tok.eos_token or "")]       # char offset where the next block starts
    # last token whose end <= block_end (the block's final newline included)
    pos = np.searchsorted(ends, np.asarray(block_end, np.int64), side="right") - 1
    keep = (pos >= 0) & (pos < len(ids))
    return dict(ids=np.asarray(ids, np.int32), ts=np.asarray(ts, np.int64)[keep], pos=pos[keep].astype(np.int64))


class RefereeDec(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        d = enc.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, len(LABELS) + 1))
        self.ptr = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, len(LABELS) - 1))

    def forward(self, ids, att, pos, pmask):
        h = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        last_i = att.sum(1) - 1
        last = h[torch.arange(h.size(0), device=h.device), last_i]
        m = att.unsqueeze(-1).to(h.dtype)
        mean = (h * m).sum(1) / m.sum(1).clamp_min(1)
        z = self.head(torch.cat([last, mean], -1))
        g = torch.gather(h, 1, pos.unsqueeze(-1).expand(-1, -1, h.size(-1)))
        g = torch.cat([g, last.unsqueeze(1).expand(-1, g.size(1), -1)], -1)
        pl = self.ptr(g).float().masked_fill(~pmask.unsqueeze(-1), -1e4)
        return z[:, :len(LABELS)].float(), z[:, len(LABELS)].float(), pl
