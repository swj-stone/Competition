"""LM member inference (GPU, inference only): shipped referee weights -> test predictions.

    python lm_infer.py --test test.csv --out lm.pkl

Members live in lm/m<i>/ (config.json + tokenizer + one *.safetensors each; ModernBERT encoders or Qwen3 decoders).
Writes a DataFrame indexed by run_id: P_<label> (mean of member log-probs, renormalised), S (mean success prob),
T1_<fault> (pointer turn of the first member).  Errors propagate."""
import argparse, glob, os, sys, time
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lmnet
import lmdec
from serial import LABELS
from transformers import AutoConfig, AutoModel, AutoTokenizer
from safetensors.torch import load_file

ap = argparse.ArgumentParser()
ap.add_argument("--test", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--budget", type=int, default=65536)
a = ap.parse_args()
t0 = time.time()
if not torch.cuda.is_available() and os.environ.get("CR_LM_ALLOW_CPU") != "1":
    raise SystemExit("lm_infer: CUDA device required")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.enable_cudnn_sdp(False)                  # cuDNN SDPA: "No valid execution plans" with padding masks
torch.backends.cuda.matmul.allow_tf32 = False
rows = pd.read_csv(a.test)
ids = rows["run_id"].values
rows = rows.to_dict("records")
members = sorted(d for d in glob.glob(os.path.join(HERE, "m*")) if os.path.isfile(os.path.join(d, "config.json")))
assert members, "no LM members shipped"
LP, SS, T1, n, cache = 0.0, 0.0, None, 0, {}
for d in members:
    cfg = AutoConfig.from_pretrained(d)
    dec = cfg.model_type.startswith("qwen")
    tok = AutoTokenizer.from_pretrained(d)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    key = (cfg.model_type, tok.vocab_size)
    if key not in cache:
        cache[key] = [(lmdec.encode if dec else lmnet.encode)(tok, r) for r in rows]
    items = cache[key]
    for path in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        enc = AutoModel.from_config(cfg, attn_implementation="sdpa")
        model = lmdec.RefereeDec(enc) if dec else lmnet.Referee(enc)
        sd = load_file(path)
        missing, unexpected = model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
        assert not missing and not unexpected, (missing[:5], unexpected[:5])
        model.to(dev)
        P, S, PT = lmnet.predict(model, items, tok.pad_token_id, dev, budget=a.budget)
        LP = LP + np.log(np.clip(P, 1e-6, 1.0)); SS = SS + S; n += 1
        if T1 is None:
            T1 = np.full((len(items), 6), -1, np.int64)
            for i, (pp, it) in enumerate(zip(PT, items)):
                if pp is not None and len(pp):
                    T1[i] = it["ts"][pp.argmax(0)]
        del model, enc, sd; torch.cuda.empty_cache()
        print("lm_infer: %s/%s done %.0fs" % (os.path.basename(d), os.path.basename(path), time.time() - t0), flush=True)
LP = LP / n; LP -= np.log(np.exp(LP).sum(1, keepdims=True))
df = pd.DataFrame(np.exp(LP), columns=["P_" + l for l in LABELS], index=ids)
df["S"] = SS / n
for j, l in enumerate(LABELS[1:]):
    df["T1_" + l] = T1[:, j]
df.to_pickle(a.out)
print("lm_infer: done %d runs, %d members, %.0fs" % (len(df), n, time.time() - t0), flush=True)
