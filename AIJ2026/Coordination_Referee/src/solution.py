#!/usr/bin/env python3
"""Coordination Referee — submission entry point.

    python solution.py --train train.csv --test test.csv --output predictions.csv

Fixed v51 pipeline (CPU submission / local CUDA training):
  1. semantic parsing of every message (assign / deliver / ack / work / iterate /
     wait / disagree / drift ...) with subtask + counterpart recovery;
  2. run-level evidence features per fault class + per-turn candidate features;
  3. per-class LightGBM candidate scorers (out-of-fold on train) that estimate
     P(this message is the fault turn);
  4. run-level LightGBM (label, success) on features + aggregated candidate scores;
  5. fault_turn = best-scoring candidate of the predicted class.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RUN_COLUMNS = ["goal", "agents", "shared_state", "messages", "artifacts", "topology"]


def _avail_cpus() -> int:
    """CPUs this process may really use: cpu_count capped by the affinity mask and a cgroup CPU quota
    (containers often report the host's cores)."""
    import math
    n = os.cpu_count() or 1
    try:
        n = min(n, len(os.sched_getaffinity(0)))
    except Exception:
        pass
    try:                                                        # cgroup v2
        q, per = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if q != "max":
            n = min(n, max(1, int(math.ceil(int(q) / int(per)))))
    except Exception:
        pass
    try:                                                        # cgroup v1
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        per = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if q > 0 and per > 0:
            n = min(n, max(1, int(math.ceil(q / per))))
    except Exception:
        pass
    return max(1, n)


def _fallback(test: pd.DataFrame, path: str, why: str) -> None:
    print("WARNING: %s -> writing constant predictions" % why, flush=True)
    ids = test["run_id"] if "run_id" in test.columns else pd.Series(
        ["row_%d" % i for i in range(len(test))])
    pd.DataFrame({"run_id": ids, "label": "clean", "success": 1,
                  "fault_turn": -1}).to_csv(path, index=False)


def _add_extra_train(train: pd.DataFrame) -> pd.DataFrame:
    """Append the organisers' public demo training set (repository data/train.csv, shipped as
    extra_train.csv.gz next to this file) to the platform train.  The FAQ explicitly allows training
    on it; it is generated with another seed.  Rows whose content (goal + messages) already occurs in
    the platform train are dropped so that no run can appear twice (duplicates would leak across the
    internal cross-fitting folds).  Disabled when the file is missing or CR_NO_EXTRA=1."""
    path = os.path.join(HERE, "extra_train.csv.gz")
    if os.environ.get("CR_NO_EXTRA") == "1" or not os.path.exists(path):
        return train
    try:
        import hashlib
        ex = pd.read_csv(path)
        need = RUN_COLUMNS + ["label", "success", "fault_turn"]
        if any(c not in ex.columns for c in need):
            return train
        key = lambda df: [hashlib.md5((str(g) + "|" + str(m)).encode()).hexdigest()
                          for g, m in zip(df["goal"], df["messages"])]
        seen = set(key(train))
        ex = ex[[k not in seen for k in key(ex)]].copy()
        ex["run_id"] = "extra_" + ex["run_id"].astype(str)
        cols = [c for c in train.columns if c in ex.columns]
        out = pd.concat([train, ex[cols]], ignore_index=True)
        print("extra training runs: +%d (train %d -> %d)" % (len(ex), len(train), len(out)), flush=True)
        return out
    except Exception as exc:                                    # pragma: no cover
        print("WARNING: extra train not used (%s)" % exc, flush=True)
        return train


def _lm_start(args):
    """Start the LM member (GPU inference of shipped weights) on the test file; joined by the pipeline."""
    import subprocess
    import tempfile
    if os.environ.get("CR_LM_PRECOMPUTED"):                  # local tests: LM output produced elsewhere
        return None, os.environ["CR_LM_PRECOMPUTED"]
    out = os.path.join(tempfile.mkdtemp(prefix="cr_lm_"), "lm.pkl")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    proc = subprocess.Popen([sys.executable, os.path.join(HERE, "lm", "lm_infer.py"), "--test", args.test, "--out", out],
                            env=env)
    return proc, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=os.environ.get("TRAIN_PATH", "train.csv"))
    ap.add_argument("--test", default=os.environ.get("TEST_PATH", "test.csv"))
    ap.add_argument("--output", default=os.environ.get("OUTPUT_PATH", "predictions.csv"))
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("CR_N_JOBS", min(8, _avail_cpus()))))
    args = ap.parse_args()

    random.seed(0)
    np.random.seed(0)

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    miss_te = [c for c in RUN_COLUMNS if c not in test.columns]
    miss_tr = [c for c in RUN_COLUMNS + ["label", "success", "fault_turn"]
               if c not in train.columns]
    if miss_te or miss_tr or len(train) < 50 or len(test) == 0:
        _fallback(test, args.output, "reduced input (test misses %s, train misses %s)"
                  % (miss_te, miss_tr))
        return
    train = _add_extra_train(train)
    for df in (train, test):
        for c in RUN_COLUMNS:
            df[c] = df[c].where(df[c].notna(), None)

    import pipeline
    pipeline.LM_PROC = _lm_start(args)
    out = pipeline.fit_predict(train, test, n_jobs=args.jobs)
    out["success"] = out["success"].astype(int)
    out["fault_turn"] = out["fault_turn"].astype(int)
    out.to_csv(args.output, index=False)
    print("wrote %s (%d rows)" % (args.output, len(out)), flush=True)
    print(out["label"].value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
