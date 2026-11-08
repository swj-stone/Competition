"""Prepare and run the retained Coordination Referee submission with the active Python."""
import argparse
import csv
import gzip
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "assets" / "submission_v84b_v81_lm2.zip"
RUNTIME = ROOT / "runtime" / "v84b"
DIGEST = "918ae44f7fbb01756594c165e19dd8969e7d599ec3dfbe1dea53952129bbd8c4"
FEATURES = {"run_id", "goal", "agents", "shared_state", "messages", "artifacts", "topology"}
TARGETS = {"label", "success", "fault_turn"}
LABELS = {"clean", "dropped_handoff", "duplicated_work", "deadlock", "conflict",
          "goal_drift", "runaway_loop"}
csv.field_size_limit(128 * 1024 * 1024)


def sha(stream):
    return hashlib.file_digest(stream, "sha256").hexdigest()


def verified_archive():
    with ARCHIVE.open("rb") as f:
        if sha(f) != DIGEST:
            raise ValueError(f"Wrong or damaged archive: {ARCHIVE}")
    print(f"Verified original archive (including both LM checkpoints and CSET weights): {DIGEST}",
          flush=True)


def verify_tree(z):
    expected = set(z.namelist())
    files = set()
    for p in RUNTIME.rglob("*"):
        if p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction()):
            raise ValueError(f"Unexpected link in extracted runtime: {p}")
        if p.is_file():
            files.add(p.relative_to(RUNTIME).as_posix())
    if files != expected:
        raise ValueError(f"Runtime file list changed (missing: {sorted(expected-files)}; "
                         f"unexpected: {sorted(files-expected)}). Move {RUNTIME} aside and prepare again.")
    for name in z.namelist():
        with z.open(name) as original, (RUNTIME / name).open("rb") as local:
            if sha(local) != sha(original):
                raise ValueError(f"Extracted file differs from original: {name}. Move {RUNTIME} aside and prepare again.")
    print(f"Verified {len(files)} extracted files against the original archive.", flush=True)


def prepare():
    verified_archive()
    with ZipFile(ARCHIVE) as z:
        if RUNTIME.exists():
            verify_tree(z)
        else:
            RUNTIME.parent.mkdir(parents=True, exist_ok=True)
            # Only the exact archive with the pinned hash may be extracted here.
            with tempfile.TemporaryDirectory(dir=RUNTIME.parent) as temp:
                staging = Path(temp) / "v84b"
                z.extractall(staging)
                staging.rename(RUNTIME)
            print(f"Prepared {len(z.namelist())} files in {RUNTIME}", flush=True)


def csv_ids(path, role):
    """Check competition schemas and reject accidental constant-prediction fallback inputs."""
    if not path.is_file():
        raise ValueError(f"{role} CSV does not exist: {path}")
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if len(cols) != len(set(cols)):
            raise ValueError(f"{path}: duplicate column names")
        wanted = {"run_id"} | TARGETS if role == "predictions" else FEATURES
        if role == "train":
            wanted = wanted | TARGETS
        if not wanted.issubset(cols):
            raise ValueError(f"{path}: missing columns {sorted(wanted - set(cols))}")
        if role == "predictions" and set(cols) != wanted:
            raise ValueError(f"{path}: unexpected prediction columns")
        if role == "test" and TARGETS.intersection(cols):
            raise ValueError(f"{path}: remove label, success and fault_turn from test inputs")
        ids = set()
        for i, row in enumerate(reader, 2):
            rid = row["run_id"]
            if None in row or None in row.values() or not rid.strip() or rid in ids:
                raise ValueError(f"{path}: malformed row, blank or duplicate run_id at record {i}")
            ids.add(rid)
            if role != "test":
                label = row["label"]
                turn = int(row["fault_turn"])
                if (label not in LABELS or row["success"] not in {"0", "1"} or
                        turn < -1 or (label == "clean" and turn != -1)):
                    raise ValueError(f"{path}: invalid targets for run {rid}")
        if not ids or (role == "train" and len(ids) < 50):
            raise ValueError(f"{path}: too few {role} records ({len(ids)})")
        return ids


def check(test_ids, output):
    pred_ids = csv_ids(output, "predictions")
    if pred_ids != test_ids:
        raise ValueError(f"run_id coverage mismatch: missing {len(test_ids-pred_ids)}, "
                         f"unexpected {len(pred_ids-test_ids)}")
    print(f"Checked prediction schema, labels, turns and coverage for {len(pred_ids)} runs.",
          flush=True)


def run_full_pipeline(cmd, job, env):
    # The archive returns first-pass predictions if too few pseudo-labels qualify.
    # In that branch the original driver need not wait for the LM. The shared pipe
    # stays open until its writers finish, and the fusion marker proves the LM was used.
    fused = False
    with subprocess.Popen(cmd, cwd=job, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace") as proc:
        for line in proc.stdout:
            print(line, end="", flush=True)
            fused |= "LM member (w=0.20, a=0.00):" in line
        if proc.wait() != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    if not fused:
        raise ValueError("No v84b LM fusion recorded. Fewer than 20 high-confidence test runs "
                         "can trigger the archive's first-pass-only path. No output published.")


def run(args):
    train, test, output = (p.expanduser().resolve() for p in (args.train, args.test, args.output))
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if args.lm_budget is not None and args.lm_budget < 1:
        raise ValueError("--lm-budget must be positive")
    if output.exists() or output.is_relative_to(RUNTIME):
        raise ValueError("Choose a new output filename outside the extracted runtime.")
    if output.suffix.lower() != ".csv":
        raise ValueError("Use a .csv filename for the uncompressed predictions.")
    test_ids = csv_ids(test, "test")
    if len(test_ids) < 20:
        raise ValueError("Full v84b requires at least 20 high-confidence test runs; use a larger test set.")
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("CR_")}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONIOENCODING="utf-8")
    if args.profile == "supplied-only":
        env.update(CR_NO_EXTRA="1", CR_CSET_MODE="train")
    # The LM subprocess normally starts alongside a long tree model fit. Fail early on CPU.
    subprocess.run([sys.executable, "-c",
                    "import torch; import lightgbm, catboost, sklearn, transformers, safetensors; "
                    "import sys; sys.exit(0 if torch.cuda.is_available() and "
                    "torch.cuda.is_bf16_supported() else 'Full v84b needs a CUDA GPU with bfloat16 support.')"],
                   env=env, check=True)
    prepare()
    # --train can point to the demo CSV inside the runtime just extracted above.
    if csv_ids(train, "train").intersection(test_ids):
        raise ValueError("Train and test run_id sets overlap; supply separate inputs.")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Work on temporary predictions; publish only after the exit code and format checks pass.
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="referee-") as job:
        env.update(TMPDIR=job, TMP=job, TEMP=job)
        candidate = Path(job) / "predictions.csv"
        if args.lm_budget is not None:
            precomputed = Path(job) / "lm.pkl"
            lm_cmd = [sys.executable, "-B", str(RUNTIME / "lm" / "lm_infer.py"),
                      "--test", str(test), "--out", str(precomputed), "--budget", str(args.lm_budget)]
            print(f"Running both LMs first with token budget {args.lm_budget}.", flush=True)
            subprocess.run(lm_cmd, cwd=job, check=True,
                           env=dict(env, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                                    TOKENIZERS_PARALLELISM="false"))
            env["CR_LM_PRECOMPUTED"] = str(precomputed)
        cmd = [sys.executable, "-B", str(RUNTIME / "solution.py"), "--train", str(train),
               "--test", str(test), "--output", str(candidate), "--jobs", str(args.jobs)]
        print(f"Running v84b, profile={args.profile}, jobs={args.jobs}", flush=True)
        run_full_pipeline(cmd, job, env)
        check(test_ids, candidate)
        candidate.rename(output)
    print(f"Wrote {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cmds = parser.add_subparsers(dest="action", required=True)
    cmds.add_parser("verify", help="check the SHA-256 of the packaged original ZIP")
    cmds.add_parser("prepare", help="verify and unpack the ZIP, or verify the existing runtime")
    r = cmds.add_parser("run", help="run the complete v84b pipeline; requires its CUDA environment")
    r.add_argument("--train", type=Path, required=True)
    r.add_argument("--jobs", type=int, default=4)
    r.add_argument("--lm-budget", type=int, help="run LM inference first with this token budget "
                   "(try 8192 for limited VRAM); omit for original concurrent execution / 65536")
    r.add_argument("--profile", choices=("supplied-only", "archive-default"), default="supplied-only")
    c = cmds.add_parser("check", help="check predictions against the unlabeled test CSV, without scoring")
    for sub in (r, c):
        sub.add_argument("--test", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "verify":
        verified_archive()
    elif args.action == "prepare":
        prepare()
    elif args.action == "check":
        check(csv_ids(args.test, "test"), args.output)
    else:
        run(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, csv.Error, subprocess.CalledProcessError) as error:
        sys.exit(f"Error: {error}")
