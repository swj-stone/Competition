"""Create a small offline I/O exercise from the public demo data in the v84b ZIP.

Run from the reproduction folder: python examples/make_smoke.py --out work/smoke
Never pass truth.csv as input. Run with bundled extra data disabled because it
contains the smoke rows. This is not the official test or an estimate of quality.
"""
import argparse
import csv
import gzip
import hashlib
import io
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_TRAIN_SHA = "ac241c58a16cfafba759ff065b8d6bf6d30da677cf692907d4dbf2d9e0d31d9b"
LABELS = ("clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop")
FEATURES = ("run_id", "goal", "agents", "shared_state", "messages", "artifacts", "topology")
TARGETS = ("run_id", "label", "success", "fault_turn")
TRAIN = (*FEATURES, "label", "fault_turn", "success")


def to_bytes(rows, columns):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def split_demo(archive):
    with zipfile.ZipFile(archive) as bundle:
        compressed = bundle.read("extra_train.csv.gz")
    if hashlib.sha256(compressed).hexdigest() != PUBLIC_TRAIN_SHA:
        raise ValueError("The archived public demo training data failed the SHA-256 check.")
    counts, train, test = Counter(), [], []
    seen_ids, seen_runs = set(), set()
    csv.field_size_limit(2**28)
    with gzip.open(io.BytesIO(compressed), "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != TRAIN:
            raise ValueError("Unexpected header in the public training CSV.")
        for row in reader:
            label, run_id = row["label"], row["run_id"]
            if label not in LABELS or counts[label] >= 85:
                continue
            content = (row["goal"], row["messages"])
            if run_id in seen_ids or content in seen_runs:
                continue
            seen_ids.add(run_id)
            seen_runs.add(content)
            (train if counts[label] < 80 else test).append(row)
            counts[label] += 1
            if len(train) == 560 and len(test) == 35:
                break
    if any(counts[label] != 85 for label in LABELS):
        raise ValueError("Not enough distinct examples for the fixed split.")
    return {"train.csv": to_bytes(train, TRAIN), "test.csv": to_bytes(test, FEATURES),
            "truth.csv": to_bytes(test, TARGETS)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", type=Path,
                        default=ROOT / "assets" / "submission_v84b_v81_lm2.zip")
    parser.add_argument("--out", type=Path, default=ROOT / "work" / "smoke")
    args = parser.parse_args()
    files = split_demo(args.archive)
    args.out.mkdir(parents=True, exist_ok=True)
    # An identical rerun is safe. Never overwrite different files, even partly.
    for name, data in files.items():
        path = args.out / name
        if path.exists() and path.read_bytes() != data:
            raise SystemExit(f"Refusing to overwrite {path}; choose another --out directory.")
    for name, data in files.items():
        path = args.out / name
        if not path.exists():
            with path.open("xb") as dest:
                dest.write(data)
        print(hashlib.sha256(data).hexdigest(), path)
    print("Public demo only: 560 training rows and 35 unlabeled inputs. "
          "Use CR_NO_EXTRA=1 and CR_CSET_MODE=train; keep truth.csv out of the model.")


if __name__ == "__main__":
    main()
