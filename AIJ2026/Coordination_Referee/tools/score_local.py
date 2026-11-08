"""Score labeled development predictions; official Robustness and Score are unavailable."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

LABELS = ("clean", "dropped_handoff", "duplicated_work", "deadlock", "conflict", "goal_drift", "runaway_loop")


def read(path):
    rows = {}
    csv.field_size_limit(2**30)
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"run_id", "label", "success", "fault_turn"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"Missing ID or target columns: {path}")
        for row in reader:
            key, label = row["run_id"], row["label"]
            if not key or key in rows or label not in LABELS or row["success"] not in ("0", "1"):
                raise ValueError(f"Invalid ID or targets in {path}: {key!r}")
            turn = int(row["fault_turn"])
            if turn < -1 or (label == "clean" and turn != -1):
                raise ValueError(f"Invalid fault_turn for {key!r}: {turn}")
            rows[key] = (label, int(row["success"]), turn)
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def f1(tp, actual, predicted):
    return 2 * tp / (actual + predicted) if actual + predicted else 0.0


def score(truth, pred):
    if truth.keys() != pred.keys():
        raise ValueError("Truth and predictions must contain exactly the same run IDs")
    actual, predicted, correct = Counter(), Counter(), Counter()
    actual_success = predicted_success = correct_success = hits = 0
    for key, (label, success, turn) in truth.items():
        p_label, p_success, p_turn = pred[key]
        actual[label] += 1
        predicted[p_label] += 1
        correct[label] += p_label == label
        actual_success += success
        predicted_success += p_success
        correct_success += success * p_success
        if label != "clean" and turn < 0:
            raise ValueError(f"Missing true fault onset for {key!r}")
        hits += label != "clean" and p_label == label and p_turn >= 0 and abs(p_turn - turn) <= 2
    faults = len(truth) - actual["clean"]
    if not faults:
        raise ValueError("No true faults: joint hit@2 has no denominator")
    macro = sum(f1(correct[c], actual[c], predicted[c]) for c in LABELS) / len(LABELS)
    succ = f1(correct_success, actual_success, predicted_success)
    hit = hits / faults
    return dict(rows=len(truth), true_faults=faults, hits=hits, macro_f1=macro, success_f1=succ,
                joint_hit_at_2=hit, local_K=0.50*macro + 0.15*succ + 0.10*hit,
                official_robustness_f1=None, official_score=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(score(read(args.truth), read(args.pred)), indent=2))
