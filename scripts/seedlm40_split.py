"""SeedLM-40, part A: name the diagnostic and calibration split (the extra pool's training part, two folds).

The 434 training-side real tasks with reference solutions are the only real tasks this project may read one
by one. They are training data (coders saw some of the descriptions, and rounds 3-4 wrote operators for their
failures), so they calibrate judges, selectors and abstention thresholds; they do not measure how a coder
generalizes. Two folds by a hash of the task id let a threshold be chosen on one fold and checked on the other.

  python scripts/seedlm40_split.py      # writes data/seedlm/diagnostic_split.json
"""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylittlellm.rulecoder.tasks import ROOT
from seedlm2_real import held_out

BANK = ROOT / "outputs/rule_pilot/teacher/round3/all_tasks.json"


def fold(task_id):
    return "a" if int(hashlib.sha256(("diag:" + task_id).encode()).hexdigest(), 16) % 2 == 0 else "b"


def main():
    items = [i for i in json.loads(BANK.read_text()) if i["source"] == "extra" and not held_out(i["task_id"])]
    folds = {"a": [], "b": []}
    for item in items:
        folds[fold(item["task_id"])].append(item["task_id"])
    record = {"what": "extra pool, training part: real tasks with reference solutions, no development-fifth, "
                      "validation or test task", "use": "calibration and diagnostics only (training-side data)",
              "tasks": len(items), "folds": folds}
    (ROOT / "data/seedlm/diagnostic_split.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({"tasks": len(items), "fold_a": len(folds["a"]), "fold_b": len(folds["b"])}))


if __name__ == "__main__":
    main()
