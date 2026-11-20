"""SeedLM-23's paired measure: each base coder fine-tuned with and without the unlikelihood term. Counts only.

For each base, the two arms took the same steps over the same batches from the same checkpoint
(scripts/seedlm23_finetune.py), so the difference between them is the effect of being told which programs
that fit the example are wrong. From SeedLM-15's cache (first program at beam width 32 that reproduces the
visible example) and the width-32 beam statistics (scripts/seedlm3_discovery.py), per base and on average:
held-out teacher tasks right (all, and SeedLM-21's new-combination and pieces-seen parts), development tasks
right, wrong programs that fit the example per task, and tasks where some verified program is right.

  python scripts/seedlm23_compare.py --pairs s23-u1:s23-m1 s23-u2:s23-m2 s23-u3:s23-m3
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylittlellm.rulecoder.tasks import ROOT
from seedlm21_compare import counts, labels


def beam(coder):
    path = ROOT / "reports/seedlm" / f"discovery3_{coder}.json"
    if not path.exists():
        return {}
    stats = json.loads(path.read_text())["teacher"]["by_width"]["32"]
    return {"wrong_fitting_per_task": stats["spurious"], "some_verified_right": stats["oracle"],
            "first_verified_right": stats["first"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", required=True, help="treated:control")
    parser.add_argument("--output", default="reports/seedlm/contrast_compare.json")
    args = parser.parse_args()
    task_labels = labels()
    report, differences = {"pairs": {}}, {}
    for pair in args.pairs:
        treated, control = pair.split(":")
        sides = {name: dict(counts(name, task_labels), **beam(name)) for name in (treated, control)}
        diff = {k: round(sides[treated][k] - sides[control][k], 2) for k in sides[treated] if k in sides[control]}
        report["pairs"][pair] = {"treated": sides[treated], "control": sides[control], "difference": diff}
        for k, v in diff.items():
            differences.setdefault(k, []).append(v)
    report["mean_difference"] = {k: round(statistics.mean(v), 2) for k, v in differences.items()}
    text = json.dumps(report, indent=1)
    print(text)
    (ROOT / args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
