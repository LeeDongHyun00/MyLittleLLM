"""SeedLM-21's measure: held-out teacher programs with a new piece, written by each coder. Counts only.

Each held-out teacher task's own program is labelled by the coarsest piece that neither the synthetic
library programs nor any training program has (scripts/seedlm5_fragments.py: a new token, else a new
edge, else a new twig, else a new whole program, else nothing new), relative to the control coders'
training data. For each coder, SeedLM-15's cache (outputs/seedlm/v15/<coder>.json: the first program
at beam width 32 that reproduces the visible example, judged on every case) gives whether it wrote a
correct program; this script counts them per label and per arm.

  python scripts/seedlm21_compare.py --arm s21-y1 s21-y2 s21-y3 --control s12-a1 s12-a2 s12-a3
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylittlellm.rulecoder.tasks import ROOT
from seedlm3_discovery import teacher_cases
from seedlm5_fragments import first_unknown, pieces
from seedlm15_ensemble_scaling import CACHE
from seedlm21_coverage import known_pieces

NEW = ("tokens", "edges", "twigs")


def labels():
    known = known_pieces()
    out = []
    for case in teacher_cases():
        try:
            out.append(first_unknown(pieces(case["gold"]), known))
        except (ValueError, IndexError, KeyError):
            out.append(None)
    return out


def counts(coder, task_labels):
    record = json.loads((CACHE / f"{coder}.json").read_text())
    rows = record["teacher"]
    assert len(rows) == len(task_labels), (coder, len(rows), len(task_labels))
    correct = [bool(r and r["correct"]) for r in rows]
    return {"new_combination": sum(c for c, l in zip(correct, task_labels) if l in NEW),
            "pieces_seen": sum(c for c, l in zip(correct, task_labels) if l is not None and l not in NEW),
            "teacher": sum(correct),
            "development": sum(bool(r and r["correct"]) for r in record["development"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", nargs="+", required=True)
    parser.add_argument("--control", nargs="+", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    task_labels = labels()
    report = {"tasks": len(task_labels), "new_combination_tasks": sum(l in NEW for l in task_labels),
              "pieces_seen_tasks": sum(l is not None and l not in NEW for l in task_labels)}
    for side in ("arm", "control"):
        coders = getattr(args, side)
        per = {c: counts(c, task_labels) for c in coders}
        report[side] = {"coders": per,
                        "mean": {k: round(statistics.mean(p[k] for p in per.values()), 2) for k in next(iter(per.values()))}}
    report["difference_of_means"] = {k: round(report["arm"]["mean"][k] - report["control"]["mean"][k], 2)
                                     for k in report["arm"]["mean"]}
    text = json.dumps(report, indent=1)
    print(text)
    if args.output:
        (ROOT / args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
