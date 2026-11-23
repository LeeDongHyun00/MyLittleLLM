"""Round 5 pilot, stage 2: decontaminate the invented exercises, run the shipped engine, build the bank.

Exercises (outputs/rule_round5/tasks_part*.json, scripts/round5_task_check.py) are dropped when their
description is close to an evaluation task's (MBPP+ validation and test, HumanEval+, the development fifth)
or to a training-side real task's (the extra pool's training part: the pilot's transfer target, which the
round's teachers must not have reproduced), by SeedLM-3's rules (a shared run of six words, or content-word
Jaccard >= 0.6, after the opening boilerplate). The shipped engine then tries each kept exercise: those it
fails by their asserts (no program, or a wrong one) form outputs/rule_round5/bank_imagined.json, those it
solves regression.json, in round 4's bank format. Counts only are printed.

  python scripts/round5_bank.py              # every tasks_part*.json
  python scripts/round5_bank.py --parts 2     # the second pilot's exercises only
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylittlellm.rulecoder import backinfer as bi
from mylittlellm.rulecoder.limits import cap_memory
from mylittlellm.rulecoder.sandbox import run_public_asserts
from mylittlellm.rulecoder.tasks import ROOT
from round5_task_check import problem
from seedlm2_real import held_out
from seedlm3_decontaminate import body, content, evaluation_descriptions, shingles

DIR = ROOT / "outputs/rule_round5"
BANK = ROOT / "outputs/rule_pilot/teacher/round3/all_tasks.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=int, nargs="*")
    args = parser.parse_args()
    cap_memory()
    tasks = []
    for path in sorted(DIR.glob("tasks_part*.json")):
        part = int(path.stem[len("tasks_part"):])
        if args.parts and part not in args.parts:
            continue
        tasks += [t for t in json.loads(path.read_text()) if problem(t, part) is None]
    training = [i["description"] for i in json.loads(BANK.read_text())
                if i["source"] == "extra" and not held_out(i["task_id"])]
    references = [body(t) for t in evaluation_descriptions() + training]
    reference_shingles = set().union(*(shingles(w) for w in references))
    reference_content = [content(w) for w in references]
    kept, dropped = [], 0
    for task in tasks:
        words = body(task["description"])
        mine = content(words)
        if shingles(words) & reference_shingles or any(
                mine and len(mine & other) / len(mine | other) >= 0.6 for other in reference_content):
            dropped += 1
            continue
        kept.append(task)
    bank, regression = [], []
    counts = {"written": len(tasks), "dropped_close_to_a_real_task": dropped, "kept": len(kept),
              "solved": 0, "wrong": 0, "none": 0}
    for task in kept:
        visible = task["asserts"][0]
        found = bi.synthesize(task["description"], visible, task["entry_point"], weights={})
        if found is None:
            status = "none"
        else:
            ok = run_public_asserts(found["code"], "\n".join(task["asserts"]))["status"] == "pass"
            status = "solved" if ok else "wrong"
        counts[status] += 1
        entry = {"task_id": task["task_id"], "source": "imagined", "category": task["category"],
                 "description": task["description"], "entry_point": task["entry_point"], "visible_assert": visible,
                 "asserts": task["asserts"], "reference_code": task["reference_code"],
                 "baseline": {"status": status, "expr": found.get("expr") if found else None,
                              "confident": bool(found and found.get("confident"))}}
        (regression if status == "solved" else bank).append(entry)
    (DIR / "bank_imagined.json").write_text(json.dumps(bank, indent=1) + "\n")
    (DIR / "regression.json").write_text(json.dumps(regression, indent=1) + "\n")
    (DIR / "bank_summary.json").write_text(json.dumps(counts, indent=1) + "\n")
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
