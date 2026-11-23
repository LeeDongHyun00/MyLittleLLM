"""Check teacher-invented practice exercises (round 5 pilot): format, and the reference passes its asserts.

Each task (outputs/rule_round5/tasks_part<k>.json) is {"task_id": "Imag/<k>-<nnn>", "category", "description",
"entry_point", "asserts": three "assert f(...) == value" lines, "reference_code"}. A task passes when the
fields are well formed, every assert parses as a call of the entry point with literal arguments, the
expected values are not all equal, the arguments are short, and the reference solution passes all three
asserts in the sandbox.

  python scripts/round5_task_check.py --part 1
"""
import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mylittlellm.rulecoder import backinfer as bi
from mylittlellm.rulecoder.sandbox import run_public_asserts

DIR = Path(__file__).resolve().parents[1] / "outputs/rule_round5"
CATEGORIES = {"numbers", "sequences", "strings", "nested", "predicates"}


def problem(task, part):
    for key in ("task_id", "category", "description", "entry_point", "asserts", "reference_code"):
        if not task.get(key):
            return f"missing {key}"
    if not re.fullmatch(rf"Imag/{part}-\d{{3}}", task["task_id"]):
        return f"task_id must look like Imag/{part}-001"
    if task["category"] not in CATEGORIES:
        return f"category must be one of {sorted(CATEGORIES)}"
    text = task["description"].strip()
    if "\n" in text or not text.lower().startswith("write"):
        return "description must be one sentence starting with 'Write'"
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", task["entry_point"]):
        return "entry_point must be snake_case"
    if len(task["asserts"]) != 3 or not all(a.startswith("assert ") for a in task["asserts"]):
        return "need exactly three 'assert ...' lines"
    examples = [bi.parse_example(a, task["entry_point"]) for a in task["asserts"]]
    if any(e is None or not e["args"] for e in examples):
        return "every assert must call the entry point with literal arguments and compare with =="
    if len({repr(e["expected"]) for e in examples}) < 2:
        return "the three asserts must give at least two different outputs"
    if any(len(repr(e["args"])) > 200 for e in examples):
        return "arguments longer than 200 characters"
    if f"def {task['entry_point']}(" not in task["reference_code"]:
        return "reference_code must define the entry point"
    result = run_public_asserts(task["reference_code"], "\n".join(task["asserts"]))
    if result.get("status") != "pass":
        return f"reference fails its asserts: {result.get('detail', result.get('status'))}"[:200]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", type=int, required=True)
    args = parser.parse_args()
    path = DIR / f"tasks_part{args.part}.json"
    tasks = json.loads(path.read_text()) if path.exists() else []
    good, bad, seen = 0, 0, {"task_id": set(), "description": set(), "entry_point": set()}
    for task in tasks:
        issue = problem(task, args.part)
        if not issue:
            for key in seen:
                value = str(task[key]).strip().lower()
                if value in seen[key]:
                    issue = f"duplicate {key}"
                seen[key].add(value)
        if issue:
            bad += 1
            print(f"{task.get('task_id')}: {issue}")
        else:
            good += 1
    by_category = {c: sum(t.get("category") == c for t in tasks) for c in sorted(CATEGORIES)}
    print(json.dumps({"part": args.part, "tasks": len(tasks), "good": good, "with_problems": bad,
                      "by_category": by_category}))


if __name__ == "__main__":
    main()
