"""Check teacher-written sketch tasks (SeedLM-34): valid teacher tasks whose solution has the assigned sketch.

Each assignment (outputs/sketch_tasks/in_part<k>.json) gives a sketch id, the sketch (prefix tokens with "?"
for holes of at most 5 tokens) and two training tasks that have it. Each written task
(out_part<k>.json: {"id", "sketch_id", "category", "description", "entry_point", "inputs", "expression"})
must pass SeedLM-3's teacher-task check (scripts/seedlm3_task_check.py: three inputs, runs, outputs not all
equal, constants stated, writable in pyx), have exactly the assigned sketch, and repeat no training task's
expression or description.

  python scripts/seedlm34_task_check.py --part 1            # problems, then counts
  python scripts/seedlm34_task_check.py --all --summary
  python scripts/seedlm34_task_check.py --sketch "sum(x * x for x in a0 if x % 2)" --args 1     # a helper
  python scripts/seedlm34_task_check.py --all --accept      # copy the passing tasks to data/seedlm/sketch_tasks_part<k>.json
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylittlellm.seedlm import sketch as sk
from seedlm3_task_check import check

DIR = Path(__file__).resolve().parents[1] / "outputs/sketch_tasks"
DATA = Path(__file__).resolve().parents[1] / "data/seedlm"
FIELDS = ("id", "sketch_id", "category", "description", "entry_point", "inputs", "expression")


def training_texts():
    path = DIR / "training_texts.json"
    return json.loads(path.read_text()) if path.exists() else {"expressions": [], "descriptions": []}


def verdict(task, assigned, seen):
    missing = [k for k in ("id", "sketch_id", "description", "entry_point", "inputs", "expression") if not task.get(k)]
    if missing:
        return f"missing {missing}"
    if task["sketch_id"] not in assigned:
        return f"unknown sketch_id {task['sketch_id']}"
    try:
        ok, reason, details = check(task)
    except Exception as error:      # a malformed task must not stop the check of the others
        return f"{type(error).__name__}: {error}"[:200]
    if not ok:
        return reason
    found, _ = sk.sketch_of(details["tokens"])
    want = tuple(assigned[task["sketch_id"]])
    if found != want:
        return f"sketch {' '.join(found)} is not the assigned {' '.join(want)}"
    if task["expression"].replace(" ", "") in seen["expressions"]:
        return "repeats a training task's expression"
    if task["description"].strip().lower() in seen["descriptions"]:
        return "repeats a training task's description"
    return None


def check_part(part, verbose=True, accept=False):
    assignments = json.loads((DIR / f"in_part{part}.json").read_text())
    assigned = {a["sketch_id"]: a["sketch"] for a in assignments}
    path = DIR / f"out_part{part}.json"
    tasks = json.loads(path.read_text()) if path.exists() else []
    texts = training_texts()
    seen = {"expressions": set(texts["expressions"]), "descriptions": set(texts["descriptions"])}
    good, bad, ids, per_sketch, kept = 0, 0, set(), {}, []
    for task in tasks:
        problem = verdict(task, assigned, seen)
        if not problem and task["id"] in ids:
            problem = "duplicate id"
        ids.add(task.get("id"))
        if problem:
            bad += 1
            if verbose:
                print(f"{task.get('id')}: {problem}")
        else:
            good += 1
            per_sketch[task["sketch_id"]] = per_sketch.get(task["sketch_id"], 0) + 1
            kept.append({k: task[k] for k in FIELDS if k in task})
    short = sum(per_sketch.get(s, 0) < 4 for s in assigned)
    if accept:
        (DATA / f"sketch_tasks_part{part}.json").write_text(json.dumps(kept, indent=1) + "\n")
    return {"part": part, "sketches": len(assigned), "tasks": len(tasks), "good": good, "with_problems": bad,
            "sketches_with_fewer_than_4_good": short}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--accept", action="store_true", help="write the passing tasks to data/seedlm")
    parser.add_argument("--sketch", help="print the pyx tokens and sketch of this expression, then stop")
    parser.add_argument("--args", type=int, default=1, help="number of arguments a0.. the expression uses")
    parser.add_argument("--literals", default="[]", help="the description's literals, as a Python list")
    args = parser.parse_args()
    if args.sketch:
        import ast
        from mylittlellm.seedlm import pyx
        tokens = pyx.serialize(args.sketch, args.args, ast.literal_eval(args.literals))
        print("tokens:", " ".join(tokens))
        print("sketch:", " ".join(sk.sketch_of(tokens)[0]))
        return
    parts = sorted(int(p.stem[len("in_part"):]) for p in DIR.glob("in_part*.json")) if args.all else [args.part]
    for part in parts:
        print(json.dumps(check_part(part, verbose=not args.summary, accept=args.accept)))


if __name__ == "__main__":
    main()
