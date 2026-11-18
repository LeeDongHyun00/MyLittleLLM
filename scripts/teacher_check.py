"""Try an operator pack on development tasks: which bank failures it fixes and what it breaks.

A teacher tool. It reads bank files written by round3_bank.py, which hold train and extra-pool
tasks only, and refuses any validation or test task, so a pack cannot be tuned on held-out data.
Train tasks are judged by the hidden tests, extra-pool tasks by their own asserts.

  python scripts/teacher_check.py --pack r3_strings --bank strings            # the failures of one category
  python scripts/teacher_check.py --pack r3_strings --bank strings --regress  # plus tasks that were solved before
  python scripts/teacher_check.py --pack r3_strings --bank strings --only Mbpp/104 MbppX/321 --show
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import time

BANK = Path(__file__).resolve().parents[1] / "outputs/rule_pilot/teacher/round3"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack", default="none", help="pack module name(s) under oppacks/, comma-separated")
    parser.add_argument("--bank", required=True, help="category name (bank_<name>.json) or a path to a bank file")
    parser.add_argument("--regress", action="store_true", help="also re-check every task the core library solves")
    parser.add_argument("--only", nargs="*", help="restrict to these task ids")
    parser.add_argument("--show", action="store_true", help="print the generated code of every found program")
    parser.add_argument("--output", help="write per-task results as JSON")
    args = parser.parse_args()

    os.environ["RULE_PACKS"] = args.pack        # must be set before the rule engine is imported
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mylittlellm.rulecoder import backinfer as bi, scoring
    from mylittlellm.rulecoder.sandbox import run_public_asserts
    from mylittlellm.rulecoder.tasks import load_split

    split = load_split()
    held_out = set(split["val"]) | set(split["test"])
    path = Path(args.bank) if args.bank.endswith(".json") else BANK / f"bank_{args.bank}.json"
    items = json.loads(path.read_text())
    if args.regress:
        items = items + json.loads((BANK / "regression.json").read_text())
    if args.only:
        items = [item for item in items if item["task_id"] in set(args.only)]
    if any(item["task_id"] in held_out for item in items):
        raise SystemExit("Refusing to run: the bank contains validation or test tasks")

    def judge(item, found):
        if item["source"] == "extra":
            return run_public_asserts(found["code"], "\n".join(item["asserts"]))["status"] == "pass"
        with (BANK / "scoring" / (item["task_id"].split("/")[1] + ".pkl")).open("rb") as stream:
            stored = pickle.load(stream)
        return scoring.score(stored["problem"], found["code"], stored["oracle"])["solved"]

    started, results = time.monotonic(), []
    for item in items:
        before = item["baseline"]
        try:
            found = bi.synthesize(item["description"], item["visible_assert"], item["entry_point"], weights={})
        except Exception as error:       # an operator that raises something other than TypeError/ValueError
            print(f"CRASH   {item['task_id']}: {type(error).__name__}: {error}")
            found = None
        if not found:
            status = "none"
        elif before["status"] == "solved" and found["expr"] == before["expr"]:
            status = "solved"           # same program as the one already judged correct
        else:
            status = "solved" if judge(item, found) else "wrong"
        results.append({"task_id": item["task_id"], "source": item["source"], "category": item["category"],
                        "before": before["status"], "after": status, "expr": found["expr"] if found else None,
                        "ops": found["ops"] if found else [], "pack_ops": [
                            name for name in (found["ops"] if found else []) if bi.OPERATORS[name].pack != "core"]})
        changed = before["status"] != status or (found and found["expr"] != before["expr"])
        if changed or args.only:
            label = {"solved": "SOLVED ", "wrong": "WRONG  ", "none": "NONE   "}[status]
            print(f"{label} {item['task_id']:11s} was {before['status']:6s} | {found['expr'] if found else '-'}")
            if args.show and found:
                print(found["code"])

    def count(source, before, after):
        return sum(r["source"] == source and r["before"] in before and r["after"] == after for r in results)

    print(f"\n{len(items)} tasks in {time.monotonic() - started:.0f}s, operators loaded: {len(bi.OPERATORS)} "
          f"(packs: {', '.join(bi.ACTIVE_PACKS) or 'none'})")
    for source in ("train", "extra"):
        total = sum(r["source"] == source for r in results)
        if total:
            print(f"  {source:5s} {total:4d} tasks | newly solved {count(source, ('wrong', 'none'), 'solved'):3d} "
                  f"| still wrong {count(source, ('wrong', 'none'), 'wrong'):3d} "
                  f"| still nothing found {count(source, ('wrong', 'none'), 'none'):3d} "
                  f"| BROKEN (was solved) {count(source, ('solved',), 'wrong') + count(source, ('solved',), 'none'):3d}")
    used = {}
    for r in results:
        for name in r["pack_ops"]:
            slot = used.setdefault(name, {"solved": 0, "wrong": 0})
            slot["solved" if r["after"] == "solved" else "wrong"] += 1
    if used:
        print("  pack operators in final programs (solved / wrong):")
        for name, slot in sorted(used.items(), key=lambda kv: (-kv[1]["solved"], kv[0])):
            print(f"    {name:32s} {slot['solved']:3d} / {slot['wrong']:3d}")
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
