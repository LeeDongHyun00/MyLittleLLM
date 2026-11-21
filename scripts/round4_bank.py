"""Round 4 failure bank: training-pool tasks the current rule engine still fails (for operator teachers).

Round 3 (docs/rule_round3.md) gave teachers the tasks the engine failed on and kept the operators that
helped; the engine went from 57 to 68 on the locked test split. This script rebuilds the bank for the
engine as it is now, from the extra pool's training part only (the held-out development fifth that the
SeedLM experiments measure on is left out, as are the MBPP+ train tasks, whose tests sit with the locked
split). A task fails when the engine finds no program (`none`) or finds one that reproduces the visible
example but fails the task's asserts (`wrong`). Categories are round 3's.

Writes outputs/rule_round4/bank_<category>.json (teacher-visible copies), regression.json (the tasks it
solves, with the program it finds, for checking that a pack breaks nothing) and prints counts.

  python scripts/round4_bank.py
"""
import collections
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylittlellm.rulecoder import backinfer as bi
from mylittlellm.rulecoder.limits import cap_memory
from mylittlellm.rulecoder.sandbox import run_public_asserts
from mylittlellm.rulecoder.tasks import ROOT
from seedlm2_real import held_out

BANK = ROOT / "outputs/rule_pilot/teacher/round3/all_tasks.json"
OUT = ROOT / "outputs/rule_round4"


def main():
    cap_memory()
    items = [i for i in json.loads(BANK.read_text()) if i["source"] == "extra" and not held_out(i["task_id"])]
    banks, counts, regression = collections.defaultdict(list), collections.Counter(), []
    for index, item in enumerate(items):
        found = bi.synthesize(item["description"], item["visible_assert"], item["entry_point"], weights={})
        if found is None:
            status = "none"
        else:
            ok = run_public_asserts(found["code"], "\n".join(item["asserts"]))["status"] == "pass"
            status = "solved" if ok else "wrong"
        counts[(item["category"], status)] += 1
        entry = {k: item[k] for k in ("task_id", "source", "category", "description", "entry_point",
                                      "visible_assert", "asserts", "reference_code")}
        entry["baseline"] = {"status": status, "expr": found.get("expr") if found else None,
                             "confident": bool(found and found.get("confident"))}
        (regression if status == "solved" else banks[item["category"]]).append(entry)
        if index % 50 == 0:
            print(index, len(items), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for category, entries in banks.items():
        (OUT / f"bank_{category}.json").write_text(json.dumps(entries, indent=1) + "\n")
    (OUT / "regression.json").write_text(json.dumps(regression, indent=1) + "\n")
    summary = {"tasks": len(items), "by_status": dict(collections.Counter(s for (_, s) in counts.elements())),
               "by_category": {f"{c}/{s}": n for (c, s), n in sorted(counts.items())}}
    (OUT / "bank_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
