"""Would a human solution from the training pool already solve a held-out task? Counts only.

SeedLM-5's coverage diagnostic (scripts/seedlm5_coverage.py) ran the coder's training programs (pyx
one-liners) as they are on the held-out development fifth and found 4 of the 29 engine failures
covered. The training pool also carries each task's reference solution: the human-written MBPP
function, usually several statements long, which no SeedLM data has used. MBPP holds many near
relatives, so a reference solution may solve another task outright.

Each training-pool reference solution (MBPP training split and the extra pool's training part) is
called, in one sandboxed process per solution, on the visible example of every held-out task with
the same number of arguments; where its output matches the visible output, it is run on all of that
task's asserts. Reported: held-out tasks with a matching reference (visible example only) and with a
reference passing every assert, split by whether the engine solves the task, and among the covered
engine failures, how many have the covering reference among the 10 training tasks whose description
shares the most words with theirs (a retrieval that a method could use without the hidden asserts).

  python scripts/seedlm14_reference_coverage.py [--verdicts holdout4_s5-var]
"""
import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylittlellm.rulecoder import backinfer as bi
from mylittlellm.rulecoder.consensus import canonical
from mylittlellm.rulecoder.sandbox import run_calls, run_public_asserts
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import language
from seedlm2_real import held_out

BANK = ROOT / "outputs/rule_pilot/teacher/round3/all_tasks.json"
STOP = set("write a an the function to python of in and given for from with that is are by list string number "
           "numbers find check whether".split())


def words(text):
    return {w for w in language.words(text) if w not in STOP and not w.isdigit()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verdicts", default="holdout4_s5-var")
    parser.add_argument("--output", default="reports/seedlm/reference_coverage.json")
    args = parser.parse_args()
    engine = {t: v["engine"] for t, v in
              json.loads((ROOT / "outputs/seedlm/verdicts" / f"{args.verdicts}.json").read_text()).items()}
    bank = json.loads(BANK.read_text())
    train = [i for i in bank if i["source"] in ("train", "extra")
             and not (i["source"] == "extra" and held_out(i["task_id"])) and i.get("reference_code")]
    tasks = []
    for item in bank:
        if item["source"] != "extra" or not held_out(item["task_id"]) or item["task_id"] not in engine:
            continue
        example = bi.parse_example(item["visible_assert"], item["entry_point"])
        if example is None or not example["args"] or example["mode"] != "eq":
            continue
        tasks.append({"item": item, "args": example["args"], "expected": canonical(example["expected"])})
    matches = {t["item"]["task_id"]: [] for t in tasks}
    for ref in train:
        params = re.search(rf"def\s+{re.escape(ref['entry_point'])}\s*\(([^)]*)\)", ref["reference_code"])
        if not params:
            continue
        arity = len([p for p in params.group(1).split(",") if p.strip() and not p.strip().startswith("*")])
        fitting = [t for t in tasks if len(t["args"]) == arity]
        if not fitting:
            continue
        outcome = run_calls(ref["reference_code"], ref["entry_point"], [t["args"] for t in fitting], per_call=0.5)
        if outcome.get("status") != "ok":
            continue
        for t, (status, value) in zip(fitting, outcome["results"]):
            if status == "ok" and value == t["expected"]:
                matches[t["item"]["task_id"]].append(ref)
    counts = {"tasks": len(tasks), "engine_fails": sum(not engine[t["item"]["task_id"]] for t in tasks),
              "visible_match": {"engine_fails": 0, "engine_solves": 0},
              "all_asserts": {"engine_fails": 0, "engine_solves": 0},
              "all_asserts_and_retrieved_top10": {"engine_fails": 0, "engine_solves": 0},
              "matching_references_per_task_median": None}
    per_task_matches = []
    for t in tasks:
        item = t["item"]
        part = "engine_solves" if engine[item["task_id"]] else "engine_fails"
        refs = matches[item["task_id"]]
        per_task_matches.append(len(refs))
        if not refs:
            continue
        counts["visible_match"][part] += 1
        alias = f"\n\n{item['entry_point']} = {{name}}\n"
        passing = [r for r in refs
                   if run_public_asserts(r["reference_code"] + alias.format(name=r["entry_point"]),
                                         "\n".join(item["asserts"]))["status"] == "pass"]
        if not passing:
            continue
        counts["all_asserts"][part] += 1
        mine = words(item["description"])
        ranked = sorted(train, key=lambda r: -len(mine & words(r["description"])))[:10]
        if any(r["task_id"] in {p["task_id"] for p in passing} for r in ranked):
            counts["all_asserts_and_retrieved_top10"][part] += 1
    per_task_matches.sort()
    counts["matching_references_per_task_median"] = per_task_matches[len(per_task_matches) // 2]
    counts["references"] = len(train)
    (ROOT / args.output).write_text(json.dumps(counts, indent=1) + "\n")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
