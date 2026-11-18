"""Where could a learned model help the rule engine? An oracle decomposition on development tasks.

For every development task (train split and extra pool) the shipped engine does not solve, the
search is re-run keeping thousands of matches, and an oracle (the task's own tests) looks for a
correct program among them:
  ranking : a correct program is among the matches, just not ranked first;
  gating  : a correct program appears only when every keyword gate is open (the description does
            not use the words the operator was gated on);
  beyond  : no correct program within two operators even with all gates open.
A learned reader of descriptions can fix ranking and gating; only deeper search or new operators
reach "beyond". Validation and test tasks are not used.
"""
import argparse
import ast
import copy
import json
from pathlib import Path
import pickle
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mylittlellm.rulecoder import backinfer as bi, scoring
from mylittlellm.rulecoder.limits import Timeout, time_limit
from mylittlellm.rulecoder.pipeline import same_output
from mylittlellm.rulecoder.sandbox import run_public_asserts
from mylittlellm.rulecoder.tasks import ROOT

BANK = ROOT / "outputs/rule_pilot/teacher/round3"


def assert_cases(asserts, entry_point):
    cases = []
    for line in asserts:
        example = bi.parse_example(line, entry_point)
        if example is None:
            return None
        cases.append((example["args"], example["expected"], example["mode"]))
    return cases


def quick_ok(code, entry_point, cases):
    scope = {}
    try:
        with time_limit(1.0):
            exec(code, scope)
            for args, expected, mode in cases:
                value = scope[entry_point](*copy.deepcopy(list(args)))
                if mode == "set":
                    if not (isinstance(value, (list, tuple, set)) and set(value) == set(expected)):
                        return False
                elif not same_output(value, expected):
                    return False
        return True
    except (Exception, Timeout):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=8.0)
    parser.add_argument("--matches", type=int, default=3000)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    bi.MAX_MATCHES = args.matches
    bi.MAX_COLLECTED = args.matches
    status = {r["task_id"]: r["after"] for r in json.loads((BANK / "attribution_selected.json").read_text())["results"]}
    items = [i for i in json.loads((BANK / "all_tasks.json").read_text()) if status.get(i["task_id"]) != "solved"]
    items = items[:args.limit] if args.limit else items
    problems = None
    counts, rows = {}, []
    original_named_in = bi.Op.named_in
    started = time.monotonic()
    for item in items:
        if item["source"] == "extra":
            cases = assert_cases(item["asserts"], item["entry_point"])

            def judge(code):
                return run_public_asserts(code, "\n".join(item["asserts"]))["status"] == "pass"
        else:
            with (BANK / "scoring" / (item["task_id"].split("/")[1] + ".pkl")).open("rb") as stream:
                stored = pickle.load(stream)
            problem, oracle = stored["problem"], stored["oracle"]
            cases = [(inputs, out, "eq") for inputs, out in zip(problem["base_input"], oracle["base"])]

            def judge(code, problem=problem, oracle=oracle):
                return scoring.score(problem, code, oracle)["solved"]
        example = bi.parse_example(item["visible_assert"], item["entry_point"])
        verdict, program = "unparsed", None
        if example is not None and example["args"] and cases:
            verdict = "beyond"
            for gates_open in (False, True):
                bi.Op.named_in = (lambda self, text: True) if gates_open else original_named_in
                matches = bi.search(example, item["description"], args.budget, weights={})
                hit = None
                for rank, match in enumerate(matches):
                    code = bi.to_code(item["entry_point"], len(example["args"]), match)
                    if quick_ok(code, item["entry_point"], cases) and judge(code):
                        hit, program = rank, match["program"]
                        break
                if hit is not None:
                    verdict = "gating" if gates_open else "ranking"
                    break
            bi.Op.named_in = original_named_in
        counts.setdefault(item["source"], {}).setdefault(verdict, 0)
        counts[item["source"]][verdict] += 1
        rows.append({"task_id": item["task_id"], "source": item["source"], "category": item["category"],
                     "verdict": verdict, "program": program})
        print(item["task_id"], verdict, f"{time.monotonic() - started:.0f}s", flush=True)
    report = {"tasks": len(items), "counts": counts, "settings": vars(args)}
    (ROOT / "reports/seedlm/ceiling_dev.json").write_text(json.dumps(report, indent=1) + "\n")
    (ROOT / "outputs/seedlm/ceiling_dev_rows.json").write_text(json.dumps(rows, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
