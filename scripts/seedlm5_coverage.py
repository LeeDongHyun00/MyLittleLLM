"""Does the training data already hold a solution? Program coverage of held-out tasks, counts only.

For each source of training programs (generic teacher tasks, SeedLM-5's seeded variants, development
one-liners, real development rows; training parts only, the way the coders saw them), every program is
tried as-is on every held-out development task with the same number of arguments: the task's own
literals and visible output fill the program's constant slots, as they would if the coder wrote those
tokens there. A task is covered by a source when one of its programs passes all of the task's asserts.
This separates "the knowledge is not in the data" from "the coder does not use it".

Engine verdicts come from the local per-task file of a held-out run (`seedlm4_tasks.py holdout
--verdicts`). Only counts are reported.

  python scripts/seedlm5_coverage.py [--verdicts holdout4_s4-post_verdicts]
"""
import argparse
import ast
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylittlellm.rulecoder import backinfer as bi
from mylittlellm.rulecoder.limits import Timeout, time_limit
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import pyx
from seedlm2_real import held_out
from seedlm3_train import oneliner_items, teacher_held, teacher_task_items
from seedlm4_train import variant_held

BANK = ROOT / "outputs/rule_pilot/teacher/round3/all_tasks.json"
SECONDS = 0.25


def pools():
    """{source: {(tokens, n_args)}}, training parts only."""
    out = {"teacher": set(), "variants": set(), "oneliners": set(), "real": set()}
    for task in teacher_task_items():
        if not teacher_held(task["id"]):
            out["teacher"].add((tuple(task["tokens"]), len(ast.literal_eval(task["inputs"][0]))))
    for task in teacher_task_items("variant_tasks"):
        if not variant_held(task):
            out["variants"].add((tuple(task["tokens"]), len(ast.literal_eval(task["inputs"][0]))))
    for entry in oneliner_items():
        if entry["part"] == "train":
            out["oneliners"].add((tuple(entry["tokens"]), len(entry["example"]["args"])))
    for row in json.loads((ROOT / "outputs/seedlm/v2/real_dev.json").read_text()):
        if row["part"] == "train":
            out["real"].add((tuple(row["program"]), len(ast.literal_eval(row["args"]))))
    return out


def passes(tokens, examples, literals, output):
    for example in examples:
        try:
            with time_limit(SECONDS):
                value = pyx.execute(list(tokens), example["args"], literals, output)
        except (Exception, Timeout):
            return False
        if not bi._matches(value, example):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verdicts", default="holdout4_s4-post_verdicts")
    parser.add_argument("--output", default="reports/seedlm/coverage_holdout.json")
    args = parser.parse_args()
    engine = {t: v["engine"] for t, v in
              json.loads((ROOT / "outputs/seedlm/verdicts" / f"{args.verdicts}.json").read_text()).items()}
    sources = pools()
    covered = {name: {"engine_fails": 0, "engine_solves": 0} for name in list(sources) + ["any"]}
    counts = {"tasks": 0, "engine_fails": 0}
    for item in json.loads(BANK.read_text()):
        if item["source"] != "extra" or not held_out(item["task_id"]) or item["task_id"] not in engine:
            continue
        examples = [bi.parse_example(line, item["entry_point"]) for line in item["asserts"]]
        if any(e is None or not e["args"] for e in examples):
            continue
        counts["tasks"] += 1
        part = "engine_solves" if engine[item["task_id"]] else "engine_fails"
        counts["engine_fails"] += part == "engine_fails"
        literals = bi.description_literals(item["description"])
        visible = examples[0]["expected"]
        output = visible if isinstance(visible, str) else None
        n_args = len(examples[0]["args"])
        hit_any = False
        for name, programs in sources.items():
            hit = any(passes(tokens, examples, literals, output) for tokens, n in programs if n == n_args)
            covered[name][part] += hit
            hit_any |= hit
        covered["any"][part] += hit_any
    report = {"verdicts": args.verdicts, **counts,
              "programs": {name: len(p) for name, p in sources.items()}, "covered": covered}
    (ROOT / args.output).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
