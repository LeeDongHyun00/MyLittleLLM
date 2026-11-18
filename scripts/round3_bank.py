"""Build the failure bank for a teacher round: development tasks the rule engine cannot solve yet.

Two sources, neither of which touches the validation or test split:
  train : the MBPP+ train split, judged by the hidden tests;
  extra : MBPP tasks that are not part of MBPP+ at all, judged by their own three asserts.
Extra tasks that look like a validation or test task (same function name, or a very similar
description) are dropped automatically, so a rule mined from them is not a copy of a held-out task.
Everything written here stays under outputs/ because it contains dataset text.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import pickle
import re
import sys

os.environ["RULE_PACKS"] = "none"       # the bank describes what the core library fails on
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mylittlellm.rulecoder import backinfer as bi, scoring
from mylittlellm.rulecoder.sandbox import run_public_asserts
from mylittlellm.rulecoder.tasks import ROOT, load_split, parse_prompt, read_tasks

EXTRA = ROOT / "data/raw/rule_pilot/mbpp.jsonl"
OUT = ROOT / "outputs/rule_pilot/teacher/round3"
GEN = ROOT / "outputs/rule_pilot/gen"
SCORED = ROOT / "outputs/rule_pilot/scored"
SIMILAR = 0.6       # word-set Jaccard above which an extra task counts as a copy of a held-out task


def words(text):
    return frozenset(re.findall(r"[a-z0-9]+", text.lower())) - {"a", "an", "the", "to", "of", "write", "function",
                                                                "python", "given", "find", "in", "from", "for"}


def category(example):
    """Group tasks by the shape of their visible example, so each teacher sees one kind of problem."""
    if example is None:
        return "unparsed"
    args, expected = example["args"], example["expected"]
    if isinstance(expected, bool):
        return "predicates"

    def nested(value):
        return isinstance(value, dict) or (isinstance(value, (list, tuple, set))
                                           and any(isinstance(v, (list, tuple, dict, set)) for v in value))
    if any(nested(arg) for arg in args):
        return "nested"
    if any(isinstance(arg, str) for arg in args) and not any(isinstance(arg, (list, tuple, set)) for arg in args):
        return "strings"
    if all(bi.is_num(arg) for arg in args):
        return "numbers"
    return "sequences"


def entry_point_of(code, first_assert):
    """The function the asserts call: defined by the reference code and named in the first assert."""
    try:
        defined = [node.name for node in ast.walk(ast.parse(code)) if isinstance(node, ast.FunctionDef)]
    except SyntaxError:
        return None
    called = [name for name in defined if re.search(rf"\b{re.escape(name)}\(", first_assert)]
    return called[0] if called else None


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-run", default="engine-only-t", help="stored engine-only run over the train split")
    parser.add_argument("--pipeline-run", default="s15-branch-dev-t", help="stored 1.5B pipeline run over train")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scoring").mkdir(exist_ok=True)
    tasks, split = read_tasks(), load_split()
    held_out = split["val"] + split["test"]
    held_names = {tasks[t]["entry_point"] for t in held_out}
    held_words = [words(parse_prompt(tasks[t]["prompt"])[0]) for t in held_out]
    plus_numbers = {int(t.split("/")[1]) for t in tasks}

    bank, counts = [], {"train": {}, "extra": {}}

    # ---- train split: status from the stored engine-only run, scored on the hidden tests
    engine = {r["task_id"]: r for r in read_jsonl(GEN / f"{args.engine_run}.jsonl")}
    engine_scored = {r["task_id"]: r for r in read_jsonl(SCORED / f"{args.engine_run}.jsonl")}
    pipeline = {r["task_id"]: r for r in read_jsonl(SCORED / f"{args.pipeline_run}.jsonl")}
    problems = scoring.load_problems()
    oracle = scoring.groundtruth(problems)
    for task_id in split["train"]:
        task = tasks[task_id]
        description, visible = parse_prompt(task["prompt"])
        example = bi.parse_example(visible, task["entry_point"])
        inferred = [a for a in engine[task_id]["attempts"] if a.get("source") == "backinfer"]
        status = "solved" if engine_scored[task_id]["solved"] else ("wrong" if inferred else "none")
        bank.append({"task_id": task_id, "source": "train", "category": category(example),
                     "description": description, "entry_point": task["entry_point"], "visible_assert": visible,
                     "reference_code": task["canonical_solution"].strip("\n"),
                     "baseline": {"status": status, "expr": inferred[-1]["inference"]["expr"] if inferred else None},
                     "pipeline_1p5b_solved": bool(pipeline[task_id]["solved"])})
        with (OUT / "scoring" / (task_id.split("/")[1] + ".pkl")).open("wb") as stream:
            pickle.dump({"problem": problems[task_id], "oracle": oracle[task_id]}, stream)
    del oracle

    # ---- extra pool: MBPP tasks outside MBPP+, judged by their own asserts
    dropped = {"in_mbpp_plus": 0, "setup_code": 0, "no_entry_point": 0, "looks_like_held_out": 0}
    for row in map(json.loads, EXTRA.read_text().splitlines()):
        if row["task_id"] in plus_numbers:
            dropped["in_mbpp_plus"] += 1
            continue
        if row["test_setup_code"].strip():
            dropped["setup_code"] += 1
            continue
        asserts = [line.strip() for line in row["test_list"]]
        entry_point = entry_point_of(row["code"], asserts[0])
        if entry_point is None:
            dropped["no_entry_point"] += 1
            continue
        mine = words(row["text"])
        if entry_point in held_names or any(len(mine & other) / max(1, len(mine | other)) >= SIMILAR
                                            for other in held_words):
            dropped["looks_like_held_out"] += 1
            continue
        try:
            example = bi.parse_example(asserts[0], entry_point)
        except (ValueError, SyntaxError):
            example = None
        found = bi.synthesize(row["text"], asserts[0], entry_point, weights={}) if example else None
        if not found:
            status = "none"
        else:
            status = "solved" if run_public_asserts(found["code"], "\n".join(asserts))["status"] == "pass" else "wrong"
        bank.append({"task_id": f"MbppX/{row['task_id']}", "source": "extra", "category": category(example),
                     "description": row["text"], "entry_point": entry_point, "visible_assert": asserts[0],
                     "asserts": asserts, "reference_code": row["code"].replace("\r\n", "\n").expandtabs(4),
                     "baseline": {"status": status, "expr": found["expr"] if found else None}})

    for item in bank:
        slot = counts[item["source"]].setdefault(item["category"], {"solved": 0, "wrong": 0, "none": 0})
        slot[item["baseline"]["status"]] += 1
    categories = sorted({item["category"] for item in bank} - {"unparsed"})
    for name in categories:
        failures = [item for item in bank if item["category"] == name and item["baseline"]["status"] != "solved"]
        (OUT / f"bank_{name}.json").write_text(json.dumps(failures, indent=1) + "\n")
    solved = [item for item in bank if item["baseline"]["status"] == "solved"]
    (OUT / "regression.json").write_text(json.dumps(solved, indent=1) + "\n")
    (OUT / "all_tasks.json").write_text(json.dumps(bank, indent=1) + "\n")
    summary = {"similarity_threshold": SIMILAR, "extra_dropped": dropped, "counts": counts,
               "tasks": {"train": sum(i["source"] == "train" for i in bank),
                         "extra": sum(i["source"] == "extra" for i in bank)},
               "failures_per_category": {name: sum(i["category"] == name and i["baseline"]["status"] != "solved"
                                                   for i in bank) for name in categories},
               "regression_tasks": len(solved)}
    report = ROOT / "reports/rule_pilot/round3_bank_summary.json"
    report.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
