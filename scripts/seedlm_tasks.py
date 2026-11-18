"""Use SeedLM proposals on benchmark tasks: does a learned proposer add to exhaustive back-inference?

Counts only are printed for the locked test split.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from mylittlellm.rulecoder import backinfer as bi, scoring
from mylittlellm.rulecoder.tasks import ROOT, load_split, parse_prompt, read_tasks
from mylittlellm.seedlm import propose, synth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    net, vocabulary = propose.load(ROOT / "outputs/seedlm" / f"{args.model}.pt")
    arities = propose.arity_table(vocabulary)
    tasks, split = read_tasks(), load_split()
    problems = scoring.load_problems()
    oracle = scoring.groundtruth(problems)
    report = {}
    for part in args.splits:
        stats = {"tasks": 0, "usable_example": 0, "search_found": 0, "search_correct": 0, "model_found": 0,
                 "model_top1_correct": 0, "model_found_where_search_failed": 0,
                 "model_correct_where_search_failed": 0, "same_program_as_search": 0}
        details = []
        for task_id in split[part]:
            task = tasks[task_id]
            description, asserts = parse_prompt(task["prompt"])
            example = bi.parse_example(asserts, task["entry_point"])
            stats["tasks"] += 1
            if example is None or example["mode"] != "eq" or not example["args"] or \
                    isinstance(example["expected"], bool) or example["expected"] is None:
                continue
            stats["usable_example"] += 1
            searched = bi.synthesize(description, asserts, task["entry_point"], weights={})
            search_ok = bool(searched) and scoring.score(problems[task_id], searched["code"], oracle[task_id])["solved"]
            stats["search_found"] += bool(searched)
            stats["search_correct"] += search_ok
            named = [name for name, op in bi.OPERATORS.items() if op.named_in(description.lower())]
            proposals = propose.propose(net, vocabulary, example["args"], example["expected"], named, args.width,
                                        arities)
            if not proposals:
                continue
            stats["model_found"] += 1
            expr, helpers, _ = synth.to_expression(proposals[0]["program"])
            code = bi.to_code(task["entry_point"], len(example["args"]), {"expr": expr, "helpers": helpers})
            solved = scoring.score(problems[task_id], code, oracle[task_id])["solved"]
            stats["model_top1_correct"] += solved
            stats["same_program_as_search"] += bool(searched) and searched["expr"] == expr
            if not searched:
                stats["model_found_where_search_failed"] += 1
                stats["model_correct_where_search_failed"] += solved
            if part != "test":
                details.append({"task_id": task_id, "program": " ".join(proposals[0]["program"]), "solved": solved,
                                "search_found": bool(searched), "search_correct": search_ok})
        report[part] = stats
        print(part, json.dumps(stats), flush=True)
        if part != "test":
            (ROOT / "outputs/seedlm" / f"tasks_{args.model}_{part}.json").write_text(json.dumps(details, indent=1))
    (ROOT / "reports/seedlm" / f"tasks_{args.model}.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
