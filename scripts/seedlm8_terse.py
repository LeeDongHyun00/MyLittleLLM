"""Does the coder lose its way when the statement is short? The same held-out teacher tasks, read twice.

Held-out teacher tasks are written explicitly (every argument named in order, every constant
stated); real development tasks are terse MBPP-style statements. A subagent rewrote the 139
held-out teacher descriptions in MBPP's style, keeping their meaning and constants
(data/seedlm/teacher_holdout_terse.json). With the program, the examples and the coder fixed,
this compares, for each description style:
  first      the first verified program at beam width 32 is correct on all three cases
  gold_lp    the coder's log-probability of the teacher's program (serialized with that
             statement's own literals), as quartiles and as a paired mean difference

  python scripts/seedlm8_terse.py --model s5-var
"""
import argparse
import ast
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

from mylittlellm.rulecoder import backinfer as bi
from mylittlellm.rulecoder.limits import cap_memory
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import coder3, coder4, language, post, pyx
from seedlm3_discovery import value_of
from seedlm3_train import named_in, teacher_held, teacher_task_items

TERSE = ROOT / "data/seedlm/teacher_holdout_terse.json"


def cases(task, description):
    """The task read with a given description: examples, literals, gold tokens and a judge, or None."""
    try:
        pairs = [(list(ast.literal_eval(i)), ast.literal_eval(o)) for i, o in zip(task["inputs"], task["outputs"])]
    except (ValueError, SyntaxError):
        return None
    literals = bi.description_literals(description)
    output = pairs[0][1] if isinstance(pairs[0][1], str) else None
    try:
        gold = pyx.serialize(task["expression"], len(pairs[0][0]), literals, output)
    except pyx.Unsupported:
        gold = None

    def judge(program):
        for args, expected in pairs:
            ok, value = value_of(program, args, literals, output)
            if not ok or not bi._matches(value, {"expected": expected, "mode": "eq"}):
                return False
        return True

    return {"description": description, "literals": literals, "args": pairs[0][0], "expected": pairs[0][1],
            "mode": "eq", "gold": gold, "judge": judge}


def run(net, vocabulary, grammar, case, width):
    prefix = vocabulary.condition(language.words(case["description"]), case["literals"], case["args"],
                                  case["expected"], named_in(case["description"]))
    beam = coder4.beam_search(net, vocabulary, grammar, prefix, case["args"], case["literals"], case["expected"],
                              case["mode"], width)
    verified = [post.to_prefix(p) for p, _ in beam
                if coder4.reproduces(p, case["args"], case["literals"], case["expected"], case["mode"])]
    first = bool(verified) and case["judge"](verified[0])
    gold = None
    if case["gold"] is not None:
        try:
            gold = coder4.program_log_prob(net, vocabulary, prefix, post.to_postfix(list(case["gold"])), case["args"],
                                           case["literals"], case["expected"], case["mode"])
        except (KeyError, ValueError):
            gold = None
    return first, gold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--width", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cap_memory()      # a program's runaway allocation fails instead of killing the run
    net, vocabulary = coder4.load(ROOT / "outputs/seedlm" / f"{args.model}.pt")
    grammar = coder4.Grammar(vocabulary)
    terse = {row["id"]: row["terse"] for row in json.loads(TERSE.read_text())}
    tasks = [t for t in teacher_task_items() if teacher_held(t["id"]) and t["id"] in terse]
    first = {"original": 0, "terse": 0, "both": 0, "only_original": 0, "only_terse": 0}
    gold = {"original": [], "terse": []}
    words = {"original": [], "terse": []}
    for task in tasks:
        outcome = {}
        for style, description in (("original", task["description"]), ("terse", terse[task["id"]])):
            case = cases(task, description)
            if case is None:
                break
            outcome[style] = run(net, vocabulary, grammar, case, args.width)
            words[style].append(len(language.words(description)))
        if len(outcome) < 2:
            continue
        (a, ga), (b, gb) = outcome["original"], outcome["terse"]
        first["original"] += a
        first["terse"] += b
        first["both"] += a and b
        first["only_original"] += a and not b
        first["only_terse"] += b and not a
        if ga is not None and gb is not None:
            gold["original"].append(ga)
            gold["terse"].append(gb)
    report = {"model": args.model, "width": args.width, "tasks": len(tasks), "first_correct": first,
              "words_median": {k: statistics.median(v) for k, v in words.items()},
              "gold_log_prob_quartiles": {k: [round(q, 2) for q in statistics.quantiles(v, n=4)]
                                          for k, v in gold.items() if len(v) >= 4},
              "gold_paired_mean_drop_nats": round(statistics.mean(a - b for a, b in zip(gold["original"],
                                                                                          gold["terse"])), 2)
              if gold["original"] else None, "gold_scored": len(gold["original"])}
    (ROOT / "reports/seedlm" / f"terse_{args.model}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
