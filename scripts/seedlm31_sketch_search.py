"""SeedLM-31: does forcing the structure of known sketches let a coder reach programs its own beam does not?

For each task, a prefix coder runs its own beam (width 32) and, separately, a sketch-held beam
(mylittlellm/seedlm/sketch.py): the K sketches the naive Bayes ranker puts first for the task's description
words, argument count and output kind, each decoded with its fixed tokens forced and its holes (at most 5
tokens each) filled by the coder, 4 hypotheses per sketch. The sketch library is built from the
teacher-written training programs only. Reported per set, counts only: tasks where some verified program
(one that reproduces the visible example) is right on every case or passes the task's tests, for the plain
beam, the sketch search and both; the first verified program's accuracy for the plain beam; verified
candidates per task; seconds per task; and, on the held-out teacher tasks, how often the program's own
sketch is among the K ranked.

  python scripts/seedlm31_sketch_search.py --model s20-p5 [--sketches 32 --per-sketch 4] [--limit N]
"""
import argparse
import ast
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

from mylittlellm.rulecoder.limits import cap_memory
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import coder3, language, sketch as sk
from seedlm3_train import named_in, oneliner_items, teacher_held, teacher_task_items
from seedlm4_train import variant_held


def training_examples():
    """(tokens, description words, number of arguments, visible output) of every teacher-written training
    program: the training parts of the teacher tasks, variants and composed tasks, and the development
    one-liners."""
    out = []
    for prefix, held in (("teacher_tasks", lambda t: teacher_held(t["id"])), ("variant_tasks", variant_held),
                         ("compose_tasks", lambda t: teacher_held(t["id"]))):
        for task in teacher_task_items(prefix):
            if held(task):
                continue
            try:
                args, output = ast.literal_eval(task["inputs"][0]), ast.literal_eval(task["outputs"][0])
            except (ValueError, SyntaxError):
                continue
            out.append((task["tokens"], language.words(task["description"]), len(args), output))
    for entry in oneliner_items():
        if entry["part"] == "train":
            out.append((entry["tokens"], language.words(entry["item"]["description"]), len(entry["example"]["args"]),
                        entry["example"]["expected"]))
    return out


def engine_failures():
    """Per development task, in development_cases() order: whether the rule engine fails it (SeedLM-4's
    local verdicts; counts only leave this script)."""
    from seedlm2_real import held_out
    verdicts = json.loads((ROOT / "outputs/seedlm/verdicts/holdout4_s5-var.json").read_text())
    bank = json.loads((ROOT / "outputs/rule_pilot/teacher/round3/all_tasks.json").read_text())
    return [not verdicts[i["task_id"]]["engine"] for i in bank if i["source"] == "extra" and held_out(i["task_id"])]


def measure(net, vocabulary, grammar, library, cases, k, per_sketch, golds=None, engine_fails=None):
    counts = {"tasks": 0, "plain_some_right": 0, "sketch_some_right": 0, "union_some_right": 0,
              "plain_first_right": 0, "sketch_only": 0, "plain_verified": 0, "sketch_verified": 0,
              "seconds_plain": 0.0, "seconds_sketch": 0.0}
    gold_in_top = 0
    failing = {"engine_failures": 0, "plain_some_right": 0, "union_some_right": 0}
    for index, case in enumerate(cases):
        if case is None:
            continue
        counts["tasks"] += 1
        words = language.words(case["description"])
        args, literals, expected = case["args"], case["literals"], case["expected"]
        prefix = vocabulary.condition(words, literals, args, expected, named_in(case["description"]))
        string_output = isinstance(expected, str)
        started = time.time()
        plain = [p for p, _ in coder3.wide_beam_search(net, vocabulary, grammar, prefix, len(args), len(literals),
                                                       string_output, 32)
                 if coder3.reproduces(p, args, literals, expected, case["mode"])]
        counts["seconds_plain"] += time.time() - started
        ranked = library.rank(words, len(args), expected)[:k]
        started = time.time()
        found = sk.sketch_search(net, vocabulary, grammar, prefix, len(args), len(literals), string_output, ranked,
                                 per_sketch=per_sketch)
        held = [p for p, _, _ in found if coder3.reproduces(p, args, literals, expected, case["mode"])]
        counts["seconds_sketch"] += time.time() - started
        plain_right = [bool(case["judge"](p)) for p in plain]
        sketch_right = any(case["judge"](p) for p in held)
        counts["plain_some_right"] += any(plain_right)
        counts["sketch_some_right"] += sketch_right
        counts["union_some_right"] += any(plain_right) or sketch_right
        counts["sketch_only"] += sketch_right and not any(plain_right)
        counts["plain_first_right"] += bool(plain_right and plain_right[0])
        if engine_fails is not None and engine_fails[index]:
            failing["engine_failures"] += 1
            failing["plain_some_right"] += any(plain_right)
            failing["union_some_right"] += any(plain_right) or sketch_right
        counts["plain_verified"] += len(plain)
        counts["sketch_verified"] += len(held)
        if golds is not None:
            try:
                gold_in_top += sk.sketch_of(golds[index])[0] in ranked
            except (ValueError, IndexError, KeyError):
                pass
        if counts["tasks"] % 20 == 0:
            print(counts["tasks"], json.dumps({k2: round(v, 1) if isinstance(v, float) else v for k2, v in counts.items()}),
                  flush=True)
    n = max(1, counts["tasks"])
    for key in ("plain_verified", "sketch_verified", "seconds_plain", "seconds_sketch"):
        counts[key + "_per_task"] = round(counts.pop(key) / n, 2)
    if golds is not None:
        counts["gold_sketch_in_ranked"] = gold_in_top
    if engine_fails is not None:
        counts["on_engine_failures"] = failing
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="s20-p5")
    parser.add_argument("--sketches", type=int, default=32)
    parser.add_argument("--per-sketch", type=int, default=4)
    parser.add_argument("--limit", type=int, help="only the first N tasks of each set (a timing check)")
    parser.add_argument("--sets", nargs="+", default=["teacher", "development"])
    parser.add_argument("--train-check", type=int, default=0,
                        help="instead of the held-out sets, N training teacher tasks (a check of the machinery)")
    parser.add_argument("--output", default="reports/seedlm/sketch_search.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    cap_memory()
    net, vocabulary = coder3.load(ROOT / "outputs/seedlm" / f"{args.model}.pt")
    grammar = coder3.Grammar(vocabulary)
    library = sk.Library(training_examples())
    report = {"model": args.model, "sketches": args.sketches, "per_sketch": args.per_sketch,
              "library_sketches": len(library.sketches)}
    if args.train_check:
        from seedlm23_mine import right
        tasks = [t for t in teacher_task_items() if not teacher_held(t["id"])][:args.train_check]
        cases = []
        for t in tasks:
            pairs = [(list(ast.literal_eval(i)), ast.literal_eval(o)) for i, o in zip(t["inputs"], t["outputs"])]
            from mylittlellm.rulecoder import backinfer as bi
            literals = bi.description_literals(t["description"])
            cases.append({"description": t["description"], "literals": literals, "args": pairs[0][0],
                          "expected": pairs[0][1], "mode": "eq",
                          "judge": lambda p, pairs=pairs, literals=literals: right(p, pairs, literals)})
        report["train_check"] = measure(net, vocabulary, grammar, library, cases, args.sketches, args.per_sketch,
                                        [t["tokens"] for t in tasks])
        print(json.dumps(report["train_check"]))
        return
    from seedlm3_discovery import teacher_cases
    from seedlm15_ensemble_scaling import development_cases
    for name in args.sets:
        cases = teacher_cases() if name == "teacher" else development_cases()
        golds = [c["gold"] for c in cases] if name == "teacher" else None
        fails = engine_failures() if name == "development" else None
        if args.limit:
            cases, golds = cases[:args.limit], (golds[:args.limit] if golds else None)
            fails = fails[:args.limit] if fails else None
        report[name] = measure(net, vocabulary, grammar, library, cases, args.sketches, args.per_sketch, golds, fails)
        print(name, json.dumps(report[name]), flush=True)
    if not args.limit:
        (ROOT / args.output).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
