"""Wrong programs that fit one example, mined from a coder's own beam (SeedLM-23).

With one visible example, the coder's answer is the most probable program in its beam that reproduces
the example. SeedLM-21's beam statistics found its errors to be wrong programs that fit the example by
accident and come before the right one; maximum likelihood never shows the coder such programs. This
script runs a coder's beam (width 16) on training tasks with three known cases, the first one visible,
and keeps for each task the verified programs that come before the first one right on every case (all
verified ones when none is right), each paired with a right program (the coder's own first right one,
or else the task's own). Sources (training data only):
  teacher  - training parts of the teacher tasks and SeedLM-5's variants (written descriptions; a coder
             trained on them usually has them by heart, so they give few pairs)
  composed - the training part of SeedLM-6's composed tasks (written descriptions), for coders that did
             not train on them
  pysynth - a fresh draw of generated programs over training inputs (mylittlellm/seedlm/pysynth.py,
            seed 23, not the draw SeedLM-21 trained on), described like the grafts

  python scripts/seedlm23_mine.py --model s12-a1 --sources composed pysynth --limit 1200 600 [--count-only]
"""
import argparse
import ast
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

from mylittlellm.rulecoder import backinfer as bi
from mylittlellm.rulecoder.limits import Timeout, cap_memory, time_limit
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import coder3, coder4, language, post, pysynth, pyx, pyx_words
from seedlm3_train import named_in, teacher_held, teacher_task_items
from seedlm4_train import variant_held

OUT = ROOT / "outputs/seedlm/v23"
MAX_NEGATIVES = 3


def right(program, cases, literals):
    """Whether a prefix program gives every case's output. A verdict string "o" stays the visible case's
    output on the other cases too (as scripts/seedlm3_discovery.py judges)."""
    output = cases[0][1] if isinstance(cases[0][1], str) else None
    for args, expected in cases:
        try:
            with time_limit(coder3.EXECUTION_SECONDS):
                value = pyx.execute(list(program), args, literals, output)
        except (Exception, Timeout):
            return False
        if not bi._matches(value, {"expected": expected, "mode": "eq"}):
            return False
    return True


def teacher_tasks(source="teacher"):
    if source == "composed":
        items = [t for t in teacher_task_items("compose_tasks") if not teacher_held(t["id"])]
    else:
        items = [t for t in teacher_task_items() if not teacher_held(t["id"])]
        items += [t for t in teacher_task_items("variant_tasks") if not variant_held(t)]
    out = []
    for task in items:
        try:
            cases = [(list(ast.literal_eval(i)), ast.literal_eval(o)) for i, o in zip(task["inputs"], task["outputs"])]
        except (ValueError, SyntaxError):
            continue
        if len(cases) < 3:
            continue
        literals = bi.description_literals(task["description"])
        out.append({"source": source, "id": task["id"], "words": language.words(task["description"]),
                    "named": named_in(task["description"]), "literals": literals, "cases": cases,
                    "gold": list(task["tokens"])})
    return out


def pysynth_tasks(count, describer):
    from seedlm6_grafts import programs
    rng = random.Random(23)
    out = []
    for row in pysynth.generate(programs(), count, seed=23):
        cases = [(list(ast.literal_eval(c)), ast.literal_eval(o)) for c, o in zip(row["cases"], row["outputs"])]
        words = pyx_words.describe(rng, row["tokens"], row["literals"], cases[0][0], describer=describer)
        out.append({"source": "pysynth", "id": f"p{len(out)}", "words": words, "named": named_in(" ".join(words)),
                    "literals": row["literals"], "cases": cases, "gold": list(row["tokens"])})
    return out


def mine(net, vocabulary, grammar, task, width=16):
    """(positive, negatives, counts) for one task; negatives are prefix programs."""
    args, expected = task["cases"][0]
    prefix = vocabulary.condition(task["words"], task["literals"], args, expected, task["named"])
    verified, positive = [], None
    for postfix, _ in coder4.beam_search(net, vocabulary, grammar, prefix, args, task["literals"], expected, "eq", width):
        if not coder4.reproduces(postfix, args, task["literals"], expected, "eq"):
            continue
        program = post.to_prefix(postfix)
        if right(program, task["cases"], task["literals"]):
            positive = program
            break
        verified.append(program)
    first_right = positive is not None and not verified
    if positive is None:
        positive = task["gold"]
    return positive, verified[:MAX_NEGATIVES], {"first_right": first_right, "wrong_first": bool(verified)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sources", nargs="+", default=["composed", "pysynth"])
    parser.add_argument("--limit", type=int, nargs="+", default=[2000], help="tasks per source (one number, or one per source)")
    parser.add_argument("--count-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    cap_memory()
    net, vocabulary = coder4.load(ROOT / "outputs/seedlm" / f"{args.model}.pt")
    grammar = coder4.Grammar(vocabulary)
    describer = language.Describer(bi.OPERATORS, language.load_phrases())
    started = time.time()
    pairs, counts = [], {}
    limits = args.limit * len(args.sources) if len(args.limit) == 1 else args.limit
    for source, limit in zip(args.sources, limits):
        tasks = pysynth_tasks(limit, describer) if source == "pysynth" else teacher_tasks(source)
        random.Random(0).shuffle(tasks)
        tasks = tasks[:limit]
        c = {"tasks": len(tasks), "first_right": 0, "wrong_first": 0, "no_verified_right": 0, "pairs": 0}
        for task in tasks:
            try:
                positive, negatives, flags = mine(net, vocabulary, grammar, task)
            except (KeyError, ValueError):
                continue        # a token the coder's vocabulary lacks
            c["first_right"] += flags["first_right"]
            c["wrong_first"] += flags["wrong_first"]
            c["no_verified_right"] += positive is task["gold"] and flags["wrong_first"]
            for negative in negatives:
                c["pairs"] += 1
                pairs.append({"source": source, "id": task["id"], "words": task["words"], "named": task["named"],
                              "literals": repr(task["literals"]), "args": repr(task["cases"][0][0]),
                              "expected": repr(task["cases"][0][1]), "positive": positive, "negative": negative})
        counts[source] = c
        print(source, json.dumps(c), round(time.time() - started), "s", flush=True)
    if not args.count_only:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"pairs_{args.model}.json").write_text(json.dumps(pairs) + "\n")
    print(json.dumps({"model": args.model, "counts": counts, "seconds": round(time.time() - started)}))


if __name__ == "__main__":
    main()
