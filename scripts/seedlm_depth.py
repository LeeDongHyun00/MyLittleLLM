"""Exhaustive search against SeedLM proposals on unseen synthetic programs, by program depth.

Exhaustive back-inference covers every program of up to two operators and nothing deeper. A
learned proposer has no depth limit but looks at only a beam of candidates. This measures, for
programs of one, two and three operators, how often each finds a program that reproduces the
example, given the same hints about which operators the description names.
"""
import argparse
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mylittlellm.rulecoder import backinfer as bi
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import data, propose, synth


def describe(names):
    """Text that names these operators, so the search is given the hints the model is given."""
    return ". ".join(" ".join(bi.OPERATORS[name].alternatives[0]) for name in names if bi.OPERATORS[name].alternatives)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--per-depth", type=int, default=80)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--budget", type=float, default=2.0, help="seconds the exhaustive search may use")
    parser.add_argument("--seed", type=int, default=424242, help="not used by any training or evaluation set")
    args = parser.parse_args()
    net, vocabulary = propose.load(ROOT / "outputs/seedlm" / f"{args.model}.pt")
    arities = propose.arity_table(vocabulary)
    rng = random.Random(args.seed)
    groups = {1: [], 2: [], 3: []}
    while any(len(group) < args.per_depth for group in groups.values()):
        example = synth.sample(rng)
        if example is None:
            continue
        depth = sum("/" in token for token in example["program"])
        named = data.named_for_training(example, vocabulary, rng)
        tokens, _ = vocabulary.encode(example["args"], example["expected"], named, example["program"])
        if len(tokens) <= data.MAX_LENGTH and len(groups[depth]) < args.per_depth:
            groups[depth].append((example, named))

    report = {"model": args.model, "width": args.width, "search_budget_seconds": args.budget, "depths": {}}
    for depth, group in groups.items():
        stats = {"examples": len(group), "search_found": 0, "model_found": 0, "both": 0, "search_only": 0,
                 "model_only": 0, "neither": 0, "search_seconds": 0.0, "model_seconds": 0.0}
        for example, named in group:
            target = {"args": example["args"], "expected": example["expected"], "mode": "eq"}
            started = time.monotonic()
            searched = bool(bi.search(target, describe(named), args.budget, weights={}))
            stats["search_seconds"] += time.monotonic() - started
            started = time.monotonic()
            proposed = bool(propose.propose(net, vocabulary, example["args"], example["expected"], named,
                                            args.width, arities))
            stats["model_seconds"] += time.monotonic() - started
            stats["search_found"] += searched
            stats["model_found"] += proposed
            stats[("both" if proposed else "search_only") if searched else ("model_only" if proposed else "neither")] += 1
        stats["search_seconds"] = round(stats["search_seconds"] / len(group), 3)
        stats["model_seconds"] = round(stats["model_seconds"] / len(group), 3)
        report["depths"][str(depth)] = stats
        print(depth, json.dumps(stats), flush=True)
    (ROOT / "reports/seedlm" / f"depth_{args.model}.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
