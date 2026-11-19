"""Pairs of training tasks for SeedLM-6's teachers to compose into new tasks.

Each pair holds two training-part tasks (generic teacher tasks or SeedLM-5 variants; never a held-out
one): a host and a donor, with their descriptions, one-line programs and example inputs and outputs.
A teacher writes one new task whose one-line program joins a computation of the donor into the host
(or the other way round) in a way the training programs never do (scripts/seedlm6_novelty.py checks
it). Half of the pairs are chosen so the donor's output has the type of one of the host's arguments,
which makes feeding one into the other natural; the other half are random.

  python scripts/seedlm6_pairs.py [--parts 6] [--per-part 220]   writes outputs/seedlm/v6/pairs_part*.json
"""
import argparse
import ast
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylittlellm.rulecoder.tasks import ROOT
from seedlm3_train import teacher_held, teacher_task_items
from seedlm4_train import variant_held

OUT = ROOT / "outputs/seedlm/v6"


def entries():
    items = [t for t in teacher_task_items() if not teacher_held(t["id"])]
    items += [t for t in teacher_task_items("variant_tasks") if not variant_held(t)]
    out = []
    for task in items:
        try:
            args = [ast.literal_eval(text) for text in task["inputs"]]
            outputs = [ast.literal_eval(text) for text in task["outputs"]]
        except (ValueError, SyntaxError):
            continue
        out.append({"id": task["id"], "category": task.get("category", ""), "description": task["description"],
                    "expression": task["expression"], "examples": [f"{a!r} -> {o!r}" for a, o in
                                                                    zip(task["inputs"], outputs)],
                    "arg_types": sorted({type(x).__name__ for a in args[:1] for x in a}),
                    "output_type": type(outputs[0]).__name__})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=int, default=6)
    parser.add_argument("--per-part", type=int, default=220)
    parser.add_argument("--seed", type=int, default=6)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    items = entries()
    by_arg_type = {}
    for item in items:
        for name in item["arg_types"]:
            by_arg_type.setdefault(name, []).append(item)
    pairs, used = [], set()
    while len(pairs) < args.parts * args.per_part:
        donor = rng.choice(items)
        if len(pairs) % 2 == 0 and by_arg_type.get(donor["output_type"]):
            host = rng.choice(by_arg_type[donor["output_type"]])
        else:
            host = rng.choice(items)
        key = (host["id"], donor["id"])
        if host["id"] == donor["id"] or key in used or host["category"] == donor["category"] and rng.random() < 0.5:
            continue
        used.add(key)
        pairs.append({"pair": len(pairs) + 1, "host": host, "donor": donor})
    OUT.mkdir(parents=True, exist_ok=True)
    for part in range(args.parts):
        chunk = pairs[part::args.parts]
        (OUT / f"pairs_part{part + 1}.json").write_text(json.dumps(chunk, indent=1) + "\n")
    print(json.dumps({"training_tasks": len(items), "pairs": len(pairs), "parts": args.parts}))


if __name__ == "__main__":
    main()
