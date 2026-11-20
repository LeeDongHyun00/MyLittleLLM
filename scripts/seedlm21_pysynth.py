"""Generated programs that mix Python's nodes with the rule library (SeedLM-21), cached for training.

mylittlellm/seedlm/pysynth.py writes them over the input cases of the training teacher tasks and
variants (the hosts SeedLM-6's grafts use); this script caches a deterministic set and shows samples.

  python scripts/seedlm21_pysynth.py --count 30000 [--seed 0] [--save]
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
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import pysynth, pyx


def cache_path(count, seed, library="all"):
    return ROOT / "outputs/seedlm/v21" / f"pysynth_{count}_{seed}_{library}.json"


def parsed(row):
    return dict(row, cases=[list(ast.literal_eval(text)) for text in row["cases"]],
                outputs=[ast.literal_eval(text) for text in row["outputs"]])


def make(count, seed, library="all", families=None):
    from seedlm5_fragments import training_programs
    from seedlm6_grafts import programs        # here: seedlm6_grafts imports the trainer
    return pysynth.generate(programs(), count, seed, families,
                            pysynth.library_from(training_programs(), library))


def load_or_make(count, seed=0, library="all"):
    path = cache_path(count, seed, library)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(make(count, seed, library)))
    return [parsed(r) for r in json.loads(path.read_text())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--library", default="all", choices=["all", "training", "sqrt"])
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    started = time.time()
    rows = make(args.count, args.seed, args.library)
    if args.save:
        path = cache_path(args.count, args.seed, args.library)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows))
    lengths = sorted(len(r["tokens"]) for r in rows)
    print(json.dumps({"programs": len(rows), "seconds": round(time.time() - started, 1),
                      "tokens_quartiles": [lengths[len(lengths) // 4], lengths[len(lengths) // 2],
                                           lengths[3 * len(lengths) // 4]],
                      "with_library": sum(any("/" in t and t[0] not in "@$" for t in r["tokens"]) for r in rows),
                      "with_comprehension": sum(any(t.startswith("@Comp") for t in r["tokens"]) for r in rows)}))
    for row in random.Random(1).sample(rows, min(12, len(rows))):
        print(pyx.to_code("f", len(ast.literal_eval(row["cases"][0])), row["tokens"], row["literals"]).splitlines()[-1].strip(),
              "|", row["cases"][0][:50], "->", row["outputs"][0][:40])


if __name__ == "__main__":
    main()
