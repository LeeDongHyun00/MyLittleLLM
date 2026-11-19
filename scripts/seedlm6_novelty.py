"""Check composed tasks: valid as teacher tasks, and holding a composition the coder's training lacks.

SeedLM-6's teachers write tasks that join a computation from one training task with one from
another. A task counts only if its program has at least one parent -> child link (an "edge") or one
node-with-children combination (a "twig") that no training program has: the training parts of the
teacher tasks, SeedLM-5's variants, the development one-liners, the real development rows, and a
sample of the synthetic library programs (scripts/seedlm5_fragments.py defines the pieces).

  python scripts/seedlm6_novelty.py data/seedlm/compose_tasks_part1.json     per-task verdicts
  python scripts/seedlm6_novelty.py --summary data/seedlm/compose_tasks_part*.json

The seen pieces are computed once and cached in outputs/seedlm/v6/seen_pieces.json.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylittlellm.rulecoder.tasks import ROOT
from seedlm3_task_check import check
from seedlm5_fragments import pieces, synthetic_programs, training_programs

CACHE = ROOT / "outputs/seedlm/v6/seen_pieces.json"


def seen_pieces(refresh=False):
    if CACHE.exists() and not refresh:
        data = json.loads(CACHE.read_text())
        return {name: {tuple(x) for x in values} for name, values in data.items()}
    seen = {"edges": set(), "twigs": set()}
    for program in training_programs() + synthetic_programs(100000):
        found = pieces(program)
        seen["edges"] |= found["edges"]
        seen["twigs"] |= found["twigs"]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps({name: sorted(map(list, values)) for name, values in seen.items()}))
    return seen


def verdict(task, seen):
    ok, reason, details = check(task)
    if not ok:
        return {"id": task.get("id"), "ok": False, "reason": reason}
    found = pieces(details["tokens"])
    new_edges = sorted(found["edges"] - seen["edges"])
    new_twigs = sorted(found["twigs"] - seen["twigs"])
    return {"id": task.get("id"), "ok": True, "novel": bool(new_edges or new_twigs),
            "new_edges": [f"{p} [{s}] <- {c}" for p, s, c in new_edges], "new_twigs": len(new_twigs),
            "tokens": len(details["tokens"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--summary", action="store_true", help="counts only")
    parser.add_argument("--refresh", action="store_true", help="recompute the cached seen pieces")
    args = parser.parse_args()
    seen = seen_pieces(args.refresh)
    total = valid = novel = 0
    for name in args.files:
        for task in json.loads(Path(name).read_text()):
            result = verdict(task, seen)
            total += 1
            valid += result["ok"]
            novel += result.get("novel", False)
            if not args.summary:
                print(json.dumps(result))
    print(json.dumps({"tasks": total, "valid": valid, "valid_and_novel": novel}))


if __name__ == "__main__":
    main()
