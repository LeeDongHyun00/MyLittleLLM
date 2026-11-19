"""Two coders on the same held-out teacher programs, task by task, by novelty and by an arm's coverage.

Reads the local per-task outcomes that `scripts/seedlm5_fragments.py --model` keeps
(outputs/seedlm/verdicts/fragments_<model>.json) and reports, for the programs with a new link or
twig (SeedLM-6's primary measure) and for the others, how many both coders solve, only the base, and
only the arm; and, with --arm, the same split by whether the arm's added data holds the program's new
pieces. Counts only.

  python scripts/seedlm6_compare.py --base s5-var --model s6-graft --arm graft
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylittlellm.rulecoder.tasks import ROOT
from seedlm3_discovery import teacher_cases
from seedlm5_fragments import added_pieces, pieces, synthetic_programs, training_programs

LOCAL = ROOT / "outputs/seedlm/verdicts"
NEW = ("tokens", "edges", "twigs")


def tally(pairs):
    return {"both": sum(a and b for a, b in pairs), "only_base": sum(a and not b for a, b in pairs),
            "only_model": sum(b and not a for a, b in pairs), "neither": sum(not a and not b for a, b in pairs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="s5-var")
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", choices=["graft", "compose"])
    parser.add_argument("--set", default="teacher_holdout")
    args = parser.parse_args()
    base = json.loads((LOCAL / f"fragments_{args.base}.json").read_text())[args.set]
    model = json.loads((LOCAL / f"fragments_{args.model}.json").read_text())[args.set]
    assert len(base) == len(model) and all(a["level"] == b["level"] and a["index"] == b["index"]
                                           for a, b in zip(base, model))
    report = {"base": args.base, "model": args.model, "set": args.set,
              "new_composition": tally([(a["correct"], b["correct"]) for a, b in zip(base, model)
                                        if a["level"] in NEW]),
              "all_pieces_seen": tally([(a["correct"], b["correct"]) for a, b in zip(base, model)
                                        if a["level"] not in NEW])}
    if args.arm and args.set == "teacher_holdout":
        known = {"edges": set(), "twigs": set()}
        for program in training_programs() + synthetic_programs(100000):
            found = pieces(program)
            known["edges"] |= found["edges"]
            known["twigs"] |= found["twigs"]
        added = added_pieces(args.arm)
        split = {"covered": [], "uncovered": []}
        tasks = teacher_cases()
        for a, b in zip(base, model):
            if a["level"] not in NEW:
                continue
            task = tasks[a["index"]]
            found = pieces(task["gold"])
            new = (found["edges"] - known["edges"]) | (found["twigs"] - known["twigs"])
            covered = bool(new) and new <= (added["edges"] | added["twigs"])
            split["covered" if covered else "uncovered"].append((a["correct"], b["correct"]))
        report["by_coverage"] = {name: tally(pairs) for name, pairs in split.items()}
    out = ROOT / "reports/seedlm" / f"compare_{args.base}_{args.model}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
