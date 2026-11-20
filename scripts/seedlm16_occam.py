"""The coder vote with ties broken by simplicity: among equally supported behaviours, the shortest program.

SeedLM-15 found that the order in which tied behaviours are compared moves the thirteen-coder vote
by up to 8 held-out teacher tasks. Occam's razor is the standard tie-break in program synthesis: of the
programs consistent with the evidence, prefer the simplest. Scored over SeedLM-15's cached proposals
(outputs/seedlm/v15/), on the same random subsets.

  python scripts/seedlm16_occam.py [--subsets 200]
"""
import argparse
import json
from pathlib import Path
import random
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylittlellm.rulecoder import consensus
from mylittlellm.rulecoder.tasks import ROOT
from seedlm15_ensemble_scaling import CACHE


def occam_correct(rows):
    """The vote over the given proposals, ties to the fewest tokens, then to the earlier proposal."""
    present = [r for r in rows if r is not None]
    if not present:
        return False
    support = [sum(i == j or consensus.compare(a["behaviour"], b["behaviour"]) == "agree"
                   for j, b in enumerate(present)) for i, a in enumerate(present)]
    best = max(support)
    tied = [i for i, s in enumerate(support) if s == best]
    return present[min(tied, key=lambda i: (len(present[i]["program"]), i))]["correct"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subsets", type=int, default=200)
    args = parser.parse_args()
    coders = sorted(p.stem for p in CACHE.glob("*.json"))
    data = {c: json.loads((CACHE / f"{c}.json").read_text()) for c in coders}
    rng = random.Random(0)          # the seed SeedLM-15's scoring used
    report = {"coders": coders, "subsets_per_size": args.subsets}
    for set_name in ("teacher", "development"):
        n_tasks = len(data[coders[0]][set_name])
        by_size = {}
        for k in range(1, len(coders) + 1):
            totals = []
            for _ in range(args.subsets):
                members = rng.sample(coders, k)
                totals.append(sum(occam_correct([data[c][set_name][t] for c in members]) for t in range(n_tasks)))
            by_size[k] = {"mean": round(statistics.mean(totals), 2), "sd": round(statistics.pstdev(totals), 2),
                          "min": min(totals), "max": max(totals)}
        report[set_name] = {"tasks": n_tasks, "vote_by_size": by_size}
        print(set_name, json.dumps({k: v["mean"] for k, v in by_size.items()}), "13:", by_size[len(coders)],
              flush=True)
    (ROOT / "reports/seedlm/ensemble_occam.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
