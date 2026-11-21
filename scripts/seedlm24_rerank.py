"""SeedLM-24's measure: the verifier's choice among a coder's verified programs, against the coder's first.

For each evaluation coder, its width-32 beam's verified programs on the held-out teacher tasks and the
development fifth (scripts/seedlm24_verifier_data.py heldout) are scored by the verifier; the chosen program
is the one with the highest probability of being right (ties: beam order). Reported per coder and on
average: tasks right with the coder's first verified program, with the verifier's choice, and with some
verified program (the ceiling); also, not for choosing, the verifier combined with the coder's log-probability.
Counts only.

  python scripts/seedlm24_rerank.py --verifier verifier-1 --coders s12-a1 s12-a2 s12-a3
"""
import argparse
import ast
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import verifier

DATA = ROOT / "outputs/seedlm/v24"


def choose(probabilities, candidates, weight):
    scores = [math.log(max(p, 1e-9)) + weight * c["log_prob"] for p, c in zip(probabilities, candidates)]
    return max(range(len(candidates)), key=lambda i: (scores[i], -i))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verifier", required=True)
    parser.add_argument("--coders", nargs="+", required=True)
    parser.add_argument("--output", default="reports/seedlm/verifier_rerank.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    net, vocabulary, tokens = verifier.load(ROOT / "outputs/seedlm" / f"{args.verifier}.pt")
    report = {"verifier": args.verifier, "coders": {}}
    for coder in args.coders:
        record = json.loads((DATA / f"heldout_{coder}.json").read_text())
        entry = {}
        for set_name, rows in record.items():
            counts = {"tasks": len(rows), "first": 0, "verifier": 0, "verifier_plus_coder": 0, "some_right": 0,
                      "changed": 0, "fixed": 0, "broken": 0}
            for row in rows:
                if not row or not row["candidates"]:
                    continue
                candidates = row["candidates"]
                literals, a, expected, probes = (ast.literal_eval(row[k]) for k in ("literals", "args", "expected", "probes"))
                encoded = []
                for c in candidates:
                    try:
                        encoded.append(verifier.encode(vocabulary, tokens, row["words"], literals, a, expected,
                                                       row["named"], c["program"], probes, c["behaviour"]))
                    except KeyError:
                        encoded.append(None)
                usable = [i for i, e in enumerate(encoded) if e is not None]
                probabilities = [0.0] * len(candidates)
                if usable:
                    for i, p in zip(usable, verifier.right_probability(net, tokens, [encoded[i] for i in usable])):
                        probabilities[i] = p
                pick = choose(probabilities, candidates, 0.0)
                both = choose(probabilities, candidates, 1.0)
                counts["first"] += candidates[0]["right"]
                counts["verifier"] += candidates[pick]["right"]
                counts["verifier_plus_coder"] += candidates[both]["right"]
                counts["some_right"] += any(c["right"] for c in candidates)
                counts["changed"] += pick != 0
                counts["fixed"] += candidates[pick]["right"] and not candidates[0]["right"]
                counts["broken"] += candidates[0]["right"] and not candidates[pick]["right"]
            entry[set_name] = counts
        report["coders"][coder] = entry
        print(coder, json.dumps(entry), flush=True)
    report["mean_gain"] = {s: round(statistics.mean(report["coders"][c][s]["verifier"] - report["coders"][c][s]["first"]
                                                    for c in args.coders), 2) for s in ("teacher", "development")}
    text = json.dumps(report, indent=1)
    print(text)
    (ROOT / args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
