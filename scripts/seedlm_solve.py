"""Solve one task with the SeedLM coder ensemble: description + one visible assert in, Python function out.

Every coder writes its first program that reproduces the visible example (beam width 32); the programs
are run on 24 probe inputs, and the behaviour the most coders share wins, ties going to the shortest
program (SeedLM-10, 15 and 16: on held-out teacher tasks thirteen coders solve 72.9 of 139 this way,
the best single coder 62). The rule engine's own answer can be shown alongside.

  python scripts/seedlm_solve.py --description "Write a function to find the sum of the squares of the odd numbers in a list." \\
      --assert "assert odd_square_sum([1, 2, 3]) == 10"
  python scripts/seedlm_solve.py --task task.json [--coders s12-a1 s12-a2 ...] [--engine]
"""
import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

from mylittlellm.rulecoder import backinfer as bi, consensus
from mylittlellm.rulecoder.limits import cap_memory
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import coder4, data2, pyx, select
from seedlm10_ensemble import first_verified

CODERS = ["s4-post", "s5-var", "s6-comp", "s6-graft", "s8-all", "s9-early", "s9-early-last",
          "s12-a1", "s12-a2", "s12-a3", "s12-b1", "s12-b2", "s12-b3"]


def solve(description, visible_assert, coders, entry_point=None):
    """(code or None, details) for one task."""
    entry_point = entry_point or re.search(r"assert\s+(?:not\s+)?\(?\s*([A-Za-z_]\w*)\s*\(", visible_assert).group(1)
    example = bi.parse_example(visible_assert, entry_point)
    if example is None or not example["args"] or len(example["args"]) > data2.MAX_ARGS:
        return None, {"reason": "the visible assert could not be read as one call with 1 to 4 arguments"}
    args, expected, mode = example["args"], example["expected"], example["mode"]
    literals = bi.description_literals(description)
    output = expected if isinstance(expected, str) else None
    proposals = []
    for name, net, vocabulary, grammar in coders:
        program = first_verified(net, vocabulary, grammar, description, literals, args, expected, mode)
        if program is not None:
            proposals.append((name, program))
    if not proposals:
        return None, {"reason": "no coder found a program that reproduces the example"}
    probes = consensus.probe_inputs(list(args), 24)
    results = [select.behaviour(p, probes, literals, output) for _, p in proposals]
    support = [sum(i == j or consensus.compare(a, b) == "agree" for j, b in enumerate(results))
               for i, a in enumerate(results)]
    best = max(support)
    pick = min((i for i, s in enumerate(support) if s == best), key=lambda i: (len(proposals[i][1]), i))
    code = pyx.to_code(entry_point, len(args), proposals[pick][1], literals, output)
    agreeing = [proposals[j][0] for j in range(len(proposals))
                if j == pick or consensus.compare(results[pick], results[j]) == "agree"]
    return code, {"entry_point": entry_point, "coders_with_a_program": len(proposals), "of": len(coders),
                  "agreeing": agreeing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--description")
    parser.add_argument("--assert", dest="visible_assert")
    parser.add_argument("--task", help="a JSON file with description, visible_assert and optionally entry_point")
    parser.add_argument("--coders", nargs="+", default=CODERS)
    parser.add_argument("--engine", action="store_true", help="also show the rule engine's program")
    args = parser.parse_args()
    if args.task:
        task = json.loads(Path(args.task).read_text())
        description, visible_assert, entry_point = task["description"], task["visible_assert"], task.get("entry_point")
    else:
        description, visible_assert, entry_point = args.description, args.visible_assert, None
    if not description or not visible_assert:
        parser.error("give --description and --assert, or --task")
    torch.set_num_threads(4)
    cap_memory()      # a program's runaway allocation fails instead of killing the run
    coders = []
    for name in args.coders:
        net, vocabulary = coder4.load(ROOT / "outputs/seedlm" / f"{name}.pt")
        coders.append((name, net, vocabulary, coder4.Grammar(vocabulary)))
    code, details = solve(description, visible_assert, coders, entry_point)
    print(json.dumps(details, ensure_ascii=False))
    print(code if code else "# no program")
    if args.engine:
        found = bi.synthesize(description, visible_assert, details.get("entry_point") or entry_point, weights={})
        print("# engine:", "confident" if found and found.get("confident") else "not confident" if found else "none")
        print(found["code"] if found else "# no program")


if __name__ == "__main__":
    main()
