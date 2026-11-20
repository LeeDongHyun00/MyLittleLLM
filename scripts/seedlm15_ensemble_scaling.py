"""How does the coder vote grow with the number of coders? Proposals cached once, subsets scored after.

SeedLM-10 (seven coders) and SeedLM-12 (six other coders) both found the vote of separately trained
coders about 9 tasks above the best single coder on held-out teacher tasks. This script caches every
coder's first verified program (beam width 32) and its behaviour on the 24 consensus probes for the
held-out teacher tasks and the held-out development fifth, judges each proposal, and then scores the
vote (SeedLM-10's rule, ties to the earlier coder in a random order) for random subsets of every size.

  python scripts/seedlm15_ensemble_scaling.py cache --coders s5-var s8-all ...   (outputs/seedlm/v15/)
  python scripts/seedlm15_ensemble_scaling.py score [--subsets 200]
"""
import argparse
import json
from pathlib import Path
import random
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mylittlellm.rulecoder import backinfer as bi, consensus
from mylittlellm.rulecoder.limits import cap_memory
from mylittlellm.rulecoder.sandbox import run_public_asserts
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import data2, pyx, select
from seedlm2_real import held_out

BANK = ROOT / "outputs/rule_pilot/teacher/round3/all_tasks.json"
CACHE = ROOT / "outputs/seedlm/v15"


def development_cases():
    out = []
    for item in json.loads(BANK.read_text()):
        if item["source"] != "extra" or not held_out(item["task_id"]):
            continue
        example = bi.parse_example(item["visible_assert"], item["entry_point"])
        if example is None or not example["args"] or len(example["args"]) > data2.MAX_ARGS:
            out.append(None)
            continue
        literals = bi.description_literals(item["description"])
        expected = example["expected"]
        output = expected if isinstance(expected, str) else None

        def judge(program, item=item, n=len(example["args"]), literals=literals, output=output):
            code = pyx.to_code(item["entry_point"], n, program, literals, output)
            return run_public_asserts(code, "\n".join(item["asserts"]))["status"] == "pass"

        out.append({"description": item["description"], "literals": literals, "args": example["args"],
                    "expected": expected, "mode": example["mode"], "judge": judge})
    return out


def cache(coders):
    import torch
    from mylittlellm.seedlm import coder3, coder4
    from seedlm10_ensemble import first_verified
    from seedlm3_discovery import teacher_cases
    torch.set_num_threads(4)
    cap_memory()      # a program's runaway allocation fails instead of killing the run
    sets = {"teacher": teacher_cases(), "development": development_cases()}
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in coders:
        path = CACHE / f"{name}.json"
        if path.exists():
            continue
        if torch.load(ROOT / "outputs/seedlm" / f"{name}.pt", weights_only=False).get("kind") == "post-coder":
            net, vocabulary = coder4.load(ROOT / "outputs/seedlm" / f"{name}.pt")
            grammar = coder4.Grammar(vocabulary)
            propose = first_verified
        else:                       # SeedLM-3's prefix coders (SeedLM-17: a different architecture)
            net, vocabulary = coder3.load(ROOT / "outputs/seedlm" / f"{name}.pt")
            grammar = coder3.Grammar(vocabulary)
            propose = first_verified_prefix
        record = {}
        for set_name, tasks in sets.items():
            rows = []
            for task in tasks:
                if task is None:
                    rows.append(None)
                    continue
                program = propose(net, vocabulary, grammar, task["description"], task["literals"],
                                  task["args"], task["expected"], task["mode"])
                if program is None:
                    rows.append(None)
                    continue
                output = task["expected"] if isinstance(task["expected"], str) else None
                probes = consensus.probe_inputs(list(task["args"]), 24)
                rows.append({"program": program, "correct": bool(task["judge"](program)),
                             "behaviour": select.behaviour(program, probes, task["literals"], output)})
            record[set_name] = rows
        path.write_text(json.dumps(record) + "\n")
        print(name, {k: sum(bool(r and r["correct"]) for r in v) for k, v in record.items()}, flush=True)


def first_verified_prefix(net, vocabulary, grammar, description, literals, args, expected, mode, width=32):
    """SeedLM-3's prefix coder: its first program at beam width 32 that reproduces the example."""
    from mylittlellm.seedlm import coder3, language
    from seedlm3_train import named_in
    prefix = vocabulary.condition(language.words(description), literals, args, expected, named_in(description))
    for program, _ in coder3.wide_beam_search(net, vocabulary, grammar, prefix, len(args), len(literals),
                                              isinstance(expected, str), width):
        if coder3.reproduces(program, args, literals, expected, mode):
            return list(program)
    return None


def vote_correct(rows):
    """SeedLM-10's vote over the given proposals (in order); whether the chosen one is correct."""
    present = [r for r in rows if r is not None]
    if not present:
        return False
    support = [sum(i == j or consensus.compare(a["behaviour"], b["behaviour"]) == "agree"
                   for j, b in enumerate(present)) for i, a in enumerate(present)]
    best = max(support)
    return present[support.index(best)]["correct"]


def score(subsets, seed=0):
    coders = sorted(p.stem for p in CACHE.glob("*.json"))
    data = {c: json.loads((CACHE / f"{c}.json").read_text()) for c in coders}
    rng = random.Random(seed)
    report = {"coders": coders, "subsets_per_size": subsets}
    for set_name in ("teacher", "development"):
        n_tasks = len(data[coders[0]][set_name])
        singles = {c: sum(bool(r and r["correct"]) for r in data[c][set_name]) for c in coders}
        by_size = {}
        for k in range(1, len(coders) + 1):
            totals = []
            for _ in range(subsets):      # the full set too: only the order (ties) varies
                members = rng.sample(coders, k)
                totals.append(sum(vote_correct([data[c][set_name][t] for c in members]) for t in range(n_tasks)))
            by_size[k] = {"mean": round(statistics.mean(totals), 2),
                          "sd": round(statistics.pstdev(totals), 2), "min": min(totals), "max": max(totals)}
        oracle = sum(any(data[c][set_name][t] and data[c][set_name][t]["correct"] for c in coders)
                     for t in range(n_tasks))
        report[set_name] = {"tasks": n_tasks, "single": singles, "vote_by_size": by_size, "oracle": oracle}
        print(set_name, json.dumps(report[set_name]), flush=True)
    (ROOT / "reports/seedlm/ensemble_scaling.json").write_text(json.dumps(report, indent=1) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["cache", "score"])
    parser.add_argument("--coders", nargs="+")
    parser.add_argument("--subsets", type=int, default=200)
    args = parser.parse_args()
    if args.command == "cache":
        cache(args.coders)
    else:
        score(args.subsets)


if __name__ == "__main__":
    main()
