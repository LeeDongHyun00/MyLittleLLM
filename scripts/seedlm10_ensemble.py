"""Agreement across separately trained coders: does the behaviour most coders share win?

SeedLM-7 found that agreement inside one coder's beam is not evidence: the beam's wrong programs are
variants of one mistake. Coders trained on different data mixes, or in different runs, make partly
different mistakes. Here each of several post-coders proposes its first verified program (beam width
32); each proposal is run on 24 probe inputs (the consensus probe grammar); two proposals agree when at
least 3 probes give both a value and none differ. The ensemble takes the proposal that the most coders
agree with (itself included); ties go to the earliest coder in the given order (the adopted coder
first). Counted on each set:
  single   each coder's own first verified program is correct
  vote     the ensemble's choice is correct
  oracle   some coder's proposal is correct
On the extra pool's held-out fifth, the engine-backed method is also scored: the engine's program
unless the engine marks it not confident, then the ensemble's choice (as pyx_override does with one
coder). Sets: held-out teacher tasks, held-out engine failures with a teacher one-liner (beyond), and
the held-out development fifth (counts only; per-task verdicts stay local).

  python scripts/seedlm10_ensemble.py --coders s5-var s8-all s6-graft s6-comp s9-early s9-early-last s4-post
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

from mylittlellm.rulecoder import backinfer as bi, consensus
from mylittlellm.rulecoder.limits import cap_memory
from mylittlellm.rulecoder.sandbox import run_public_asserts
from mylittlellm.rulecoder.tasks import ROOT, parse_prompt
from mylittlellm.seedlm import coder4, data2, language, post, pyx, select
from seedlm2_real import held_out
from seedlm3_discovery import beyond_cases, teacher_cases
from seedlm3_train import named_in

BANK = ROOT / "outputs/rule_pilot/teacher/round3/all_tasks.json"


def first_verified(net, vocabulary, grammar, description, literals, args, expected, mode, width=32):
    prefix = vocabulary.condition(language.words(description), literals, args, expected, named_in(description))
    for postfix, _ in coder4.beam_search(net, vocabulary, grammar, prefix, args, literals, expected, mode, width):
        if coder4.reproduces(postfix, args, literals, expected, mode):
            return post.to_prefix(postfix)
    return None


def vote(proposals, args, literals, output):
    """Index of the proposal most coders agree with (ties: the earliest), or None if there is none."""
    present = [i for i, p in enumerate(proposals) if p is not None]
    if not present:
        return None
    probes = consensus.probe_inputs(list(args), 24)
    if not probes or len(present) == 1:
        return present[0]
    results = {i: select.behaviour(proposals[i], probes, literals, output) for i in present}
    support = {i: sum(consensus.compare(results[i], results[j]) == "agree" or i == j for j in present)
               for i in present}
    best = max(support.values())
    return next(i for i in present if support[i] == best)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coders", nargs="+", required=True)
    parser.add_argument("--sets", nargs="+", default=["teacher", "beyond", "development"])
    parser.add_argument("--output", default="reports/seedlm/ensemble_holdout.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    cap_memory()      # a program's runaway allocation fails instead of killing the run
    coders = []
    for name in args.coders:
        net, vocabulary = coder4.load(ROOT / "outputs/seedlm" / f"{name}.pt")
        coders.append((name, net, vocabulary, coder4.Grammar(vocabulary)))
    report = {"coders": args.coders}
    for set_name in args.sets:
        if set_name == "development":
            continue
        tasks = teacher_cases() if set_name == "teacher" else beyond_cases()
        entry = {"tasks": len(tasks), "single": {n: 0 for n in args.coders}, "vote": 0, "oracle": 0,
                 "vote_changed": 0}
        for task in tasks:
            output = task["expected"] if isinstance(task["expected"], str) else None
            proposals = [first_verified(net, vocabulary, grammar, task["description"], task["literals"],
                                        task["args"], task["expected"], task["mode"])
                         for _, net, vocabulary, grammar in coders]
            verdicts = [p is not None and task["judge"](p) for p in proposals]
            for (name, *_), ok in zip(coders, verdicts):
                entry["single"][name] += ok
            pick = vote(proposals, task["args"], task["literals"], output)
            entry["vote"] += pick is not None and verdicts[pick]
            entry["oracle"] += any(verdicts)
            entry["vote_changed"] += pick is not None and pick != 0 and proposals[0] is not None
        report[set_name] = entry
        print(set_name, json.dumps(entry), flush=True)
    if "development" in args.sets:
        items = [i for i in json.loads(BANK.read_text()) if i["source"] == "extra" and held_out(i["task_id"])]
        entry = {"tasks": len(items), "engine": 0, "single": {n: 0 for n in args.coders}, "vote": 0,
                 "oracle": 0, "engine_override_vote": 0, "engine_override_first": 0,
                 "vote_versus_engine": [0, 0], "override_vote_versus_engine": [0, 0]}
        per_task = {}
        for item in items:
            description, asserts = item["description"], item["visible_assert"]
            example = bi.parse_example(asserts, item["entry_point"])
            found = bi.synthesize(description, asserts, item["entry_point"], weights={}, keep=40)
            engine_code = found["code"] if found else None
            confident = bool(found and found.get("confident"))
            codes = [None] * len(coders)
            pick = None
            if example is not None and example["args"] and len(example["args"]) <= data2.MAX_ARGS:
                ex_args, expected, mode = example["args"], example["expected"], example["mode"]
                literals = bi.description_literals(description)
                output = expected if isinstance(expected, str) else None
                proposals = [first_verified(net, vocabulary, grammar, description, literals, ex_args, expected, mode)
                             for _, net, vocabulary, grammar in coders]
                codes = [pyx.to_code(item["entry_point"], len(ex_args), p, literals, output) if p else None
                         for p in proposals]
                pick = vote(proposals, ex_args, literals, output)
            cache = {}

            def judge(code):
                if code not in cache:
                    cache[code] = bool(code) and run_public_asserts(code, "\n".join(item["asserts"]))["status"] == "pass"
                return cache[code]

            engine_ok = judge(engine_code)
            singles = [judge(c) for c in codes]
            vote_code = codes[pick] if pick is not None else None
            override_vote = engine_code if confident or not vote_code else vote_code
            override_first = engine_code if confident or not codes[0] else codes[0]
            entry["engine"] += engine_ok
            for (name, *_), ok in zip(coders, singles):
                entry["single"][name] += ok
            entry["vote"] += judge(vote_code)
            entry["oracle"] += any(singles)
            entry["engine_override_vote"] += judge(override_vote)
            entry["engine_override_first"] += judge(override_first)
            entry["vote_versus_engine"][0] += judge(vote_code) and not engine_ok
            entry["vote_versus_engine"][1] += engine_ok and not judge(vote_code)
            entry["override_vote_versus_engine"][0] += judge(override_vote) and not engine_ok
            entry["override_vote_versus_engine"][1] += engine_ok and not judge(override_vote)
            per_task[item["task_id"]] = {"engine": engine_ok, "vote": judge(vote_code),
                                         "override_vote": judge(override_vote)}
        local = ROOT / "outputs/seedlm/verdicts"
        local.mkdir(parents=True, exist_ok=True)
        (local / "ensemble_development.json").write_text(json.dumps(per_task) + "\n")
        report["development"] = entry
        print("development", json.dumps(entry), flush=True)
    (ROOT / args.output).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
