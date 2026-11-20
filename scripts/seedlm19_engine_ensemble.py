"""The engine and the diverse coder ensemble as two experts of equal weight.

On the development fifth the vote of eighteen coders of two architectures solves 64.6 tasks without the
engine, and the engine alone 65 (SeedLM-18): two systems of about the same accuracy that learned in
different ways, which the consensus round says is when agreement and disagreement carry information.

Rule (pre-registered): the engine side is s5-var's pyx_tie_fill (the engine's program, ties broken by
s5-var, s5-var's program where the engine has none). It counts as half of the ensemble, n/2 votes, plus
the coders that agree with it on the 24 probes; any other behaviour counts the coders that agree with it,
plus n/2 if the engine side agrees too. The engine side keeps the task unless another behaviour has
strictly more votes, so it is overturned only by more than half of the coders agreeing with each other
and not with it. Among behaviours, ties go to the shortest program. A second rule is reported but
cannot be chosen: the coders' vote wherever the engine marks its program not confident.

Uses SeedLM-15's cached proposals (outputs/seedlm/v15/). Counts only; per-task verdicts stay local.

  python scripts/seedlm19_engine_ensemble.py
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
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import coder4, data2
from seedlm15_ensemble_scaling import CACHE
from seedlm2_real import held_out
from seedlm4_tasks import post_methods

BANK = ROOT / "outputs/rule_pilot/teacher/round3/all_tasks.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/seedlm/engine_ensemble_holdout.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    cap_memory()      # a program's runaway allocation fails instead of killing the run
    coders = sorted(p.stem for p in CACHE.glob("*.json"))
    cached = {c: json.loads((CACHE / f"{c}.json").read_text())["development"] for c in coders}
    n = len(coders)
    s5_net, s5_vocabulary = coder4.load(ROOT / "outputs/seedlm/s5-var.pt")
    s5_grammar = coder4.Grammar(s5_vocabulary)
    items = [i for i in json.loads(BANK.read_text()) if i["source"] == "extra" and held_out(i["task_id"])]
    counts = {"tasks": len(items), "coders": n, "engine": 0, "base": 0, "equal_weight": 0, "unconfident_vote": 0,
              "equal_weight_changed": 0, "equal_weight_versus_base": [0, 0], "oracle_base_or_coders": 0}
    per_task = {}
    for index, item in enumerate(items):
        task = {"prompt": '"""\n' + item["description"] + "\n" + item["visible_assert"] + '\n"""\n',
                "entry_point": item["entry_point"]}
        methods = post_methods(task, s5_net, s5_vocabulary, s5_grammar, 32, 40)
        base, engine = methods["pyx_tie_fill"], methods["engine"]
        cache = {}

        def judge(code):
            if code not in cache:
                cache[code] = bool(code) and run_public_asserts(code, "\n".join(item["asserts"]))["status"] == "pass"
            return cache[code]

        base_ok, engine_ok = judge(base), judge(engine)
        rows = [cached[c][index] for c in coders]
        present = [r for r in rows if r is not None]
        equal_ok, unconfident_ok = base_ok, base_ok
        example = bi.parse_example(item["visible_assert"], item["entry_point"])
        if present and example is not None and example["args"] and len(example["args"]) <= data2.MAX_ARGS:
            probes = consensus.probe_inputs(list(example["args"]), 24)
            base_behaviour = consensus.run_engine_code(base, item["entry_point"], probes, per_call=0.1) if base \
                else None

            def agree(a, b):
                return a is not None and b is not None and consensus.compare(a, b) == "agree"

            support = [sum(i == j or agree(a["behaviour"], b["behaviour"]) for j, b in enumerate(present))
                       for i, a in enumerate(present)]
            base_votes = n / 2 + sum(agree(base_behaviour, r["behaviour"]) for r in present) if base else 0
            votes = [s + (n / 2 if agree(base_behaviour, r["behaviour"]) else 0) for s, r in zip(support, present)]
            top = max(votes)
            if top > base_votes:
                winner = min((i for i, v in enumerate(votes) if v == top), key=lambda i: (len(present[i]["program"]), i))
                equal_ok = present[winner]["correct"]
                counts["equal_weight_changed"] += 1
            found = bi.synthesize(item["description"], item["visible_assert"], item["entry_point"], weights={}, keep=40)
            if not (found and found.get("confident")):
                best = max(support)
                pick = min((i for i, s in enumerate(support) if s == best), key=lambda i: (len(present[i]["program"]), i))
                unconfident_ok = present[pick]["correct"]
        counts["engine"] += engine_ok
        counts["base"] += base_ok
        counts["equal_weight"] += equal_ok
        counts["unconfident_vote"] += unconfident_ok
        counts["equal_weight_versus_base"][0] += equal_ok and not base_ok
        counts["equal_weight_versus_base"][1] += base_ok and not equal_ok
        counts["oracle_base_or_coders"] += base_ok or any(r["correct"] for r in present)
        per_task[item["task_id"]] = {"base": base_ok, "equal_weight": equal_ok, "unconfident_vote": unconfident_ok}
    local = ROOT / "outputs/seedlm/verdicts"
    local.mkdir(parents=True, exist_ok=True)
    (local / "engine_ensemble_development.json").write_text(json.dumps(per_task) + "\n")
    (ROOT / args.output).write_text(json.dumps(counts, indent=1) + "\n")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
