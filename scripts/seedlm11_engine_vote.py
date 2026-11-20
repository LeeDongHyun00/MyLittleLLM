"""The engine and the coder ensemble together: override the engine only on a strong independent majority.

SeedLM-10's vote of seven coders beat every coder on held-out teacher tasks (70 vs 62) and every
coder alone on the development fifth (62 vs 57), but replacing the engine's unconfident programs with
it lost (63 vs 67). The engine is the strongest single source on real tasks (65 of 94). The consensus
round's asymmetry says disagreement alone does not tell who is wrong; a majority of separately trained
coders that agree with each other and not with the engine is the strongest disagreement available.

On each task of the extra pool's held-out fifth, with s5-var's methods as the base
(scripts/seedlm4_tasks.py):
  base       pyx_tie_fill: the engine's program, ties broken by s5-var; s5-var's program if the engine has none
  override   base, replaced by the coders' vote only if at least 4 of the 7 coders agree with the vote's
             program and it conflicts with the base program on the probes   (the pre-registered method)
  with       the engine's program as an eighth voter, ties to it           (reported, not chosen on)
  fill       the engine's program, else the coders' vote instead of s5-var's program (reported)
Counts only; per-task verdicts and the proposals stay local (outputs/).

  python scripts/seedlm11_engine_vote.py
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
from mylittlellm.seedlm import coder4, data2, pyx, select
from seedlm10_ensemble import first_verified
from seedlm2_real import held_out
from seedlm4_tasks import post_methods

BANK = ROOT / "outputs/rule_pilot/teacher/round3/all_tasks.json"
CODERS = ["s5-var", "s8-all", "s6-graft", "s6-comp", "s9-early", "s9-early-last", "s4-post"]
MAJORITY = 4


def code_behaviour(code, entry_point, probes):
    return consensus.run_engine_code(code, entry_point, probes, per_call=0.1) if code else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/seedlm/engine_vote_holdout.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    cap_memory()      # a program's runaway allocation fails instead of killing the run
    coders = []
    for name in CODERS:
        net, vocabulary = coder4.load(ROOT / "outputs/seedlm" / f"{name}.pt")
        coders.append((name, net, vocabulary, coder4.Grammar(vocabulary)))
    items = [i for i in json.loads(BANK.read_text()) if i["source"] == "extra" and held_out(i["task_id"])]
    counts = {"tasks": len(items), "engine": 0, "base": 0, "override": 0, "with": 0, "fill": 0,
              "override_changed": 0, "override_versus_base": [0, 0]}
    per_task = {}
    for item in items:
        prompt = '"""\n' + item["description"] + "\n" + item["visible_assert"] + '\n"""\n'
        task = {"prompt": prompt, "entry_point": item["entry_point"]}
        _, s5_net, s5_vocabulary, s5_grammar = coders[0]
        methods = post_methods(task, s5_net, s5_vocabulary, s5_grammar, 32, 40)
        base, engine = methods["pyx_tie_fill"], methods["engine"]
        override = with_engine = fill = base
        example = bi.parse_example(item["visible_assert"], item["entry_point"])
        if example is not None and example["args"] and len(example["args"]) <= data2.MAX_ARGS:
            ex_args, expected, mode = example["args"], example["expected"], example["mode"]
            literals = bi.description_literals(item["description"])
            output = expected if isinstance(expected, str) else None
            proposals = [first_verified(net, vocabulary, grammar, item["description"], literals, ex_args, expected,
                                        mode) for _, net, vocabulary, grammar in coders]
            codes = [pyx.to_code(item["entry_point"], len(ex_args), p, literals, output) if p else None
                     for p in proposals]
            probes = consensus.probe_inputs(list(ex_args), 24)
            present = [i for i, p in enumerate(proposals) if p is not None]
            if present and probes:
                results = {i: select.behaviour(proposals[i], probes, literals, output) for i in present}
                support = {i: sum(i == j or consensus.compare(results[i], results[j]) == "agree" for j in present)
                           for i in present}
                pick = max(present, key=lambda i: (support[i], -i))
                base_results = code_behaviour(base, item["entry_point"], probes)
                conflicts = base_results is None or consensus.compare(results[pick], base_results) == "conflict"
                if support[pick] >= MAJORITY and conflicts:
                    override = codes[pick]
                # The engine as an eighth voter: its program's support is the coders that agree with it plus
                # itself; it keeps the task unless some coder behaviour has strictly more support.
                engine_results = code_behaviour(engine, item["entry_point"], probes)
                engine_support = 1 + (sum(consensus.compare(engine_results, results[j]) == "agree" for j in present)
                                      if engine_results else 0)
                rival = max(present, key=lambda i: (support[i] + (engine_results is not None and
                                                                    consensus.compare(results[i], engine_results)
                                                                    == "agree"), -i))
                rival_support = support[rival] + (engine_results is not None and
                                                  consensus.compare(results[rival], engine_results) == "agree")
                with_engine = engine if engine and engine_support >= rival_support else codes[rival]
                fill = engine or codes[pick]
        cache = {}

        def judge(code):
            if code not in cache:
                cache[code] = bool(code) and run_public_asserts(code, "\n".join(item["asserts"]))["status"] == "pass"
            return cache[code]

        verdict = {name: judge(code) for name, code in (("engine", engine), ("base", base), ("override", override),
                                                         ("with", with_engine), ("fill", fill))}
        for name, ok in verdict.items():
            counts[name] += ok
        counts["override_changed"] += override != base
        counts["override_versus_base"][0] += verdict["override"] and not verdict["base"]
        counts["override_versus_base"][1] += verdict["base"] and not verdict["override"]
        per_task[item["task_id"]] = verdict
    local = ROOT / "outputs/seedlm/verdicts"
    local.mkdir(parents=True, exist_ok=True)
    (local / "engine_vote_development.json").write_text(json.dumps(per_task) + "\n")
    report = {"coders": CODERS, "majority": MAJORITY, "development": counts}
    (ROOT / args.output).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
