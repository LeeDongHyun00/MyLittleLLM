"""The coder ensemble where a single coder already helps the engine: breaking its ties, filling its gaps.

On the development fifth the best methods use one coder in two places: to choose among the engine's
equally scored programs (pyx_tie) and to answer where the engine has no program (pyx_fill). Overriding
the engine's unconfident programs with the ensemble lost (SeedLM-10/11), but averaging separately
trained coders is the textbook way to reduce one model's variance, and in SeedLM-12 single coders of
the same recipe differed by up to 8 tasks.

  ens_tie_fill   among the engine's tied best programs, the one with the highest mean log-probability
                 over the ensemble's coders; where the engine has no program, the coders' vote
                 (SeedLM-10); the pre-registered method
  ens_override   ens_tie_fill, but where the engine marks its program not confident, the coders' vote
                 (reported, cannot be chosen)
  base           s5-var's pyx_tie_fill and pyx_override, for comparison

Counts only; per-task verdicts stay local.

  python scripts/seedlm13_ensemble_tie.py --coders s12-a1 s12-a2 s12-a3 s12-b1 s12-b2 s12-b3
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

from mylittlellm.rulecoder import backinfer as bi
from mylittlellm.rulecoder.limits import cap_memory
from mylittlellm.rulecoder.sandbox import run_public_asserts
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import coder4, data2, language, post, pyx
from seedlm10_ensemble import first_verified, vote
from seedlm2_real import held_out
from seedlm3_train import named_in
from seedlm4_tasks import post_methods

BANK = ROOT / "outputs/rule_pilot/teacher/round3/all_tasks.json"


def primary(match):
    return round(2.0 * match["support"] + match["learned"] - 0.3 * match["size"], 6)


def mean_log_prob(coders, description, literals, args, expected, mode, program):
    """The ensemble's mean log-probability of a prefix pyx program, or None if a coder cannot write it."""
    if not program or "?" in program:
        return None
    scores = []
    for _, net, vocabulary, _ in coders:
        prefix = vocabulary.condition(language.words(description), literals, args, expected, named_in(description))
        try:
            scores.append(coder4.program_log_prob(net, vocabulary, prefix, post.to_postfix(list(program)), args,
                                                  literals, expected, mode))
        except (KeyError, ValueError):
            return None
    return statistics.mean(scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coders", nargs="+", required=True)
    parser.add_argument("--output", default="reports/seedlm/ensemble_tie_holdout.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    cap_memory()      # a program's runaway allocation fails instead of killing the run
    coders = []
    for name in args.coders:
        net, vocabulary = coder4.load(ROOT / "outputs/seedlm" / f"{name}.pt")
        coders.append((name, net, vocabulary, coder4.Grammar(vocabulary)))
    s5_net, s5_vocabulary = coder4.load(ROOT / "outputs/seedlm/s5-var.pt")
    s5_grammar = coder4.Grammar(s5_vocabulary)
    items = [i for i in json.loads(BANK.read_text()) if i["source"] == "extra" and held_out(i["task_id"])]
    names = ["engine", "pyx_tie_fill", "pyx_override", "ens_tie_fill", "ens_override"]
    solved = {n: 0 for n in names}
    versus = {n: [0, 0] for n in names}
    per_task = {}
    for item in items:
        description, asserts, entry_point = item["description"], item["visible_assert"], item["entry_point"]
        task = {"prompt": '"""\n' + description + "\n" + asserts + '\n"""\n', "entry_point": entry_point}
        base = post_methods(task, s5_net, s5_vocabulary, s5_grammar, 32, 40)
        codes = {"engine": base["engine"], "pyx_tie_fill": base["pyx_tie_fill"], "pyx_override": base["pyx_override"],
                 "ens_tie_fill": base["engine"], "ens_override": base["engine"]}
        example = bi.parse_example(asserts, entry_point)
        if example is not None and example["args"] and len(example["args"]) <= data2.MAX_ARGS:
            ex_args, expected, mode = example["args"], example["expected"], example["mode"]
            literals = bi.description_literals(description)
            output = expected if isinstance(expected, str) else None
            found = bi.synthesize(description, asserts, entry_point, weights={}, keep=40)
            engine = (found.get("alternatives") or [found]) if found else []
            confident = bool(found and found.get("confident"))
            top = [m for m in engine if primary(m) == primary(engine[0])] if engine else []
            scored = [(s, m) for m in top for s in [mean_log_prob(coders, description, literals, ex_args, expected,
                                                                    mode, m.get("program") or [])] if s is not None]
            tie = max(scored, key=lambda s: s[0])[1]["code"] if scored else (engine[0]["code"] if engine else None)
            proposals = [first_verified(net, vocabulary, grammar, description, literals, ex_args, expected, mode)
                         for _, net, vocabulary, grammar in coders]
            pick = vote(proposals, ex_args, literals, output)
            voted = pyx.to_code(entry_point, len(ex_args), proposals[pick], literals, output) if pick is not None \
                else None
            codes["ens_tie_fill"] = tie or voted
            codes["ens_override"] = codes["ens_tie_fill"] if confident or not voted else voted
        cache = {}

        def judge(code):
            if code not in cache:
                cache[code] = bool(code) and run_public_asserts(code, "\n".join(item["asserts"]))["status"] == "pass"
            return cache[code]

        verdict = {n: judge(codes[n]) for n in names}
        for n in names:
            solved[n] += verdict[n]
            versus[n][0] += verdict[n] and not verdict["engine"]
            versus[n][1] += verdict["engine"] and not verdict[n]
        per_task[item["task_id"]] = verdict
    local = ROOT / "outputs/seedlm/verdicts"
    local.mkdir(parents=True, exist_ok=True)
    (local / "ensemble_tie_development.json").write_text(json.dumps(per_task) + "\n")
    report = {"coders": args.coders, "tasks": len(items), "solved": solved, "versus_engine": versus}
    (ROOT / args.output).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
