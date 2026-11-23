"""SeedLM-34 diagnosis: why teacher-written sketch tasks did not change selection. Counts and means only.

Exploration after the registered measure (reports/seedlm/seedlm34_protocol.json), on the SeedLM-32 candidates:
  ranks    for tasks with a right candidate, the rank of the best right one under the six-coder mean with the
           sketch-task coders (k) and with the controls (o)
  shift    how much the three fine-tuned coders' log-probability moved from o to k, for right and wrong
           candidates, by source (plain beam or sketch search) and by whether the candidate's sketch was taught
  fit      (--fit N) whether the fine-tuned coders write N of the sketch tasks they were trained on (first
           verified program at width 32, right on all three inputs): did 400 steps teach them at all?

  python scripts/seedlm34_diagnose.py --output reports/seedlm/seedlm34_diagnosis.json [--fit 120]
"""
import argparse
import ast
import json
from pathlib import Path
import random
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import sketch as sk
from seedlm3_train import teacher_task_items
from seedlm32_candidates import OUT
from seedlm32_select import ensemble

PAIRS = [("s34-k1", "s34-o1"), ("s34-k2", "s34-o2"), ("s34-k3", "s34-o3")]


def taught_sketches():
    by_id = {}
    for path in sorted((ROOT / "outputs/sketch_tasks").glob("in_part*.json")):
        by_id.update({a["sketch_id"]: tuple(a["sketch"]) for a in json.loads(path.read_text())})
    return {by_id[t["sketch_id"]] for t in teacher_task_items("sketch_tasks") if t["sketch_id"] in by_id}


def sketch_key(program):
    try:
        return sk.sketch_of(list(program))[0]
    except (KeyError, ValueError, IndexError):
        return None


def shift(item):
    """Mean over the three pairs of the k coder's log-probability minus its control's (None if any is missing)."""
    k, o = item.get("scores_k") or {}, item.get("scores_o") or {}
    values = [k.get(a) - o.get(b) for a, b in PAIRS if k.get(a) is not None and o.get(b) is not None]
    return statistics.mean(values) if len(values) == len(PAIRS) else None


def rank_of_right(candidates, field):
    order = sorted(candidates, key=lambda c: ensemble(c, field), reverse=True)
    return next(i + 1 for i, c in enumerate(order) if c["right"])


def candidates_report(set_name, taught):
    rows = [r for r in json.loads((OUT / f"candidates_{set_name}.json").read_text()) if r is not None]
    report = {"tasks": len(rows)}
    ranked = [r for r in rows if any(c["right"] for c in r["candidates"])]
    ranks = [(rank_of_right(r["candidates"], "scores_k"), rank_of_right(r["candidates"], "scores_o")) for r in ranked]
    report["tasks_with_a_right_candidate"] = len(ranked)
    report["right_ranked_first"] = {"k": sum(a == 1 for a, _ in ranks), "o": sum(b == 1 for _, b in ranks)}
    report["median_rank_of_best_right"] = {"k": statistics.median(a for a, _ in ranks),
                                           "o": statistics.median(b for _, b in ranks)}
    report["k_ranks_right_higher_lower_same"] = [sum(a < b for a, b in ranks), sum(a > b for a, b in ranks),
                                                 sum(a == b for a, b in ranks)]
    sketch_only = [r for r in ranked if not any(c["right"] and c["source"] == "plain" for c in r["candidates"])]
    report["tasks_right_only_by_sketch_search"] = len(sketch_only)
    report["of_those_right_sketch_taught"] = sum(
        any(c["right"] and sketch_key(c["program"]) in taught for c in r["candidates"]) for r in sketch_only)
    groups = {}
    for r in rows:
        for c in r["candidates"]:
            value = shift(c)
            if value is None:
                continue
            name = f"{c['source']}/{'right' if c['right'] else 'wrong'}/" \
                   f"{'taught' if sketch_key(c['program']) in taught else 'untaught'}"
            groups.setdefault(name, []).append(value)
    report["log_prob_shift_k_minus_o"] = {k: {"candidates": len(v), "mean": round(statistics.mean(v), 3)}
                                         for k, v in sorted(groups.items())}
    return report


def fit_report(count, seed=34):
    import torch
    from mylittlellm.rulecoder import backinfer as bi
    from mylittlellm.rulecoder.limits import cap_memory
    from mylittlellm.seedlm import coder4
    from seedlm10_ensemble import first_verified
    from seedlm3_discovery import value_of
    torch.set_num_threads(4)
    cap_memory()
    tasks = teacher_task_items("sketch_tasks")
    random.Random(seed).shuffle(tasks)
    tasks = tasks[:count]
    out = {"tasks": len(tasks)}
    for name in [n for pair in PAIRS for n in pair]:
        net, vocabulary = coder4.load(ROOT / "outputs/seedlm" / f"{name}.pt")
        grammar = coder4.Grammar(vocabulary)
        right = 0
        for task in tasks:
            literals = bi.description_literals(task["description"])
            cases = [(list(ast.literal_eval(i)), ast.literal_eval(o)) for i, o in zip(task["inputs"], task["outputs"])]
            output = cases[0][1] if isinstance(cases[0][1], str) else None
            try:
                program = first_verified(net, vocabulary, grammar, task["description"], literals, cases[0][0],
                                         cases[0][1], "eq")
            except KeyError:
                program = None
            if program is None:
                continue
            ok = True
            for args, expected in cases:
                good, value = value_of(program, args, literals, output)
                ok = ok and good and bi._matches(value, {"expected": expected, "mode": "eq"})
            right += ok
        out[name] = right
        print(name, right, flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=int, default=0)
    parser.add_argument("--output")
    args = parser.parse_args()
    taught = taught_sketches()
    report = {"taught_sketches": len(taught),
              "teacher": candidates_report("teacher", taught),
              "development": candidates_report("development", taught)}
    if args.fit:
        report["fit_on_trained_sketch_tasks"] = fit_report(args.fit)
    text = json.dumps(report, indent=1)
    print(text)
    if args.output:
        (ROOT / args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
