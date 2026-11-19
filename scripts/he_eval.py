"""The whole system on HumanEval+, a benchmark nothing in this repository was tuned on.

  targets  : which tasks the consensus step would sample (corroboration gate), written to a file
  evaluate : solved counts for
               plain     - the student's greedy answer
               engine    - the rule engine alone (shipped library, no model)
               pipeline  - the student's answer, replaced by the engine's program only when the answer
                           fails the visible example (branch-and-select without prompt rules)
               seedlm    - the pipeline with SeedLM-2 tie-breaking (the gated method) in place of the engine,
                           if --seedlm; also SeedLM-2 alone and the engine with its tie-breaking
               consensus - the pipeline plus corroboration-gated consensus, if --samples (lambda 0.5 and 0)
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mylittlellm.rulecoder import backinfer as bi, humaneval as he, scoring
from mylittlellm.rulecoder.pipeline import public_check
from mylittlellm.rulecoder.stats import mcnemar_exact, wilson
from mylittlellm.rulecoder.tasks import ROOT

OUT = ROOT / "outputs/rule_pilot"


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def as_mbpp(task):
    """The MBPP-shaped view the rule-engine code expects: description plus the visible assert."""
    return {"task_id": task["task_id"], "entry_point": task["entry_point"],
            "prompt": '"""\n' + task["description"] + "\n" + task["visible_assert"] + '\n"""\n'}


def pipeline_rows(tasks, base):
    """Branch-and-select without prompt rules: the plain answer, else the engine's program."""
    rows = {}
    for task_id, row in base.items():
        task = tasks[task_id]
        out = dict(row)
        if task["visible_assert"] and not row["visible_ok"]:
            found = bi.synthesize(task["description"], task["visible_assert"], task["entry_point"], weights={})
            if found and public_check(found["code"], task["visible_assert"])["ok"]:
                out.update(final_code=found["code"], chosen="backinfer", visible_ok=True)
        rows[task_id] = out
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["targets", "evaluate"])
    parser.add_argument("--base", default="he15-base")
    parser.add_argument("--seedlm")
    parser.add_argument("--seedlm-method", default="tie",
                        help="the SeedLM-2 method that passed its own validation gate (reports/seedlm/seedlm2_protocol.json)")
    parser.add_argument("--reviewer", help="SeedLM-3 reviewer checkpoint, for the reviewer methods")
    parser.add_argument("--pyx", help="SeedLM-3 pyx coder checkpoint (reports/seedlm/seedlm3_pyx_protocol.json)")
    parser.add_argument("--pyx-method", default="pyx_tie_fill")
    parser.add_argument("--samples")
    parser.add_argument("--output", default="he_summary.json")
    args = parser.parse_args()
    tasks = he.read_tasks()
    base = {r["task_id"]: r for r in read_jsonl(OUT / "gen" / f"{args.base}.jsonl")}
    pipeline = pipeline_rows(tasks, base)

    if args.command == "targets":
        sys.path.insert(0, str(ROOT / "scripts"))
        from consensus_sample import target_reason
        targets = {}
        for task_id, row in pipeline.items():
            task = tasks[task_id]
            if not task["visible_assert"]:
                continue
            reason = target_reason(row, as_mbpp(task), 10, 24, 3)
            if reason:
                targets[task_id] = reason
        path = OUT / "samples" / f"{args.base}_targets.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sorted(targets), indent=1) + "\n")
        counts = {}
        for reason in targets.values():
            counts[reason] = counts.get(reason, 0) + 1
        print(len(targets), "targets", counts)
        return

    problems = scoring.load_problems("humaneval")
    oracle = scoring.groundtruth(problems, "humaneval")
    cache = {}

    def solved(task_id, code):
        if (task_id, code) not in cache:
            cache[(task_id, code)] = bool(code) and scoring.score(problems[task_id], code, oracle[task_id],
                                                                   "humaneval")["solved"]
        return cache[(task_id, code)]

    finals = {"plain": {t: r["final_code"] for t, r in base.items()},
              "pipeline": {t: r["final_code"] for t, r in pipeline.items()}}
    engine = {}
    for task_id, task in tasks.items():
        found = bi.synthesize(task["description"], task["visible_assert"], task["entry_point"], weights={}) \
            if task["visible_assert"] else None
        engine[task_id] = found["code"] if found else ""
    finals["engine"] = engine
    if args.seedlm:
        import torch
        from mylittlellm.seedlm import propose2
        from seedlm2_tasks import code_for, load_reviewer, methods_for
        torch.set_num_threads(4)
        net, vocabulary = propose2.load(ROOT / "outputs/seedlm" / f"{args.seedlm}.pt")
        arities = propose2.arity_table(vocabulary)
        reviewer = load_reviewer(args.reviewer)
        picks = {task_id: methods_for(as_mbpp(task), net, vocabulary, arities, 32, 40, reviewer)
                 if task["visible_assert"] else {} for task_id, task in tasks.items()}
        chosen = {}
        for task_id, row in pipeline.items():
            task = tasks[task_id]
            chosen[task_id] = row["final_code"]
            if task["visible_assert"] and not (row["visible_ok"] and row["chosen"] != "backinfer"):
                picked = picks[task_id].get(args.seedlm_method)
                if picked and picked[0]:
                    code = code_for(as_mbpp(task), *picked)
                    if public_check(code, task["visible_assert"])["ok"]:
                        chosen[task_id] = code
                    elif row["chosen"] == "backinfer":
                        chosen[task_id] = base[task_id]["final_code"]
        finals["seedlm"] = chosen
        # The coder on its own and the engine with the chosen method, without the student (diagnostics).
        alone, engine_with = {}, {}
        for task_id, task in tasks.items():
            alone[task_id] = engine_with[task_id] = ""
            for target, name in ((alone, "model"), (engine_with, args.seedlm_method)):
                program = picks[task_id].get(name)
                if program and program[0]:
                    target[task_id] = code_for(as_mbpp(task), *program)
        finals["seedlm_alone"] = alone
        finals["engine_with_seedlm"] = engine_with
    if args.pyx:
        import torch
        from mylittlellm.seedlm import coder3
        from seedlm3_tasks import pyx_methods
        torch.set_num_threads(4)
        net, vocabulary = coder3.load(ROOT / "outputs/seedlm" / f"{args.pyx}.pt")
        grammar = coder3.Grammar(vocabulary)
        alone, chosen = {}, {}
        for task_id, task in tasks.items():
            codes = pyx_methods(as_mbpp(task), net, vocabulary, grammar) if task["visible_assert"] else {}
            alone[task_id] = codes.get(args.pyx_method) or ""
            row = pipeline[task_id]
            chosen[task_id] = row["final_code"]
            # As in the MBPP+ pipelines: the method replaces the engine where the student's answer fails
            # the visible example; if it has nothing that reproduces it, the student's first answer stays.
            if task["visible_assert"] and not (row["visible_ok"] and row["chosen"] != "backinfer"):
                code = alone[task_id]
                if code and public_check(code, task["visible_assert"])["ok"]:
                    chosen[task_id] = code
                elif row["chosen"] == "backinfer":
                    chosen[task_id] = base[task_id]["final_code"]
        finals["pyx_alone"] = alone
        finals["pyx_pipeline"] = chosen
    if args.samples:
        sys.path.insert(0, str(ROOT / "scripts"))
        from consensus_select import choose
        samples = {}
        for sample in read_jsonl(OUT / "samples" / f"{args.samples}.jsonl"):
            samples.setdefault(sample["task_id"], []).append(sample)
        targets = set(json.loads((OUT / "samples" / f"{args.base}_targets.json").read_text()))
        for lam in (0.5, 0.0):
            chosen = {}
            for task_id, row in pipeline.items():
                chosen[task_id] = row["final_code"]
                if task_id in targets and samples.get(task_id):
                    view = as_mbpp(tasks[task_id])
                    code, _, _ = choose(view, row, base[task_id], sorted(samples[task_id], key=lambda s: s["index"]),
                                        lam, 10, 24, 3)
                    chosen[task_id] = code
            finals[f"consensus_lambda{lam:g}"] = chosen
    report = {"tasks": len(tasks), "with_visible_example": sum(bool(t["visible_assert"]) for t in tasks.values()),
              "methods": {}}
    plain_solved = {t: solved(t, c) for t, c in finals["plain"].items()}
    for name, codes in finals.items():
        verdict = {t: solved(t, c) for t, c in codes.items()}
        wins = sum(verdict[t] and not plain_solved[t] for t in verdict)
        losses = sum(plain_solved[t] and not verdict[t] for t in verdict)
        count = sum(verdict.values())
        report["methods"][name] = {"solved": count, "pct": round(100 * count / len(tasks), 1),
                                   "ci95": wilson(count, len(tasks)), "vs_plain": [wins, losses],
                                   "mcnemar_p": round(mcnemar_exact(wins, losses), 4)}
        print(name, json.dumps(report["methods"][name]), flush=True)
    (ROOT / "reports/rule_pilot" / args.output).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
