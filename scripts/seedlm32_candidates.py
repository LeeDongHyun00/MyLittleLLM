"""SeedLM-32, step 1: every candidate program for each held-out task, kept locally for choosing among them.

Per task (the held-out teacher tasks, or the development fifth): the rule engine's program and whether it
is confident; the prefix coder's own beam (width 32) and the sketch-held search (SeedLM-31: 64 ranked
sketches, 6 hypotheses each), every program that reproduces the visible example, with the coder's
log-probability, the sketch it was held to and that sketch's ranker score; each candidate's behaviour on
the 24 consensus probes; and whether it is right (every case, or the task's tests). Everything per task
stays in outputs/seedlm/v32/ (gitignored); only counts leave this script.

  python scripts/seedlm32_candidates.py --set teacher        # then --set development
"""
import argparse
import ast
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

from mylittlellm.rulecoder import backinfer as bi, consensus
from mylittlellm.rulecoder.limits import cap_memory
from mylittlellm.rulecoder.sandbox import run_calls, run_public_asserts
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import coder3, language, select, sketch as sk
from seedlm2_real import held_out
from seedlm3_train import named_in, teacher_held, teacher_task_items
from seedlm31_sketch_search import training_examples

OUT = ROOT / "outputs/seedlm/v32"
BANK = ROOT / "outputs/rule_pilot/teacher/round3/all_tasks.json"


def teacher_tasks():
    """Held-out teacher tasks in teacher_cases() order, with what the engine needs."""
    from seedlm3_discovery import teacher_cases
    cases = teacher_cases()
    tasks = []
    for task in teacher_task_items():
        if not teacher_held(task["id"]):
            continue
        try:
            pairs = [(list(ast.literal_eval(i)), ast.literal_eval(o)) for i, o in zip(task["inputs"], task["outputs"])]
        except (ValueError, SyntaxError):
            continue
        if not pairs:
            continue
        tasks.append((task, pairs))
    assert len(tasks) == len(cases)
    out = []
    for (task, pairs), case in zip(tasks, cases):
        args, expected = pairs[0]
        visible = f"assert {task['entry_point']}({', '.join(repr(a) for a in args)}) == {expected!r}"

        def engine_right(code, task=task, pairs=pairs):
            result = run_calls(code, task["entry_point"], [p[0] for p in pairs])
            if result.get("status") != "ok":
                return False
            return all(status == "ok" and value == consensus.canonical(expected)
                       for (status, value), (_, expected) in zip(result["results"], pairs))
        out.append((case, task["description"], visible, task["entry_point"], engine_right))
    return out


def development_tasks():
    from seedlm15_ensemble_scaling import development_cases
    items = [i for i in json.loads(BANK.read_text()) if i["source"] == "extra" and held_out(i["task_id"])]
    out = []
    for item, case in zip(items, development_cases()):
        def engine_right(code, item=item):
            return run_public_asserts(code, "\n".join(item["asserts"]))["status"] == "pass"
        out.append((case, item["description"], item["visible_assert"], item["entry_point"], engine_right))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=["teacher", "development"], required=True)
    parser.add_argument("--model", default="s20-p5")
    parser.add_argument("--sketches", type=int, default=64)
    parser.add_argument("--per-sketch", type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cap_memory()
    net, vocabulary = coder3.load(ROOT / "outputs/seedlm" / f"{args.model}.pt")
    grammar = coder3.Grammar(vocabulary)
    library = sk.Library(training_examples())
    tasks = teacher_tasks() if args.set == "teacher" else development_tasks()
    rows, started = [], time.time()
    counts = {"tasks": 0, "engine_right": 0, "engine_confident": 0, "engine_none": 0, "any_right": 0,
              "candidates_per_task": 0.0}
    for case, description, visible, entry_point, engine_right in tasks:
        if case is None:
            rows.append(None)
            continue
        counts["tasks"] += 1
        found = bi.synthesize(description, visible, entry_point, weights={})
        engine = None
        if found:
            engine = {"program": found.get("program") or None, "confident": bool(found.get("confident")),
                      "right": bool(engine_right(found["code"]))}
        words = language.words(description)
        a, literals, expected, mode = case["args"], case["literals"], case["expected"], case["mode"]
        prefix = vocabulary.condition(words, literals, a, expected, named_in(description))
        string_output = isinstance(expected, str)
        probes = consensus.probe_inputs(list(a), 24)
        output = expected if string_output else None
        candidates, seen = [], set()

        def add(program, score, source, sketch=None, sketch_score=None):
            key = tuple(program)
            if key in seen or not coder3.reproduces(program, a, literals, expected, mode):
                return
            seen.add(key)
            candidates.append({"program": list(program), "source": source, "score": round(float(score), 4),
                               "sketch": list(sketch) if sketch else None, "sketch_score": sketch_score,
                               "right": bool(case["judge"](program)),
                               "behaviour": select.behaviour(program, probes, literals, output)})

        for program, score in coder3.wide_beam_search(net, vocabulary, grammar, prefix, len(a), len(literals),
                                                      string_output, 32):
            add(program, score, "plain")
        ranked = library.rank(words, len(a), expected)[:args.sketches]
        for program, score, g in sk.sketch_search(net, vocabulary, grammar, prefix, len(a), len(literals),
                                                  string_output, ranked, per_sketch=args.per_sketch):
            add(program, score, "sketch", ranked[g], round(library.score(ranked[g], words, len(a), expected), 4))
        if engine and engine["program"]:
            try:
                if coder3.reproduces(engine["program"], a, literals, expected, mode):
                    engine["behaviour"] = select.behaviour(engine["program"], probes, literals, output)
            except Exception:
                pass
        rows.append({"engine": engine, "candidates": candidates, "n_args": len(a), "string_output": string_output})
        counts["engine_right"] += bool(engine and engine["right"])
        counts["engine_confident"] += bool(engine and engine["confident"])
        counts["engine_none"] += engine is None
        counts["any_right"] += bool(engine and engine["right"]) or any(c["right"] for c in candidates)
        counts["candidates_per_task"] += len(candidates)
        if counts["tasks"] % 20 == 0:
            print(counts["tasks"], round(time.time() - started), json.dumps(counts), flush=True)
    counts["candidates_per_task"] = round(counts["candidates_per_task"] / max(1, counts["tasks"]), 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"candidates_{args.set}.json").write_text(json.dumps(rows) + "\n")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
