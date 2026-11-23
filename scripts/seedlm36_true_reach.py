"""SeedLM-36: how many "right" candidates are right only by coincidence? Counts only.

A candidate counts as right when it reproduces every case its task states: three inputs for a teacher task,
the task's asserts for a development task. Among the seventy-odd candidates the sketch search adds per task
(SeedLM-31), some pass those few cases by accident: a held-out teacher task about counting the set bits of
an integer is "solved" by (num_divisor_count(n) + num_unset_bit_count(n)) % 10. Such programs are wrong on
the hidden tests that count in the end (MBPP+), and a selector that passes them over is right to do so.

Here a candidate is truly right when, besides passing the stated cases, it behaves like a gold program on
the 24 consensus probes (rulecoder.consensus.compare gives "agree": at least three probes where both return a
value, and no difference): the teacher's own expression for a teacher task, the task's reference solution for
a development task (run in the sandbox). With --judge no_conflict, only a difference counts against it (probes
where either side raises say nothing), which is closer to failing hidden tests. Reported for the
reach (some candidate right), each source, the engine, and the selection rules of SeedLM-32 (six-coder mean)
and SeedLM-34/35 (fields scores_k, scores_o; reviewer r35-a-m10 with lambda 1).

  python scripts/seedlm36_true_reach.py --output reports/seedlm/true_reach.json
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylittlellm.rulecoder import consensus
from mylittlellm.rulecoder.limits import cap_memory
from mylittlellm.rulecoder.sandbox import run_calls
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import select
from seedlm32_candidates import BANK, OUT, development_tasks, held_out, teacher_tasks
from seedlm32_select import ensemble
from seedlm35_review_select import value as review_value

RULES = {
    "six_coder_mean": lambda c: ensemble(c),
    "six_coder_mean_k": lambda c: ensemble(c, "scores_k"),
    "six_coder_mean_o": lambda c: ensemble(c, "scores_o"),
    "coder_reviewer": lambda c: review_value(c, "r35-a-m10", "coder_reviewer", 1.0),
}


def gold_behaviours(set_name, tasks):
    out = []
    if set_name == "teacher":
        for case, *_ in tasks:
            output = case["expected"] if isinstance(case["expected"], str) else None
            probes = consensus.probe_inputs(list(case["args"]), 24)
            out.append(select.behaviour(case["gold"], probes, case["literals"], output))
        return out
    items = [i for i in json.loads(BANK.read_text()) if i["source"] == "extra" and held_out(i["task_id"])]
    for item, (case, *_) in zip(items, tasks):
        if case is None:
            out.append(None)
            continue
        probes = consensus.probe_inputs(list(case["args"]), 24)
        result = run_calls(item["reference_code"], item["entry_point"], probes)
        out.append([tuple(r) for r in result["results"]] if result.get("status") == "ok" else None)
    return out


def report_for(set_name, judge="agree"):
    tasks = teacher_tasks() if set_name == "teacher" else development_tasks()
    rows = json.loads((OUT / f"candidates_{set_name}.json").read_text())
    golds = gold_behaviours(set_name, tasks)
    counts = {"tasks": 0, "gold_unavailable": 0, "some_right": 0, "some_truly_right": 0,
              "some_right_or_engine": 0, "some_truly_right_or_engine": 0}
    for source in ("plain", "sketch", "engine"):
        counts[f"{source}_right"] = counts[f"{source}_truly_right"] = 0
    counts["right_only_by_sketch"] = counts["right_only_by_sketch_truly"] = 0
    for rule in RULES:
        counts[f"{rule}_alone_right"] = counts[f"{rule}_alone_truly_right"] = 0
        counts[f"{rule}_system_right"] = counts[f"{rule}_system_truly_right"] = 0
    for row, gold in zip(rows, golds):
        if row is None:
            continue
        counts["tasks"] += 1
        if gold is None:
            counts["gold_unavailable"] += 1
            continue

        def truly(c):
            if not (c and c.get("right") and c.get("behaviour")):
                return False
            verdict = consensus.compare(c["behaviour"], gold)
            return verdict == "agree" if judge == "agree" else verdict != "conflict"

        cands, engine = row["candidates"], row["engine"]
        counts["some_right"] += any(c["right"] for c in cands)
        counts["some_truly_right"] += any(truly(c) for c in cands)
        counts["some_right_or_engine"] += any(c["right"] for c in cands) or bool(engine and engine.get("right"))
        counts["some_truly_right_or_engine"] += any(truly(c) for c in cands) or truly(engine)
        for source in ("plain", "sketch"):
            counts[f"{source}_right"] += any(c["right"] for c in cands if c["source"] == source)
            counts[f"{source}_truly_right"] += any(truly(c) for c in cands if c["source"] == source)
        counts["engine_right"] += bool(engine and engine.get("right"))
        counts["engine_truly_right"] += truly(engine)
        if any(c["right"] for c in cands) and not any(c["right"] for c in cands if c["source"] == "plain"):
            counts["right_only_by_sketch"] += 1
            counts["right_only_by_sketch_truly"] += any(truly(c) for c in cands)
        for rule, key in RULES.items():
            pick = max(cands, key=key) if cands else None
            counts[f"{rule}_alone_right"] += bool(pick and pick["right"])
            counts[f"{rule}_alone_truly_right"] += truly(pick)
            if engine and engine["confident"]:
                choice = engine
            else:
                pool = cands + ([engine] if engine and engine.get("program") and "behaviour" in engine else [])
                choice = max(pool, key=key) if pool else engine
            counts[f"{rule}_system_right"] += bool(choice and choice.get("right"))
            counts[f"{rule}_system_truly_right"] += truly(choice)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", choices=["agree", "no_conflict"], default="agree")
    parser.add_argument("--output")
    args = parser.parse_args()
    cap_memory()
    report = {"judge": args.judge, "teacher": report_for("teacher", args.judge),
              "development": report_for("development", args.judge)}
    text = json.dumps(report, indent=1)
    print(text)
    if args.output:
        (ROOT / args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
