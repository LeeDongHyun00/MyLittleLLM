"""How much is corroboration worth? Compare the pipeline's answer with rule-engine programs on probes.

For every task whose final pipeline answer is a model answer that passes the visible example,
the answer and the engine's programs (which reproduce the same example) are run on probe inputs.
The answer is "corroborated" if some engine program gives the same value on every probe where
both return one. Accuracy is then counted separately for corroborated and uncorroborated answers.
The locked test split gets counts only.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mylittlellm.rulecoder import backinfer as bi, consensus as cs
from mylittlellm.rulecoder.tasks import ROOT, parse_prompt, read_tasks

OUT = ROOT / "outputs/rule_pilot"


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def engine_programs(task, asserts, keep):
    description, _ = parse_prompt(task["prompt"])
    found = bi.synthesize(description, asserts, task["entry_point"], weights={}, keep=keep)
    if not found:
        return []
    return found.get("alternatives") or [{"code": found["code"], "expr": found["expr"]}]


def study(run, split, tasks, keep, probes_per_task, minimum):
    rows = {r["task_id"]: r for r in read_jsonl(OUT / "gen" / f"{run}.jsonl") if r["split"] == split}
    verdicts = {r["task_id"]: r for r in read_jsonl(OUT / "scored" / f"{run}.jsonl") if r["split"] == split}
    records = []
    for task_id, row in rows.items():
        if not row["visible_ok"] or row["chosen"] == "backinfer":
            continue
        task = tasks[task_id]
        _, asserts = parse_prompt(task["prompt"])
        example = bi.parse_example(asserts, task["entry_point"])
        if example is None or not example["args"]:
            continue
        probes = cs.probe_inputs(example["args"], probes_per_task)
        model = cs.run_model_code(row["final_code"], task["entry_point"], probes)
        relations = [cs.compare(model, cs.run_engine_code(p["code"], task["entry_point"], probes), minimum)
                     for p in engine_programs(task, asserts, keep)]
        records.append({"task_id": task_id, "solved": bool(verdicts[task_id]["solved"]),
                        "engine_programs": len(relations), "agree": relations.count("agree"),
                        "conflict": relations.count("conflict"), "unknown": relations.count("unknown"),
                        "model_values": sum(status == "ok" for status, _ in model)})
    return records


def summarise(records):
    def cell(group):
        return {"tasks": len(group), "solved": sum(r["solved"] for r in group)}
    return {"answers": cell(records),
            "corroborated": cell([r for r in records if r["agree"]]),
            "engine_disagrees": cell([r for r in records if not r["agree"] and r["conflict"]]),
            "engine_undecided": cell([r for r in records if r["engine_programs"] and not r["agree"] and not r["conflict"]]),
            "no_engine_program": cell([r for r in records if not r["engine_programs"]])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", type=int, default=10, help="engine programs compared per task")
    parser.add_argument("--probes", type=int, default=24)
    parser.add_argument("--minimum", type=int, default=3, help="probes both must answer to count as agreement")
    args = parser.parse_args()
    tasks = read_tasks()
    report = {"library": {"packs": bi.ACTIVE_PACKS, "engine": sorted(bi.ENGINE), "operators": len(bi.OPERATORS)},
              "settings": vars(args), "runs": {}}
    for run, split in (("s15-branch-dev-r3", "train"), ("s15-branch-dev-r3", "val"),
                       ("s15-branch-test-r3", "test"), ("s05-branch-test-r3", "test")):
        started = time.monotonic()
        records = study(run, split, tasks, args.keep, args.probes, args.minimum)
        report["runs"][f"{run}:{split}"] = summarise(records)
        print(run, split, json.dumps(report["runs"][f"{run}:{split}"]), f"{time.monotonic() - started:.0f}s", flush=True)
        if split != "test":
            (OUT / f"consensus_{run}_{split}.json").write_text(json.dumps(records, indent=1))
    (ROOT / "reports/rule_pilot/consensus_corroboration.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
