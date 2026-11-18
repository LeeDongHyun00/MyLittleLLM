"""Why are some corroborated answers still wrong? Split them into probe-coverage misses and shared errors.

Development splits only (the reference solution is used). For each pipeline answer that a
rule-engine program corroborates but that fails the hidden tests, the reference solution is run on
the same probes:
  coverage miss : the reference agrees with the answer on every probe, so the error lies on inputs
                  the probe grammar never makes (typically edge cases of the hidden tests);
  shared error  : the reference disagrees on some probe, so the model and the engine made the same
                  mistake on ordinary inputs (their errors are correlated).
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mylittlellm.rulecoder import backinfer as bi, consensus as cs
from mylittlellm.rulecoder.tasks import ROOT, parse_prompt, read_tasks

OUT = ROOT / "outputs/rule_pilot"


def main():
    tasks = read_tasks()
    gen = {r["task_id"]: r for r in map(json.loads, open(OUT / "gen/s15-branch-dev-r3.jsonl"))}
    summary = {}
    for split in ("train", "val"):
        records = json.loads((OUT / f"consensus_s15-branch-dev-r3_{split}.json").read_text())
        counts = {"corroborated_wrong": 0, "coverage_miss": 0, "shared_error": 0, "reference_unusable": 0}
        for record in records:
            if not record["agree"] or record["solved"]:
                continue
            counts["corroborated_wrong"] += 1
            task = tasks[record["task_id"]]
            _, asserts = parse_prompt(task["prompt"])
            example = bi.parse_example(asserts, task["entry_point"])
            probes = cs.probe_inputs(example["args"], 24)
            answer = cs.run_model_code(gen[record["task_id"]]["final_code"], task["entry_point"], probes)
            reference = cs.run_model_code(task["prompt"] + task["canonical_solution"], task["entry_point"], probes)
            relation = cs.compare(answer, reference, 1)
            if relation == "conflict":
                counts["shared_error"] += 1
            elif relation == "agree":
                counts["coverage_miss"] += 1
            else:
                counts["reference_unusable"] += 1
        summary[split] = counts
        print(split, json.dumps(counts))
    (ROOT / "reports/rule_pilot/consensus_error_kinds.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
