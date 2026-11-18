"""Real development tasks as SeedLM-2 supervision: description, example and a program known to be correct.

Sources (never the validation or test split):
  - train-split and extra-pool tasks whose shipped-engine program is correct (judged by hidden tests
    or by the task's asserts);
  - tasks where the engine's first program was wrong but the oracle run found a correct one among
    the matches (ranking and gating failures): exactly the mistakes a reader should learn to avoid.
A fixed fifth of the extra pool is held out for evaluation.
"""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mylittlellm.rulecoder import backinfer as bi
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import language

BANK = ROOT / "outputs/rule_pilot/teacher/round3"
OUT = ROOT / "outputs/seedlm/v2"


def held_out(task_id):
    return task_id.startswith("MbppX/") and int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % 5 == 0


def main():
    items = {i["task_id"]: i for i in json.loads((BANK / "all_tasks.json").read_text())}
    status = {r["task_id"]: r for r in json.loads((BANK / "attribution_selected.json").read_text())["results"]}
    oracle = {r["task_id"]: r for r in json.loads((ROOT / "outputs/seedlm/ceiling_dev_rows.json").read_text())}
    rows, counts = [], {}
    for task_id, item in items.items():
        example = bi.parse_example(item["visible_assert"], item["entry_point"])
        if example is None or not example["args"] or len(example["args"]) > 4:
            continue
        program = None
        if status[task_id]["after"] == "solved":
            found = bi.synthesize(item["description"], item["visible_assert"], item["entry_point"], weights={})
            if found and found["expr"] == status[task_id]["expr"]:
                program, origin = found["program"], "engine"
        elif oracle.get(task_id, {}).get("program"):
            program, origin = oracle[task_id]["program"], "oracle_" + oracle[task_id]["verdict"]
        if not program or "?" in program:
            continue
        literals = bi.description_literals(item["description"])
        if any(t.startswith("c") and "/" not in t and int(t[1:]) >= len(literals) for t in program):
            continue
        part = "holdout" if held_out(task_id) else "train"
        rows.append({"task_id": task_id, "part": part, "origin": origin, "words": language.words(item["description"]),
                     # Python literals, not JSON: a tuple must stay a tuple.
                     "description": item["description"], "literals": repr(literals), "args": repr(example["args"]),
                     "expected": repr(example["expected"]), "mode": example["mode"], "program": program,
                     "named": [k for k, op in bi.OPERATORS.items() if op.named_in(item["description"].lower())]})
        counts[(part, origin)] = counts.get((part, origin), 0) + 1
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "real_dev.json").write_text(json.dumps(rows, indent=1))
    print({f"{a}:{b}": c for (a, b), c in sorted(counts.items())})


if __name__ == "__main__":
    main()
