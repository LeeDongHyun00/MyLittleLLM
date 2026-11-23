"""SeedLM-37, step 1: the engine's matches on the training-side real tasks, judged strictly. Training data only.

On the development fifth, a sixth of the engine's assert-passing answers differ from the reference solution
on probes (SeedLM-36). Whether a better choice among the engine's own matches could fix them is a question
for training-side tasks, where every task may be read: the extra pool's training part (434 tasks with
reference solutions; the held-out fifth and the MBPP+ splits are left out, as in round 4).

For each task the engine runs as the deployed system calls it (bi.synthesize, weights={}, up to 40 matches);
every match gets its features, its assert verdict and its strict verdict (no conflict with the reference
solution on the 24 probes, scripts/seedlm3_tasks.strict_agrees). Writes outputs/seedlm/v37/engine_train.json
(local) and prints counts.

  python scripts/seedlm37_engine_strict.py
"""
import collections
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylittlellm.rulecoder import backinfer as bi, consensus
from mylittlellm.rulecoder.limits import cap_memory
from mylittlellm.rulecoder.sandbox import run_calls, run_public_asserts
from mylittlellm.rulecoder.tasks import ROOT
from seedlm2_real import held_out
from seedlm3_tasks import reference_probe_results

BANK = ROOT / "outputs/rule_pilot/teacher/round3/all_tasks.json"
OUT = ROOT / "outputs/seedlm/v37"
FEATURES = ("support", "learned", "size", "knowledge", "specific", "all_args")


def main():
    cap_memory()
    items = [i for i in json.loads(BANK.read_text()) if i["source"] == "extra" and not held_out(i["task_id"])]
    rows, counts = [], collections.Counter()
    for index, item in enumerate(items):
        found = bi.synthesize(item["description"], item["visible_assert"], item["entry_point"], weights={}, keep=40)
        row = {"task_id": item["task_id"], "matches": []}
        if found is not None:
            reference = reference_probe_results(item)
            example = bi.parse_example(item["visible_assert"], item["entry_point"])
            probes = consensus.probe_inputs(list(example["args"]), 24)
            row["confident"] = bool(found["confident"])
            for match in found.get("alternatives", []):
                right = run_public_asserts(match["code"], "\n".join(item["asserts"]))["status"] == "pass"
                result = run_calls(match["code"], item["entry_point"], probes)
                behaviour = [tuple(r) for r in result["results"]] if result.get("status") == "ok" else None
                strict = right and (reference is None or behaviour is None
                                    or consensus.compare(behaviour, reference) != "conflict")
                row["matches"].append({"expr": match["expr"], "program": list(match["program"]), "right": right,
                                       "strict": strict, "behaviour": behaviour,
                                       **{k: match.get(k) for k in FEATURES}})
        rows.append(row)
        first = row["matches"][0] if row["matches"] else None
        counts["tasks"] += 1
        counts["first_right"] += bool(first and first["right"])
        counts["first_strict"] += bool(first and first["strict"])
        counts["any_right"] += any(m["right"] for m in row["matches"])
        counts["any_strict"] += any(m["strict"] for m in row["matches"])
        counts["first_right_not_strict"] += bool(first and first["right"] and not first["strict"])
        counts["first_right_not_strict_other_strict"] += bool(first and first["right"] and not first["strict"]
                                                              and any(m["strict"] for m in row["matches"]))
        counts["first_wrong_other_strict"] += bool(first and not first["right"]
                                                   and any(m["strict"] for m in row["matches"]))
        if index % 50 == 0:
            print(index, len(items), dict(counts), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "engine_train.json").write_text(json.dumps(rows) + "\n")
    print(json.dumps(dict(counts), indent=1))


if __name__ == "__main__":
    main()
