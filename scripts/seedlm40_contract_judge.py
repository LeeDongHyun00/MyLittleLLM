"""SeedLM-40, part C: does an inferred input contract remove the strict judge's artifacts? Training split only.

For every training-side real task whose engine answer passes the task's asserts (SeedLM-37's run,
outputs/seedlm/v37/engine_train.json), the engine program and the reference solution are compared on probe
inputs twice: the 24 plain probes (SeedLM-36's strict judge) and up to 24 probes that satisfy the contract
inferred from all the task's asserts (mylittlellm/rulecoder/contracts.py), drawn from 96 candidates. Counts are
printed; per-task results stay local (outputs/seedlm/v40/contract_train.json) for reading.

  python scripts/seedlm40_contract_judge.py
"""
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylittlellm.rulecoder import backinfer as bi, consensus, contracts
from mylittlellm.rulecoder.limits import cap_memory
from mylittlellm.rulecoder.sandbox import run_calls
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import select
from seedlm2_real import held_out

BANK = ROOT / "outputs/rule_pilot/teacher/round3/all_tasks.json"


def contract_probes(item, count=24, pool=96):
    examples = [bi.parse_example(a, item["entry_point"]) for a in item["asserts"]]
    examples = [e["args"] for e in examples if e and e["args"]]
    visible = bi.parse_example(item["visible_assert"], item["entry_point"])
    contract = contracts.infer(examples or [visible["args"]])
    return contracts.in_domain_probes(contract, consensus.probe_inputs(list(visible["args"]), pool), count), contract


def main():
    cap_memory()
    items = [i for i in json.loads(BANK.read_text()) if i["source"] == "extra" and not held_out(i["task_id"])]
    rows = json.loads((ROOT / "outputs/seedlm/v37/engine_train.json").read_text())
    out, counts, sizes = [], {"lenient_right": 0, "plain_strict_pass": 0, "contract_strict_pass": 0,
                              "conflict_to_pass": 0, "pass_to_conflict": 0, "under_3_in_domain_probes": 0}, []
    for item, row in zip(items, rows):
        first = row["matches"][0] if row["matches"] else None
        if not (first and first["right"]):
            continue
        counts["lenient_right"] += 1
        counts["plain_strict_pass"] += first["strict"]
        probes, contract = contract_probes(item)
        sizes.append(len(probes))
        counts["under_3_in_domain_probes"] += len(probes) < 3
        visible = bi.parse_example(item["visible_assert"], item["entry_point"])
        literals = bi.description_literals(item["description"])
        output = visible["expected"] if isinstance(visible["expected"], str) else None
        mine = select.behaviour(first["program"], probes, literals, output) if probes else []
        result = run_calls(item["reference_code"], item["entry_point"], probes) if probes else {"status": "none"}
        reference = [tuple(r) for r in result["results"]] if result.get("status") == "ok" else None
        verdict = consensus.compare(mine, reference) if reference is not None else "unknown"
        passed = verdict != "conflict"
        counts["contract_strict_pass"] += passed
        counts["conflict_to_pass"] += passed and not first["strict"]
        counts["pass_to_conflict"] += (not passed) and first["strict"]
        diffs = [(p, a, b) for p, (sa, a), (sb, b) in zip(probes, mine, reference or [])
                 if sa == "ok" and sb == "ok" and a != b]
        out.append({"task_id": item["task_id"], "expr": first["expr"], "plain_strict": first["strict"],
                    "contract_strict": passed, "contract": contract, "probes": len(probes),
                    "first_difference": [repr(diffs[0][0])[:200], diffs[0][1][:80], diffs[0][2][:80]] if diffs else None})
    counts["median_in_domain_probes"] = statistics.median(sizes) if sizes else 0
    (ROOT / "outputs/seedlm/v40").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs/seedlm/v40/contract_train.json").write_text(json.dumps(out) + "\n")
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
