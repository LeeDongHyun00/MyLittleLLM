"""Round 5 pilot, stage 4: does the pack solve real tasks its teachers never saw? Counts only.

Runs the shipped engine with a candidate pack on top (as scripts/teacher_check.py --round 5 loads it) on the
extra pool's training part (434 real tasks with reference solutions) and compares each task's first program
with the shipped engine's (outputs/seedlm/v37/engine_train.json, SeedLM-37): newly solved and broken, under
the lenient judge (asserts) and the strict one (asserts, then no conflict with the reference on the probes).
The registered primary measure is newly solved minus broken, strict (reports/seedlm/round5_pilot_protocol.json).

  python scripts/round5_measure.py --pack r5_pilot --output reports/seedlm/round5_pilot_transfer.json
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data/rule_pilot/operator_packs.json").read_text())
    os.environ["RULE_PACKS"] = ",".join(manifest.get("active", []) + args.pack.split(","))
    os.environ["RULE_RETIRE"] = ",".join(manifest.get("retired", {}))
    os.environ["RULE_ENGINE"] = ",".join(manifest.get("engine", []))
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    from mylittlellm.rulecoder import backinfer as bi, consensus
    from mylittlellm.rulecoder.limits import cap_memory
    from mylittlellm.rulecoder.sandbox import run_calls, run_public_asserts
    from seedlm2_real import held_out
    from seedlm3_tasks import reference_probe_results
    cap_memory()
    items = [i for i in json.loads((ROOT / "outputs/rule_pilot/teacher/round3/all_tasks.json").read_text())
             if i["source"] == "extra" and not held_out(i["task_id"])]
    before = json.loads((ROOT / "outputs/seedlm/v37/engine_train.json").read_text())
    assert [r["task_id"] for r in before] == [i["task_id"] for i in items]
    counts = {"tasks": len(items), "operators_loaded": len(bi.OPERATORS), "changed_first_program": 0,
              "lenient": {"before": 0, "after": 0, "newly_solved": 0, "broken": 0},
              "strict": {"before": 0, "after": 0, "newly_solved": 0, "broken": 0},
              "first_programs_using_the_pack": 0}
    pack_ops = {key for key, op in bi.OPERATORS.items() if op.pack in args.pack.split(",")}
    for item, old in zip(items, before):
        found = bi.synthesize(item["description"], item["visible_assert"], item["entry_point"], weights={})
        first = old["matches"][0] if old["matches"] else None
        old_lenient, old_strict = bool(first and first["right"]), bool(first and first["strict"])
        if found is None:
            new_lenient = new_strict = False
        elif first and found["expr"] == first["expr"]:
            new_lenient, new_strict = old_lenient, old_strict
        else:
            counts["changed_first_program"] += 1
            counts["first_programs_using_the_pack"] += any(op in pack_ops for op in found["ops"])
            new_lenient = run_public_asserts(found["code"], "\n".join(item["asserts"]))["status"] == "pass"
            new_strict = new_lenient
            if new_lenient:
                reference = reference_probe_results(item)
                example = bi.parse_example(item["visible_assert"], item["entry_point"])
                result = run_calls(found["code"], item["entry_point"], consensus.probe_inputs(list(example["args"]), 24))
                if reference is not None and result.get("status") == "ok":
                    new_strict = consensus.compare([tuple(r) for r in result["results"]], reference) != "conflict"
        for judge, old_ok, new_ok in (("lenient", old_lenient, new_lenient), ("strict", old_strict, new_strict)):
            counts[judge]["before"] += old_ok
            counts[judge]["after"] += new_ok
            counts[judge]["newly_solved"] += new_ok and not old_ok
            counts[judge]["broken"] += old_ok and not new_ok
    counts["primary_newly_solved_minus_broken_strict"] = counts["strict"]["newly_solved"] - counts["strict"]["broken"]
    text = json.dumps(counts, indent=1)
    print(text)
    if args.output:
        (ROOT / args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
