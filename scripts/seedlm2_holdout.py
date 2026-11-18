"""SeedLM-2 on the held-out fifth of the extra pool (development data the model never trained on).

Same four methods as seedlm2_tasks.py (engine, model, rerank, open), judged by each task's own three
asserts. The held-out tasks include ones where the engine ranked or gated the right program out,
so this is where a reader of descriptions should show its worth before the validation split is used.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from mylittlellm.rulecoder.sandbox import run_public_asserts
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import propose2
from seedlm2_real import held_out
from seedlm2_tasks import code_for, methods_for

BANK = ROOT / "outputs/rule_pilot/teacher/round3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--keep", type=int, default=40)
    args = parser.parse_args()
    torch.set_num_threads(4)
    net, vocabulary = propose2.load(ROOT / "outputs/seedlm" / f"{args.model}.pt")
    arities = propose2.arity_table(vocabulary)
    items = [i for i in json.loads((BANK / "all_tasks.json").read_text())
             if i["source"] == "extra" and held_out(i["task_id"])]
    status = {r["task_id"]: r["after"] for r in json.loads((BANK / "attribution_selected.json").read_text())["results"]}
    names = ["engine", "model", "rerank", "open"]
    solved = {n: 0 for n in names}
    by_status = {}
    for item in items:
        task = {"prompt": '"""\n' + item["description"] + "\n" + item["visible_assert"] + '\n"""\n',
                "entry_point": item["entry_point"]}
        chosen = methods_for(task, net, vocabulary, arities, args.width, args.keep)
        group = by_status.setdefault(status.get(item["task_id"], "?"), {n: 0 for n in names + ["tasks"]})
        group["tasks"] += 1
        for name in names:
            program, literals = chosen.get(name, (None, ()))
            ok = False
            if program:
                code = code_for(task, program, literals)
                ok = run_public_asserts(code, "\n".join(item["asserts"]))["status"] == "pass"
            solved[name] += ok
            group[name] += ok
    report = {"model": args.model, "tasks": len(items), "solved": solved, "by_engine_status": by_status}
    (ROOT / "reports/seedlm" / f"holdout2_{args.model}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
