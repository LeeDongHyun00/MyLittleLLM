"""Seeds for SeedLM-5's teacher variants: the training-pool tasks the shipped engine cannot solve.

The extra pool's held-out fifth, the MBPP+ validation and test splits and HumanEval+ are never seeds.
A training-pool task is unsolved when no correct library program is known for it
(outputs/seedlm/v2/real_dev.json, training part). Each seed carries what a teacher needs to write
neighbouring tasks: the description, the entry point, its asserts and the reference solution.

  python scripts/seedlm5_seeds.py [--parts 6]   writes outputs/seedlm/v5/seeds_part*.json
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylittlellm.rulecoder.tasks import ROOT
from seedlm2_real import held_out

BANK = ROOT / "outputs/rule_pilot/teacher/round3/all_tasks.json"
REAL = ROOT / "outputs/seedlm/v2/real_dev.json"
OUT = ROOT / "outputs/seedlm/v5"


def seeds():
    solved = {row["task_id"] for row in json.loads(REAL.read_text()) if row["part"] == "train"}
    out = []
    for item in json.loads(BANK.read_text()):
        if item["source"] == "extra" and held_out(item["task_id"]):
            continue
        if item["source"] not in ("extra", "train") or item["task_id"] in solved:
            continue
        out.append({"seed": item["task_id"], "category": item["category"], "description": item["description"],
                    "entry_point": item["entry_point"],
                    "asserts": item.get("asserts") or [item["visible_assert"]],
                    "reference_code": item["reference_code"]})
    return sorted(out, key=lambda s: s["seed"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=int, default=6)
    args = parser.parse_args()
    items = seeds()
    OUT.mkdir(parents=True, exist_ok=True)
    for part in range(args.parts):
        chunk = items[part::args.parts]
        (OUT / f"seeds_part{part + 1}.json").write_text(json.dumps(chunk, indent=1) + "\n")
        print(f"part {part + 1}: {len(chunk)} seeds")
    print(f"{len(items)} seeds")


if __name__ == "__main__":
    main()
