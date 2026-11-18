"""Select the operators of a teacher round by their measured effect on development tasks.

Teachers write many candidate operators; which ones stay is decided by data, not by the teacher
(the RuleFit step: many candidate rules, few survivors). Every candidate pack is run on all
development tasks (train split and extra pool), and each new operator is credited with the final
programs it appears in. The retirement rule below was fixed before any pack had been written:

  ungated operator : kept only if it appears in at least one correct final program and in at
                     least as many correct as wrong ones (it costs every task some accidental
                     matches, so it must show a benefit);
  gated operator   : retired only if it appears in two or more wrong final programs and in more
                     wrong than correct ones (the keyword gate already limits its reach, so an
                     unused definition is allowed to stay).

The validation and test splits play no part here.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "outputs/rule_pilot/teacher/round3"
MANIFEST = ROOT / "data/rule_pilot/operator_packs.json"
REPORT = ROOT / "reports/rule_pilot/round3_attribution.json"


def check(pack, output):
    command = [sys.executable, str(ROOT / "scripts/teacher_check.py"), "--pack", pack,
               "--bank", str(BANK / "all_tasks.json"), "--output", str(output)]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(Path(output).read_text()), completed.stdout


def tally(results, source=None):
    rows = [r for r in results if source is None or r["source"] == source]
    return {"tasks": len(rows), "solved": sum(r["after"] == "solved" for r in rows),
            "newly_solved": sum(r["before"] != "solved" and r["after"] == "solved" for r in rows),
            "broken": sum(r["before"] == "solved" and r["after"] != "solved" for r in rows),
            "wrong_program": sum(r["after"] == "wrong" for r in rows),
            "nothing_found": sum(r["after"] == "none" for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--packs", required=True, help="candidate packs, comma-separated")
    parser.add_argument("--round", type=int, default=3)
    args = parser.parse_args()
    packs = [name for name in args.packs.split(",") if name]

    candidate, _ = check(",".join(packs), BANK / "attribution_all.json")
    stats = {op["key"]: dict(op, right=0, wrong=0) for op in candidate["operators"]}
    for result in candidate["results"]:
        for key in result["pack_ops"]:
            stats[key]["right" if result["after"] == "solved" else "wrong"] += 1
    retired = {}
    for key, op in sorted(stats.items()):
        if op["gated"]:
            drop = op["wrong"] >= 2 and op["wrong"] > op["right"]
            reason = "gated, more wrong than right final programs"
        else:
            drop = op["right"] == 0 or op["wrong"] > op["right"]
            reason = "ungated, no correct final program" if op["right"] == 0 else "ungated, more wrong than right"
        if drop:
            retired[key] = f"{reason} on development tasks ({op['right']} right, {op['wrong']} wrong)"

    MANIFEST.write_text(json.dumps({"round": args.round, "active": packs, "retired": retired}, indent=1) + "\n")
    selected, _ = check("manifest", BANK / "attribution_selected.json")
    report = {
        "rule": "see scripts/round3_select.py",
        "packs": {name: {"operators": sum(op["pack"] == name for op in stats.values()),
                         "gated": sum(op["pack"] == name and op["gated"] for op in stats.values()),
                         "retired": sum(stats[key]["pack"] == name for key in retired)} for name in packs},
        "all_candidates": {"train": tally(candidate["results"], "train"), "extra": tally(candidate["results"], "extra")},
        "after_selection": {"train": tally(selected["results"], "train"), "extra": tally(selected["results"], "extra")},
        "operators": sorted(({"key": key, "pack": op["pack"], "gated": op["gated"], "right": op["right"],
                              "wrong": op["wrong"], "retired": key in retired} for key, op in stats.items()),
                            key=lambda row: (row["pack"], -row["right"], row["key"])),
    }
    REPORT.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("packs", "all_candidates", "after_selection")}, indent=1))
    print(f"{len(stats)} candidate operators, {len(retired)} retired; manifest written to {MANIFEST}")


if __name__ == "__main__":
    main()
