"""Select round 4's operators by their measured effect on the extra pool's training part (434 tasks).

Round 3's rules (scripts/round3_select.py), fixed before any round-4 pack was written, applied to the
training tasks only (the bank failures and the regression set of round4_bank.py; never the held-out
development fifth, validation or test):

  ungated operator : kept only if it appears in at least one correct final program and in at
                     least as many correct as wrong ones;
  gated operator   : retired only if it appears in two or more wrong final programs and in more
                     wrong than correct ones;
  operators a teacher flags as written from memory of benchmark tasks are retired before anything
  is measured. Search features stay as shipped.

Writes the candidate library (outputs/rule_round4/manifest_candidate.json: the shipped manifest plus the
round-4 packs and retirements; the shipped manifest is not touched) and reports/seedlm/round4_attribution.json.

  python scripts/round4_select.py --packs r4_numbers,r4_sequences,r4_strings [--memory key ...]
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "outputs/rule_round4"
MANIFEST = ROOT / "data/rule_pilot/operator_packs.json"


def run(packs, retire, output):
    command = [sys.executable, str(ROOT / "scripts/teacher_check.py"), "--round", "4", "--pack", packs,
               "--bank", str(BANK / "all_tasks.json"), "--output", str(output)]
    environment = dict(os.environ, RULE_RETIRE_EXTRA=",".join(retire))
    completed = subprocess.run(command, capture_output=True, text=True, check=True, env=environment)
    return json.loads(Path(output).read_text()), completed.stdout


def tally(results):
    return {"tasks": len(results), "solved": sum(r["after"] == "solved" for r in results),
            "newly_solved": sum(r["before"] != "solved" and r["after"] == "solved" for r in results),
            "broken": sum(r["before"] == "solved" and r["after"] != "solved" for r in results),
            "wrong_program": sum(r["after"] == "wrong" for r in results),
            "nothing_found": sum(r["after"] == "none" for r in results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--packs", required=True)
    parser.add_argument("--memory", nargs="*", default=[])
    args = parser.parse_args()
    items = [i for path in sorted(BANK.glob("bank_*.json"))
             if not path.stem.startswith("bank_r4_") and path.stem != "bank_summary"
             for i in json.loads(path.read_text())]
    items += json.loads((BANK / "regression.json").read_text())
    (BANK / "all_tasks.json").write_text(json.dumps(items) + "\n")
    candidate, _ = run(args.packs, args.memory, BANK / "attribution_all.json")
    stats = {op["key"]: dict(op, right=0, wrong=0) for op in candidate["operators"] if op["pack"].startswith("r4_")}
    for result in candidate["results"]:
        for key in result["pack_ops"]:
            if key in stats:
                stats[key]["right" if result["after"] == "solved" else "wrong"] += 1
    retired = {key: "a teacher flagged it as written from memory of benchmark tasks" for key in args.memory}
    for key, op in sorted(stats.items()):
        if key in retired:
            continue
        if op["gated"]:
            drop = op["wrong"] >= 2 and op["wrong"] > op["right"]
            reason = "gated, more wrong than right final programs"
        else:
            drop = op["right"] == 0 or op["wrong"] > op["right"]
            reason = "ungated, no correct final program" if op["right"] == 0 else "ungated, more wrong than right"
        if drop:
            retired[key] = f"{reason} on training tasks ({op['right']} right, {op['wrong']} wrong)"
    final, _ = run(args.packs, list(retired), BANK / "attribution_selected.json")
    shipped = json.loads(MANIFEST.read_text())
    manifest = {"round": 4, "active": shipped["active"] + [p for p in args.packs.split(",") if p],
                "engine": shipped.get("engine", []), "retired": dict(shipped.get("retired", {}), **retired)}
    (BANK / "manifest_candidate.json").write_text(json.dumps(manifest, indent=1) + "\n")
    report = {"packs": args.packs.split(","), "operators": len(stats), "retired": retired,
              "kept": sorted(k for k in stats if k not in retired),
              "all_candidates": tally(candidate["results"]), "selected": tally(final["results"]),
              "per_operator": {k: {"gated": v["gated"], "right": v["right"], "wrong": v["wrong"]}
                               for k, v in sorted(stats.items())}}
    (ROOT / "reports/seedlm/round4_attribution.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("operators", "all_candidates", "selected")}, indent=1))
    print("retired", len(retired), "kept", len(report["kept"]))


if __name__ == "__main__":
    main()
