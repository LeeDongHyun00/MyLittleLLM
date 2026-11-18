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

Search features that a teacher asked for (flags read by search() in backinfer.py) are chosen the
same way, before retirement: every subset is run with all candidate operators and the subset that
solves the most development tasks wins, ties going to the smaller subset.

Operators a teacher flags as written from memory of benchmark tasks rather than for a bank item
(--memory) are retired before anything is measured, whatever their effect.

The validation and test splits play no part here.
"""
import itertools
import os
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "outputs/rule_pilot/teacher/round3"
MANIFEST = ROOT / "data/rule_pilot/operator_packs.json"
REPORT = ROOT / "reports/rule_pilot/round3_attribution.json"
MEMORY = []


def check(pack, output, engine=()):
    command = [sys.executable, str(ROOT / "scripts/teacher_check.py"), "--pack", pack,
               "--bank", str(BANK / "all_tasks.json"), "--output", str(output)]
    environment = dict(os.environ, RULE_ENGINE=",".join(engine), RULE_RETIRE=",".join(MEMORY))
    if pack == "manifest":
        environment.pop("RULE_ENGINE")      # the manifest names its own features
    completed = subprocess.run(command, capture_output=True, text=True, check=True, env=environment)
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
    parser.add_argument("--features", default="tie_break,description_constants,keep_best,named_pairs",
                        help="candidate search features, comma-separated")
    parser.add_argument("--memory", nargs="*", default=[],
                        help="operator keys a teacher flagged as written from memory of benchmark tasks")
    parser.add_argument("--reuse-features", action="store_true",
                        help="take the feature comparison from the last run instead of repeating it")
    args = parser.parse_args()
    packs = [name for name in args.packs.split(",") if name]
    features = [name for name in args.features.split(",") if name]
    MEMORY[:] = args.memory         # out of every measurement, not only out of the final manifest

    # 1. search features: the subset that solves the most development tasks, smaller subset on ties.
    # The 16 runs take ten minutes, so their outcome is cached and reused when nothing changed.
    cache = BANK / "features_tried.json"
    if args.reuse_features and cache.exists():
        tried = json.loads(cache.read_text())
    else:
        tried = []
        for size in range(len(features) + 1):
            for subset in itertools.combinations(features, size):
                result, _ = check(",".join(packs), BANK / "attribution_features.json", subset)
                solved = sum(r["after"] == "solved" for r in result["results"])
                tried.append({"features": list(subset), "solved": solved})
                print("features", subset or "(none)", "solved", solved, flush=True)
        cache.write_text(json.dumps(tried, indent=1) + "\n")
    engine = min(tried, key=lambda row: (-row["solved"], len(row["features"])))["features"]

    # 2. operators, measured with the chosen features
    candidate, _ = check(",".join(packs), BANK / "attribution_all.json", engine)
    stats = {op["key"]: dict(op, right=0, wrong=0) for op in candidate["operators"]}
    for result in candidate["results"]:
        for key in result["pack_ops"]:
            stats[key]["right" if result["after"] == "solved" else "wrong"] += 1
    retired = {key: "a teacher flagged it as written from memory of benchmark tasks" for key in args.memory}
    defined = "".join((ROOT / "mylittlellm/rulecoder/oppacks" / f"{name}.py").read_text() for name in packs)
    unknown = [key for key in args.memory if f'"{key.split("/")[0]}"' not in defined]
    if unknown:
        raise SystemExit(f"--memory names operators no candidate pack defines: {unknown}")
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
            retired[key] = f"{reason} on development tasks ({op['right']} right, {op['wrong']} wrong)"

    MANIFEST.write_text(json.dumps({"round": args.round, "active": packs, "engine": engine, "retired": retired},
                                   indent=1) + "\n")
    selected, _ = check("manifest", BANK / "attribution_selected.json")
    report = {
        "rule": "see scripts/round3_select.py",
        "features_tried": tried,
        "features_chosen": engine,
        "packs": {name: {"operators": sum(op["pack"] == name for op in stats.values()),
                         "gated": sum(op["pack"] == name and op["gated"] for op in stats.values()),
                         "retired_by_data": sum(key in stats and stats[key]["pack"] == name for key in retired)}
                  for name in packs},
        "retired_from_memory": args.memory,
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
