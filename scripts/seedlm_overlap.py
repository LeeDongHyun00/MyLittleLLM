"""Do different coders discover the same tasks? Overlap of held-out fixes beyond the engine, counts only.

Reads the local per-task verdicts written by `seedlm3_tasks.py holdout --verdicts` and
`seedlm4_tasks.py holdout --verdicts` (outputs/seedlm/verdicts/, never committed) and reports, for one
method, how many held-out tasks each coder solves where the engine fails, how many of those every
coder shares, and how many any coder reaches. Task identities are not printed.

  python scripts/seedlm_overlap.py --runs holdout3_s3-pyx holdout3_s3b-pyx holdout4_s4-post [--method pyx_model]
"""
import argparse
from itertools import combinations
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mylittlellm.rulecoder.tasks import ROOT

LOCAL = ROOT / "outputs/seedlm/verdicts"


def fixes(verdicts, method):
    return {task for task, v in verdicts.items() if v[method] and not v["engine"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--method", default="pyx_model")
    parser.add_argument("--output", default="reports/seedlm/overlap_holdout.json")
    args = parser.parse_args()
    runs = {run: json.loads((LOCAL / f"{run}.json").read_text()) for run in args.runs}
    tasks = set.intersection(*(set(v) for v in runs.values()))
    engine_fails = {t for t in tasks if not next(iter(runs.values()))[t]["engine"]}
    found = {run: fixes({t: v[t] for t in tasks}, args.method) for run, v in runs.items()}
    report = {"method": args.method, "tasks": len(tasks), "engine_fails": len(engine_fails),
              "fixes": {run: len(f) for run, f in found.items()},
              "pairwise_shared": {f"{a} & {b}": len(found[a] & found[b]) for a, b in combinations(args.runs, 2)},
              "shared_by_all": len(set.intersection(*found.values())),
              "reached_by_any": len(set.union(*found.values()))}
    (ROOT / args.output).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
