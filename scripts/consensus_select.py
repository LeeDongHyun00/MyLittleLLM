"""Corroboration-gated consensus: choose among the pipeline answer, extra samples and engine programs.

The rule is fixed in reports/rule_pilot/consensus_protocol.json. Non-target tasks keep the pipeline
answer. For a target task every candidate that passes the visible example is run on probe inputs,
and the candidate c with the highest
    s(c) = (model candidates agreeing with c, c included) / (number of model candidates)
           + lambda * [c is an engine program, or some engine program agrees with c]
is chosen; ties keep the earlier candidate (pipeline answer first). The output is a run in the
usual format, so `rule_pilot.py score` and `summary` work on it.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mylittlellm.rulecoder import backinfer as bi, consensus as cs
from mylittlellm.rulecoder.pipeline import public_check
from mylittlellm.rulecoder.tasks import ROOT, parse_prompt, read_tasks

OUT = ROOT / "outputs/rule_pilot"
RUNS = {"train": "s15-branch-dev-r3", "val": "s15-branch-dev-r3", "test": "s15-branch-test-r3"}


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def choose(task, row, plain, samples, lam, keep, probes, minimum):
    """Return (code, source, details) for one target task."""
    _, asserts = parse_prompt(task["prompt"])
    example = bi.parse_example(asserts, task["entry_point"])
    if example is None or not example["args"]:
        return row["final_code"], row["chosen"], {"reason": "no example arguments"}
    inputs = cs.probe_inputs(example["args"], probes)
    pool, seen = [], set()

    def add(code, source, model):
        if code and code not in seen:
            seen.add(code)
            run = cs.run_model_code if model else cs.run_engine_code
            pool.append({"code": code, "source": source, "model": model,
                         "results": run(code, task["entry_point"], inputs)})

    if row["visible_ok"]:
        add(row["final_code"], row["chosen"], row["chosen"] != "backinfer")
    # Runs of the plain condition never checked the example, so it is checked here.
    if plain and public_check(plain["final_code"], asserts)["ok"]:
        add(plain["final_code"], "plain", True)
    for sample in samples:
        if sample["visible_ok"]:
            add(sample["code"], f"sample{sample['index']}", True)
    description, _ = parse_prompt(task["prompt"])
    found = bi.synthesize(description, asserts, task["entry_point"], weights={}, keep=keep)
    for rank, program in enumerate((found.get("alternatives") or [found]) if found else []):
        add(program["code"], f"engine{rank}", False)
    if not pool:
        return row["final_code"], row["chosen"], {"reason": "empty pool"}
    models = [c for c in pool if c["model"]]
    engines = [c for c in pool if not c["model"]]
    best, best_score = None, -1.0
    for candidate in pool:
        support = sum(1 for other in models
                      if other is candidate or cs.compare(candidate["results"], other["results"], minimum) == "agree")
        backed = (not candidate["model"]) or any(
            cs.compare(candidate["results"], e["results"], minimum) == "agree" for e in engines)
        score = (support / len(models) if models else 0.0) + lam * backed
        if score > best_score + 1e-9:
            best, best_score = candidate, score
    return best["code"], best["source"], {"pool": len(pool), "model_candidates": len(models),
                                          "engine_programs": len(engines), "score": round(best_score, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, help="name of a consensus_sample.py run")
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--keep", type=int, default=10)
    parser.add_argument("--probes", type=int, default=24)
    parser.add_argument("--minimum", type=int, default=3)
    parser.add_argument("--run", help="pipeline run to start from (default: the 1.5B round-3 runs)")
    parser.add_argument("--plain-run", default="s15-base", help="run holding the plain-prompt greedy answers")
    parser.add_argument("--k", type=int, help="use only the first k samples of each task (cost curve)")
    args = parser.parse_args()
    runs = {split: args.run for split in RUNS} if args.run else RUNS
    tasks = read_tasks()
    folder = OUT / "samples"
    targets = json.loads((folder / f"{args.samples}_targets.json").read_text())
    samples = {}
    for sample in read_jsonl(folder / f"{args.samples}.jsonl"):
        samples.setdefault(sample["task_id"], []).append(sample)
    plain = {r["task_id"]: r for r in read_jsonl(OUT / "gen" / f"{args.plain_run}.jsonl")}
    target = OUT / "gen" / f"{args.output_name}.jsonl"
    if target.exists():
        raise SystemExit(f"{target} already exists")
    counts = {}
    with target.open("w") as stream:
        for split in args.splits:
            for row in read_jsonl(OUT / "gen" / f"{runs[split]}.jsonl"):
                if row["split"] != split:
                    continue
                out = dict(row, name=args.output_name, condition="consensus", derived_from=[runs[split], args.samples])
                reason = targets.get(split, {}).get(row["task_id"])
                if reason:
                    drawn = sorted(samples.get(row["task_id"], []), key=lambda s: s["index"])
                    if args.k is not None:
                        drawn = [s for s in drawn if s["index"] < args.k]
                    if len(drawn) < 1 and args.k != 0:
                        raise SystemExit(f"no samples for {row['task_id']}: sampling has not finished")
                    code, source, details = choose(tasks[row["task_id"]], row, plain.get(row["task_id"]), drawn,
                                                   args.lam, args.keep, args.probes, args.minimum)
                    # Sample costs are attempts too, so call and token counts stay honest.
                    extra = [{"round": len(row["attempts"]) + i, "code": s["code"], "raw_code": s["code"],
                              "source": f"sample{s['index']}", "llm": s["llm"],
                              "check": {"ok": s["visible_ok"], "kind": "sample", "message": ""}, "hints": []}
                             for i, s in enumerate(drawn)]
                    _, asserts = parse_prompt(tasks[row["task_id"]]["prompt"])
                    out.update(final_code=code, chosen=source, attempts=row["attempts"] + extra,
                               visible_ok=public_check(code, asserts)["ok"],
                               consensus={"reason": reason, **details})
                    key = (split, reason, "changed" if code != row["final_code"] else "kept")
                    counts[key] = counts.get(key, 0) + 1
                else:
                    out["consensus"] = {"reason": "corroborated"}
                stream.write(json.dumps(out) + "\n")
    for key, value in sorted(counts.items()):
        print(*key, value)


if __name__ == "__main__":
    main()
