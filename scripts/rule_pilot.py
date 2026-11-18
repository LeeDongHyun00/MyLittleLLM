"""Rule-distillation pilot on MBPP+: generate with a CPU student, score, attribute, summarize.

Development pilot, NOT a leaderboard submission. The test split is locked: no failure
diagnosis is ever produced for it and no rule may be induced or selected from it.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mylittlellm.rulecoder import rules as rulelib
from mylittlellm.rulecoder.llm import server_model
from mylittlellm.rulecoder.pipeline import solve
from mylittlellm.rulecoder.stats import mcnemar_exact, wilson
from mylittlellm.rulecoder.tasks import ROOT, load_split, read_tasks, split_of, write_split

OUT = ROOT / "outputs/rule_pilot"
CONDITIONS = {"base", "advice", "rules", "verify", "probes", "full"}


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def task_ids(split, part):
    if part == "all":
        return sorted((t for ids in split.values() for t in ids), key=lambda t: int(t.split("/")[1]))
    return sorted(split[part], key=lambda t: int(t.split("/")[1]))


def cmd_split(_):
    record = write_split()
    print(json.dumps(record["counts"]))


def cmd_generate(args):
    tasks, split = read_tasks(), load_split()
    ids = task_ids(split, args.split)
    if args.tasks_file:
        wanted = set(json.loads(Path(args.tasks_file).read_text()))
        ids = [t for t in ids if t in wanted]
    ids = ids[:args.limit] if args.limit else ids
    path = OUT / "gen" / f"{args.name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["task_id"], r["draw"]) for r in read_jsonl(path)} if path.exists() else set()
    bank_rules = []
    if args.condition in {"rules", "full"}:
        bank_rules = rulelib.load_rules(args.bank, args.rule_ids)
    checkers = None
    if args.condition in {"probes", "full"}:
        from mylittlellm.rulecoder.checkers import active_families, make_checkers
        checkers = make_checkers(active_families(args.bank))
    autofix = None
    if args.condition == "full":
        from mylittlellm.rulecoder.codemods import autofix
    model = server_model()
    with path.open("a") as stream:
        for draw in range(args.draws):
            for task_id in ids:
                if (task_id, draw) in done:
                    continue
                # Only rules whose trigger matches this task are candidates for the prompt.
                fired = [r for r in bank_rules if rulelib.applies(r, tasks[task_id])]
                if args.random_subsets:
                    # Per-task random rule subsets feed the RuleFit-style attribution fit.
                    keep = set(rulelib.random_subset([r["id"] for r in fired], f"{args.name}:{task_id}:{draw}"))
                    fired = [r for r in fired if r["id"] in keep]
                fired = fired[:args.max_rules]
                used = [r["id"] for r in fired] if bank_rules else None
                if args.condition == "advice":
                    block = rulelib.render(rulelib.GENERIC_ADVICE)
                else:
                    block = rulelib.render([r["text"] for r in fired]) if fired else None
                start = time.monotonic()
                result = solve(tasks[task_id], rule_block=block,
                               verify=args.condition in {"verify", "probes", "full"},
                               checkers=checkers, autofix=autofix, max_repairs=args.max_repairs)
                row = {"name": args.name, "task_id": task_id, "split": split_of(task_id, split), "draw": draw,
                       "condition": args.condition, "model": model["id"], "rule_ids": used,
                       "fired_rule_ids": [r["id"] for r in bank_rules if rulelib.applies(r, tasks[task_id])]
                       if bank_rules else None,
                       "rule_block_sha256": hashlib.sha256(block.encode()).hexdigest() if block else None,
                       "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                       "wall_seconds": round(time.monotonic() - start, 3), **result}
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                last = result["attempts"][-1]["llm"]
                print(args.name, task_id, f"calls={len(result['attempts'])}",
                      f"tokens={last['completion_tokens']}", f"{row['wall_seconds']}s", flush=True)


def cmd_autofix(args):
    """Apply the codemods to an existing run's answers; costs no model call."""
    from mylittlellm.rulecoder.codemods import autofix
    tasks = read_tasks()
    source, target = OUT / "gen" / f"{args.name}.jsonl", OUT / "gen" / f"{args.output_name}.jsonl"
    if target.exists():
        raise SystemExit(f"{target} already exists")
    changed = 0
    with target.open("w") as stream:
        for row in read_jsonl(source):
            fixed = autofix(row["final_code"], tasks[row["task_id"]]["entry_point"])
            changed += fixed != row["final_code"]
            row.update(name=args.output_name, condition=row["condition"] + "+autofix", final_code=fixed)
            stream.write(json.dumps(row) + "\n")
    print(f"{args.output_name}: codemods changed {changed} answers")


def cmd_first_attempt(args):
    """Derive the single-shot run that a looped run contains as its first attempt.

    Decoding is greedy and seeded, so the first attempt of a verify/full run is exactly what the
    matching no-loop condition would produce; deriving it avoids generating every answer twice.
    """
    source, target = OUT / "gen" / f"{args.name}.jsonl", OUT / "gen" / f"{args.output_name}.jsonl"
    if target.exists():
        raise SystemExit(f"{target} already exists")
    fallback = {}
    if args.fallback:
        fallback = {(r["task_id"], r["draw"]): r for r in read_jsonl(OUT / "gen" / f"{args.fallback}.jsonl")}
    with target.open("w") as stream:
        for row in read_jsonl(source):
            if fallback and not row.get("fired_rule_ids"):
                # No rule fired, so the prompt is byte-identical to the fallback run's prompt. Reusing
                # that answer keeps cache-induced decoding noise out of the comparison.
                same = dict(fallback[(row["task_id"], row["draw"])])
                same.update(name=args.output_name, condition=args.condition, derived_from=args.fallback,
                            rule_ids=[], fired_rule_ids=[])
                stream.write(json.dumps(same) + "\n")
                continue
            first = dict(row["attempts"][0])
            code = first["raw_code"]
            first.update(code=code, autofixed=False)
            first.pop("check", None)
            first.pop("hints", None)
            row.update(name=args.output_name, condition=args.condition, final_code=code, attempts=[first],
                       wall_seconds=first["llm"]["elapsed_seconds"], derived_from=args.name)
            stream.write(json.dumps(row) + "\n")
    print(f"{args.output_name}: derived from first attempts of {args.name}")


def cmd_score(args):
    from mylittlellm.rulecoder import scoring
    problems = scoring.load_problems()
    oracle = scoring.groundtruth(problems)
    source, target = OUT / "gen" / f"{args.name}.jsonl", OUT / "scored" / f"{args.name}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["task_id"], r["draw"]) for r in read_jsonl(target)} if target.exists() else set()
    # Scoring is deterministic, so an answer identical to one already scored reuses that verdict.
    known = {}
    for reference in args.reuse or []:
        verdicts = {(r["task_id"], r["draw"]): r for r in read_jsonl(OUT / "scored" / f"{reference}.jsonl")}
        for row in read_jsonl(OUT / "gen" / f"{reference}.jsonl"):
            verdict = verdicts.get((row["task_id"], row["draw"]))
            if verdict:
                known[(row["task_id"], row["final_code"])] = verdict
    with target.open("a") as stream:
        for row in read_jsonl(source):
            if (row["task_id"], row["draw"]) in done:
                continue
            problem = problems[row["task_id"]]
            reused = known.get((row["task_id"], row["final_code"]))
            if reused:
                result = {k: reused[k] for k in ("base", "plus", "solved")}
            else:
                result = scoring.score(problem, row["final_code"], oracle[row["task_id"]])
            scored = {k: row.get(k) for k in ("name", "task_id", "split", "draw", "condition", "model",
                                              "rule_ids", "fired_rule_ids")}
            scored.update(base=result["base"], plus=result["plus"], solved=result["solved"],
                          calls=len(row["attempts"]),
                          completion_tokens=sum(a["llm"]["completion_tokens"] for a in row["attempts"]),
                          prompt_tokens_processed=sum(a["llm"]["prompt_tokens_processed"] or 0
                                                      for a in row["attempts"]),
                          wall_seconds=row["wall_seconds"],
                          # Model compute only; wall time also contains queueing and sandbox checks.
                          model_seconds=round(sum((a["llm"]["prompt_ms"] or 0) + (a["llm"]["predicted_ms"] or 0)
                                                  for a in row["attempts"]) / 1000, 3),
                          first_public_ok=row["attempts"][0].get("check", {}).get("ok"))
            # The locked test split never gets a per-test diagnosis, so nobody can tune on it.
            scored["verdict_reused"] = bool(reused)
            if args.diagnose and not result["solved"] and row["split"] != "test":
                scored["diagnosis"] = reused.get("diagnosis") if reused else scoring.diagnose(
                    problem, row["final_code"], oracle[row["task_id"]], result)
            stream.write(json.dumps(scored) + "\n")
            stream.flush()
    rows = read_jsonl(target)
    for part in ("train", "val", "test"):
        subset = [r for r in rows if r["split"] == part]
        if subset:
            print(args.name, part, f"n={len(subset)}", f"base={sum(r['base'] == 'pass' for r in subset)}",
                  f"solved(base+plus)={sum(r['solved'] for r in subset)}")


def cmd_export_failures(args):
    """Write the wrong-example bank (development splits only) for the teacher."""
    tasks = read_tasks()
    gen = {(r["task_id"], r["draw"]): r for r in read_jsonl(OUT / "gen" / f"{args.name}.jsonl")}
    items = []
    for row in read_jsonl(OUT / "scored" / f"{args.name}.jsonl"):
        if row["split"] not in args.splits or row["solved"]:
            continue
        if row["split"] == "test":
            raise SystemExit("Refusing to export locked test failures")
        items.append({"task_id": row["task_id"], "prompt": tasks[row["task_id"]]["prompt"],
                      "student_code": gen[(row["task_id"], row["draw"])]["final_code"],
                      "base": row["base"], "plus": row["plus"], "diagnosis": row.get("diagnosis"),
                      "rule_ids": row.get("rule_ids")})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(items, indent=1) + "\n")
    print(f"exported {len(items)} failures to {args.output}")


def cmd_attribute(args):
    """RuleFit-style fit: paired outcome change vs. the no-rule baseline ~ rule inclusion indicators.

    A rule can only be included where its trigger fired, so "fired" indicators enter the fit as
    nuisance features; otherwise a rule would be credited with the difficulty of the tasks it targets.
    """
    baseline = {r["task_id"]: r["solved"] for r in read_jsonl(OUT / "scored" / f"{args.baseline}.jsonl")}
    rows = [r for r in read_jsonl(OUT / "scored" / f"{args.name}.jsonl") if r["split"] in args.splits]
    rule_ids = sorted({i for r in rows for i in r["fired_rule_ids"]})
    features = [[1.0 if i in r["rule_ids"] else 0.0 for i in rule_ids] +
                [1.0 if i in r["fired_rule_ids"] else 0.0 for i in rule_ids] for r in rows]
    targets = [float(r["solved"]) - float(baseline[r["task_id"]]) for r in rows]
    penalties = [0.0, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08]
    penalty, errors = rulelib.cross_validated_penalty(features, targets, penalties)
    intercept, coef = rulelib.lasso(features, targets, penalty)
    report = {"name": args.name, "baseline": args.baseline, "splits": args.splits, "observations": len(rows),
              "penalty": penalty, "cv_errors": errors, "intercept": round(intercept, 4), "rules": []}
    for j, rule_id in enumerate(rule_ids):
        fired = [r for r in rows if rule_id in r["fired_rule_ids"]]
        used = [r for r in fired if rule_id in r["rule_ids"]]
        unused = [r for r in fired if rule_id not in r["rule_ids"]]
        change = lambda group: sum(float(r["solved"]) - float(baseline[r["task_id"]]) for r in group) / len(group) \
            if group else None
        with_rule, without_rule = change(used), change(unused)
        support = len(used) / len(rows)
        report["rules"].append({
            "id": rule_id, "fired": len(fired), "included": len(used), "support": round(support, 3),
            "solved_when_included": sum(r["solved"] for r in used),
            "wins": sum(r["solved"] and not baseline[r["task_id"]] for r in used),
            "losses": sum(not r["solved"] and baseline[r["task_id"]] for r in used),
            "ucb_reward": round(rulelib.ucb_reward(sum(r["solved"] for r in used), len(used), len(rows)), 4),
            # Difference in outcome change between tasks that got the rule and tasks that could have.
            "contrast": None if with_rule is None or without_rule is None else round(with_rule - without_rule, 4),
            "lasso_coefficient": round(coef[j], 4),
            "importance": round(rulelib.rule_importance(coef[j], support), 4)})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=1) + "\n")
    for rule in report["rules"]:
        print(rule)
    print("penalty", penalty, "intercept", report["intercept"], "n", len(rows),
          "mean change", round(sum(targets) / len(targets), 4))


def cmd_probe_stats(args):
    """How well does each probe family predict hidden-test failure? Development splits only."""
    from mylittlellm.rulecoder.pipeline import public_check
    from mylittlellm.rulecoder.probes import FAMILIES, run_probes
    from mylittlellm.rulecoder.tasks import parse_prompt
    tasks = read_tasks()
    scored = {r["task_id"]: r for r in read_jsonl(OUT / "scored" / f"{args.name}.jsonl")}
    records = []
    for row in read_jsonl(OUT / "gen" / f"{args.name}.jsonl"):
        if row["split"] == "test":
            continue
        task = tasks[row["task_id"]]
        asserts = parse_prompt(task["prompt"])[1]
        visible_ok = public_check(row["final_code"], asserts)["ok"]
        findings = run_probes(row["final_code"], task["entry_point"], asserts) if visible_ok else []
        records.append({"task_id": row["task_id"], "split": row["split"], "visible_ok": visible_ok,
                        "solved": scored[row["task_id"]]["solved"], "findings": findings})
    report = {"name": args.name, "families": {}}
    for part in ("train", "val"):
        eligible = [r for r in records if r["split"] == part and r["visible_ok"]]
        hidden_failures = sum(not r["solved"] for r in eligible)
        report[part] = {"tasks": sum(r["split"] == part for r in records), "visible_ok": len(eligible),
                        "hidden_failures_among_visible_ok": hidden_failures}
        for family in FAMILIES:
            fired = [r for r in eligible if any(f["family"] == family for f in r["findings"])]
            caught = sum(not r["solved"] for r in fired)
            report["families"].setdefault(family, {})[part] = {
                "fired": len(fired), "hidden_failures_caught": caught,
                "precision": round(caught / len(fired), 3) if fired else None,
                "recall": round(caught / hidden_failures, 3) if hidden_failures else None}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"report": report, "records": records}, indent=1) + "\n")
    print(json.dumps(report, indent=1))


def cmd_summary(args):
    reference = None
    table = []
    for name in args.names:
        rows = {r["task_id"]: r for r in read_jsonl(OUT / "scored" / f"{name}.jsonl")
                if r["split"] == args.split and r["draw"] == 0}
        n = len(rows)
        solved, base = sum(r["solved"] for r in rows.values()), sum(r["base"] == "pass" for r in rows.values())
        entry = {"name": name, "split": args.split, "n": n, "model": next(iter(rows.values()))["model"],
                 "condition": next(iter(rows.values()))["condition"],
                 "base_pass": base, "base_pct": round(100 * base / n, 1), "base_ci95": wilson(base, n),
                 "plus_pass": solved, "plus_pct": round(100 * solved / n, 1), "plus_ci95": wilson(solved, n),
                 "mean_llm_calls": round(sum(r["calls"] for r in rows.values()) / n, 2),
                 "mean_completion_tokens": round(sum(r["completion_tokens"] for r in rows.values()) / n, 1),
                 "mean_prompt_tokens_processed": round(sum(r["prompt_tokens_processed"]
                                                           for r in rows.values()) / n, 1),
                 "mean_model_seconds": round(sum(r["model_seconds"] for r in rows.values()) / n, 2),
                 # Median is reported too: a few tasks ran while another job competed for the CPU.
                 "median_wall_seconds": round(statistics.median(r["wall_seconds"] for r in rows.values()), 2),
                 "mean_wall_seconds": round(sum(r["wall_seconds"] for r in rows.values()) / n, 2)}
        if reference is None:
            reference = rows
        elif set(rows) == set(reference):
            wins = sum(rows[t]["solved"] and not reference[t]["solved"] for t in rows)
            losses = sum(not rows[t]["solved"] and reference[t]["solved"] for t in rows)
            entry["vs_first"] = {"wins": wins, "losses": losses, "net_pct": round(100 * (wins - losses) / n, 1),
                                 "mcnemar_p": round(mcnemar_exact(wins, losses), 4)}
        table.append(entry)
        print(json.dumps(entry))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(table, indent=1) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("split").set_defaults(func=cmd_split)
    gen = sub.add_parser("generate")
    gen.add_argument("--name", required=True)
    gen.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    gen.add_argument("--split", required=True, choices=["train", "val", "test", "all"])
    gen.add_argument("--bank", type=Path, default=rulelib.BANK)
    gen.add_argument("--rule-ids", nargs="*")
    gen.add_argument("--random-subsets", action="store_true")
    gen.add_argument("--max-rules", type=int, default=6,
                     help="cap on rules shown per task; bank order is the priority order")
    gen.add_argument("--draws", type=int, default=1)
    gen.add_argument("--tasks-file")
    gen.add_argument("--max-repairs", type=int, default=2)
    gen.add_argument("--limit", type=int)
    gen.set_defaults(func=cmd_generate)
    fix = sub.add_parser("autofix")
    fix.add_argument("--name", required=True)
    fix.add_argument("--output-name", required=True)
    fix.set_defaults(func=cmd_autofix)
    first = sub.add_parser("first-attempt")
    first.add_argument("--name", required=True)
    first.add_argument("--output-name", required=True)
    first.add_argument("--condition", required=True, choices=["base", "rules"])
    first.add_argument("--fallback", help="run whose answers are used where no rule fired (same prompt)")
    first.set_defaults(func=cmd_first_attempt)
    score = sub.add_parser("score")
    score.add_argument("--name", required=True)
    score.add_argument("--diagnose", action="store_true")
    score.add_argument("--reuse", nargs="*", help="already scored runs whose identical answers need no re-run")
    score.set_defaults(func=cmd_score)
    export = sub.add_parser("export-failures")
    export.add_argument("--name", required=True)
    export.add_argument("--splits", nargs="+", default=["train"], choices=["train", "val"])
    export.add_argument("--output", required=True)
    export.set_defaults(func=cmd_export_failures)
    attribute = sub.add_parser("attribute")
    attribute.add_argument("--name", required=True)
    attribute.add_argument("--baseline", required=True)
    attribute.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    attribute.add_argument("--output", required=True)
    attribute.set_defaults(func=cmd_attribute)
    probe = sub.add_parser("probe-stats")
    probe.add_argument("--name", required=True)
    probe.add_argument("--output", required=True)
    probe.set_defaults(func=cmd_probe_stats)
    summary = sub.add_parser("summary")
    summary.add_argument("--names", nargs="+", required=True)
    summary.add_argument("--split", default="test", choices=["train", "val", "test"])
    summary.add_argument("--output")
    summary.set_defaults(func=cmd_summary)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
