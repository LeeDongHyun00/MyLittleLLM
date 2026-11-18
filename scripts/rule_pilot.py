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
from mylittlellm.rulecoder.pipeline import solve, solve_branching
from mylittlellm.rulecoder.stats import mcnemar_exact, wilson
from mylittlellm.rulecoder.tasks import ROOT, load_split, read_tasks, split_of, write_split

OUT = ROOT / "outputs/rule_pilot"
CONDITIONS = {"base", "advice", "rules", "verify", "probes", "full", "infer", "branch"}


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
    if args.condition in {"rules", "full", "branch"}:
        bank_rules = rulelib.load_rules(args.bank, args.rule_ids)
    checkers = None
    if args.condition in {"probes", "full"}:
        from mylittlellm.rulecoder.checkers import active_families, make_checkers
        checkers = make_checkers(active_families(args.bank))
    autofix = None
    if args.condition == "full":
        from mylittlellm.rulecoder.codemods import autofix
    model = server_model()
    proposer = load_proposer(args.seedlm) if args.condition == "branch" else None
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
                if args.condition == "branch":
                    result = solve_branching(tasks[task_id], rule_block=block, proposer=proposer)
                else:
                    result = solve(tasks[task_id], rule_block=block,
                                   verify=args.condition in {"verify", "probes", "full", "infer"},
                                   checkers=checkers, autofix=autofix, infer=args.condition == "infer",
                                   max_repairs=args.max_repairs)
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


def cmd_compose(args):
    """Overlay a partial run on a complete one: tasks absent from the overlay keep the base answer.

    Used when a condition changes the prompt for only some tasks (a rule fired); the other
    tasks have a byte-identical prompt, so their base answer is the answer of that condition.
    """
    target = OUT / "gen" / f"{args.output_name}.jsonl"
    if target.exists():
        raise SystemExit(f"{target} already exists")
    overlay = {(r["task_id"], r["draw"]): r for r in read_jsonl(OUT / "gen" / f"{args.overlay}.jsonl")}
    with target.open("w") as stream:
        for row in read_jsonl(OUT / "gen" / f"{args.base}.jsonl"):
            if row["split"] not in args.splits:
                continue
            chosen = dict(overlay.get((row["task_id"], row["draw"]), row))
            chosen.update(name=args.output_name, condition=args.condition,
                          derived_from=args.overlay if (row["task_id"], row["draw"]) in overlay else args.base)
            stream.write(json.dumps(chosen) + "\n")
    print(f"{args.output_name}: {len(overlay)} overlay answers on top of {args.base}")


def cmd_learn_operators(args):
    """Credit assignment for the operator rules: send hidden-test outcomes back to the operators.

    For every development task the search returns several programs that all reproduce the
    visible example. Each is run on the hidden tests; operators that took part in programs
    which generalised gain weight, operators behind coincidences lose weight. The weights
    only re-rank matches, so a wrong weight can never make the search return a non-match.
    """
    import math
    from mylittlellm.rulecoder import backinfer, scoring
    from mylittlellm.rulecoder.tasks import parse_prompt
    tasks, split = read_tasks(), load_split()
    problems = scoring.load_problems()
    oracle = scoring.groundtruth(problems)
    counts, examples = {}, 0
    for part in args.splits:
        for task_id in split[part]:
            description, asserts = parse_prompt(tasks[task_id]["prompt"])
            result = backinfer.synthesize(description, asserts, tasks[task_id]["entry_point"], weights={},
                                          keep=args.top)
            if not result:
                continue
            for match in result["alternatives"]:
                solved = scoring.score(problems[task_id], match["code"], oracle[task_id])["solved"]
                examples += 1
                for name in set(match["ops"]):
                    win, loss = counts.get(name, (0, 0))
                    counts[name] = (win + solved, loss + (not solved))
    # Smoothed log-odds, shrunk towards zero for rarely seen operators and clipped.
    weights = {name: round(max(-1.5, min(1.5, 0.5 * math.log((win + 1) / (loss + 1)))), 3)
               for name, (win, loss) in sorted(counts.items())}
    record = {"learned_from": {"splits": args.splits, "top_matches_per_task": args.top, "programs_scored": examples},
              "counts": {name: {"generalised": win, "coincidence": loss} for name, (win, loss) in sorted(counts.items())},
              "operators": weights}
    backinfer.WEIGHTS.write_text(json.dumps(record, indent=1) + "\n")
    print(f"scored {examples} programs; learned weights for {len(weights)} operators")
    for name in sorted(weights, key=weights.get)[:8] + sorted(weights, key=weights.get)[-8:]:
        print(f"  {name:28s} {weights[name]:+.2f}  {counts[name]}")


def cmd_infer(args):
    """Apply example back-inference to an existing run: no model call is made.

    Equivalent to `generate --condition infer --max-repairs 0` because the first attempt of that
    condition is the answer the source run already holds.
    """
    from mylittlellm.rulecoder.pipeline import back_infer, public_check
    from mylittlellm.rulecoder.tasks import parse_prompt
    tasks = read_tasks()
    source, target = OUT / "gen" / f"{args.name}.jsonl", OUT / "gen" / f"{args.output_name}.jsonl"
    if target.exists():
        raise SystemExit(f"{target} already exists")
    stats = {"tasks": 0, "visible_failures": 0, "program_found": 0, "confident": 0, "knowledge": 0}
    with target.open("w") as stream:
        for row in read_jsonl(source):
            if row["split"] not in args.splits:
                continue
            stats["tasks"] += 1
            task = tasks[row["task_id"]]
            asserts = parse_prompt(task["prompt"])[1]
            first = row["attempts"][0]
            # Inference is tried on the first answer; later repair attempts of the source run are
            # kept only where inference finds nothing, as in the live pipeline.
            if not public_check(first.get("raw_code", first["code"]), asserts)["ok"]:
                stats["visible_failures"] += 1
                inferred = back_infer(task, asserts, 1)
                if inferred and inferred["check"]["ok"]:
                    stats["program_found"] += 1
                    stats["confident"] += inferred["inference"]["confident"]
                    stats["knowledge"] += inferred["inference"]["knowledge"]
                    row["attempts"] = [first, inferred]
                    row["final_code"] = inferred["code"]
            row.update(name=args.output_name, condition=row["condition"] + "+infer", derived_from=args.name)
            stream.write(json.dumps(row) + "\n")
    print(args.output_name, json.dumps(stats))


def load_proposer(name):
    """The from-scratch model as a pipeline branch; PyTorch is imported only when it is asked for."""
    if not name:
        return None
    from mylittlellm.seedlm.propose import proposer
    return proposer(ROOT / "outputs/seedlm" / f"{name}.pt")


def cmd_select(args):
    """Offline form of the branch-and-select pipeline, built from runs that already exist.

    Candidates are taken in the given order (rule-conditioned answer first, plain answer next).
    The first one that passes the visible example is kept; if none does, back-inference is
    tried. Generation is greedy and seeded, so this reproduces `generate --condition branch`
    without spending the model calls a second time.
    """
    from mylittlellm.rulecoder.pipeline import back_infer, public_check, seed_infer
    from mylittlellm.rulecoder.tasks import parse_prompt
    tasks = read_tasks()
    proposer = load_proposer(args.seedlm)
    problems = oracle = None
    if args.public == "base":
        # The setting where a task's original tests are public, not just its first example.
        from mylittlellm.rulecoder import backinfer, scoring
        backinfer.MAX_MATCHES = 400     # with several examples many more candidates can be told apart
        problems = scoring.load_problems()
        oracle = scoring.groundtruth(problems)

    def check(task_id, code, asserts):
        result = public_check(code, asserts)
        if result["ok"] and problems is not None:
            ok = scoring.base_passes(problems[task_id], code, oracle[task_id])
            result = {"ok": ok, "kind": "public_tests", "message": "" if ok else "fails a public test"}
        return result
    runs = [{(r["task_id"], r["draw"]): r for r in read_jsonl(OUT / "gen" / f"{name}.jsonl")}
            for name in args.candidates]
    target = OUT / "gen" / f"{args.output_name}.jsonl"
    if target.exists():
        raise SystemExit(f"{target} already exists")
    chosen = {}
    with target.open("w") as stream:
        for key, row in runs[0].items():
            if row["split"] not in args.splits:
                continue
            task = tasks[row["task_id"]]
            asserts = parse_prompt(task["prompt"])[1]
            attempts, seen = [], set()
            for name, run in zip(args.candidates, [] if args.engine_only else runs):
                first = dict(run[key]["attempts"][0])
                code = first.get("raw_code", first["code"])
                if code in seen:
                    continue    # same prompt, same answer: not a separate branch
                seen.add(code)
                first.update(round=len(attempts), code=code, raw_code=code, source=name,
                             check=check(row["task_id"], code, asserts), hints=[])
                attempts.append(first)
                if first["check"]["ok"]:
                    break
            if not any(a["check"]["ok"] for a in attempts):
                if problems is None:
                    inferred = back_infer(task, asserts, len(attempts))
                else:
                    task_id = row["task_id"]
                    inferred = back_infer(task, asserts, len(attempts),
                                          accept=lambda code: check(task_id, code, asserts)["ok"],
                                          examples=list(zip(problems[task_id]["base_input"], oracle[task_id]["base"])))
                if inferred and inferred["check"]["ok"]:
                    attempts.append(inferred)
            if proposer and not any(a["check"]["ok"] for a in attempts):
                proposed = seed_infer(task, asserts, len(attempts), proposer)
                if proposed and proposed["check"]["ok"]:
                    attempts.append(proposed)
            if not attempts:
                # Rule engine alone and nothing found: there is no answer to give.
                attempts.append({"round": 0, "code": "", "raw_code": "", "source": "none", "hints": [],
                                 "check": {"ok": False, "kind": "none", "message": ""},
                                 "llm": {"finish_reason": "none", "prompt_tokens": 0, "completion_tokens": 0,
                                         "prompt_tokens_processed": 0, "prompt_ms": 0, "predicted_ms": 0,
                                         "elapsed_seconds": 0}})
            winner = next((a for a in attempts if a["check"]["ok"]), attempts[0])
            chosen[winner["source"]] = chosen.get(winner["source"], 0) + 1
            out = dict(row)
            out.update(name=args.output_name, condition="branch" if problems is None else "branch-public-tests",
                       final_code=winner["code"], attempts=attempts,
                       chosen=winner["source"], visible_ok=winner["check"]["ok"],
                       wall_seconds=round(sum(a["llm"]["elapsed_seconds"] for a in attempts), 3),
                       derived_from=args.candidates)
            stream.write(json.dumps(out) + "\n")
    print(args.output_name, "chosen branch counts:", json.dumps(chosen))


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
                                              "rule_ids", "fired_rule_ids", "chosen", "visible_ok")}
            inferred = [a for a in row["attempts"] if a.get("source") == "backinfer"]
            scored.update(base=result["base"], plus=result["plus"], solved=result["solved"],
                          calls=sum(a["llm"]["completion_tokens"] > 0 for a in row["attempts"]),
                          inferred=inferred[-1]["inference"] if inferred else None,
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


def cmd_breakdown(args):
    """Where the gain of a branch-and-select run comes from; counts only, so it is safe on the test split."""
    baseline = {r["task_id"]: r for r in read_jsonl(OUT / "scored" / f"{args.baseline}.jsonl")}
    bank = {r["id"]: r for r in json.loads(rulelib.BANK.read_text())["rules"]}
    rows = [r for r in read_jsonl(OUT / "scored" / f"{args.name}.jsonl") if r["split"] == args.split]
    report = {"name": args.name, "baseline": args.baseline, "split": args.split, "tasks": len(rows),
              "solved": sum(r["solved"] for r in rows),
              "baseline_solved": sum(baseline[r["task_id"]]["solved"] for r in rows), "branches": {}}

    def tally(group):
        return {"tasks": len(group), "passes_visible_example": sum(bool(r.get("visible_ok")) for r in group),
                "solved": sum(r["solved"] for r in group),
                "baseline_solved": sum(baseline[r["task_id"]]["solved"] for r in group),
                "fixed": sum(r["solved"] and not baseline[r["task_id"]]["solved"] for r in group),
                "broken": sum(not r["solved"] and baseline[r["task_id"]]["solved"] for r in group)}

    for branch in sorted({r.get("chosen") or "single" for r in rows}):
        report["branches"][branch] = tally([r for r in rows if (r.get("chosen") or "single") == branch])
    inferred = [r for r in rows if r.get("inferred")]
    report["inference"] = {
        "programs_used": len(inferred), "solved": sum(r["solved"] for r in inferred),
        "confident": {"n": sum(r["inferred"]["confident"] for r in inferred),
                      "solved": sum(r["solved"] for r in inferred if r["inferred"]["confident"])},
        "knowledge_operator": {"n": sum(r["inferred"]["knowledge"] for r in inferred),
                               "solved": sum(r["solved"] for r in inferred if r["inferred"]["knowledge"])},
        "generic_operators_only": {"n": sum(not r["inferred"]["knowledge"] for r in inferred),
                                   "solved": sum(r["solved"] for r in inferred if not r["inferred"]["knowledge"])}}
    # Rules written for a concept seen in a train failure vs. concepts the teacher added from its own knowledge.
    by_origin = {}
    for r in rows:
        origins = {bank[i]["provenance"] for i in (r.get("rule_ids") or []) if i in bank}
        if origins and "rules" in (r.get("chosen") or ""):
            key = "train_failure" if any("train_failure" in o for o in origins) else "category_only"
            by_origin.setdefault(key, []).append(r)
    report["rule_branch_by_origin"] = {k: tally(v) for k, v in by_origin.items()}
    print(json.dumps(report, indent=1))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=1) + "\n")


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
    gen.add_argument("--seedlm", help="SeedLM checkpoint name: a last branch of the branch condition")
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
    compose = sub.add_parser("compose")
    compose.add_argument("--base", required=True)
    compose.add_argument("--overlay", required=True)
    compose.add_argument("--output-name", required=True)
    compose.add_argument("--condition", required=True)
    compose.add_argument("--splits", nargs="+", default=["test"], choices=["train", "val", "test"])
    compose.set_defaults(func=cmd_compose)
    learn = sub.add_parser("learn-operators")
    learn.add_argument("--splits", nargs="+", default=["train"], choices=["train", "val"])
    learn.add_argument("--top", type=int, default=6)
    learn.set_defaults(func=cmd_learn_operators)
    infer = sub.add_parser("infer")
    infer.add_argument("--name", required=True)
    infer.add_argument("--output-name", required=True)
    infer.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    infer.set_defaults(func=cmd_infer)
    select = sub.add_parser("select")
    select.add_argument("--candidates", nargs="+", required=True)
    select.add_argument("--output-name", required=True)
    select.add_argument("--splits", nargs="+", default=["test"], choices=["train", "val", "test"])
    select.add_argument("--engine-only", action="store_true",
                        help="ignore every model answer: what the rule engine solves with no model at all")
    select.add_argument("--seedlm", help="SeedLM checkpoint name: a last branch where the search finds nothing")
    select.add_argument("--public", choices=["example", "base"], default="example",
                        help="what a candidate is verified against: the one example in the prompt (the standard "
                             "setting) or all of the task's original tests (a separate, easier setting)")
    select.set_defaults(func=cmd_select)
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
    breakdown = sub.add_parser("breakdown")
    breakdown.add_argument("--name", required=True)
    breakdown.add_argument("--baseline", required=True)
    breakdown.add_argument("--split", default="test", choices=["train", "val", "test"])
    breakdown.add_argument("--output")
    breakdown.set_defaults(func=cmd_breakdown)
    summary = sub.add_parser("summary")
    summary.add_argument("--names", nargs="+", required=True)
    summary.add_argument("--split", default="test", choices=["train", "val", "test"])
    summary.add_argument("--output")
    summary.set_defaults(func=cmd_summary)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
