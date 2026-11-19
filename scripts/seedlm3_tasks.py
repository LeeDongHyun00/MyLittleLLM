"""SeedLM-3's pyx coder on tasks: held-out development tasks, then (after the gate) validation and test.

Methods (reports/seedlm/seedlm3_pyx_protocol.json), each giving at most one program per task, verified
on the visible example:
  engine        the shipped rule engine
  pyx_model     the coder's first beam proposal that reproduces the example
  pyx_fill      the engine's program, else pyx_model
  pyx_tie       the coder breaks ties among the engine's best-scored programs
  pyx_tie_fill  pyx_tie, else pyx_model
  pyx_override  pyx_tie_fill, but pyx_model replaces an engine program the engine marks as not confident

  holdout --model NAME            the extra pool's held-out fifth, judged by each task's three asserts
  splits --model NAME --splits S  MBPP+ splits (EvalPlus scoring), alone and in the stored pipelines
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

from mylittlellm.rulecoder import backinfer as bi, scoring
from mylittlellm.rulecoder.pipeline import public_check
from mylittlellm.rulecoder.sandbox import run_public_asserts
from mylittlellm.rulecoder.tasks import ROOT, load_split, parse_prompt, read_tasks
from mylittlellm.seedlm import coder3, data2, language, pyx
from seedlm2_real import held_out
from seedlm2_tasks import PIPELINES, read_jsonl

OUT = ROOT / "outputs/rule_pilot"
BANK = ROOT / "outputs/rule_pilot/teacher/round3"
NAMES = ["engine", "pyx_model", "pyx_fill", "pyx_tie", "pyx_tie_fill", "pyx_override", "pyx_rerank", "pyx_margin"]
MARGIN = 2.0     # nats by which a coder-only program must beat the engine's choice (protocol addendum 2)


def pyx_methods(task, net, vocabulary, grammar, width=32, keep=40):
    """Code (or None) for each method on one task."""
    description, asserts = parse_prompt(task["prompt"])
    example = bi.parse_example(asserts, task["entry_point"])
    out = {name: None for name in NAMES}
    if example is None or not example["args"] or len(example["args"]) > data2.MAX_ARGS:
        return out
    words = language.words(description)
    literals = bi.description_literals(description)
    named = [k for k, op in bi.OPERATORS.items() if op.named_in(description.lower())]
    prefix = vocabulary.condition(words, literals, example["args"], example["expected"], named)
    found = bi.synthesize(description, asserts, task["entry_point"], weights={}, keep=keep)
    engine = (found.get("alternatives") or [found]) if found else []
    out["engine"] = engine[0]["code"] if engine else None
    output = example["expected"] if isinstance(example["expected"], str) else None
    beam = coder3.beam_search(net, vocabulary, grammar, prefix, len(example["args"]), len(literals), output is not None,
                              width)
    model = next((p for p, _ in beam if coder3.reproduces(p, example["args"], literals, example["expected"],
                                                          example["mode"])), None)
    out["pyx_model"] = pyx.to_code(task["entry_point"], len(example["args"]), model, literals, output) if model else None

    def primary(match):
        return round(2.0 * match["support"] + match["learned"] - 0.3 * match["size"], 6)

    top = [m for m in engine if primary(m) == primary(engine[0])] if engine else []
    scored = []
    for match in top:
        program = match.get("program") or []
        if program and "?" not in program:
            try:
                scored.append((coder3.program_log_prob(net, vocabulary, prefix, program), match))
            except KeyError:
                continue
    tie = max(scored, key=lambda s: s[0])[1]["code"] if scored else out["engine"]
    out["pyx_fill"] = out["engine"] or out["pyx_model"]
    out["pyx_tie"] = tie
    out["pyx_tie_fill"] = tie or out["pyx_model"]
    confident = bool(found and found.get("confident"))
    out["pyx_override"] = out["pyx_tie_fill"] if confident or not out["pyx_model"] else out["pyx_model"]
    # Full trust in the coder: every engine match and every verified beam program, by the coder's
    # log-probability; and the cautious version, where a coder-only program must win by MARGIN nats.
    pool = []
    for match in engine:
        program = match.get("program") or []
        if program and "?" not in program:
            try:
                pool.append((coder3.program_log_prob(net, vocabulary, prefix, program), match["code"], "engine"))
            except KeyError:
                continue
    for program, _ in beam:
        if coder3.reproduces(program, example["args"], literals, example["expected"], example["mode"]):
            code = pyx.to_code(task["entry_point"], len(example["args"]), program, literals, output)
            pool.append((coder3.program_log_prob(net, vocabulary, prefix, program), code, "coder"))
    best = max(pool, key=lambda s: s[0]) if pool else None
    out["pyx_rerank"] = best[1] if best else None
    tie_score = max((s[0] for s in scored), default=None)
    coder_best = max((s for s in pool if s[2] == "coder"), key=lambda s: s[0], default=None)
    if coder_best and (tie_score is None or coder_best[0] > tie_score + MARGIN):
        out["pyx_margin"] = coder_best[1]
    else:
        out["pyx_margin"] = out["pyx_tie_fill"]
    return out


def holdout(args, net, vocabulary, grammar):
    items = [i for i in json.loads((BANK / "all_tasks.json").read_text())
             if i["source"] == "extra" and held_out(i["task_id"])]
    solved = {n: 0 for n in NAMES}
    for item in items:
        task = {"prompt": '"""\n' + item["description"] + "\n" + item["visible_assert"] + '\n"""\n',
                "entry_point": item["entry_point"]}
        codes = pyx_methods(task, net, vocabulary, grammar, args.width, args.keep)
        verdicts = {}
        for name in NAMES:
            code = codes[name]
            if code not in verdicts:
                verdicts[code] = bool(code) and run_public_asserts(code, "\n".join(item["asserts"]))["status"] == "pass"
            solved[name] += verdicts[code]
    report = {"model": args.model, "tasks": len(items), "solved": solved}
    (ROOT / "reports/seedlm" / f"holdout3_{args.model}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))


def splits(args, net, vocabulary, grammar):
    tasks, split = read_tasks(), load_split()
    problems = scoring.load_problems()
    oracle = scoring.groundtruth(problems)
    verdicts = {}

    def judge(task_id, code):
        if (task_id, code) not in verdicts:
            verdicts[(task_id, code)] = bool(code) and scoring.score(problems[task_id], code, oracle[task_id])["solved"]
        return verdicts[(task_id, code)]

    report = {"model": args.model, "settings": vars(args)}
    for part in args.splits:
        started = time.monotonic()
        codes = {t: pyx_methods(tasks[t], net, vocabulary, grammar, args.width, args.keep) for t in split[part]}
        entry = {"tasks": len(split[part]),
                 "alone": {n: sum(judge(t, codes[t][n]) for t in split[part]) for n in NAMES}, "pipelines": {}}
        for run in PIPELINES[part]:
            rows = {r["task_id"]: r for r in read_jsonl(OUT / "gen" / f"{run}.jsonl") if r["split"] == part}
            stored = {r["task_id"]: r for r in read_jsonl(OUT / "scored" / f"{run}.jsonl") if r["split"] == part}
            result = {"stored": sum(r["solved"] for r in stored.values())}
            for name in NAMES:
                count = 0
                for task_id, row in rows.items():
                    if row["visible_ok"] and row["chosen"] != "backinfer":
                        count += stored[task_id]["solved"]
                        continue
                    code = codes[task_id][name]
                    _, asserts = parse_prompt(tasks[task_id]["prompt"])
                    if code and public_check(code, asserts)["ok"]:
                        count += judge(task_id, code)
                    else:
                        first = row["attempts"][0]
                        count += judge(task_id, first.get("raw_code", first["code"])) if first.get("code") else 0
                result[name] = count
            entry["pipelines"][run] = result
        entry["seconds"] = round(time.monotonic() - started)
        report[part] = entry
        print(part, json.dumps(entry), flush=True)
    (ROOT / "reports/seedlm" / f"tasks3_{args.model}_{'_'.join(args.splits)}.json").write_text(
        json.dumps(report, indent=1) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["holdout", "splits"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--splits", nargs="+", default=["val"])
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--keep", type=int, default=40)
    args = parser.parse_args()
    torch.set_num_threads(4)
    net, vocabulary = coder3.load(ROOT / "outputs/seedlm" / f"{args.model}.pt")
    grammar = coder3.Grammar(vocabulary)
    (holdout if args.command == "holdout" else splits)(args, net, vocabulary, grammar)


if __name__ == "__main__":
    main()
