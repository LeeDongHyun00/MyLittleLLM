"""SeedLM-2 on benchmark tasks: alone, as a re-ranker of the engine's matches, and as a gate opener.

Methods, each producing at most one program per task (verified on the visible example):
  engine : the shipped rule engine (keyword gates, hand-set ranking);
  model  : SeedLM-2's first proposal that reproduces the example (beam search);
  rerank : the engine's matches plus the model's proposals, ordered by the model's log-probability;
  open   : as rerank, but the search also treats every operator in the model's beam as named,
           so a gate the description's wording missed can open.
Each method is also slotted into the stored branch-and-select pipelines in place of the engine
(used only where the student's answers fail the visible example). The test split gets counts only.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from mylittlellm.rulecoder import backinfer as bi, scoring
from mylittlellm.rulecoder.pipeline import public_check
from mylittlellm.rulecoder.tasks import ROOT, load_split, parse_prompt, read_tasks
from mylittlellm.seedlm import data2, language, propose2, review

OUT = ROOT / "outputs/rule_pilot"
PIPELINES = {"val": ["s15-branch-dev-r3"], "test": ["s15-branch-test-r3", "s05-branch-test-r3"], "train": []}


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def methods_for(task, net, vocabulary, arities, width, keep, reviewer=None):
    """Each method's program for one task; with `reviewer` (net, vocabulary), also SeedLM-3's
    reviewer methods (reports/seedlm/seedlm3_review_protocol.json)."""
    description, asserts = parse_prompt(task["prompt"])
    example = bi.parse_example(asserts, task["entry_point"])
    if example is None or not example["args"] or len(example["args"]) > data2.MAX_ARGS:
        return {}
    words = language.words(description)
    literals = bi.description_literals(description)
    named = [k for k, op in bi.OPERATORS.items() if op.named_in(description.lower())]
    prefix = vocabulary.condition(words, literals, example["args"], example["expected"], named)
    out = {}
    found = bi.synthesize(description, asserts, task["entry_point"], weights={}, keep=keep)
    engine = (found.get("alternatives") or [found]) if found else []
    out["engine"] = engine[0]["program"] if engine else None
    beam = propose2.beam_search(net, vocabulary, prefix, len(example["args"]), len(literals), width, arities=arities)
    verified = [p for p, _ in beam if propose2.reproduces(p, example["args"], literals, example["expected"], example["mode"])]
    out["model"] = verified[0] if verified else None

    def best_by_model(programs):
        scored = []
        for program in programs:
            if program and "?" not in program:
                try:
                    scored.append((propose2.program_log_prob(net, vocabulary, prefix, program), program))
                except KeyError:
                    continue
        return max(scored, key=lambda s: s[0])[1] if scored else None

    out["rerank"] = best_by_model([m["program"] for m in engine] + verified)
    beam_ops = sorted({t for p, _ in beam for t in p if "/" in t})
    opened = bi.synthesize(description, asserts, task["entry_point"], weights={}, keep=keep, extra_named=beam_ops)
    opened = (opened.get("alternatives") or [opened]) if opened else []
    out["open"] = best_by_model([m["program"] for m in opened] + verified)
    # Weaker interventions, for when the model is less reliable than the engine's own ranking:
    #   gate : the model only opens keyword gates; the engine still ranks;
    #   tie  : the model only breaks ties among the engine's equally scored best programs;
    #   fill : the model answers only where the engine has no program at all.
    out["gate"] = opened[0]["program"] if opened else None

    def primary(match):
        return round(2.0 * match["support"] + match["learned"] - 0.3 * match["size"], 6)

    top = [m for m in engine if engine and primary(m) == primary(engine[0])]
    out["tie"] = best_by_model([m["program"] for m in top]) if top else None
    out["fill"] = out["engine"] if engine else out["model"]
    if reviewer is not None:
        review_net, review_vocabulary = reviewer
        pool = []
        for program in [m["program"] for m in engine] + verified:
            if program and "?" not in program and program not in pool:
                pool.append(program)
        judged = review.scores(review_net, review_vocabulary, pool, literals, example["args"], example["expected"], words)
        scored = [(s, p) for s, p in zip(judged, pool) if s is not None]
        out["review"] = max(scored, key=lambda x: x[0])[1] if scored else None
        tied = [m["program"] for m in top]
        among = [(s, p) for s, p in scored if p in tied]
        out["review_tie"] = max(among, key=lambda x: x[0])[1] if among else out["engine"]
        both = []
        for s, p in scored:
            try:
                both.append((s + propose2.program_log_prob(net, vocabulary, prefix, p), p))
            except KeyError:
                continue
        out["coder_reviewer"] = max(both, key=lambda x: x[0])[1] if both else None
    return {name: (program, literals) for name, program in out.items()}


def load_reviewer(name):
    if not name:
        return None
    return propose2.load(ROOT / "outputs/seedlm" / f"{name}.pt")


def code_for(task, program, literals):
    description, asserts = parse_prompt(task["prompt"])
    example = bi.parse_example(asserts, task["entry_point"])
    expr, helpers = data2.to_expression(program, literals)
    return bi.to_code(task["entry_point"], len(example["args"]), {"expr": expr, "helpers": helpers})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--splits", nargs="+", default=["val"])
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--keep", type=int, default=40)
    parser.add_argument("--reviewer", help="SeedLM-3 reviewer checkpoint: adds review, review_tie, coder_reviewer")
    args = parser.parse_args()
    torch.set_num_threads(4)
    net, vocabulary = propose2.load(ROOT / "outputs/seedlm" / f"{args.model}.pt")
    arities = propose2.arity_table(vocabulary)
    reviewer = load_reviewer(args.reviewer)
    tasks, split = read_tasks(), load_split()
    problems = scoring.load_problems()
    oracle = scoring.groundtruth(problems)
    verdicts = {}

    def judge(task_id, code):
        if (task_id, code) not in verdicts:
            verdicts[(task_id, code)] = scoring.score(problems[task_id], code, oracle[task_id])["solved"]
        return verdicts[(task_id, code)]

    report = {"model": args.model, "settings": vars(args)}
    for part in args.splits:
        started = time.monotonic()
        names = ["engine", "model", "rerank", "open", "gate", "tie", "fill"]
        if reviewer is not None:
            names += ["review", "review_tie", "coder_reviewer"]
        alone = {n: 0 for n in names}
        codes = {}
        for task_id in split[part]:
            task = tasks[task_id]
            chosen = methods_for(task, net, vocabulary, arities, args.width, args.keep, reviewer)
            codes[task_id] = {}
            for name in names:
                program, literals = chosen.get(name, (None, ()))
                code = code_for(task, program, literals) if program else None
                codes[task_id][name] = code
                alone[name] += bool(code) and judge(task_id, code)
        entry = {"tasks": len(split[part]), "alone": alone, "pipelines": {}}
        for run in PIPELINES[part]:
            rows = {r["task_id"]: r for r in read_jsonl(OUT / "gen" / f"{run}.jsonl") if r["split"] == part}
            stored = {r["task_id"]: r for r in read_jsonl(OUT / "scored" / f"{run}.jsonl") if r["split"] == part}
            result = {"stored": sum(r["solved"] for r in stored.values())}
            for name in names:
                solved = 0
                for task_id, row in rows.items():
                    keep_model_answer = row["visible_ok"] and row["chosen"] != "backinfer"
                    if keep_model_answer:
                        solved += stored[task_id]["solved"]
                        continue
                    code = codes[task_id][name]
                    _, asserts = parse_prompt(tasks[task_id]["prompt"])
                    if code and public_check(code, asserts)["ok"]:
                        solved += judge(task_id, code)
                    else:
                        # Nothing reproduces the example: the first model answer stays, as in the pipeline.
                        first = row["attempts"][0]
                        solved += judge(task_id, first.get("raw_code", first["code"])) if first.get("code") else 0
                result[name] = solved
            entry["pipelines"][run] = result
        entry["seconds"] = round(time.monotonic() - started)
        report[part] = entry
        print(part, json.dumps(entry), flush=True)
    suffix = f"_{args.reviewer}" if args.reviewer else ""
    (ROOT / "reports/seedlm" / f"tasks2_{args.model}{suffix}_{'_'.join(args.splits)}.json").write_text(
        json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
