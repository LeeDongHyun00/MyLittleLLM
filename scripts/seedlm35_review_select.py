"""SeedLM-35: rerank SeedLM-32's candidates with the pyx reviewer, P(description | program, example). Counts only.

  score    adds each reviewer's log-probability of the task's words to every candidate (and the engine's
           program) in outputs/seedlm/v32/candidates_<set>.json, under item["review"][<reviewer>]
  explore  every reviewer and rule on one set (registered for the held-out teacher tasks only)
  measure  one reviewer and rule (the registered measure on the development fifth)

Rules: "review" (the reviewer alone) and "coder_reviewer" (the six-coder mean of SeedLM-32 plus lambda times
the reviewer). Each is reported alone (candidates only) and as the system (the engine where it is confident,
else the rule over the candidates and the engine's program).

  python scripts/seedlm35_review_select.py score --set teacher --reviewers r35-a-m10 r35-a-m20 r35-a-m30 r35-a
  python scripts/seedlm35_review_select.py explore --set teacher --output reports/seedlm/review35_teacher.json
  python scripts/seedlm35_review_select.py measure --set development --reviewer r35-a --rule coder_reviewer --lam 1
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylittlellm.rulecoder.tasks import ROOT
from seedlm32_candidates import OUT
from seedlm32_select import ensemble

LAMBDAS = [0.25, 0.5, 1.0, 2.0]
NONE = float("-inf")


def load_reviewer(name):
    import torch
    from mylittlellm.seedlm import coder4, model as seedmodel
    checkpoint = torch.load(ROOT / "outputs/seedlm" / f"{name}.pt", weights_only=False)
    net = seedmodel.SeedLM(seedmodel.Config(**checkpoint["config"]))
    net.load_state_dict(checkpoint["state"])
    net.eval()
    return net, coder4.Vocabulary(checkpoint["vocabulary"]["words"], checkpoint["vocabulary"]["operators"])


def review_scores(net, vocabulary, programs, literals, args, expected, words, batch_size=32):
    """log P(words | program, example) per program; None for a program with a symbol outside the vocabulary."""
    import torch
    from mylittlellm.seedlm import data2, review
    out, rows = [None] * len(programs), []
    for i, program in enumerate(programs):
        try:
            rows.append((i, review.encode(vocabulary, list(program), literals, args, expected, words)))
        except (KeyError, ValueError):
            continue
    with torch.no_grad():
        for begin in range(0, len(rows), batch_size):
            chunk = rows[begin:begin + batch_size]
            width = max(len(tokens) for _, (tokens, _) in chunk)
            batch = torch.full((len(chunk), width), data2.PAD, dtype=torch.long)
            for j, (_, (tokens, _)) in enumerate(chunk):
                batch[j, :len(tokens)] = torch.tensor(tokens)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                logits = net(batch)
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            for j, (i, (tokens, start)) in enumerate(chunk):
                positions = torch.arange(start, len(tokens))
                out[i] = float(log_probs[j, positions - 1, torch.tensor(tokens[start:])].sum())
    return out


def score(set_name, reviewers):
    import torch
    from mylittlellm.seedlm import language
    from seedlm32_candidates import development_tasks, teacher_tasks
    torch.set_num_threads(4)
    path = OUT / f"candidates_{set_name}.json"
    rows = json.loads(path.read_text())
    tasks = teacher_tasks() if set_name == "teacher" else development_tasks()
    for name in reviewers:
        net, vocabulary = load_reviewer(name)
        for row, (case, description, _, _, _) in zip(rows, tasks):
            if row is None:
                continue
            items = row["candidates"] + ([row["engine"]] if row["engine"] and row["engine"].get("program") else [])
            values = review_scores(net, vocabulary, [i["program"] for i in items], case["literals"], case["args"],
                                   case["expected"], language.words(description))
            for item, value in zip(items, values):
                item.setdefault("review", {})[name] = value
        print(name, "scored", flush=True)
    path.write_text(json.dumps(rows) + "\n")


def value(item, reviewer, rule, lam):
    r = (item.get("review") or {}).get(reviewer)
    if r is None:
        return NONE
    return r if rule == "review" else ensemble(item) + lam * r


def counts(rows, reviewer, rule, lam):
    alone = system = fixed = broken = 0
    for r in rows:
        pick = max(r["candidates"], key=lambda c: value(c, reviewer, rule, lam)) if r["candidates"] else None
        alone += bool(pick and pick["right"])
        engine = r["engine"]
        if engine and engine["confident"]:
            answer = engine["right"]
        else:
            pool = r["candidates"] + ([engine] if engine and engine.get("program") and "behaviour" in engine else [])
            choice = max(pool, key=lambda c: value(c, reviewer, rule, lam)) if pool else None
            answer = bool(choice["right"]) if choice else bool(engine and engine["right"])
        system += answer
        fixed += answer and not (engine and engine["right"])
        broken += bool(engine and engine["right"]) and not answer
    return {"alone": alone, "system": system, "system_versus_engine": [fixed, broken]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["score", "explore", "measure"])
    parser.add_argument("--set", choices=["teacher", "development"], required=True)
    parser.add_argument("--reviewers", nargs="+")
    parser.add_argument("--reviewer")
    parser.add_argument("--rule", choices=["review", "coder_reviewer"])
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.command == "score":
        score(args.set, args.reviewers)
        return
    rows = [r for r in json.loads((OUT / f"candidates_{args.set}.json").read_text()) if r is not None]
    report = {"set": args.set, "tasks": len(rows)}
    if args.command == "measure":
        report.update({"reviewer": args.reviewer, "rule": args.rule, "lambda": args.lam,
                       **counts(rows, args.reviewer, args.rule, args.lam)})
    else:
        names = sorted({n for r in rows for c in r["candidates"] for n in (c.get("review") or {})})
        report["results"] = []
        for name in names:
            report["results"].append({"reviewer": name, "rule": "review", **counts(rows, name, "review", 0.0)})
            for lam in LAMBDAS:
                report["results"].append({"reviewer": name, "rule": "coder_reviewer", "lambda": lam,
                                          **counts(rows, name, "coder_reviewer", lam)})
    text = json.dumps(report, indent=1)
    print(text)
    if args.output:
        (ROOT / args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
