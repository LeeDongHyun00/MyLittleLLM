"""How near does the coder's beam come to the teacher's program? Edit distance and a one-token repair.

The discovery diagnostic (scripts/seedlm3_discovery.py) found that on held-out engine failures the
teacher's program sits far below the beam's best in log-probability. Log-probability does not say
whether the beam holds something close to it: one rare token can cost 30 nats. This script measures
closeness directly, on the same two sets (held-out teacher tasks; held-out engine failures with a
teacher one-liner):

  distance   the smallest token edit distance (Levenshtein, prefix pyx) between the teacher's program
             and any beam program, as a histogram
  repair     every beam program is changed at one position to one of the coder's own top alternatives
             there (one forward pass per program gives them all, since a position only sees what comes
             before it); counted are the tasks where such a neighbour is correct on every known case
             (oracle), and where the first neighbour that reproduces the visible example, by the
             coder's score of the change, is correct (first)
  combined   the beam's first verified program, else the first verified repair (a method)
  surprisal  where the teacher's program loses its probability: the share of its total surprisal in
             its costliest one and three tokens, and how many of its tokens cost more than 3 nats

A beam program that already reproduces the example is left as it is. Counts only: the second set
belongs to the held-out development fifth.

  python scripts/seedlm5_nearmiss.py --model s5-var [--width 32] [--alternatives 16]
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import coder3, coder4, language, post
from seedlm3_discovery import beyond_cases, teacher_cases
from seedlm3_train import named_in

BUCKETS = ["0", "1", "2", "3", "4-5", "6+"]


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        previous, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (x != y))
    return row[-1]


def bucket(d):
    return "0" if d == 0 else "1" if d == 1 else "2" if d == 2 else "3" if d == 3 else "4-5" if d <= 5 else "6+"


@torch.no_grad()
def position_alternatives(net, vocabulary, base, prefix, program, task, k):
    """For each position of a postfix program: the token written there, its log-probability, and the
    coder's k best program tokens at that position (one forward pass, as each position sees only what
    precedes it)."""
    tokens, side = coder4.encode(vocabulary, prefix, post.annotated(program, task["args"], task["literals"],
                                                                     task["expected"], task["mode"]))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = net(torch.tensor([tokens]), notes=torch.tensor([side]))[0]
    log_probs = torch.log_softmax(logits.float(), dim=-1).masked_fill(~base, float("-inf"))
    out = []
    for i in range(len(program)):
        row = log_probs[len(prefix) + i - 1]
        top = torch.topk(row, k)
        choices = [(vocabulary.symbol(int(t)), float(s)) for s, t in zip(top.values, top.indices)
                   if math.isfinite(float(s))]
        out.append((float(row[tokens[len(prefix) + i]]), choices))
    return out


@torch.no_grad()
def gold_surprisal(net, vocabulary, prefix, task):
    """-log P of each token of the teacher's program (postfix, with its own notes), or None."""
    try:
        postfix = post.to_postfix(list(task["gold"]))
        tokens, side = coder4.encode(vocabulary, prefix, post.annotated(postfix, task["args"], task["literals"],
                                                                         task["expected"], task["mode"]))
    except (KeyError, ValueError):
        return None
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = net(torch.tensor([tokens]), notes=torch.tensor([side]))[0]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return [-float(log_probs[i - 1, tokens[i]]) for i in range(len(prefix), len(tokens))]


def concentration(costs):
    """Share of the program's surprisal in its single and three costliest tokens; tokens above 3 nats."""
    total = sum(costs)
    ranked = sorted(costs, reverse=True)
    return (ranked[0] / total if total > 0 else 1.0, sum(ranked[:3]) / total if total > 0 else 1.0,
            sum(c > 3 for c in costs))


def repairs(net, vocabulary, base, prefix, beam, task, k):
    """One-token changes of every beam program, best change first, as prefix pyx programs."""
    found = []
    for program, score in beam:
        options = position_alternatives(net, vocabulary, base, prefix, program, task, k)
        for i, (current, choices) in enumerate(options):
            for token, logp in choices:
                if token == program[i]:
                    continue
                changed = program[:i] + [token] + program[i + 1:]
                try:
                    candidate = post.to_prefix(changed)
                except Exception:
                    continue
                # A change is ranked by the beam program's score plus what the new token gains or loses
                # (later tokens are not rescored: their notes would change with the value).
                found.append((score + logp - current, candidate))
    found.sort(key=lambda pair: -pair[0])
    seen, out = set(), []
    for _, candidate in found:
        key = tuple(candidate)
        if key not in seen:
            seen.add(key)
            out.append(candidate)
    return out


def measure(net, vocabulary, grammar, tasks, width, k):
    result = {"tasks": len(tasks), "distance": {b: 0 for b in BUCKETS}, "beam_first": 0, "beam_oracle": 0,
              "repair_first": 0, "repair_oracle": 0, "repair_new": 0, "combined_first": 0,
              "seconds": 0.0}
    top1, top3, hard = [], [], []
    for task in tasks:
        started = time.monotonic()
        prefix = vocabulary.condition(language.words(task["description"]), task["literals"], task["args"],
                                      task["expected"], named_in(task["description"]))
        beam = coder4.beam_search(net, vocabulary, grammar, prefix, task["args"], task["literals"], task["expected"],
                                  task["mode"], width)
        gold = list(task["gold"])
        costs = gold_surprisal(net, vocabulary, prefix, task)
        if costs:
            one, three, many = concentration(costs)
            top1.append(one)
            top3.append(three)
            hard.append(many)
        programs = [post.to_prefix(p) for p, _ in beam]
        if programs:
            result["distance"][bucket(min(edit_distance(gold, p) for p in programs))] += 1
        else:
            result["distance"]["6+"] += 1

        def reproduces(program):
            return coder3.reproduces(program, task["args"], task["literals"], task["expected"], task["mode"])

        verified = [p for p in programs if reproduces(p)]
        beam_ok = [task["judge"](p) for p in verified]
        result["beam_first"] += bool(beam_ok) and beam_ok[0]
        result["beam_oracle"] += any(beam_ok)
        unverified = [(p, s) for p, s in beam if not reproduces(post.to_prefix(p))]
        base = grammar.base_mask(len(task["args"]), len(task["literals"]), isinstance(task["expected"], str))
        neighbours = [p for p in repairs(net, vocabulary, base, prefix, unverified, task, k) if reproduces(p)]
        verdicts = [task["judge"](p) for p in neighbours]
        result["repair_first"] += bool(verdicts) and verdicts[0]
        result["repair_oracle"] += any(verdicts)
        result["repair_new"] += any(verdicts) and not any(beam_ok)
        # As a method: the beam's first verified program, else the first verified repair.
        result["combined_first"] += beam_ok[0] if beam_ok else bool(verdicts) and verdicts[0]
        result["seconds"] += time.monotonic() - started
    result["seconds"] = round(result["seconds"], 1)

    def quartiles(values):
        return [round(q, 2) for q in statistics.quantiles(values, n=4)] if len(values) >= 4 else []

    # Where the teacher's program loses its probability: in one or a few tokens, or all along it.
    result["gold_surprisal"] = {"scored": len(top1), "top1_share_quartiles": quartiles(top1),
                                "top3_share_quartiles": quartiles(top3),
                                "tokens_above_3_nats": {str(n): hard.count(n) for n in range(0, 4)} |
                                {"4+": sum(n >= 4 for n in hard)}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--alternatives", type=int, default=16)
    parser.add_argument("--sets", nargs="+", default=["teacher", "beyond"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    net, vocabulary = coder4.load(ROOT / "outputs/seedlm" / f"{args.model}.pt")
    grammar = coder4.Grammar(vocabulary)
    report = {"model": args.model, "width": args.width, "alternatives": args.alternatives}
    for name in args.sets:
        tasks = (teacher_cases if name == "teacher" else beyond_cases)()
        report[name] = measure(net, vocabulary, grammar, tasks, args.width, args.alternatives)
        print(name, json.dumps(report[name]), flush=True)
    (ROOT / "reports/seedlm" / f"nearmiss_{args.model}.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
