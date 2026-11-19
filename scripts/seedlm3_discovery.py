"""How far is the pyx coder's search from the programs it could find? Discovery as the beam widens.

On two sets the coder never trained on,
  teacher  - held-out teacher-authored tasks (three known input/output pairs; the first is visible)
  beyond   - held-out development tasks the rule engine fails and a teacher solved in one line
             (judged by the task's own tests)
and at several beam widths, it counts the tasks where
  verified - some program in the beam reproduces the visible example
  first    - the first such program is correct on every known case
  oracle   - some verified program is correct
and the mean number of verified programs that are wrong (spurious). It also reports where the teacher's
own program falls: its log-probability under the coder against the best beam program's, and whether
the beam holds it. Counts and summaries only (the beyond set belongs to the held-out development fifth).
"""
import argparse
import ast
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

from mylittlellm.rulecoder import backinfer as bi
from mylittlellm.rulecoder.limits import Timeout, time_limit
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import coder3, coder4, language, post, pyx
from seedlm2_ceiling import quick_ok
from seedlm3_oneliner_check import cases_for
from seedlm3_train import named_in, oneliner_items, teacher_held, teacher_task_items


def value_of(program, args, literals, output):
    try:
        with time_limit(coder3.EXECUTION_SECONDS):
            return True, pyx.execute(program, args, literals, output)
    except (Exception, Timeout):
        return False, None


def teacher_cases():
    """(description, literals, visible example, every case, teacher tokens, judge) per held-out teacher task."""
    out = []
    for task in teacher_task_items():
        if not teacher_held(task["id"]):
            continue
        cases = []
        for text, result in zip(task["inputs"], task["outputs"]):
            try:
                cases.append((list(ast.literal_eval(text)), ast.literal_eval(result)))
            except (ValueError, SyntaxError):
                cases = []
                break
        if not cases:
            continue
        literals = bi.description_literals(task["description"])
        output = cases[0][1] if isinstance(cases[0][1], str) else None

        def judge(program, cases=cases, literals=literals, output=output):
            for args, expected in cases:
                ok, value = value_of(program, args, literals, output)
                if not ok or not bi._matches(value, {"expected": expected, "mode": "eq"}):
                    return False
            return True

        out.append({"description": task["description"], "literals": literals, "args": cases[0][0],
                    "expected": cases[0][1], "mode": "eq", "gold": task["tokens"], "judge": judge})
    return out


def beyond_cases():
    out = []
    for entry in oneliner_items():
        if entry["part"] != "holdout":
            continue
        item, example = entry["item"], entry["example"]
        output = example["expected"] if isinstance(example["expected"], str) else None
        cases = cases_for(item) or []

        def judge(program, item=item, example=example, literals=entry["literals"], output=output, cases=cases):
            code = pyx.to_code(item["entry_point"], len(example["args"]), program, literals, output)
            return quick_ok(code, item["entry_point"], cases)

        out.append({"description": item["description"], "literals": entry["literals"], "args": example["args"],
                    "expected": example["expected"], "mode": example["mode"], "gold": entry["tokens"],
                    "judge": judge})
    return out


class Prefix:
    """SeedLM-3's coder: prefix pyx programs."""

    def __init__(self, net, vocabulary):
        self.net, self.vocabulary, self.grammar = net, vocabulary, coder3.Grammar(vocabulary)

    def search(self, prefix, task, width):
        found = coder3.wide_beam_search(self.net, self.vocabulary, self.grammar, prefix, len(task["args"]),
                                        len(task["literals"]), isinstance(task["expected"], str), width)
        return found

    def program(self, found):
        return found

    def gold(self, task):
        return list(task["gold"])

    def log_prob(self, prefix, task):
        return coder3.program_log_prob(self.net, self.vocabulary, prefix, list(task["gold"]))


class Postfix(Prefix):
    """SeedLM-4's coder: annotated postfix programs (its programs are turned back into prefix pyx to be judged)."""

    def __init__(self, net, vocabulary):
        self.net, self.vocabulary, self.grammar = net, vocabulary, coder4.Grammar(vocabulary)

    def search(self, prefix, task, width):
        found = coder4.beam_search(self.net, self.vocabulary, self.grammar, prefix, task["args"], task["literals"],
                                   task["expected"], task["mode"], width)
        return [(post.to_prefix(p), s) for p, s in found]

    def log_prob(self, prefix, task):
        return coder4.program_log_prob(self.net, self.vocabulary, prefix, post.to_postfix(list(task["gold"])),
                                       task["args"], task["literals"], task["expected"], task["mode"])


def measure(coder, vocabulary, tasks, widths):
    result = {w: {"verified": 0, "first": 0, "oracle": 0, "spurious": [], "gold_in_beam": 0, "seconds": 0.0}
              for w in widths}
    gaps, paired = [], []      # paired: one gap (or None) per task, in the sets' fixed order
    for task in tasks:
        prefix = vocabulary.condition(language.words(task["description"]), task["literals"], task["args"],
                                      task["expected"], named_in(task["description"]))
        string_output = isinstance(task["expected"], str)
        output = task["expected"] if string_output else None
        best = None
        for width in widths:
            started = time.monotonic()
            found = coder.search(prefix, task, width)
            result[width]["seconds"] += time.monotonic() - started
            if found and best is None:
                best = found[0][1]
            verified = [p for p, _ in found if coder3.reproduces(p, task["args"], task["literals"], task["expected"],
                                                                   task["mode"])]
            verdicts = [task["judge"](p) for p in verified]
            entry = result[width]
            entry["verified"] += bool(verified)
            entry["first"] += bool(verdicts) and verdicts[0]
            entry["oracle"] += any(verdicts)
            entry["spurious"].append(sum(not v for v in verdicts))
            entry["gold_in_beam"] += any(p == list(task["gold"]) for p, _ in found)
        try:
            gold = coder.log_prob(prefix, task)
        except (KeyError, ValueError):
            gold = None
        if gold is not None and best is not None:
            gaps.append(best - gold)
        paired.append(round(best - gold, 3) if gold is not None and best is not None else None)
    for width in widths:
        entry = result[width]
        entry["spurious"] = round(statistics.mean(entry["spurious"]), 2) if entry["spurious"] else 0
        entry["seconds"] = round(entry["seconds"], 1)
    quantiles = statistics.quantiles(gaps, n=4) if len(gaps) >= 4 else []
    return {"tasks": len(tasks), "by_width": result, "paired_gaps": paired,
            "gold_gap_nats": {"scored": len(gaps), "quartiles": [round(q, 2) for q in quantiles],
                              "within_5": sum(g <= 5 for g in gaps), "within_10": sum(g <= 10 for g in gaps)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--widths", type=int, nargs="+", default=[8, 32, 128, 512])
    parser.add_argument("--sets", nargs="+", default=["teacher", "beyond"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    path = ROOT / "outputs/seedlm" / f"{args.model}.pt"
    kind = torch.load(path).get("kind")
    if kind == "post-coder":
        net, vocabulary = coder4.load(path)
        coder = Postfix(net, vocabulary)
    else:
        net, vocabulary = coder3.load(path)
        coder = Prefix(net, vocabulary)
    report = {"model": args.model, "widths": args.widths}
    for name in args.sets:
        tasks = (teacher_cases if name == "teacher" else beyond_cases)()[:args.limit]
        report[name] = measure(coder, vocabulary, tasks, args.widths)
        # Per-task gaps let two coders be compared task by task; they stay local (outputs/ is not
        # committed) because the beyond set belongs to the held-out development fifth.
        local = ROOT / "outputs/seedlm/verdicts"
        local.mkdir(parents=True, exist_ok=True)
        (local / f"discovery3_{args.model}_{name}.json").write_text(json.dumps(report[name].pop("paired_gaps")) + "\n")
        print(name, json.dumps(report[name]), flush=True)
    suffix = "" if args.limit is None else f"_limit{args.limit}"
    (ROOT / "reports/seedlm" / f"discovery3_{args.model}{suffix}.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
