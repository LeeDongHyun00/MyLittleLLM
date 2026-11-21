"""SeedLM-25: choose among a coder's verified programs by how well each explains its own outputs. Counts only.

A program that is right is what the coder would write for the description on any example of the task,
while a wrong one that fits the visible example by accident should look less like the description's
program once it is asked to explain other inputs. For each verified program in a coder's width-32 beam
(SeedLM-24's held-out candidate files, outputs/seedlm/v24/heldout_<coder>.json), the program is run on
the first three probe inputs on which it runs, and the coder scores the same program as its answer to
each of those pseudo-examples (the description, the probe's arguments, the program's own output), as it
does for the visible example. The choice is the program with the highest sum of the four log-probabilities
(visible example and three pseudo-examples); ties keep beam order. A task whose candidates use the visible
output string ("o", "o!") keeps the coder's first program: their meaning changes with the example.

  python scripts/seedlm25_cycle.py --coders s12-a1 s12-a2 s12-a3
"""
import argparse
import ast
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

from mylittlellm.rulecoder.limits import Timeout, cap_memory, time_limit
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import coder4, post, pyx, synth

DATA = ROOT / "outputs/seedlm/v24"
PSEUDO = 3


def pseudo_examples(program, probes, literals):
    out = []
    for args in probes:
        try:
            with time_limit(0.1):
                value = pyx.execute(list(program), [a for a in ast.literal_eval(repr(args))], literals, None)
            text = repr(value)
            if value is None or len(text) > synth.MAX_OUT or ast.literal_eval(text) != value:
                continue
        except (Exception, Timeout):
            continue
        out.append((list(args), value))
        if len(out) == PSEUDO:
            break
    return out


def scores(net, vocabulary, row, candidates):
    """(visible log-prob, sum over pseudo-examples, number of pseudo-examples) per candidate."""
    literals, args, expected, probes = (ast.literal_eval(row[k]) for k in ("literals", "args", "expected", "probes"))
    out = []
    for candidate in candidates:
        program = candidate["program"]
        postfix = post.to_postfix(list(program))
        prefix = vocabulary.condition(row["words"], literals, args, expected, row["named"])
        visible = coder4.program_log_prob(net, vocabulary, prefix, postfix, args, literals, expected)
        total, count = 0.0, 0
        for probe_args, value in pseudo_examples(program, probes, literals):
            pseudo = vocabulary.condition(row["words"], literals, probe_args, value, row["named"])
            total += coder4.program_log_prob(net, vocabulary, pseudo, postfix, probe_args, literals, value)
            count += 1
        out.append((visible, total, count))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coders", nargs="+", required=True)
    parser.add_argument("--output", default="reports/seedlm/cycle_rerank.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    cap_memory()
    report = {"coders": {}}
    for coder in args.coders:
        net, vocabulary = coder4.load(ROOT / "outputs/seedlm" / f"{coder}.pt")
        record = json.loads((DATA / f"heldout_{coder}.json").read_text())
        entry = {}
        for set_name, rows in record.items():
            counts = {"tasks": len(rows), "first": 0, "cycle": 0, "pseudo_only": 0, "some_right": 0,
                      "kept_first_for_output_string": 0, "changed": 0, "fixed": 0, "broken": 0}
            for row in rows:
                if not row or not row["candidates"]:
                    continue
                candidates = row["candidates"]
                counts["first"] += candidates[0]["right"]
                counts["some_right"] += any(c["right"] for c in candidates)
                if any(t in ("o", "o!") for c in candidates for t in c["program"]):
                    counts["kept_first_for_output_string"] += 1
                    counts["cycle"] += candidates[0]["right"]
                    counts["pseudo_only"] += candidates[0]["right"]
                    continue
                with torch.no_grad():
                    found = scores(net, vocabulary, row, candidates)
                total = [v + s if n else float("-inf") for v, s, n in found]
                pick = max(range(len(candidates)), key=lambda i: (total[i], -i))
                mean = [s / n if n else float("-inf") for _, s, n in found]
                only = max(range(len(candidates)), key=lambda i: (mean[i], -i))
                counts["cycle"] += candidates[pick]["right"]
                counts["pseudo_only"] += candidates[only]["right"]
                counts["changed"] += pick != 0
                counts["fixed"] += candidates[pick]["right"] and not candidates[0]["right"]
                counts["broken"] += candidates[0]["right"] and not candidates[pick]["right"]
            entry[set_name] = counts
        report["coders"][coder] = entry
        print(coder, json.dumps(entry), flush=True)
    report["mean_gain"] = {s: round(statistics.mean(report["coders"][c][s]["cycle"] - report["coders"][c][s]["first"]
                                                    for c in args.coders), 2) for s in ("teacher", "development")}
    text = json.dumps(report, indent=1)
    print(text)
    (ROOT / args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
