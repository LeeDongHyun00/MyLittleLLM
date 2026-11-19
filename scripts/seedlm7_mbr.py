"""Does choosing by behaviour on probe inputs beat taking the first verified program? Counts only.

For each task of the held-out teacher tasks and the held-out engine failures with a teacher
one-liner (the sets of scripts/seedlm3_discovery.py), the coder's beam is searched at each width;
among the programs that reproduce the visible example it counts the tasks where
  first   the first (most probable) verified program is correct on every known case
  mbr     the program chosen by execution-based minimum Bayes risk on probe inputs is
          (mylittlellm/seedlm/select.py)
  oracle  some verified program is
The beyond set belongs to the held-out development fifth: counts only.

  python scripts/seedlm7_mbr.py --model s5-var [--widths 32 128]
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

from mylittlellm.rulecoder.limits import cap_memory
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import coder3, coder4, language, post, select
from seedlm3_discovery import beyond_cases, teacher_cases
from seedlm3_train import named_in


def measure(net, vocabulary, grammar, tasks, widths):
    result = {w: {"verified": 0, "first": 0, "mbr": 0, "oracle": 0, "changed": 0, "seconds": 0.0} for w in widths}
    for task in tasks:
        prefix = vocabulary.condition(language.words(task["description"]), task["literals"], task["args"],
                                      task["expected"], named_in(task["description"]))
        output = task["expected"] if isinstance(task["expected"], str) else None
        for width in widths:
            started = time.monotonic()
            beam = coder4.beam_search(net, vocabulary, grammar, prefix, task["args"], task["literals"],
                                      task["expected"], task["mode"], width)
            verified = [(post.to_prefix(p), s) for p, s in beam
                        if coder4.reproduces(p, task["args"], task["literals"], task["expected"], task["mode"])]
            entry = result[width]
            if verified:
                programs, scores = [p for p, _ in verified], [s for _, s in verified]
                pick = select.mbr_pick(programs, scores, task["args"], task["literals"], output)
                verdicts = [task["judge"](p) for p in programs]
                entry["verified"] += 1
                entry["first"] += verdicts[0]
                entry["mbr"] += verdicts[pick]
                entry["oracle"] += any(verdicts)
                entry["changed"] += pick != 0
            entry["seconds"] += time.monotonic() - started
    for entry in result.values():
        entry["seconds"] = round(entry["seconds"], 1)
    return {"tasks": len(tasks), "by_width": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--widths", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--sets", nargs="+", default=["teacher", "beyond"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    cap_memory()      # a program's runaway allocation fails instead of killing the run
    net, vocabulary = coder4.load(ROOT / "outputs/seedlm" / f"{args.model}.pt")
    grammar = coder4.Grammar(vocabulary)
    report = {"model": args.model, "widths": args.widths}
    for name in args.sets:
        report[name] = measure(net, vocabulary, grammar, (teacher_cases if name == "teacher" else beyond_cases)(),
                               args.widths)
        print(name, json.dumps(report[name]), flush=True)
    (ROOT / "reports/seedlm" / f"mbr_{args.model}.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
