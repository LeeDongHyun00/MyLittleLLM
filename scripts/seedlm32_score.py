"""SeedLM-32, step 2: score every gathered candidate under coders other than the one that wrote it.

Only agreement between coders that learned differently has changed the ranking so far (SeedLM-10 to 20),
and a coder's own probability is the best single ranking signal (SeedLM-24, 25). A sketch-only candidate is
one no coder proposes, so first-choice agreement cannot speak for it; but any coder can say how likely it
would be to write that program. Each candidate in outputs/seedlm/v32/candidates_<set>.json gets its
log-probability under six fixed coders (four post-coders of two data mixes and two prefix coders of SeedLM-3's
mix), and the number of cached coders (SeedLM-21's 24) whose first verified program behaves the same on the
24 probes. A coder that cannot write a candidate's tokens gives it no score.

  python scripts/seedlm32_score.py --set teacher
"""
import argparse
import ast
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

from mylittlellm.rulecoder import consensus
from mylittlellm.rulecoder.limits import cap_memory
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import coder3, coder4, language, post
from seedlm3_train import named_in
from seedlm15_ensemble_scaling import CACHE
from seedlm32_candidates import OUT, development_tasks, teacher_tasks

SCORERS = ["s5-var", "s12-a1", "s12-a2", "s12-a3", "s18-p1", "s18-p2"]
VOTERS = ["s12-a1", "s12-a2", "s12-a3", "s12-b1", "s12-b2", "s12-b3", "s18-p1", "s18-p2", "s18-p3", "s20-p4", "s20-p5",
          "s20-p6", "s21-y1", "s21-y2", "s21-y3", "s3-pyx", "s3b-pyx", "s4-post", "s5-var", "s6-comp", "s6-graft",
          "s8-all", "s9-early", "s9-early-last"]


def load(name):
    path = ROOT / "outputs/seedlm" / f"{name}.pt"
    if torch.load(path, weights_only=False).get("kind") == "post-coder":
        net, vocabulary = coder4.load(path)
        return "post", net, vocabulary
    net, vocabulary = coder3.load(path)
    return "prefix", net, vocabulary


@torch.no_grad()
def log_prob(kind, net, vocabulary, words, named, literals, args, expected, mode, program):
    try:
        prefix = vocabulary.condition(words, literals, args, expected, named)
        if kind == "post":
            return coder4.program_log_prob(net, vocabulary, prefix, post.to_postfix(list(program)), args, literals,
                                           expected, mode)
        return coder3.program_log_prob(net, vocabulary, prefix, list(program))
    except (KeyError, ValueError, IndexError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=["teacher", "development"], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cap_memory()
    path = OUT / f"candidates_{args.set}.json"
    rows = json.loads(path.read_text())
    tasks = teacher_tasks() if args.set == "teacher" else development_tasks()
    cached = {v: json.loads((CACHE / f"{v}.json").read_text())[args.set] for v in VOTERS}
    started = time.time()
    for name in SCORERS:
        kind, net, vocabulary = load(name)
        for index, (row, (case, description, _, _, _)) in enumerate(zip(rows, tasks)):
            if row is None:
                continue
            words, named = language.words(description), named_in(description)
            items = row["candidates"] + ([row["engine"]] if row["engine"] and row["engine"].get("program") else [])
            for item in items:
                item.setdefault("scores", {})[name] = log_prob(
                    kind, net, vocabulary, words, named, case["literals"], case["args"], case["expected"], case["mode"],
                    item["program"])
        print(name, round(time.time() - started), flush=True)
    for index, row in enumerate(rows):
        if row is None:
            continue
        firsts = [cached[v][index] for v in VOTERS if cached[v][index] is not None]
        items = row["candidates"] + ([row["engine"]] if row["engine"] and row["engine"].get("behaviour") else [])
        for item in items:
            item["voters"] = sum(consensus.compare(item["behaviour"], f["behaviour"]) == "agree" for f in firsts)
    path.write_text(json.dumps(rows) + "\n")
    print("scored", len(rows), round(time.time() - started))


if __name__ == "__main__":
    main()
