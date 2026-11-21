"""Candidates for SeedLM-24's verifier: every program in a coder's beam that reproduces the visible example,
with what it does on probe inputs and whether it is right.

  train    - training tasks with three known cases, the first visible: the training parts of the teacher
             tasks and variants, of SeedLM-6's composed tasks, and a fresh draw of generated programs
             (pysynth, seed 24). Right means right on every case. Written to outputs/seedlm/v24/train_<coder>.json
  heldout  - the held-out teacher tasks (right on every case) and the held-out development fifth (passes the
             task's own tests), at beam width 32, for the rerank measure. Per-task data stays local:
             outputs/seedlm/v24/heldout_<coder>.json

  python scripts/seedlm24_verifier_data.py train --model s12-a1 --part 0 --parts 2
  python scripts/seedlm24_verifier_data.py heldout --model s12-a3
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

from mylittlellm.rulecoder import backinfer as bi, consensus
from mylittlellm.rulecoder.limits import cap_memory
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import coder4, language, post, select, verifier
from seedlm3_train import named_in
from seedlm23_mine import pysynth_tasks, right, teacher_tasks

OUT = ROOT / "outputs/seedlm/v24"
MAX_CANDIDATES = 8


def candidates(net, vocabulary, grammar, words, named, literals, args, expected, mode, judge, width):
    """Verified programs in beam order: {"program", "log_prob", "right", "behaviour"}."""
    prefix = vocabulary.condition(words, literals, args, expected, named)
    probes = consensus.probe_inputs(list(args), verifier.PROBES)
    output = expected if isinstance(expected, str) else None
    out = []
    for postfix, score in coder4.beam_search(net, vocabulary, grammar, prefix, args, literals, expected, mode, width):
        if not coder4.reproduces(postfix, args, literals, expected, mode):
            continue
        program = post.to_prefix(postfix)
        out.append({"program": program, "log_prob": round(float(score), 4), "right": bool(judge(program)),
                    "behaviour": select.behaviour(program, probes, literals, output)})
        if len(out) >= MAX_CANDIDATES:
            break
    return out, probes


def in_part(task_id, part, parts):
    return int(hashlib.sha256(("verifier:" + task_id).encode()).hexdigest(), 16) % parts == part


def train_mode(args, net, vocabulary, grammar):
    describer = language.Describer(bi.OPERATORS, language.load_phrases())
    tasks = teacher_tasks("teacher") + teacher_tasks("composed") + \
        [dict(t, source="pysynth", id="q" + t["id"]) for t in pysynth_tasks(args.pysynth, describer, seed=24)]
    tasks = [t for t in tasks if in_part(t["id"], args.part, args.parts)]
    random.Random(0).shuffle(tasks)
    rows, started = [], time.time()
    for index, task in enumerate(tasks):
        visible_args, visible_expected = task["cases"][0]
        try:
            found, probes = candidates(net, vocabulary, grammar, task["words"], task["named"], task["literals"],
                                       visible_args, visible_expected, "eq",
                                       lambda p, t=task: right(p, t["cases"], t["literals"]), args.width)
        except (KeyError, ValueError):
            continue
        if found:
            rows.append({"id": task["id"], "source": task["source"], "words": task["words"], "named": task["named"],
                         "literals": repr(task["literals"]), "args": repr(visible_args),
                         "expected": repr(visible_expected), "probes": repr(probes), "candidates": found})
        if index % 200 == 0:
            print(index, len(tasks), len(rows), round(time.time() - started), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"train_{args.model}_part{args.part}.json").write_text(json.dumps(rows) + "\n")
    count = sum(len(r["candidates"]) for r in rows)
    print(json.dumps({"model": args.model, "part": args.part, "tasks": len(tasks), "with_candidates": len(rows),
                      "candidates": count, "right": sum(c["right"] for r in rows for c in r["candidates"]),
                      "seconds": round(time.time() - started)}))


def heldout_mode(args, net, vocabulary, grammar):
    from seedlm3_discovery import teacher_cases
    from seedlm15_ensemble_scaling import development_cases
    record = {}
    for set_name, cases in (("teacher", teacher_cases()), ("development", development_cases())):
        rows = []
        for case in cases:
            if case is None:
                rows.append(None)
                continue
            found, probes = candidates(net, vocabulary, grammar, language.words(case["description"]),
                                       named_in(case["description"]), case["literals"], case["args"],
                                       case["expected"], case["mode"], case["judge"], 32)
            rows.append({"words": language.words(case["description"]), "named": named_in(case["description"]),
                         "literals": repr(case["literals"]), "args": repr(case["args"]),
                         "expected": repr(case["expected"]), "probes": repr(probes), "candidates": found})
        record[set_name] = rows
        print(set_name, len(rows), "first right", sum(bool(r and r["candidates"] and r["candidates"][0]["right"]) for r in rows),
              "some right", sum(bool(r and any(c["right"] for c in r["candidates"])) for r in rows), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"heldout_{args.model}.json").write_text(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "heldout"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--pysynth", type=int, default=1000)
    parser.add_argument("--width", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cap_memory()
    net, vocabulary = coder4.load(ROOT / "outputs/seedlm" / f"{args.model}.pt")
    grammar = coder4.Grammar(vocabulary)
    (train_mode if args.mode == "train" else heldout_mode)(args, net, vocabulary, grammar)


if __name__ == "__main__":
    main()
