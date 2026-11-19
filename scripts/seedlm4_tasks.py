"""SeedLM-4's coder on tasks, with SeedLM-3's eight methods (scripts/seedlm3_tasks.py) unchanged in meaning:

  engine        the shipped rule engine
  pyx_model     the coder's first beam proposal that reproduces the visible example
  pyx_fill      the engine's program, else pyx_model
  pyx_tie       the coder breaks ties among the engine's best-scored programs
  pyx_tie_fill  pyx_tie, else pyx_model
  pyx_override  pyx_tie_fill, but pyx_model replaces an engine program the engine marks as not confident
  pyx_rerank    every engine match and every verified beam program, by the coder's log-probability
  pyx_margin    pyx_tie_fill, unless the coder's best verified program beats the engine's best tied one by 2 nats

The coder writes annotated postfix programs (mylittlellm/seedlm/post.py); an engine program is scored
by writing it in postfix and annotating it on the visible example.

  holdout --model NAME            the extra pool's held-out fifth, judged by each task's three asserts
  splits --model NAME --splits S  MBPP+ splits (EvalPlus scoring), alone and in the stored pipelines

With --mbr, two more methods choose among the coder's verified programs by their behaviour on probe
inputs (execution minimum Bayes risk, mylittlellm/seedlm/select.py) instead of taking the first:
  pyx_mbr           the chosen program
  pyx_override_mbr  pyx_override with the chosen program in place of pyx_model's
(reports get the suffix _mbr).
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

from mylittlellm.rulecoder.limits import cap_memory
from mylittlellm.rulecoder import backinfer as bi
from mylittlellm.rulecoder.tasks import ROOT, parse_prompt
from mylittlellm.seedlm import coder4, data2, language, post, pyx, select
from seedlm3_tasks import MARGIN, NAMES, holdout, splits

MBR_NAMES = ["pyx_mbr", "pyx_override_mbr"]


def post_methods(task, net, vocabulary, grammar, width=32, keep=40):
    """Code (or None) for each method on one task."""
    description, asserts = parse_prompt(task["prompt"])
    example = bi.parse_example(asserts, task["entry_point"])
    out = {name: None for name in NAMES}
    if example is None or not example["args"] or len(example["args"]) > data2.MAX_ARGS:
        return out
    args, expected, mode = example["args"], example["expected"], example["mode"]
    words = language.words(description)
    literals = bi.description_literals(description)
    named = [k for k, op in bi.OPERATORS.items() if op.named_in(description.lower())]
    prefix = vocabulary.condition(words, literals, args, expected, named)
    found = bi.synthesize(description, asserts, task["entry_point"], weights={}, keep=keep)
    engine = (found.get("alternatives") or [found]) if found else []
    out["engine"] = engine[0]["code"] if engine else None
    output = expected if isinstance(expected, str) else None
    beam = coder4.beam_search(net, vocabulary, grammar, prefix, args, literals, expected, mode, width)
    verified = [(p, s) for p, s in beam if coder4.reproduces(p, args, literals, expected, mode)]

    def code(postfix):
        return pyx.to_code(task["entry_point"], len(args), post.to_prefix(postfix), literals, output)

    def score(program):
        """The coder's log-probability of an engine program, or None if it cannot write it."""
        if not program or "?" in program:
            return None
        try:
            postfix = post.to_postfix(program)
            return coder4.program_log_prob(net, vocabulary, prefix, postfix, args, literals, expected, mode)
        except (KeyError, ValueError):
            return None

    out["pyx_model"] = code(verified[0][0]) if verified else None

    def primary(match):
        return round(2.0 * match["support"] + match["learned"] - 0.3 * match["size"], 6)

    top = [m for m in engine if primary(m) == primary(engine[0])] if engine else []
    scored = [(s, m) for m in top for s in [score(m.get("program") or [])] if s is not None]
    tie = max(scored, key=lambda s: s[0])[1]["code"] if scored else out["engine"]
    out["pyx_fill"] = out["engine"] or out["pyx_model"]
    out["pyx_tie"] = tie
    out["pyx_tie_fill"] = tie or out["pyx_model"]
    confident = bool(found and found.get("confident"))
    out["pyx_override"] = out["pyx_tie_fill"] if confident or not out["pyx_model"] else out["pyx_model"]
    # The beam's own scores are the same log-probabilities program_log_prob computes (both sum the
    # coder's log-probabilities of the program tokens with the annotations in context).
    pool = [(s, m["code"], "engine") for m in engine for s in [score(m.get("program") or [])] if s is not None]
    pool += [(s, code(p), "coder") for p, s in verified]
    best = max(pool, key=lambda s: s[0]) if pool else None
    out["pyx_rerank"] = best[1] if best else None
    tie_score = max((s[0] for s in scored), default=None)
    coder_best = max(((s, c) for s, c, source in pool if source == "coder"), key=lambda s: s[0], default=None)
    if coder_best and (tie_score is None or coder_best[0] > tie_score + MARGIN):
        out["pyx_margin"] = coder_best[1]
    else:
        out["pyx_margin"] = out["pyx_tie_fill"]
    if "pyx_mbr" in out:
        pick = select.mbr_pick([post.to_prefix(p) for p, _ in verified], [s for _, s in verified], args, literals,
                               output) if verified else None
        out["pyx_mbr"] = code(verified[pick][0]) if verified else None
        out["pyx_override_mbr"] = out["pyx_tie_fill"] if confident or not out["pyx_mbr"] else out["pyx_mbr"]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["holdout", "splits"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--splits", nargs="+", default=["val"])
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--keep", type=int, default=40)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--verdicts", action="store_true",
                        help="also keep per-task held-out verdicts locally (outputs/seedlm/verdicts)")
    parser.add_argument("--mbr", action="store_true", help="add the two execution-MBR methods")
    args = parser.parse_args()
    if args.mbr:
        NAMES.extend(MBR_NAMES)      # the list seedlm3_tasks' loops read
        args.suffix += "_mbr"
    torch.set_num_threads(4)
    cap_memory()      # a program's runaway allocation fails instead of killing the run
    net, vocabulary = coder4.load(ROOT / "outputs/seedlm" / f"{args.model}.pt")
    grammar = coder4.Grammar(vocabulary)
    if args.command == "holdout":
        holdout(args, net, vocabulary, grammar, post_methods, "holdout4")
    else:
        splits(args, net, vocabulary, grammar, post_methods, "tasks4")


if __name__ == "__main__":
    main()
