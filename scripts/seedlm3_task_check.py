"""Check teacher-authored tasks for SeedLM-3: a description, inputs, and a one-line pyx-writable solution.

Each task is {"id", "category", "description", "entry_point", "inputs": [three argument tuples as
Python literals, the first is the visible example], "expression": one Python expression over
a0..a3}. The check runs the expression on every input, and requires that
  - every input runs without error and the outputs are not all equal;
  - arguments and outputs are short enough for the model's context (SeedLM's byte limits);
  - every constant other than a small integer or a short separator is stated in the description
    (or is the visible verdict string or its conventional opposite), so it can be read;
  - the expression can be written as pyx tokens (mylittlellm/seedlm/pyx.py), at most MAX_TOKENS long.

  check FILE [--output FILE]   per-task verdicts and a summary
"""
import argparse
import ast
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mylittlellm.rulecoder import backinfer as bi
from mylittlellm.rulecoder.limits import Timeout, time_limit
from mylittlellm.seedlm import pyx, synth

MAX_TOKENS = 32


def check(task):
    """(ok, reason, details) for one authored task."""
    try:
        inputs = [ast.literal_eval(text) for text in task["inputs"]]
    except (ValueError, SyntaxError) as error:
        return False, f"inputs are not Python literals: {error}", {}
    inputs = [list(i) if isinstance(i, tuple) else None for i in inputs]
    if len(inputs) < 3 or any(i is None or not i or len(i) > 4 for i in inputs):
        return False, "need three inputs, each a tuple of 1 to 4 arguments", {}
    if len({len(i) for i in inputs}) != 1:
        return False, "inputs disagree on the number of arguments", {}
    n_args = len(inputs[0])
    outputs = []
    for args in inputs:
        if len(repr(args)) > synth.MAX_IN:
            return False, f"arguments longer than {synth.MAX_IN} characters: {args!r}"[:200], {}
        try:
            value = _plain(task["expression"], copy.deepcopy(args))
        except (Exception, Timeout) as error:
            return False, f"error on {args!r}: {type(error).__name__}: {error}"[:200], {}
        if isinstance(value, (set, frozenset, map, filter, zip)) or value is None:
            return False, f"output type {type(value).__name__} (return a list, tuple, dict, str, number or bool)", {}
        if len(repr(value)) > synth.MAX_OUT:
            return False, f"output longer than {synth.MAX_OUT} characters: {value!r}"[:200], {}
        outputs.append(value)
    if all(repr(o) == repr(outputs[0]) for o in outputs):
        return False, "every input gives the same output; pick inputs that tell programs apart", {}
    literals = bi.description_literals(task["description"])
    visible = outputs[0] if isinstance(outputs[0], str) else None
    try:
        tokens = pyx.serialize(task["expression"], n_args, literals, visible)
    except pyx.Unsupported as error:
        return False, f"not writable as pyx: {error} (state the constant in the description, or simplify)", {}
    if len(tokens) > MAX_TOKENS:
        return False, f"{len(tokens)} tokens, more than {MAX_TOKENS}", {}
    # The tokens must mean what the expression means, with literals read from the description.
    for args, value in zip(inputs, outputs):
        try:
            with time_limit(1.0):
                again = pyx.execute(tokens, copy.deepcopy(args), literals, visible)
        except (Exception, Timeout) as error:
            return False, f"pyx run failed: {type(error).__name__}: {error}"[:200], {}
        if repr(again) != repr(value):
            return False, f"pyx value differs on {args!r}", {}
    return True, "", {"tokens": tokens, "outputs": [repr(o) for o in outputs], "n_args": n_args}


def _plain(expression, args):
    scope = pyx.environment()
    scope.update({f"a{i}": v for i, v in enumerate(args)})
    with time_limit(1.0):
        return eval(expression, scope)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("check")
    run.add_argument("input")
    run.add_argument("--output")
    args = parser.parse_args()
    tasks = json.loads(Path(args.input).read_text())
    results, ids = [], set()
    for task in tasks:
        missing = [k for k in ("id", "description", "entry_point", "inputs", "expression") if not task.get(k)]
        if missing:
            ok, reason, details = False, f"missing {missing}", {}
        elif task["id"] in ids:
            ok, reason, details = False, "duplicate id", {}
        else:
            ok, reason, details = check(task)
        ids.add(task.get("id"))
        results.append(dict(task, ok=ok, reason=reason, **details))
        if not ok:
            print("FAIL", task.get("id"), reason)
    good = [r for r in results if r["ok"]]
    print(json.dumps({"tasks": len(results), "ok": len(good),
                      "median_tokens": sorted(len(r["tokens"]) for r in good)[len(good) // 2] if good else None}))
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
