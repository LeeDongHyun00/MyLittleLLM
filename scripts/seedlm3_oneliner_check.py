"""How far could a reader that composes get? Check teacher-written one-line programs on development tasks.

A one-liner is a single Python expression over the task's arguments a0..a3 that may call the shipped
operator library (as name_arity, e.g. txt_scs_length_2(a0, a1)), Python builtins, comprehensions,
conditional expressions and arithmetic. It is the language SeedLM-3 would have to generate. This
script judges teacher-written one-liners against each development task's own tests (the three
asserts of the extra pool, the original MBPP tests for train tasks) and reports their size and
whether every constant could have been read from the description or the visible example.

  list    : the library as callable names with arity and one teacher phrase
  show    : the source of one operator
  check   : judge a JSON list of {"task_id", "expression"} and print a summary
Validation, test and HumanEval tasks are never loaded.
"""
import argparse
import ast
import builtins
import copy
import inspect
import json
import math
from pathlib import Path
import pickle
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mylittlellm.rulecoder import backinfer as bi
from mylittlellm.rulecoder.limits import Timeout, time_limit
from mylittlellm.rulecoder.pipeline import same_output
from mylittlellm.rulecoder.tasks import ROOT

BANK = ROOT / "outputs/rule_pilot/teacher/round3"
PHRASES = ROOT / "data/seedlm/operator_phrases.json"
SAFE_BUILTINS = ["abs", "all", "any", "bool", "chr", "dict", "divmod", "enumerate", "filter", "float", "int", "len",
                 "list", "map", "max", "min", "ord", "pow", "range", "reversed", "round", "set", "sorted", "str",
                 "sum", "tuple", "zip", "isinstance", "frozenset"]


def callable_name(key):
    name, arity = key.split("/")
    return f"{name}_{arity}"


def environment():
    scope = {"__builtins__": {name: getattr(builtins, name) for name in SAFE_BUILTINS}, "math": math, "re": re}
    for key, op in bi.OPERATORS.items():
        scope[callable_name(key)] = op.fn
    return scope


def tasks():
    return {item["task_id"]: item for item in json.loads((BANK / "all_tasks.json").read_text())}


def cases_for(item):
    """(args, expected, mode) triples: the extra pool's asserts, or the original MBPP base tests."""
    if item["source"] == "extra":
        cases = []
        for line in item["asserts"]:
            example = bi.parse_example(line, item["entry_point"])
            if example is None:
                return None
            cases.append((example["args"], example["expected"], example["mode"]))
        return cases
    with (BANK / "scoring" / (item["task_id"].split("/")[1] + ".pkl")).open("rb") as stream:
        stored = pickle.load(stream)
    return [(inputs, out, "eq") for inputs, out in zip(stored["problem"]["base_input"], stored["oracle"]["base"])]


def judge(expression, item, scope=None):
    """Returns (passes, n_args, error)."""
    example = bi.parse_example(item["visible_assert"], item["entry_point"])
    if example is None:
        return False, 0, "visible example unparsed"
    n_args = len(example["args"])
    params = ", ".join(f"a{i}" for i in range(n_args))
    source = f"def {item['entry_point']}({params}):\n    return {expression}\n"
    scope = dict(environment() if scope is None else scope)
    cases = cases_for(item)
    if not cases:
        return False, n_args, "no cases"
    try:
        with time_limit(2.0):
            exec(source, scope)
            for args, expected, mode in cases:
                value = scope[item["entry_point"]](*copy.deepcopy(list(args)))
                if mode == "set":
                    if not (isinstance(value, (list, tuple, set)) and set(value) == set(expected)):
                        return False, n_args, f"wrong on {args!r}: {value!r}"
                elif not same_output(value, expected):
                    return False, n_args, f"wrong on {args!r}: {value!r}"
    except Timeout:
        return False, n_args, "timeout"
    except Exception as error:  # the teacher's expression is untrusted code under test
        return False, n_args, f"{type(error).__name__}: {error}"
    return True, n_args, ""


def readable_constants(expression, item):
    """Constants not readable from the description or the visible example (small integers are free)."""
    example = bi.parse_example(item["visible_assert"], item["entry_point"]) or {"args": [], "expected": None}
    visible = item["description"] + " " + repr(example["expected"]) + " " + repr(example["args"])
    literals = bi.description_literals(item["description"])
    unread = []
    for node in ast.walk(ast.parse(expression, mode="eval")):
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, int) and -1 <= value <= 10:
                continue
            if isinstance(value, str) and (value in visible or value in item["description"]):
                continue
            if value in literals:
                continue
            unread.append(value)
    return unread


def size(expression):
    return sum(1 for _ in ast.walk(ast.parse(expression, mode="eval")))


def library_calls(expression):
    names = {callable_name(k): k for k in bi.OPERATORS}
    return sorted({names[n.id] for n in ast.walk(ast.parse(expression, mode="eval"))
                   if isinstance(n, ast.Name) and n.id in names})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    show = sub.add_parser("show")
    show.add_argument("name", help="callable name, e.g. txt_scs_length_2")
    check = sub.add_parser("check")
    check.add_argument("input")
    check.add_argument("--output")
    args = parser.parse_args()

    if args.command == "list":
        phrases = json.loads(PHRASES.read_text()) if PHRASES.exists() else {}
        for key in sorted(bi.OPERATORS):
            print(f"{callable_name(key)}\t{key.split('/')[1]}\t{(phrases.get(key) or [''])[0]}")
        return
    if args.command == "show":
        for key, op in bi.OPERATORS.items():
            if callable_name(key) == args.name:
                print(op.helper or inspect.getsource(op.fn))
                return
        print("no such operator")
        return

    bank = tasks()
    rows, out = json.loads(Path(args.input).read_text()), []
    scope = environment()
    for row in rows:
        item = bank.get(row["task_id"])
        if item is None or not row.get("expression"):
            out.append(dict(row, passes=False, error="unknown task or no expression"))
            continue
        try:
            passes, _, error = judge(row["expression"], item, scope)
            entry = dict(row, passes=passes, error=error, size=size(row["expression"]),
                         unread_constants=readable_constants(row["expression"], item),
                         library=library_calls(row["expression"]))
        except SyntaxError as error:
            entry = dict(row, passes=False, error=f"SyntaxError: {error}")
        out.append(entry)
        print(("PASS" if entry["passes"] else "FAIL"), row["task_id"], entry.get("size"), entry.get("error", "")[:100],
              entry.get("unread_constants") or "")
    passed = [e for e in out if e["passes"]]
    clean = [e for e in passed if not e.get("unread_constants")]
    print(json.dumps({"rows": len(out), "pass": len(passed), "pass_with_readable_constants": len(clean),
                      "median_size": sorted(e["size"] for e in clean)[len(clean) // 2] if clean else None}))
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
