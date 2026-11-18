"""HumanEval+ as a second, untouched benchmark: its examples read into the same form as MBPP+'s.

No rule, operator, phrase, prompt or threshold in this repository was written or tuned while
looking at HumanEval+, so it is a clean check of generalisation. A task's examples are the calls
its docstring shows with their results (">>> f(x)" followed by the result line, or "f(x) == y",
"f(x) -> y", "f(x) ➞ y", "f(x) => y", "f(x) should return y"); the first one is the visible example.
"""
import ast
import gzip
import json
import re

from .tasks import ROOT

DATASET = ROOT / "data/raw/rule_pilot/HumanEvalPlus.jsonl.gz"
ARROWS = r"(?:==>|==|➞|->|=>|should return|returns)"


def _docstring(prompt, entry_point):
    try:
        tree = ast.parse(prompt)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            return ast.get_docstring(node) or ""
    return ""


def _literal(text):
    try:
        return ast.literal_eval(text.strip().rstrip(".,"))
    except (ValueError, SyntaxError):
        return _NO


_NO = object()


def examples(prompt, entry_point):
    """(call text, expected value) pairs stated in the docstring, in order."""
    doc = _docstring(prompt, entry_point)
    lines = doc.splitlines()
    found = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(">>>") and entry_point + "(" in stripped:
            call = stripped[3:].strip()
            following = lines[i + 1].strip() if i + 1 < len(lines) else ""
            value = _literal(following) if following and not following.startswith(">>>") else _NO
            if value is not _NO:
                found.append((call, value))
            continue
        match = re.match(rf"^\s*({re.escape(entry_point)}\(.*\))\s*{ARROWS}\s*(.+?)\s*$", line)
        if match:
            value = _literal(match.group(2))
            if value is not _NO:
                found.append((match.group(1), value))
    usable = []
    for call, value in found:
        try:
            node = ast.parse(call, mode="eval").body
        except SyntaxError:
            continue
        if isinstance(node, ast.Call) and all(_literal(ast.unparse(a)) is not _NO for a in node.args) and not node.keywords:
            usable.append((call, value))
    return usable


def description(prompt, entry_point):
    """The docstring without its example lines, as one line of text."""
    doc = _docstring(prompt, entry_point)
    kept = []
    skip_next = False
    for line in doc.splitlines():
        stripped = line.strip()
        if skip_next:
            skip_next = False
            continue
        if stripped.startswith(">>>"):
            skip_next = True
            continue
        if entry_point + "(" in stripped or stripped.lower().startswith(("example", "for example")):
            continue
        kept.append(stripped)
    return " ".join(part for part in kept if part)


def read_tasks(path=DATASET):
    """task_id -> {prompt, entry_point, description, asserts (all examples), visible_assert (the first)}."""
    tasks = {}
    with gzip.open(path, "rt") as stream:
        for row in map(json.loads, stream):
            pairs = examples(row["prompt"], row["entry_point"])
            asserts = [f"assert {call} == {value!r}" for call, value in pairs]
            tasks[row["task_id"]] = {"task_id": row["task_id"], "prompt": row["prompt"], "entry_point": row["entry_point"],
                                     "description": description(row["prompt"], row["entry_point"]),
                                     "asserts": asserts, "visible_assert": asserts[0] if asserts else ""}
    return tasks
