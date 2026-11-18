"""Subprocess execution of untrusted student code with hard limits.

Only the example assert that is visible in the task prompt is executed here.
Hidden tests are never run from this module, so the repair loop cannot see them.
"""
import ast
import json
import os
import pickle
import resource
import subprocess
import sys
import tempfile

MEMORY_BYTES = 2 * 1024 ** 3
CLIP = 300

_DRIVER = r'''
import ast, contextlib, io, json, pickle, sys, traceback

CLIP = %d

def clip(value):
    text = repr(value)
    return text if len(text) <= CLIP else text[:CLIP] + "...(truncated)"

def brief(exc):
    frames = traceback.extract_tb(exc.__traceback__)
    line = next((f.lineno for f in reversed(frames) if f.filename == "<solution>"), None)
    where = f" (solution line {line})" if line else ""
    return f"{type(exc).__name__}: {exc}"[:CLIP] + where

def main():
    job = pickle.load(open(sys.argv[1], "rb"))
    scope = {"__name__": "solution"}
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            exec(compile(job["code"], "<solution>", "exec"), scope)
    except BaseException as exc:
        return {"status": "error", "stage": "load", "detail": brief(exc)}
    if job["mode"] == "call":
        fn = scope.get(job["entry_point"])
        if not callable(fn):
            return {"status": "error", "stage": "entry_point", "detail": f"function {job['entry_point']} is not defined"}
        try:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                value = fn(*job["args"])
        except BaseException as exc:
            return {"status": "exception", "detail": brief(exc)}
        return {"status": "ok", "repr": clip(value), "printed": bool(sink.getvalue().strip())}
    # Some examples call math.isclose. Give `math` to the assert only: solution functions keep
    # their own globals, so a missing `import math` inside the solution is still caught.
    import math
    harness = {"math": math, **scope}
    for node in ast.parse(job["asserts"]).body:
        source = ast.unparse(node)
        test = node.test if isinstance(node, ast.Assert) else None
        try:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                # Evaluate both sides separately so the feedback can show got vs expected.
                if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
                    got = eval(compile(ast.Expression(test.left), "<assert>", "eval"), harness)
                    want = eval(compile(ast.Expression(test.comparators[0]), "<assert>", "eval"), harness)
                    if not got == want:
                        return {"status": "fail", "assert": source, "got": clip(got), "expected": clip(want),
                                "left": ast.unparse(test.left)}
                else:
                    exec(compile(ast.Module([node], []), "<assert>", "exec"), harness)
        except AssertionError:
            return {"status": "fail", "assert": source}
        except BaseException as exc:
            return {"status": "error", "stage": "assert", "assert": source, "detail": brief(exc)}
    return {"status": "pass", "printed": bool(sink.getvalue().strip())}

print("\n__RESULT__" + json.dumps(main()))
''' % CLIP


def syntax_error(code):
    """Return a one-line syntax error message, or None when the code parses."""
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg} (line {exc.lineno})"
    return None


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
    os.setsid()


def _run(job, timeout):
    with tempfile.TemporaryDirectory() as tmp:
        job_path = os.path.join(tmp, "job.pkl")
        with open(job_path, "wb") as stream:
            pickle.dump(job, stream)
        try:
            proc = subprocess.run([sys.executable, "-I", "-c", _DRIVER, job_path], cwd=tmp, capture_output=True,
                                  text=True, timeout=timeout, preexec_fn=_limits, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "detail": f"no result within {timeout} seconds"}
    marker = proc.stdout.rfind("\n__RESULT__")
    if marker < 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["no output"]
        return {"status": "error", "stage": "crash", "detail": tail[0][:CLIP]}
    return json.loads(proc.stdout[marker + len("\n__RESULT__"):])


def run_public_asserts(code, asserts, timeout=10):
    """Run the prompt-visible example asserts against candidate code."""
    return _run({"mode": "asserts", "code": code, "asserts": asserts}, timeout)


def run_call(code, entry_point, args, timeout=10):
    """Call the entry point once; used for diagnosis and edge-case probes."""
    return _run({"mode": "call", "code": code, "entry_point": entry_point, "args": list(args)}, timeout)
