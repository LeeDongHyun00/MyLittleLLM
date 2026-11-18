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

# One canonical text per value, shared by the sandbox and by in-process callers, so that two
# programs can be compared by their outputs: equal values give equal text (sets and dicts in any
# order, floats to nine significant digits), and long values are hashed to keep messages small.
CANONICAL = r'''
import hashlib as _hashlib


def canonical(value, depth=0):
    if depth > 8:
        return "<deep>"
    if value is None or isinstance(value, (bool, str)):
        text = repr(value)
    elif isinstance(value, int):
        text = repr(value) if abs(value) < 10 ** 60 else "int:" + _hashlib.sha1(repr(value).encode()).hexdigest()
    elif isinstance(value, float):
        text = "nan" if value != value else format(value, ".9g")
    elif isinstance(value, complex):
        text = "complex(" + canonical(value.real) + "," + canonical(value.imag) + ")"
    elif isinstance(value, list):
        text = "[" + ",".join(canonical(v, depth + 1) for v in value) + "]"
    elif isinstance(value, tuple):
        text = "(" + ",".join(canonical(v, depth + 1) for v in value) + ")"
    elif isinstance(value, (set, frozenset)):
        text = "{" + ",".join(sorted(canonical(v, depth + 1) for v in value)) + "}"
    elif isinstance(value, dict):
        text = "{" + ",".join(sorted(canonical(k, depth + 1) + ":" + canonical(v, depth + 1)
                                     for k, v in value.items())) + "}"
    else:
        text = type(value).__name__ + ":" + repr(value)[:200]
    if len(text) > 400:
        text = "sha1:" + _hashlib.sha1(text.encode()).hexdigest()
    return text
'''

_DRIVER = (r'''
import ast, contextlib, io, json, pickle, sys, traceback

CLIP = %d
''' + CANONICAL + r'''

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
    if job["mode"] == "calls":
        # Many inputs in one process: a probe that hangs is cut off alone, the others still run.
        import copy, signal
        fn = scope.get(job["entry_point"])
        if not callable(fn):
            return {"status": "error", "stage": "entry_point", "detail": f"function {job['entry_point']} is not defined"}

        def interrupt(signum, frame):
            raise TimeoutError("probe call took too long")

        signal.signal(signal.SIGALRM, interrupt)
        results = []
        for args in job["inputs"]:
            try:
                signal.setitimer(signal.ITIMER_REAL, job["per_call"])
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    value = fn(*copy.deepcopy(args))
                signal.setitimer(signal.ITIMER_REAL, 0)
                results.append(["ok", canonical(value)])
            except BaseException as exc:
                signal.setitimer(signal.ITIMER_REAL, 0)
                results.append(["exc", type(exc).__name__])
        return {"status": "ok", "results": results}
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
''') % CLIP


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


def run_calls(code, entry_point, inputs, per_call=0.5):
    """Call the entry point on many argument lists in one subprocess; canonical outputs or exception names."""
    job = {"mode": "calls", "code": code, "entry_point": entry_point, "inputs": [list(a) for a in inputs],
           "per_call": per_call}
    return _run(job, timeout=5 + per_call * len(inputs))
