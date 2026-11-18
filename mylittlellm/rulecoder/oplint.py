"""Property checks for operators: what the search computes must be what the emitted code computes.

An operator has two forms, a Python function used while searching and a code template pasted
into the answer. If they disagree the engine verifies one program and ships another. Operators
are written by teachers (people or models), so this is checked mechanically on generated inputs:
  - same result from the function and from the emitted code, including its type;
  - no argument is modified;
  - a second call gives the same result;
  - an operator that returns a bool is gated (one True/False example fits half of all predicates);
  - a call stays cheap.
"""
import copy
import math
import random
import time

from . import backinfer as bi

SLOW_SECONDS = 0.05
WORDS = "red green blue level radar Python java code data 12 3.5 a1b2 hello world noon Abc xyz".split()


def value_pool(seed=7):
    """Small values of every shape the tasks use, with the edge cases hidden tests like."""
    rng = random.Random(seed)
    pool = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 17, 25, 28, 36, 97, 100, 153, 360, 1000, 1024, 2020,
            -1, -3, -12, 0.0, 0.5, 1.5, 2.0, 3.14, 10.25, -2.5, "", "a", " ", "ab", "aa", "abc", "ABC", "a b",
            "racecar", "Hello World", "hello_world", "helloWorld", "  padded  ", "one two three", "a,b,c",
            "x1y22z333", "1234", "-17", "3.5", "aeiou", "The Quick brown fox", "ab12!@", "2020-11-18",
            'say "hi" and "bye"', "(a[b]{c})", "192.168.001.010", "user@example.com",
            [], (), [0], [1], (5,), [1, 2, 3], [3, 1, 2], (1, 2, 3), [1, 1, 2, 2, 3], [5, -2, 0, 7], [2, 4, 6],
            [1, 3, 5], [1.5, 2.5], [10, 20, 30, 40, 50], (4, 4, 4), [0, 0, 1], ["a", "b"], ["b", "a", "c"],
            ("x", "y"), ["apple", "fig", "kiwi"], ["aa", "b", "ccc"], ["1", "22", "3"], [1, "a", 2.5, "b"],
            [[1, 2], [3, 4]], [[1, 2, 3], [4, 5, 6]], [(1, 2), (3, 4)], ((1, 2), (3, 4)), [[1], [2, 3], []],
            [(1, "a"), (2, "b")], [("a", 1), ("b", 2), ("a", 3)], [[3, 1], [2, 2], [1, 3]], [[1, 2], [1, 2]],
            {"a": 1, "b": 2}, {"x": 10, "y": 5, "z": 7}, {1: "one", 2: "two"}, {}, {"k": [1, 2], "m": [3]},
            [{"a": 1}, {"a": 2}], {1, 2, 3}, None, True, False]
    for _ in range(60):
        kind = rng.choice(["ints", "ints", "tints", "strs", "word", "text", "rows", "pairs", "int", "float"])
        if kind == "int":
            pool.append(rng.randint(-20, 200))
        elif kind == "float":
            pool.append(round(rng.uniform(-5, 50), 2))
        elif kind in ("ints", "tints"):
            values = [rng.randint(-5, 20) for _ in range(rng.randint(1, 7))]
            pool.append(tuple(values) if kind == "tints" else values)
        elif kind == "strs":
            pool.append([rng.choice(WORDS) for _ in range(rng.randint(1, 5))])
        elif kind == "word":
            pool.append(rng.choice(WORDS))
        elif kind == "text":
            pool.append(" ".join(rng.choice(WORDS) for _ in range(rng.randint(2, 5))))
        elif kind == "rows":
            width = rng.randint(1, 3)
            pool.append([[rng.randint(0, 9) for _ in range(width)] for _ in range(rng.randint(1, 4))])
        else:
            pool.append([(rng.randint(0, 9), rng.randint(0, 9)) for _ in range(rng.randint(1, 4))])
    return pool


def same(left, right):
    """Equality that also compares types at every level and tolerates float rounding."""
    if type(left) is not type(right):
        return False
    if isinstance(left, float):
        return left == right or (math.isnan(left) and math.isnan(right)) or math.isclose(left, right, rel_tol=1e-9)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    return left == right


def emitted(op):
    """The operator as the engine would ship it: a compiled one-line function."""
    names = [f"a{i}" for i in range(op.arity)]
    code = bi.to_code("_emitted", op.arity, {"expr": bi.render(op, names), "helpers": [op.helper] if op.helper else []})
    scope = {}
    exec(code, scope)       # library templates are trusted code; inputs are generated here
    return scope["_emitted"]


def argument_sets(arity, pool, rng, trials):
    """Every pool value for unary operators; random tuples plus related ones for the others.

    Random pairs almost never contain a container together with one of its own members, a valid
    index or a second container of the same kind, which is exactly what most binary operators need.
    """
    if arity == 1:
        return [(value,) for value in pool]
    sets = [tuple(rng.choice(pool) for _ in range(arity)) for _ in range(trials)]
    containers = [v for v in pool if isinstance(v, (list, tuple, str, dict)) and len(v) > 0]
    for value in containers:
        members = list(value)[:3] + [0, 1, 2, len(value), -1]
        if isinstance(value, str):
            members += [value[:1], value[-1:], value[:2], " "]
        for member in members:
            if arity == 2:
                sets += [(value, member), (member, value)]
            else:
                sets += [(value, member, rng.choice(members)), (value, 0, len(value)), (value, value, value)]
        same_kind = [other for other in containers if type(other) is type(value)]
        sets += [(value, rng.choice(same_kind))] * (arity == 2)
        sets += [(value, rng.choice(same_kind), rng.choice(same_kind))] * (arity == 3)
    numbers = [v for v in pool if isinstance(v, (int, float)) and not isinstance(v, bool)]
    sets += [tuple(rng.choice(numbers) for _ in range(arity)) for _ in range(trials // 3)]
    return sets


def lint(operators=None, seed=7, trials=1500):
    """Return a list of (operator key, problem, example) for every violated property."""
    operators = bi.OPERATORS if operators is None else operators
    pool, rng, problems = value_pool(seed), random.Random(seed), []
    for key, op in operators.items():
        try:
            shipped = emitted(op)
        except Exception as error:
            problems.append((key, f"template does not compile: {type(error).__name__}: {error}", None))
            continue
        applied = False
        for args in argument_sets(op.arity, pool, rng, trials):
            before = copy.deepcopy(args)
            started = time.perf_counter()
            try:
                result = op.fn(*args)
            except Exception:
                continue        # the operator does not apply to these arguments
            elapsed = time.perf_counter() - started
            applied = True
            shown = repr(before)[:160]
            if not same(args, before):
                problems.append((key, "modifies its arguments", shown))
                break
            if elapsed > SLOW_SECONDS:
                problems.append((key, f"slow call ({elapsed:.2f}s)", shown))
                break
            if isinstance(result, bool) and not op.gated:
                problems.append((key, "returns a bool but is not gated", shown))
                break
            try:
                again = op.fn(*copy.deepcopy(before))
            except Exception as error:
                problems.append((key, f"second call raised {type(error).__name__}", shown))
                break
            if not same(result, again):
                problems.append((key, "two calls disagree", shown))
                break
            try:
                code_result = shipped(*copy.deepcopy(before))
            except Exception as error:
                problems.append((key, f"emitted code raised {type(error).__name__}: {error}", shown))
                break
            if not same(result, code_result) and not (isinstance(result, set) and result == code_result):
                problems.append((key, f"emitted code returns {code_result!r:.80}, search computed {result!r:.80}",
                                 shown))
                break
        if not applied:
            # Not a defect of the operator, but nothing was verified either; the caller decides.
            problems.append((key, "UNTESTED: no generated input fits its guard", None))
    return problems
