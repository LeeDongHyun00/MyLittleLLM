"""Executable rules: probe a candidate with mutated copies of the visible example's arguments.

No oracle is available at inference time, so a probe can only observe crashes and hangs,
never wrong values. Which probe families are worth acting on is decided from development
data (precision of "probe fires" for "hidden tests fail"), not assumed.
"""
import ast

from .sandbox import run_call

NUMBER = (int, float)
PROBE_TIMEOUT = 3


def example_args(asserts, entry_point):
    """Literal positional arguments of the first entry-point call in the visible assert, or None."""
    try:
        tree = ast.parse(asserts)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == entry_point and not node.keywords):
            try:
                return [ast.literal_eval(arg) for arg in node.args]
            except (ValueError, SyntaxError):
                return None
    return None


def _is_number(value):
    return isinstance(value, NUMBER) and not isinstance(value, bool)


def _empty(value):
    if isinstance(value, (list, tuple, str, dict, set)) and len(value):
        yield type(value)()


def _single(value):
    if isinstance(value, (list, tuple, str)) and len(value) > 1:
        yield value[:1]


def _negative(value):
    if _is_number(value) and value > 0:
        yield -value
    elif isinstance(value, (list, tuple)) and value and all(_is_number(x) for x in value) and any(x > 0 for x in value):
        yield type(value)(-x for x in value)


def _zero(value):
    if _is_number(value) and value != 0:
        yield type(value)(0)
    elif isinstance(value, (list, tuple)) and value and all(_is_number(x) for x in value) and any(value):
        yield type(value)(type(x)(0) for x in value)


def _float(value):
    if isinstance(value, int) and not isinstance(value, bool):
        yield float(value) + 0.5
    elif isinstance(value, (list, tuple)) and value and all(isinstance(x, int) and not isinstance(x, bool) for x in value):
        yield type(value)(x + 0.5 for x in value)


def _large(value):
    if isinstance(value, int) and not isinstance(value, bool) and 0 < value < 10 ** 6:
        yield value * 20 + 7
    elif isinstance(value, (list, tuple, str)) and 0 < len(value) <= 50:
        yield value * 200


def _duplicates(value):
    if isinstance(value, (list, tuple)) and value:
        yield type(value)(x for x in value for _ in range(2))


def _nested_empty(value):
    if isinstance(value, (list, tuple)) and value and all(isinstance(x, (list, tuple, str)) and len(x) for x in value):
        yield type(value)(type(x)() for x in value)


FAMILIES = {"empty": _empty, "single": _single, "negative": _negative, "zero": _zero, "float": _float,
            "large": _large, "duplicates": _duplicates, "nested_empty": _nested_empty}


def probe_inputs(args):
    """Yield (family, mutated argument list); each probe changes exactly one argument."""
    for family, mutate in FAMILIES.items():
        for position, value in enumerate(args):
            for mutated in mutate(value):
                yield family, args[:position] + [mutated] + args[position + 1:]


def run_probes(code, entry_point, asserts, families=None):
    """Return one finding per family that crashed or hung: {family, call, problem}."""
    args = example_args(asserts, entry_point)
    if args is None:
        return []
    findings = {}
    for family, mutated in probe_inputs(args):
        if (families is not None and family not in families) or family in findings:
            continue
        outcome = run_call(code, entry_point, mutated, timeout=PROBE_TIMEOUT)
        if outcome["status"] in {"exception", "timeout"}:
            call = f"{entry_point}({', '.join(repr(a) for a in mutated)})"
            findings[family] = {"family": family, "call": call if len(call) <= 200 else call[:200] + "...",
                                "problem": outcome["status"], "detail": outcome["detail"]}
    return list(findings.values())
