"""Executable repair rules: deterministic edits that need no model call.

Each codemod came from a failure class the teacher found in development data. Unlike a
prompt rule, a codemod works whether or not the student model follows instructions.
"""
import ast
import builtins
import difflib

MODULES = ("math", "re", "itertools", "collections", "heapq", "functools", "string", "bisect",
           "operator", "cmath", "statistics", "fractions", "decimal", "datetime", "copy", "random", "sys")
HELPERS = {"Counter": "collections", "defaultdict": "collections", "deque": "collections",
           "OrderedDict": "collections", "combinations": "itertools", "permutations": "itertools",
           "product": "itertools", "groupby": "itertools", "chain": "itertools", "accumulate": "itertools",
           "zip_longest": "itertools", "combinations_with_replacement": "itertools",
           "reduce": "functools", "lru_cache": "functools", "sqrt": "math", "gcd": "math",
           "factorial": "math", "ceil": "math", "floor": "math", "comb": "math", "pi": "math",
           "isclose": "math", "heappush": "heapq", "heappop": "heapq", "heapify": "heapq",
           "nlargest": "heapq", "nsmallest": "heapq", "bisect_left": "bisect", "bisect_right": "bisect"}


def _bound_names(tree):
    """Every name the module binds itself: imports, defs, assignments, arguments, loop targets."""
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
    return bound


def add_missing_imports(code):
    """Prepend imports for standard-library names that are used but never bound."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    bound = _bound_names(tree)
    loaded = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    missing = loaded - bound - set(dir(builtins))
    lines = [f"import {name}" for name in MODULES if name in missing]
    lines += [f"from {HELPERS[name]} import {name}" for name in sorted(missing) if name in HELPERS]
    return "\n".join(lines) + "\n" + code if lines else code


def alias_entry_point(code, entry_point):
    """Bind the name the example calls to the function the answer actually defines."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if entry_point in _bound_names(tree):
        return code
    functions = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and not n.name.startswith("_")]
    if len(functions) == 1:
        target = functions[0]
    else:
        # With helpers present only a near-identical name (a spelling variant) is safe to alias.
        close = difflib.get_close_matches(entry_point, functions, n=1, cutoff=0.75)
        if not close:
            return code
        target = close[0]
    return code.rstrip("\n") + f"\n\n{entry_point} = {target}\n"


def autofix(code, entry_point):
    return alias_entry_point(add_missing_imports(code), entry_point)
