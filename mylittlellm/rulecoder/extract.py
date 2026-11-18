"""Turn a raw chat completion into candidate source code. Applied identically in every condition."""
import ast
import re

FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)(?:```|\Z)", re.S)
KEPT = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
ASSIGNS = (ast.Assign, ast.AnnAssign, ast.AugAssign)


def extract_code(text, entry_point=None):
    """Pick the fenced Python block that defines the entry point, else the first block, else all text."""
    blocks = [body for lang, body in FENCE.findall(text) if lang.lower() in {"", "python", "py", "python3"}]
    if not blocks:
        blocks = [text]
    if entry_point:
        pattern = re.compile(rf"def\s+{re.escape(entry_point)}\s*\(")
        for body in blocks:
            if pattern.search(body):
                return body.strip("\n") + "\n"
    return blocks[0].strip("\n") + "\n"


def _target_names(node):
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}


def keep_definitions(code):
    """Drop top-level demo statements (prints, asserts, sample calls) that would run on import.

    Models often append usage examples; executing them can crash or block a correct function.
    Top-level assignments survive only if a kept definition reads the assigned name.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    definitions = [n for n in tree.body if isinstance(n, KEPT)]
    used = {n.id for d in definitions for n in ast.walk(d) if isinstance(n, ast.Name)}
    kept = [n for n in tree.body
            if isinstance(n, KEPT) or (isinstance(n, ASSIGNS) and _target_names(n) & used)]
    if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for n in kept):
        return code
    return "\n\n".join(ast.unparse(n) for n in kept) + "\n"
