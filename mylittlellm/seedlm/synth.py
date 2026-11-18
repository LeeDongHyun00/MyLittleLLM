"""Self-generated training data: random inputs, random operator programs, executed outputs.

No external corpus is used. The operator library of the rule engine is the teacher: a program
is sampled from it, run on random arguments, and the model must recover the program from the
arguments and the result. Every sequence is therefore correct by construction.
"""
import random
import string

from ..rulecoder import backinfer as bi

WORDS = ("red green blue black white orange python java code data list test value number apple banana cherry "
         "quick brown fox lazy dog hello world alpha beta gamma delta one two three four five six").split()
SIGNATURES = [  # argument shapes that occur in small function tasks, with sampling weights
    (("ints",), 14), (("tints",), 8), (("text",), 9), (("word",), 6), (("strs",), 5), (("rows",), 6),
    (("pairs",), 5), (("trows",), 4), (("int",), 10), (("dict",), 3), (("mixed",), 2), (("float",), 2),
    (("ints", "int"), 9), (("tints", "tints"), 5), (("ints", "ints"), 6), (("text", "word"), 3),
    (("text", "int"), 3), (("strs", "int"), 3), (("int", "int"), 7), (("trows", "trows"), 3),
    (("rows", "int"), 3), (("float", "float"), 2), (("text", "char"), 2), (("ints", "int", "int"), 2),
    (("int", "int", "int"), 2), (("text", "word", "word"), 1), (("tints", "int"), 3), (("strs", "strs"), 2),
]
MAX_IN, MAX_OUT, MAX_PROGRAM = 110, 60, 12


def random_value(kind, rng):
    if kind == "int":
        return rng.choice([rng.randint(0, 12), rng.randint(0, 40), rng.randint(-9, 99)])
    if kind == "float":
        return round(rng.uniform(0.5, 30), rng.choice([1, 2]))
    if kind in {"ints", "tints"}:
        low = rng.choice([0, 0, -9, 1])
        values = [rng.randint(low, rng.choice([9, 20, 60])) for _ in range(rng.randint(2, 8))]
        if rng.random() < 0.3:      # duplicates make dedupe/count operators observable
            values += rng.sample(values, k=min(len(values), 2))
        return tuple(values) if kind == "tints" else values
    if kind == "word":
        return rng.choice(WORDS) if rng.random() < 0.7 else "".join(rng.choices(string.ascii_letters, k=rng.randint(3, 8)))
    if kind == "char":
        return rng.choice("aeiost ,_")
    if kind == "strs":
        return [rng.choice(WORDS) for _ in range(rng.randint(2, 5))]
    if kind == "text":
        words = [rng.choice(WORDS) for _ in range(rng.randint(2, 5))]
        style = rng.random()
        if style < 0.15:
            return "_".join(words)
        if style < 0.3:
            return "".join(w.title() for w in words)
        if style < 0.45:
            return " ".join(words) + str(rng.randint(0, 99))
        return " ".join(w.title() if rng.random() < 0.2 else w for w in words)
    if kind in {"rows", "trows"}:
        width, height = rng.randint(2, 4), rng.randint(2, 4)
        rows = [[rng.randint(0, 20) for _ in range(width)] for _ in range(height)]
        return tuple(tuple(r) for r in rows) if kind == "trows" else rows
    if kind == "pairs":
        return [(rng.randint(0, 20), rng.randint(0, 20)) for _ in range(rng.randint(2, 5))]
    if kind == "dict":
        return {word: rng.randint(0, 50) for word in rng.sample(WORDS, k=rng.randint(2, 4))}
    if kind == "mixed":
        items = [rng.randint(0, 20) for _ in range(rng.randint(2, 4))] + rng.sample(WORDS, k=2)
        rng.shuffle(items)
        return items
    raise ValueError(kind)


def _apply(rng, pool, operators, tries=40):
    """Apply a random operator to random members of the pool; return (value, tree) or None."""
    for _ in range(tries):
        op = rng.choice(operators)
        if op.arity > len(pool):
            continue
        chosen = rng.sample(pool, k=op.arity)
        try:
            value = op.fn(*(item[0] for item in chosen))
        except Exception:
            continue
        # Sets are skipped because their printed order is not reproducible across processes.
        if value is None or isinstance(value, (bool, set, frozenset)) or len(repr(value)) > MAX_OUT:
            continue
        return value, (op.name + "/" + str(op.arity), [item[1] for item in chosen])
    return None


def sample(rng, operators=None, max_depth=3):
    """One training example: {'args', 'expected', 'program' (prefix tokens), 'ops'} or None."""
    operators = operators or (bi.U + bi.B + bi.T)
    kinds = rng.choices([s for s, _ in SIGNATURES], weights=[w for _, w in SIGNATURES])[0]
    args = [random_value(kind, rng) for kind in kinds]
    if len(repr(args)) > MAX_IN:
        return None
    base = [(arg, f"a{i}") for i, arg in enumerate(args)]
    depth = rng.choices([1, 2, 3][:max_depth], weights=[5, 4, 2][:max_depth])[0]
    current = None
    for level in range(depth):
        # Later levels must consume the value just built, so the program is a connected tree.
        pool = base if current is None else [current] + base
        for _ in range(8):
            made = _apply(rng, pool, operators)
            if made is None:
                return None
            if current is None or _uses(made[1], current[1]):
                break
        else:
            return None
        if made[0] in args and level == depth - 1:
            return None     # an identity program teaches nothing
        current = made
    program = _prefix(current[1])
    if len(program) > MAX_PROGRAM or not _uses_argument(program):
        return None
    return {"args": args, "expected": current[0], "program": program,
            "ops": sorted({token for token in program if "/" in token})}


def _uses(tree, target):
    return tree is target or (isinstance(tree, tuple) and any(_uses(child, target) for child in tree[1]))


def _prefix(tree):
    if isinstance(tree, str):
        return [tree]
    return [tree[0]] + [token for child in tree[1] for token in _prefix(child)]


def _uses_argument(program):
    return any(token.startswith("a") and token[1:].isdigit() for token in program)


def execute(program, args):
    """Run a prefix program on arguments; raises if it is malformed or does not apply."""
    position = 0

    def read():
        nonlocal position
        token = program[position]
        position += 1
        if "/" not in token:
            return args[int(token[1:])]
        op = bi.OPERATORS[token]
        return op.fn(*[read() for _ in range(op.arity)])

    value = read()
    if position != len(program):
        raise ValueError("trailing tokens")
    return value


def to_expression(program):
    """Render a prefix program as the Python expression the rule engine would emit."""
    position = 0
    helpers, knowledge = [], False

    def read():
        nonlocal position, knowledge
        token = program[position]
        position += 1
        if "/" not in token:
            return token
        op = bi.OPERATORS[token]
        if op.helper:
            helpers.append(op.helper)
        knowledge = knowledge or op.kind == "knowledge"
        return bi.render(op, [read() for _ in range(op.arity)])

    expr = read()
    return expr, helpers, knowledge
