"""Random programs that mix Python's own nodes with the rule library (SeedLM-21).

SeedLM-5's fragment diagnostics found that the coder writes the parent -> child links ("edges") it has
seen and almost never one it has not. Its synthetic data (data2, 300,000 rows) are library operators
only, so every edge that touches one of Python's nodes (arithmetic, comparisons, conditionals,
indexing and slicing, builtins, methods, comprehensions) comes from a few thousand teacher-written
programs; on the held-out teacher tasks almost every edge no training program has is of that kind.

This module writes such programs at scale. It grows an expression bottom-up over one task's input
cases (a training teacher task or variant: realistic arguments, three cases), each step applying a
Python node or a library operator to the expression so far and to leaves or small side expressions,
and evaluating every new expression on all the cases. A kept program runs on every case, uses an
argument, changes its operand at every step on some case, and gives at least two different outputs.
Programs are written as Python text and turned into pyx tokens with pyx.serialize, then checked
against pyx.execute. Nothing here reads a held-out task.
"""
import ast
import math
import random

from ..rulecoder import backinfer as bi
from ..rulecoder.limits import Timeout, time_limit
from . import coder3, pyx

SECONDS = 0.2           # all evaluations of one candidate expression
MAX_REPR = 120          # a value's text (as SeedLM-6's grafts)
MAX_OUTPUT = 60         # the program's output on each case (the coder reads it as bytes)
MAX_ELEMENTS = 30       # a sequence a comprehension walks
INTS = [0, 1, 2, 3, 4, 5, 10, -1]
SHORT = ["", " ", ",", "-", "a", "_"]
LIBRARY = sorted(bi.OPERATORS.items())
# How often each kind of step is tried. Library calls and comprehensions are what connect Python's
# nodes to the operators the coder already knows, so they get the most weight.
FAMILIES = {"library": 5, "arithmetic": 3, "compare": 2, "logic": 1, "conditional": 1, "index": 2,
            "slice": 1, "builtin": 3, "method": 2, "comprehension": 3}


def value_kind(value):
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int" if abs(value) < 10 ** 12 else None
    if isinstance(value, float):
        return "float" if math.isfinite(value) else None
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, tuple):
        return "tuple"
    if isinstance(value, dict):
        return "dict"
    return None


class Expr:
    """An expression with its value in every environment of its context."""
    __slots__ = ("text", "values", "kind", "constant", "uses")

    def __init__(self, text, values, constant=False, uses=frozenset()):
        self.text, self.values, self.constant, self.uses = text, values, constant, frozenset(uses)
        kinds = {value_kind(v) for v in values} if values is not None else {value_kind(ast.literal_eval(text))}
        self.kind = kinds.pop() if len(kinds) == 1 else None

    def element_kind(self):
        """The kind every element of every value shares, for sequences; None otherwise."""
        if self.kind not in ("list", "tuple", "str"):
            return None
        kinds = {value_kind(e) for v in self.values for e in v}
        return kinds.pop() if len(kinds) == 1 else ("str" if self.kind == "str" else None)

    def nonempty(self):
        return self.kind in ("list", "tuple", "str", "dict") and all(len(v) > 0 for v in self.values)


NUMBER = ("int", "float")
SEQUENCE = ("list", "tuple", "str")


class Context:
    """The environments an expression is evaluated in (one per case, or one per case and element inside a
    comprehension), the leaves and side expressions it may use, and the variables it must use."""

    def __init__(self, envs, leaves, required=frozenset()):
        self.envs, self.leaves, self.required = envs, leaves, frozenset(required)

    def evaluate(self, text):
        try:
            code = compile(text, "<pysynth>", "eval")
        except (SyntaxError, ValueError, RecursionError):
            return None
        values = []
        try:
            with time_limit(SECONDS):
                for env in self.envs:
                    value = eval(code, env)
                    if value_kind(value) is None or len(repr(value)) > MAX_REPR:
                        return None
                    values.append(value)
        except (Exception, Timeout):
            return None
        return values


def _constants():
    return [Expr(repr(n), None, constant=True) for n in INTS]


def _pick(rng, items, kinds=None, nonempty=False, element=None):
    found = [e for e in items if (kinds is None or e.kind in kinds) and (not nonempty or e.nonempty())
             and (element is None or e.element_kind() in element)]
    return rng.choice(found) if found else None


class Step:
    """Applying one node to the current expression (which must be an operand) and to other operands."""

    def __init__(self, rng, ctx, current, library=None):
        self.rng, self.ctx, self.current = rng, ctx, current
        self.drawn = library or (LIBRARY, None)      # (operators, weights) for library steps

    def other(self, kinds=None, constant_rate=0.4, nonempty=False, element=None):
        """An operand other than the current expression: a leaf or side expression, or a constant."""
        rng = self.rng
        if kinds and "int" in kinds and rng.random() < constant_rate:
            return Expr(repr(rng.choice(INTS)), None, constant=True)
        if kinds and "str" in kinds and rng.random() < constant_rate / 2:
            return Expr(repr(rng.choice(SHORT)), None, constant=True)
        return _pick(rng, [e for e in self.ctx.leaves if e is not self.current], kinds, nonempty, element)

    # ------------------------------------------------------------------ families: (text, operands) or None
    def arithmetic(self):
        rng, x = self.rng, self.current
        if x.kind in NUMBER:
            op = rng.choice(["+", "-", "*", "//", "%", "**", "/", "+", "-", "*", "%"])
            y = self.other(["int"], constant_rate=0.7) if op == "**" else self.other(list(NUMBER), 0.6)
            if y is None or (op == "**" and not (y.constant and y.text in ("2", "3"))):
                return None
            left, right = (x, y) if rng.random() < 0.75 or y.constant else (y, x)
            return f"({left.text} {op} {right.text})", [x]
        if x.kind in SEQUENCE:
            if rng.random() < 0.5:
                y = self.other([x.kind], constant_rate=0.3)
                if y is None:
                    return None
                left, right = (x, y) if rng.random() < 0.5 else (y, x)
                return f"({left.text} + {right.text})", [x]
            return f"({x.text} * {rng.choice([2, 3])})", [x]
        return None

    def compare(self):
        rng, x = self.rng, self.current
        if x.kind in NUMBER:
            y = self.other(list(NUMBER), 0.6)
            op = rng.choice(["<", "<=", ">", ">=", "==", "!="])
        elif x.kind == "str" and rng.random() < 0.5:
            y = self.other(["str"], 0.5)
            op = rng.choice(["==", "!=", "in", "not in"])
        elif x.kind in SEQUENCE:
            element = x.element_kind()
            y = self.other([element] if element else None, 0.3)
            if y is None or (element and y.kind != element):
                return None
            return f"({y.text} {rng.choice(['in', 'not in'])} {x.text})", [x]
        else:
            y = self.other([x.kind] if x.kind else None, 0.3)
            op = rng.choice(["==", "!="])
        if y is None:
            return None
        left, right = (x, y) if rng.random() < 0.75 or y.constant else (y, x)
        return f"({left.text} {op} {right.text})", [x]

    def logic(self):
        rng, x = self.rng, self.current
        if x.kind != "bool":
            return None
        if rng.random() < 0.35:
            return f"(not {x.text})", [x]
        y = self.other(["bool"], 0)
        if y is None:
            return None
        return f"({x.text} {rng.choice(['and', 'or'])} {y.text})", [x]

    def conditional(self):
        rng, x = self.rng, self.current
        if x.kind == "bool":
            kinds = rng.choice([["int"], ["str"], ["list"]])
            a, b = self.other(kinds, 0.6), self.other(kinds, 0.6)
            if a is None or b is None or a.text == b.text:
                return None
            return f"({a.text} if {x.text} else {b.text})", [x]
        condition = self.other(["bool"], 0)
        other = self.other([x.kind] if x.kind else None, 0.5)
        if condition is None or other is None or other.kind not in (x.kind, None) or other.text == x.text:
            return None
        if rng.random() < 0.5:
            return f"({x.text} if {condition.text} else {other.text})", [x]
        return f"({other.text} if {condition.text} else {x.text})", [x]

    def index(self):
        rng, x = self.rng, self.current
        if x.kind in SEQUENCE and x.nonempty():
            if rng.random() < 0.7:
                return f"{x.text}[{rng.choice(['0', '1', '-1', '-1', '2', '-2'])}]", [x]
            i = self.other(["int"], 0)
            return (f"{x.text}[{i.text}]", [x]) if i else None
        if x.kind == "int":
            s = self.other(list(SEQUENCE), 0, nonempty=True)
            return (f"{s.text}[{x.text}]", [x]) if s else None
        return None

    def slice(self):
        rng, x = self.rng, self.current
        if x.kind in SEQUENCE:
            form = rng.choice(["[1:]", "[:-1]", "[::-1]", "[::2]", "[1::2]", "[:{n}]", "[{n}:]", "[:{n}]",
                               "[-{n}:]"])
            return x.text + form.format(n=rng.choice([2, 3])), [x]
        if x.kind == "int":
            s = self.other(list(SEQUENCE), 0)
            if s is None:
                return None
            return rng.choice([f"{s.text}[:{x.text}]", f"{s.text}[{x.text}:]"]), [x]
        return None

    def builtin(self):
        rng, x = self.rng, self.current
        element = x.element_kind()
        options = []
        if x.kind in SEQUENCE + ("dict",):
            options += ["len({x})"] * 3 + ["sorted({x})", "list(reversed({x}))", "len(set({x}))"]
            if x.kind != "list":
                options.append("list({x})")
            if x.nonempty() and element in NUMBER + ("str",):
                options += ["max({x})", "min({x})"]
            if element in NUMBER:
                options += ["sum({x})"] * 2 + ["sorted({x}, reverse=True)"]
            if element == "bool":
                options += ["any({x})", "all({x})"]
            if element in SEQUENCE and x.nonempty():
                options += ["max({x}, key=len)", "min({x}, key=len)", "sorted({x}, key=len)"]
            if x.kind == "list":
                options.append("tuple({x})")
        elif x.kind in NUMBER:
            options += ["abs({x})", "str({x})", "max({x}, {y})", "min({x}, {y})"]
            if x.kind == "float":
                options += ["round({x})", "round({x}, 2)", "int({x})"]
            else:
                options += ["float({x})", "bool({x})"]
                if all(0 <= v <= 30 for v in x.values):
                    options.append("list(range({x}))")
                if all(32 <= v < 127 for v in x.values):
                    options.append("chr({x})")
        elif x.kind == "str":
            options += ["int({x})", "len({x})"]
        if not options:
            return None
        template = rng.choice(options)
        y = self.other(list(NUMBER), 0.6) if "{y}" in template else None
        if "{y}" in template and y is None:
            return None
        return template.format(x=x.text, y=y.text if y else ""), [x]

    def method(self):
        rng, x = self.rng, self.current
        if x.kind == "str":
            template = rng.choice(["{x}.upper()", "{x}.lower()", "{x}.strip()", "{x}.split()", "{x}.title()",
                                   "{x}.swapcase()", "{x}.capitalize()", "{x}.isdigit()", "{x}.isalpha()",
                                   "{x}.isupper()", "{x}.islower()", "{x}.count({s})", "{x}.find({s})",
                                   "{x}.startswith({s})", "{x}.endswith({s})", "{x}.replace({s}, {t})",
                                   "{x}.split({s})", "{t}.join({x})", "{x}.lstrip({s})", "{x}.rstrip({s})"])
        elif x.kind in ("list", "tuple") and x.element_kind() == "str":
            template = rng.choice(["{t}.join({x})", "{t}.join({x})", "{x}.count({s})", "{x}.index({s})"])
        elif x.kind in ("list", "tuple"):
            template = rng.choice(["{x}.count({e})", "{x}.index({e})"])
        else:
            return None
        s = self.other(["str"], 0.7) if "{s}" in template else None
        t = Expr(repr(rng.choice(SHORT[:4])), None, constant=True) if "{t}" in template else None
        e = self.other([x.element_kind()], 0.3) if "{e}" in template and x.element_kind() else None
        if ("{s}" in template and s is None) or ("{e}" in template and e is None):
            return None
        return template.format(x=x.text, s=s.text if s else "", t=t.text if t else "", e=e.text if e else ""), [x]

    def library(self):
        rng, x = self.rng, self.current
        operators, weights = self.drawn
        for _ in range(12):
            key, op = rng.choices(operators, weights)[0] if weights else rng.choice(operators)
            if op.arity > 4:
                continue
            slot = rng.randrange(op.arity)
            operands = []
            for position in range(op.arity):
                if position == slot:
                    operands.append(x)
                    continue
                y = self.other(None if rng.random() < 0.5 else [x.kind] if x.kind else None, 0.3)
                if y is None:
                    break
                operands.append(y)
            else:
                return f"{pyx.callable_name(key)}({', '.join(o.text for o in operands)})", [x]
        return None

    def comprehension(self):
        return None     # set by Generator: needs the whole context


ARGUMENTS = frozenset(f"a{i}" for i in range(4))


def _names(text):
    return {node.id for node in ast.walk(ast.parse(text, mode="eval")) if isinstance(node, ast.Name)}


class Generator:
    """Writes programs over one host's cases (and its description literals)."""

    def __init__(self, rng, families=None, library=None):
        """library: (operators as (key, op) pairs, weights or None), all operators equally by default."""
        self.rng = rng
        self.families = families or FAMILIES
        self.library = library
        self.env = pyx.environment()

    def _context(self, cases, literals):
        envs = []
        for case in cases:
            env = dict(self.env)
            env.update({f"a{i}": value for i, value in enumerate(case)})
            env.update({f"c{i}": value for i, value in enumerate(literals)})
            envs.append(env)
        leaves = [Expr(f"a{i}", [case[i] for case in cases], uses={f"a{i}"}) for i in range(len(cases[0]))]
        leaves += [Expr(f"c{i}", [value] * len(cases), constant=True) for i, value in enumerate(literals)
                   if value_kind(value) in ("int", "float", "str")]
        return Context(envs, leaves)

    def _apply(self, ctx, current, families, depth, tries=12):
        """current with one more node on top (or a new expression if current is None), evaluated."""
        rng = self.rng
        names, weights = zip(*families.items())
        for _ in range(tries):
            family = rng.choices(names, weights)[0]
            base = current or rng.choice([e for e in ctx.leaves if not e.constant] or ctx.leaves)
            if family == "comprehension":
                if depth > 0:
                    continue
                made = self._comprehension(ctx, base)
            else:
                made = getattr(Step(rng, ctx, base, self.library), family)()
            if made is None:
                continue
            text, operands = made
            values = ctx.evaluate(text)
            if values is None:
                continue
            if any(o.values is not None and values == o.values for o in operands):
                continue        # the step changed nothing on any case
            return Expr(text, values, uses=_names(text) & (ARGUMENTS | ctx.required))
        return None

    def _side(self, ctx, depth):
        """A small side expression (one step over a leaf) the main expression may combine with."""
        made = self._apply(ctx, None, {k: v for k, v in self.families.items() if k != "comprehension"}, depth)
        return [made] if made is not None else []

    def _comprehension(self, ctx, base):
        """A comprehension over base (a sequence), or over range(len(base)), zip or enumerate, with an
        element expression or a condition (one or two steps over the bound variable)."""
        rng = self.rng
        if base.kind not in SEQUENCE or any(len(v) > MAX_ELEMENTS for v in base.values) or not base.nonempty():
            return None
        form = rng.choices(["plain", "range", "enumerate", "zip"], [6, 1, 1, 1])[0]
        if form == "zip":
            partner = _pick(rng, [e for e in ctx.leaves if e is not base], SEQUENCE, nonempty=True)
            if partner is None:
                return None
            head, names = f"zip({base.text}, {partner.text})", ("x", "y")
            rows = [list(zip(a, b)) for a, b in zip(base.values, partner.values)]
        elif form == "enumerate":
            head, names = f"enumerate({base.text})", ("i", "x")
            rows = [list(enumerate(v)) for v in base.values]
        elif form == "range":
            head, names = f"range(len({base.text}))", ("i",)
            rows = [[(i,) for i in range(len(v))] for v in base.values]
        else:
            head, names = base.text, ("x",)
            rows = [[(e,) for e in v] for v in base.values]
        envs, inner_leaves = [], []
        for env, row in zip(ctx.envs, rows):
            for bound in row:
                inner = dict(env)
                inner.update(dict(zip(names, bound)))
                envs.append(inner)
        if not envs:
            return None
        for position, name in enumerate(names):
            values = [dict(zip(names, bound))[name] for row in rows for bound in row]
            inner_leaves.append(Expr(name, values, uses={name}))
        counts = [len(row) for row in rows]
        for leaf in ctx.leaves:
            if leaf.constant:
                inner_leaves.append(Expr(leaf.text, [leaf.values[0]] * len(envs), constant=True))
            elif leaf.kind is not None and leaf is not base:
                inner_leaves.append(Expr(leaf.text, [v for v, n in zip(leaf.values, counts) for _ in range(n)],
                                         uses=leaf.uses))
        if form == "range":         # an index loop reads the sequence at i
            values = [v[i] for v, row in zip(base.values, rows) for (i,) in row]
            inner_leaves.insert(0, Expr(f"{base.text}[i]", values, uses={"i"}))
        inner = Context(envs, inner_leaves, required=set(names))
        steps = {k: v for k, v in self.families.items() if k != "comprehension"}
        start = rng.choice(inner_leaves[:len(names) + (form == "range")])
        element = self._apply(inner, start, steps, depth=1)
        if element is not None and rng.random() < 0.3:
            element = self._apply(inner, element, steps, depth=1) or element
        condition = None
        if rng.random() < 0.45:
            condition = self._apply(inner, rng.choice(inner_leaves[:len(names) + (form == "range")]),
                                    {"compare": 3, "method": 1, "library": 1, "logic": 1}, depth=1)
            if condition is not None and condition.kind != "bool":
                condition = None
        target = ", ".join(names)
        if element is None and condition is None:
            return None
        bound = {leaf.text: leaf for leaf in inner_leaves if leaf.text in names}
        body = element.text if element is not None else names[-1]
        body_kind = element.kind if element is not None else bound[names[-1]].kind
        tail = f" for {target} in {head}" + (f" if {condition.text}" if condition is not None else "")
        options = [f"[{body}{tail}]"]
        if body_kind in NUMBER:
            options += [f"sum({body}{tail})", f"max({body}{tail})", f"min({body}{tail})"]
        if body_kind == "bool":
            options += [f"any({body}{tail})", f"all({body}{tail})"]
        if body_kind == "str":
            options += [f"''.join({body}{tail})"]
        if condition is not None:
            options += [f"len([{body}{tail}])", f"sum(1{tail})"]
        options += [f"sorted({body}{tail})", f"len(set({body}{tail}))"]
        return rng.choice(options), [base]

    def program(self, cases, literals, steps):
        """Python text of one program over the cases, or None."""
        rng = self.rng
        ctx = self._context(cases, literals)
        for _ in range(rng.choice([0, 1, 1, 2])):
            ctx.leaves += self._side(ctx, depth=0)
        current = None
        for _ in range(steps):
            made = self._apply(ctx, current, self.families, depth=0)
            if made is None:
                break
            current = made
        if current is None or not any(n.startswith("a") for n in current.uses):
            return None
        texts = [repr(v) for v in current.values]
        if len(set(texts)) < 2 or any(len(t) > MAX_OUTPUT for t in texts):
            return None
        if any(all(repr(case[i]) == t for case, t in zip(cases, texts)) for i in range(len(cases[0]))):
            return None         # returns an argument
        return current.text, current.values


def tokens_for(text, cases, literals, values):
    """pyx tokens for the program text, if pyx writes it and gives the same values; else None."""
    try:
        tokens = pyx.serialize(text, len(cases[0]), literals)
    except (pyx.Unsupported, RecursionError, SyntaxError):
        return None
    if len(tokens) > coder3.MAX_PROGRAM:
        return None
    for case, value in zip(cases, values):
        try:
            with time_limit(SECONDS):
                again = pyx.execute(list(tokens), case, literals)
        except (Exception, Timeout):
            return None
        if repr(again) != repr(value):
            return None
    for value in values:            # stored as text, so it must read back
        try:
            if ast.literal_eval(repr(value)) != value:
                return None
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return None
    return tokens


def library_from(programs, mode):
    """Which operators the library steps draw: "all" equally; "training" those the given programs use,
    equally; "sqrt" those, in proportion to the square root of how often they are used."""
    if mode == "all":
        return None
    used = {}
    for tokens in programs:
        for token in tokens:
            if token in bi.OPERATORS:
                used[token] = used.get(token, 0) + 1
    keys = sorted(used)
    operators = [(k, bi.OPERATORS[k]) for k in keys]
    return operators, ([math.sqrt(used[k]) for k in keys] if mode == "sqrt" else None)


def generate(hosts, count, seed=0, families=None, library=None):
    """`count` distinct programs over the hosts' cases: {"text", "tokens", "cases", "outputs", "literals"}."""
    rng = random.Random(seed)
    maker = Generator(rng, families, library)
    out, seen = [], set()
    attempts = 0
    while len(out) < count and attempts < count * 60:
        attempts += 1
        host = rng.choice(hosts)
        steps = rng.choices([1, 2, 3, 4], [2, 4, 3, 1])[0]
        made = maker.program(host["cases"], host["literals"], steps)
        if made is None:
            continue
        text, values = made
        tokens = tokens_for(text, host["cases"], host["literals"], values)
        if tokens is None or (tuple(tokens), repr(host["cases"])) in seen:
            continue
        seen.add((tuple(tokens), repr(host["cases"])))
        out.append({"text": text, "tokens": tokens, "cases": [repr(tuple(c)) for c in host["cases"]],
                    "outputs": [repr(v) for v in values], "literals": host["literals"], "host": host["id"]})
    return out
