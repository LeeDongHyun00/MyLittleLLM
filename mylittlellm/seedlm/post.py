"""SeedLM-4's program form: pyx written bottom-up, each token followed by what it computes.

SeedLM-3's coder writes a program top-down, in prefix order, and never sees what any part of it
computes: it must choose `sum` before it has written, let alone run, the list it sums. A model of a
few million parameters cannot run a program in its head, and on the tasks the rule engine fails it
found the right program for 4 of 15 whatever its size (docs/seed_lm3.md).

Here the same programs are written in postfix order, operands first, and after every token the
partial program is run on the visible example. Three annotation tokens then say what the newest stack
item is: its kind, how it relates to the expected output (equal, one less, a permutation of it, its
length, ...) and its size. The model chooses the next token knowing the values it has built so far,
the way the rule engine searches, but steered by the description.

Comprehensions and key functions open a scope: the iterable is written first, then `@For:n` binds n
variables over it, then the conditions and the element, then the closing token (the comprehension's
or the key builtin's own pyx token, unchanged). An item that uses the bound variables is annotated as
a vector, its value for each of the first bindings.

  prefix   @Comp:list:1i1/3 @Mult/2 v0 #2 a0 @Gt/2 v0 #1        [v0 * 2 for v0 in a0 if v0 > 1]
  postfix  a0 @For:1 v0 #1 @Gt/2 v0 #2 @Mult/2 @Comp:list:1i1/3

Variables keep pyx's numbering by binding depth, so both orders spell a program the same way.
"""
import ast
import itertools
import math
import re
import types

from ..rulecoder import backinfer as bi
from ..rulecoder.limits import Timeout, time_limit
from . import pyx

FOR = "@For:"
FOR_TOKENS = [f"{FOR}{n}" for n in (1, 2, 3)]
MAX_POST = 36                 # a 32-token pyx program plus its scope openers
VECTOR = 8                    # bindings shown for an item inside a scope
EXECUTION_SECONDS = 0.25
KINDS = ["none", "bool", "int", "float", "str", "list", "tuple", "set", "dict", "iter", "other", "err"]
SMALL = range(-5, 21)
RELATIONS = (["eq", "eqv", "len", "perm", "rev", "set", "case", "sup", "sub", "elem", "has", "samelen", "lt", "gt",
              "not", "type", "none", "sum", "count", "all", "any", "join", "prefix", "neg"]
             + [f"add{k}" for k in range(1, 11)] + [f"sub{k}" for k in range(1, 11)]
             + [f"mul{k}" for k in range(2, 11)] + [f"div{k}" for k in range(2, 11)])
ANNOTATIONS = ([f"K:{k}" for k in KINDS] + [f"K:v{k}" for k in KINDS] + [f"R:{r}" for r in RELATIONS]
               + [f"N:{n}" for n in SMALL] + ["N:big", "N:neg", "N:T", "N:F", "N:-"])


# ---------------------------------------------------------------- tree shapes
def parse(tokens):
    """The (token, children) tree of a prefix pyx program; raises ValueError if malformed."""
    position = 0

    def read():
        nonlocal position
        if position >= len(tokens):
            raise ValueError("incomplete program")
        token = tokens[position]
        position += 1
        return token, tuple(read() for _ in range(pyx.arity(token)))

    tree = read()
    if position != len(tokens):
        raise ValueError("trailing tokens")
    return tree


def flatten(node):
    token, children = node
    out = [token]
    for child in children:
        out += flatten(child)
    return out


def is_comprehension(token):
    return token.startswith("@Comp:")


def is_key(token):
    return token.startswith("@") and ":key" in token.rsplit("/", 1)[0]


def comprehension_shape(token):
    """(kind, [(variables, conditions) per generator]) of a comprehension token."""
    _, kind, pattern = token[1:].rsplit("/", 1)[0].split(":")
    return kind, [tuple(int(x) for x in part.split("i")) for part in pattern.split(",")]


def key_layout(token):
    """(positional arguments, has a default) of a key builtin token."""
    has_default = ":default" in token
    return pyx.arity(token) - 1 - has_default, has_default


# ---------------------------------------------------------------- prefix <-> postfix
def to_postfix(tokens):
    out = []

    def emit(node):
        token, children = node
        if is_comprehension(token):
            kind, shapes = comprehension_shape(token)
            elements = 2 if kind == "dict" else 1
            rest, index = children[elements:], 0
            for count, conditions in shapes:
                emit(rest[index])
                index += 1
                out.append(f"{FOR}{count}")
                for _ in range(conditions):
                    emit(rest[index])
                    index += 1
            for element in children[:elements]:
                emit(element)
        elif is_key(token):
            body, arguments = children[0], children[1:]
            # The other positional arguments and the default sit outside the key function's scope,
            # so they come first; the first argument is the one the key function's variable ranges over.
            for argument in arguments[1:]:
                emit(argument)
            emit(arguments[0])
            out.append(f"{FOR}1")
            emit(body)
        else:
            for child in children:
                emit(child)
        out.append(token)

    emit(parse(tokens))
    return out


class Frame:
    """A scope: the variables it binds, the iterable they range over, and the items built inside it."""
    __slots__ = ("variables", "iterable", "items")

    def __init__(self, variables, iterable, items=()):
        self.variables, self.iterable, self.items = variables, iterable, tuple(items)

    def with_items(self, items):
        return Frame(self.variables, self.iterable, items)


def bound(frames):
    return sum(frame.variables for frame in frames)


def step(frames, token):
    """The frames after one postfix token (a new tuple; frames are shared, never changed); ValueError
    if the token cannot follow."""
    current = frames[-1]
    if token.startswith(FOR):
        if not current.items:
            raise ValueError("no iterable")
        return frames[:-1] + (current.with_items(current.items[:-1]),
                              Frame(int(token[len(FOR):]), current.items[-1]))
    if is_comprehension(token):
        kind, shapes = comprehension_shape(token)
        if len(frames) <= len(shapes):
            raise ValueError("comprehension closes too many scopes")
        closing = frames[len(frames) - len(shapes):]
        elements = 2 if kind == "dict" else 1
        children_tail = []
        for index, ((count, conditions), frame) in enumerate(zip(shapes, closing)):
            last = index == len(shapes) - 1
            if frame.variables != count or len(frame.items) != conditions + (elements if last else 0):
                raise ValueError("comprehension shape")
            children_tail += [frame.iterable] + list(frame.items[:conditions])
        node = (token, tuple(closing[-1].items[len(closing[-1].items) - elements:]) + tuple(children_tail))
        parent = frames[len(frames) - len(shapes) - 1]
        return frames[:len(frames) - len(shapes) - 1] + (parent.with_items(parent.items + (node,)),)
    if is_key(token):
        positional, has_default = key_layout(token)
        others = positional - 1 + has_default
        if len(frames) < 2 or current.variables != 1 or len(current.items) != 1:
            raise ValueError("key function shape")
        parent = frames[-2]
        if len(parent.items) < others:
            raise ValueError("missing arguments")
        rest = parent.items[len(parent.items) - others:]
        node = (token, (current.items[0], current.iterable) + rest)
        return frames[:-2] + (parent.with_items(parent.items[:len(parent.items) - others] + (node,)),)
    width = pyx.arity(token)
    if len(current.items) < width:
        raise ValueError("missing operands")
    node = (token, current.items[len(current.items) - width:])
    return frames[:-1] + (current.with_items(current.items[:len(current.items) - width] + (node,)),)


def start():
    return (Frame(0, None),)


def finished(frames):
    return len(frames) == 1 and len(frames[0].items) == 1


def to_prefix(postfix):
    frames = start()
    for token in postfix:
        frames = step(frames, token)
    if not finished(frames):
        raise ValueError("unfinished program")
    return flatten(frames[0].items[0])


# ---------------------------------------------------------------- running items
_CODE = {}


def _compiled(tokens, depth):
    key = (tuple(tokens), depth)
    code = _CODE.get(key)
    if code is None:
        if len(_CODE) > 200000:
            _CODE.clear()
        tree = pyx._Guarded().visit(pyx.deserialize(list(tokens), depth))
        code = compile(ast.fix_missing_locations(ast.Expression(tree)), "<post>", "eval")
        _CODE[key] = code
    return code


def _scope(args, literals, output):
    if pyx._ENV is None:
        pyx._ENV = pyx.environment()
    scope = dict(pyx._ENV)
    scope.update({f"a{i}": value for i, value in enumerate(args)})
    scope.update({f"c{i}": value for i, value in enumerate(literals)})
    if isinstance(output, str):
        scope.update(o=output, o_not=pyx.complement(output))
    return scope


def _free(tokens, depth):
    return max((int(t[1:]) for t in tokens if re.fullmatch(r"v\d", t) and int(t[1:]) < depth), default=-1)


def _bindings(frames, scope, needed, limit):
    """The first `limit` assignments of the variables v0..v{needed} the open frames bind, in iteration order."""
    out = []

    def walk(index, depth, current):
        if len(out) >= limit:
            return
        if depth > needed:
            out.append(dict(current))
            return
        frame = frames[index]
        iterable = eval(_compiled(flatten(frame.iterable), depth), {**scope, **current})
        for element in itertools.islice(iter(iterable), limit):
            if frame.variables == 1:
                values = (element,)
            else:
                values = tuple(element)
                if len(values) != frame.variables:
                    continue
            walk(index + 1, depth + frame.variables, {**current, **{f"v{depth + i}": v for i, v in enumerate(values)}})
            if len(out) >= limit:
                return

    walk(1, 0, {})
    return out


def item_value(frames, node, scope):
    """(vector?, value): the item's value, or its values over the first bindings when it uses a bound variable."""
    tokens = flatten(node)
    depth = bound(frames)
    code = _compiled(tokens, depth)
    needed = _free(tokens, depth)
    if needed < 0:
        return False, eval(code, scope)
    values = []
    for binding in _bindings(frames, scope, needed, VECTOR):
        try:
            values.append(_materialize(eval(code, {**scope, **binding})))
        except Exception as error:      # one binding outside the item's domain says little
            values.append(error)
    return True, values


def frame_value(frames, scope):
    """The elements a just-opened frame's variables range over (shown after @For:n)."""
    frame = frames[-1]
    iterable = eval(_compiled(flatten(frame.iterable), bound(frames[:-1])), scope)
    return [_materialize(e) for e in itertools.islice(iter(iterable), VECTOR)]


# ---------------------------------------------------------------- annotation
def _materialize(value, limit=200):
    if isinstance(value, (types.GeneratorType, map, filter, zip, enumerate, reversed, range, type({}.keys()),
                          type({}.values()), type({}.items()))):
        return list(itertools.islice(value, limit))
    return value


def kind(value):
    if value is None:
        return "none"
    for name, types_ in (("bool", bool), ("int", int), ("float", float), ("str", str), ("list", list),
                         ("tuple", tuple), ("set", (set, frozenset)), ("dict", dict)):
        if isinstance(value, types_):
            return name
    return "other"


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _sized(value):
    return isinstance(value, (str, list, tuple, set, frozenset, dict))


def _loose(value):
    if isinstance(value, (list, tuple)):
        return [_loose(v) for v in value]
    if isinstance(value, (set, frozenset)):
        try:
            return sorted(_loose(v) for v in value)
        except TypeError:
            return value
    if _number(value) and float(value).is_integer():
        return int(value)
    return value


def _contains(container, part):
    if isinstance(container, str):
        return isinstance(part, str) and part != "" and part in container
    if isinstance(container, (list, tuple)):
        if part in container:
            return True
        if isinstance(part, (list, tuple)) and 0 < len(part) < len(container):
            n = len(part)
            return any(list(container[i:i + n]) == list(part) for i in range(len(container) - n + 1))
        return False
    if isinstance(container, (set, frozenset, dict)):
        try:
            return part in container
        except TypeError:
            return False
    return False


def relation(value, expected, mode="eq"):
    """How a (scalar) value relates to the expected output; the first that holds."""
    if bi._matches(value, {"expected": expected, "mode": mode}):
        return "eq"
    # 1 == True in Python, but an integer where a verdict is expected is not "nearly right".
    if isinstance(value, bool) == isinstance(expected, bool) and (
            _loose(value) == _loose(expected) or (isinstance(value, list) and isinstance(expected, str)
                                                  and value == list(expected))):
        return "eqv"
    if _number(value) and _number(expected):
        difference = expected - value
        if float(difference).is_integer() and 1 <= abs(difference) <= 10:
            return f"add{int(difference)}" if difference > 0 else f"sub{int(-difference)}"
        if value and float(expected / value).is_integer() and 2 <= expected / value <= 10:
            return f"mul{int(expected / value)}"
        if expected and float(value / expected).is_integer() and 2 <= value / expected <= 10:
            return f"div{int(value / expected)}"
        if expected == -value:
            return "neg"
        return "lt" if value < expected else "gt"
    if isinstance(value, bool) and isinstance(expected, bool):
        return "not"
    if _sized(value) and isinstance(expected, int) and not isinstance(expected, bool) and len(value) == expected:
        return "len"
    if _sized(value) and _sized(expected) and not isinstance(value, dict):
        # A reversal is also a permutation; the more specific relation is the more useful hint.
        if isinstance(value, (list, tuple, str)) and isinstance(expected, (list, tuple, str)) \
                and list(value)[::-1] == list(expected):
            return "rev"
        try:
            if sorted(value) == sorted(expected) and len(value) == len(expected):
                return "perm"
        except TypeError:
            pass
        try:
            if set(value) == set(expected):
                return "set"
        except TypeError:
            pass
        if isinstance(value, str) and isinstance(expected, str) and value.lower() == expected.lower():
            return "case"
    if _contains(value, expected):
        return "sup"
    if _contains(expected, value):
        return "sub"
    if _sized(value) and _sized(expected) and len(value) == len(expected):
        return "samelen"
    if type(value) is type(expected):
        return "type"
    return "none"


def vector_relation(values, expected, mode="eq"):
    """How the item's values over the bindings relate to the expected output (what aggregating them would give)."""
    ok = [v for v in values if not isinstance(v, Exception)]
    if not ok:
        return "none"
    try:
        if bi._matches(ok, {"expected": expected, "mode": mode}) or _loose(ok) == _loose(expected):
            return "eq"
        if isinstance(expected, (list, tuple)) and len(ok) < len(expected) and _loose(ok) == _loose(list(expected[:len(ok)])):
            return "prefix"
        # Aggregates are only suggested for the values they apply to: truth values for all/any/count,
        # numbers for sum (any non-empty string is "all true", which would say nothing).
        verdicts = all(isinstance(v, bool) for v in ok)
        if isinstance(expected, bool) and verdicts:
            if all(ok) == expected:
                return "all"
            if any(ok) == expected:
                return "any"
        if _number(expected) and all(_number(v) for v in ok) and sum(ok) == expected:
            return "sum"
        if isinstance(expected, int) and not isinstance(expected, bool) and verdicts and sum(ok) == expected:
            return "count"
        if isinstance(expected, str) and all(isinstance(v, str) for v in ok) and "".join(ok) == expected:
            return "join"
        if any(bi._matches(v, {"expected": expected, "mode": "eq"}) for v in ok):
            return "has"
        if _sized(expected) and set(map(repr, ok)) == set(map(repr, expected)):
            return "set"
    except Exception:
        return "none"
    return "none"


def size(value):
    if isinstance(value, bool):
        return "T" if value else "F"
    if _number(value):
        if isinstance(value, float) and not math.isfinite(value):
            return "big"
        number = round(value)
        return str(number) if number in SMALL else "big" if number > 0 else "neg"
    if _sized(value):
        return str(len(value)) if len(value) in SMALL else "big"
    return "-"


def annotate_value(vector, value, expected, mode="eq"):
    """The three annotation symbols for an item (or a frame's elements when `vector`)."""
    if isinstance(value, Exception):
        return ["K:err", "R:none", "N:-"]
    if vector:
        ok = [v for v in value if not isinstance(v, Exception)]
        head = kind(ok[0]) if ok else "err"
        return [f"K:v{head}", f"R:{vector_relation(value, expected, mode)}", f"N:{size(value)}"]
    value = _materialize(value)
    return [f"K:{kind(value)}", f"R:{relation(value, expected, mode)}", f"N:{size(value)}"]


def annotation(frames, token, scope, expected, mode="eq"):
    """Annotation symbols after `token` took the program to `frames`."""
    try:
        with time_limit(EXECUTION_SECONDS):
            if token.startswith(FOR):
                return annotate_value(True, frame_value(frames, scope), expected, mode)
            vector, value = item_value(frames, frames[-1].items[-1], scope)
            return annotate_value(vector, value, expected, mode)
    except (Exception, Timeout) as error:
        return annotate_value(False, error if isinstance(error, Exception) else RuntimeError("timeout"), expected,
                              mode)


def annotated(postfix, args, literals, expected, mode="eq"):
    """[(token, [kind, relation, size]), ...] for a whole postfix program on one example."""
    output = expected if isinstance(expected, str) else None
    scope = _scope(args, literals, output)
    frames, out = start(), []
    for token in postfix:
        frames = step(frames, token)
        out.append((token, annotation(frames, token, scope, expected, mode)))
    return out
