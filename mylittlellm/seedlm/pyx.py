"""SeedLM-3's program language: Python expressions over the rule library, written as prefix tokens.

SeedLM-2 wrote programs as prefix trees of library operators only (`len/1 shortest/1 a0`). Asked to
solve, with one Python expression over the same library, the development tasks the rule engine
fails, teachers managed 108 of 154 (data/seedlm/oneliners_dev_part*.json). What the operators alone
lack is arithmetic with small constants, comparisons, conditionals, tuples, indexing and
comprehensions. pyx keeps the prefix form, which needs no brackets and lets a beam search check
arity, and adds Python's own nodes as tokens:

  library operator   name/arity              txt_scs_length/2
  Python node        @Kind/arity             @Add/2  @IfExp/3  @Index/2  @len/1  @.split/2  @math.sqrt/1
  comprehension      @Comp:kind:gens/arity   @Comp:gen:1i1/3 is (elt for v in iter if cond)
  leaves             a0..a3 arguments, c0..c5 description literals, v0..v3 bound variables,
                     #-1..#10 small integers, $<s> short strings, o / o! the visible output string
                     and its conventional opposite, None True False, math.pi, T:int (type names)

Bound variables are numbered by binding depth (v0 for the outermost), so a program has one spelling.
"""
import ast
import copy
import math
import re

from ..rulecoder import backinfer as bi

SMALL_INTS = range(-1, 11)
SHORT_STRING = 2
MAX_VARS = 4
MAX_TUPLE = 6
BINOPS = {ast.Add: "Add", ast.Sub: "Sub", ast.Mult: "Mult", ast.Div: "Div", ast.FloorDiv: "FloorDiv",
          ast.Mod: "Mod", ast.Pow: "Pow", ast.BitAnd: "BitAnd", ast.BitOr: "BitOr", ast.BitXor: "BitXor",
          ast.LShift: "LShift", ast.RShift: "RShift"}
UNARY = {ast.USub: "USub", ast.Not: "Not", ast.Invert: "Invert"}
BOOLOPS = {ast.And: "And", ast.Or: "Or"}
COMPARE = {ast.Eq: "Eq", ast.NotEq: "NotEq", ast.Lt: "Lt", ast.LtE: "LtE", ast.Gt: "Gt", ast.GtE: "GtE",
           ast.In: "In", ast.NotIn: "NotIn", ast.Is: "Is", ast.IsNot: "IsNot"}
BUILTINS = ["abs", "all", "any", "bool", "chr", "dict", "divmod", "enumerate", "float", "int", "len", "list", "max",
            "min", "ord", "pow", "range", "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
            "isinstance", "frozenset"]
TYPES = ["int", "str", "list", "tuple", "float", "dict", "set", "bool"]
MODULES = {"math": math, "re": re}
MODULE_CONSTANTS = {"math.pi": math.pi, "math.e": math.e, "math.inf": math.inf}
# Verdict strings come in conventional pairs; one example shows one of them (a rule a teacher writes).
PAIRS = [("Yes", "No"), ("YES", "NO"), ("yes", "no"), ("Found a match!", "Not matched!"),
         ("Matched!", "Not Matched!"), ("Equal", "Not Equal"), ("Valid", "Invalid"), ("Accept", "Discard"),
         ("Even", "Odd"), ("EVEN", "ODD"), ("Even Parity", "Odd Parity"), ("True", "False")]


class Unsupported(ValueError):
    """The expression uses something pyx does not write."""


def complement(text):
    """The conventional opposite of a verdict string ("Equal" -> "Not Equal")."""
    for left, right in PAIRS:
        if text == left:
            return right
        if text == right:
            return left
    for prefix, other in (("Not ", ""), ("Non ", ""), ("Invalid ", "Valid "), ("Valid ", "Invalid ")):
        if text.startswith(prefix):
            return other + text[len(prefix):]
    return "Not " + text


def callable_name(key):
    name, arity = key.split("/")
    return f"{name}_{arity}"


LIBRARY = {callable_name(key): key for key in bi.OPERATORS}


def arity(token):
    """Children a token takes; leaves take none (a short string such as "$/" is a leaf)."""
    if "/" in token and not token.startswith("$"):
        return int(token.rsplit("/", 1)[1])
    return 0


# ---------------------------------------------------------------- Python expression -> tokens
def serialize(source, n_args, literals=(), output=None):
    """Prefix tokens for a Python expression over a0..a{n_args-1}; raises Unsupported."""
    try:
        tree = ast.parse(source, mode="eval").body
    except SyntaxError as error:
        raise Unsupported(f"syntax: {error}") from None
    tokens = []
    _emit(tree, {}, tokens, n_args, list(literals), output)
    return tokens


def _literal_token(value, literals, output):
    for index, literal in enumerate(literals):
        if type(literal) is type(value) and literal == value:
            return f"c{index}"
    return None


def _emit(node, scope, out, n_args, literals, output):
    def emit(child, inner=None):
        _emit(child, scope if inner is None else inner, out, n_args, literals, output)

    if isinstance(node, ast.Name):
        if node.id in scope:
            out.append(scope[node.id])
        elif re.fullmatch(r"a\d", node.id) and int(node.id[1]) < n_args:
            out.append(node.id)
        elif node.id in TYPES:
            out.append(f"T:{node.id}")
        else:
            raise Unsupported(f"name {node.id}")
    elif isinstance(node, ast.Constant):
        out.append(_constant(node.value, literals, output))
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in BINOPS:
            raise Unsupported(f"operator {type(node.op).__name__}")
        out.append(f"@{BINOPS[type(node.op)]}/2")
        emit(node.left)
        emit(node.right)
    elif isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant) \
                and type(node.operand.value) is int and -node.operand.value in SMALL_INTS:
            out.append(f"#{-node.operand.value}")
            return
        if isinstance(node.op, ast.UAdd):
            emit(node.operand)
            return
        if type(node.op) not in UNARY:
            raise Unsupported(f"unary {type(node.op).__name__}")
        out.append(f"@{UNARY[type(node.op)]}/1")
        emit(node.operand)
    elif isinstance(node, ast.BoolOp):
        values = node.values
        for value in values[:-1]:
            out.append(f"@{BOOLOPS[type(node.op)]}/2")
            emit(value)
        emit(values[-1])
    elif isinstance(node, ast.Compare):
        # a < b < c is written as (a < b) and (b < c): same value for expressions without side effects.
        pairs = list(zip([node.left] + node.comparators[:-1], node.ops, node.comparators))
        for left, op, right in pairs[:-1]:
            out.append("@And/2")
            out.append(f"@{COMPARE[type(op)]}/2")
            emit(left)
            emit(right)
        left, op, right = pairs[-1]
        out.append(f"@{COMPARE[type(op)]}/2")
        emit(left)
        emit(right)
    elif isinstance(node, ast.IfExp):
        out.append("@IfExp/3")
        emit(node.test)
        emit(node.body)
        emit(node.orelse)
    elif isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        if len(node.elts) > MAX_TUPLE or any(isinstance(e, ast.Starred) for e in node.elts):
            raise Unsupported("long or starred display")
        out.append(f"@{type(node).__name__}/{len(node.elts)}")
        for element in node.elts:
            emit(element)
    elif isinstance(node, ast.Subscript):
        if isinstance(node.slice, ast.Slice):
            parts = [node.slice.lower, node.slice.upper] + ([node.slice.step] if node.slice.step else [])
            out.append("@Slice/3" if len(parts) == 2 else "@Slice3/4")
            emit(node.value)
            for part in parts:
                if part is None:
                    out.append("None")
                else:
                    emit(part)
        else:
            out.append("@Index/2")
            emit(node.value)
            emit(node.slice)
    elif isinstance(node, ast.Call):
        _emit_call(node, scope, out, n_args, literals, output)
    elif isinstance(node, ast.Attribute):
        name = f"{node.value.id}.{node.attr}" if isinstance(node.value, ast.Name) else ""
        if name not in MODULE_CONSTANTS:
            raise Unsupported(f"attribute {node.attr}")
        out.append(name)
    elif isinstance(node, (ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp)):
        _emit_comprehension(node, scope, out, n_args, literals, output)
    else:
        raise Unsupported(type(node).__name__)


def _constant(value, literals, output):
    if isinstance(value, bool) or value is None:
        return repr(value)
    found = _literal_token(value, literals, output)
    if isinstance(value, int):
        if value in SMALL_INTS:
            return f"#{value}"
        if found:
            return found
        raise Unsupported(f"integer {value}")
    if isinstance(value, float):
        if found:
            return found
        if value in (0.5, 0.0, 1.0, 2.0):
            return f"#{value}"
        raise Unsupported(f"float {value}")
    if isinstance(value, str):
        if isinstance(output, str) and value == output:
            return "o"
        if isinstance(output, str) and value == complement(output):
            return "o!"
        if found:
            return found
        if len(value) <= SHORT_STRING:
            return "$" + value
        raise Unsupported(f"string {value!r}")
    raise Unsupported(f"constant {value!r}")


def _bind(target, scope):
    """Bind a comprehension or lambda target; returns the extended scope and how many names."""
    names = [target.id] if isinstance(target, ast.Name) else \
        [e.id for e in target.elts] if isinstance(target, ast.Tuple) and all(isinstance(e, ast.Name) for e in target.elts) \
        else None
    if names is None:
        raise Unsupported("nested target")
    # The next index is one past the deepest variable in scope (a shadowed name keeps its depth).
    depth = 1 + max((int(token[1:]) for token in scope.values()), default=-1)
    if depth + len(names) > MAX_VARS:
        raise Unsupported("too many variables")
    inner = dict(scope)
    for offset, name in enumerate(names):
        inner[name] = f"v{depth + offset}"
    return inner, len(names)


def _emit_comprehension(node, scope, out, n_args, literals, output):
    kind = {ast.ListComp: "list", ast.GeneratorExp: "gen", ast.SetComp: "set", ast.DictComp: "dict"}[type(node)]
    scopes, pattern, current = [], [], scope
    for generator in node.generators:
        if generator.is_async:
            raise Unsupported("async")
        before = current
        current, count = _bind(generator.target, current)
        scopes.append((before, current))
        pattern.append(f"{count}i{len(generator.ifs)}")
    width = (2 if kind == "dict" else 1) + sum(1 + len(g.ifs) for g in node.generators)
    out.append(f"@Comp:{kind}:{','.join(pattern)}/{width}")
    if kind == "dict":
        _emit(node.key, current, out, n_args, literals, output)
        _emit(node.value, current, out, n_args, literals, output)
    else:
        _emit(node.elt, current, out, n_args, literals, output)
    for generator, (before, after) in zip(node.generators, scopes):
        _emit(generator.iter, before, out, n_args, literals, output)
        for condition in generator.ifs:
            _emit(condition, after, out, n_args, literals, output)


def _as_lambda(function):
    """A function argument (key=len, map(str, ...)) as a one-parameter lambda."""
    if isinstance(function, ast.Lambda):
        if len(function.args.args) != 1 or function.args.vararg or function.args.kwarg:
            raise Unsupported("lambda arity")
        return function.args.args[0].arg, function.body
    if isinstance(function, (ast.Name, ast.Attribute)):
        return "__p", ast.Call(func=function, args=[ast.Name(id="__p", ctx=ast.Load())], keywords=[])
    raise Unsupported("function argument")


def _emit_call(node, scope, out, n_args, literals, output):
    function = node.func
    if isinstance(function, ast.Name) and function.id in LIBRARY:
        key = LIBRARY[function.id]
        if node.keywords or len(node.args) != bi.OPERATORS[key].arity:
            raise Unsupported(f"call shape {function.id}")
        out.append(key)
        for argument in node.args:
            _emit(argument, scope, out, n_args, literals, output)
        return
    if isinstance(function, ast.Name) and function.id in ("map", "filter"):
        if node.keywords or len(node.args) != 2:
            raise Unsupported(f"{function.id} shape")
        if function.id == "filter" and isinstance(node.args[0], ast.Constant) and node.args[0].value is None:
            name, body = "__p", ast.Name(id="__p", ctx=ast.Load())
        else:
            name, body = _as_lambda(node.args[0])
        target = ast.Name(id=name, ctx=ast.Store())
        if function.id == "map":
            rewritten = ast.GeneratorExp(elt=body, generators=[ast.comprehension(target=target, iter=node.args[1],
                                                                                   ifs=[], is_async=0)])
        else:
            rewritten = ast.GeneratorExp(elt=ast.Name(id=name, ctx=ast.Load()),
                                         generators=[ast.comprehension(target=target, iter=node.args[1],
                                                                       ifs=[body], is_async=0)])
        _emit(rewritten, scope, out, n_args, literals, output)
        return
    if isinstance(function, ast.Name) and function.id in BUILTINS:
        keywords = {k.arg: k.value for k in node.keywords}
        if set(keywords) - {"key", "reverse", "default"} or any(k is None for k in keywords):
            raise Unsupported("keywords")
        name = function.id
        children = []
        inner = scope
        if "key" in keywords:
            parameter, body = _as_lambda(keywords["key"])
            inner, _ = _bind(ast.Name(id=parameter, ctx=ast.Store()), scope)
            name += ":key"
        if "reverse" in keywords:
            flag = keywords["reverse"]
            if not (isinstance(flag, ast.Constant) and isinstance(flag.value, bool)):
                raise Unsupported("reverse")
            if flag.value:
                name += ":reverse"
        if "default" in keywords:
            name += ":default"
        width = len(node.args) + ("key" in keywords) + ("default" in keywords)
        out.append(f"@{name}/{width}")
        if "key" in keywords:
            _emit(body, inner, out, n_args, literals, output)
        for argument in node.args:
            if isinstance(argument, ast.Starred):
                raise Unsupported("starred argument")
            _emit(argument, scope, out, n_args, literals, output)
        if "default" in keywords:
            _emit(keywords["default"], scope, out, n_args, literals, output)
        return
    if isinstance(function, ast.Attribute):
        if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
            raise Unsupported("method keywords")
        if isinstance(function.value, ast.Name) and function.value.id in MODULES:
            out.append(f"@{function.value.id}.{function.attr}/{len(node.args)}")
        else:
            out.append(f"@.{function.attr}/{1 + len(node.args)}")
            _emit(function.value, scope, out, n_args, literals, output)
        for argument in node.args:
            _emit(argument, scope, out, n_args, literals, output)
        return
    raise Unsupported("call")


# ---------------------------------------------------------------- tokens -> Python expression
def deserialize(tokens):
    """The Python expression (an ast node) a token sequence spells; raises ValueError if malformed."""
    position = 0

    def read(depth):
        nonlocal position
        if position >= len(tokens):
            raise ValueError("incomplete program")
        token = tokens[position]
        position += 1
        return _node(token, depth, read)

    node = read(0)
    if position != len(tokens):
        raise ValueError("trailing tokens")
    return node


def _name(identifier):
    return ast.Name(id=identifier, ctx=ast.Load())


def _node(token, depth, read):
    if token in bi.OPERATORS:
        return ast.Call(func=_name(callable_name(token)), args=[read(depth) for _ in range(arity(token))], keywords=[])
    if not token.startswith("@"):
        return _leaf(token)
    head, width = token[1:].rsplit("/", 1)
    width = int(width)
    if head.startswith("Comp:"):
        return _comprehension(head, depth, read)
    if head in BINOPS.values():
        op = next(k for k, v in BINOPS.items() if v == head)
        return ast.BinOp(left=read(depth), op=op(), right=read(depth))
    if head in UNARY.values():
        op = next(k for k, v in UNARY.items() if v == head)
        return ast.UnaryOp(op=op(), operand=read(depth))
    if head in BOOLOPS.values():
        op = next(k for k, v in BOOLOPS.items() if v == head)
        return ast.BoolOp(op=op(), values=[read(depth), read(depth)])
    if head in COMPARE.values():
        op = next(k for k, v in COMPARE.items() if v == head)
        left = read(depth)
        return ast.Compare(left=left, ops=[op()], comparators=[read(depth)])
    if head == "IfExp":
        test, body, orelse = read(depth), read(depth), read(depth)
        return ast.IfExp(test=test, body=body, orelse=orelse)
    if head in ("Tuple", "List", "Set"):
        elements = [read(depth) for _ in range(width)]
        if head == "Set":
            return ast.Set(elts=elements) if elements else ast.Call(func=_name("set"), args=[], keywords=[])
        return getattr(ast, head)(elts=elements, ctx=ast.Load())
    if head == "Index":
        value = read(depth)
        return ast.Subscript(value=value, slice=read(depth), ctx=ast.Load())
    if head in ("Slice", "Slice3"):
        value = read(depth)
        parts = [read(depth) for _ in range(width - 1)]
        parts = [None if isinstance(p, ast.Constant) and p.value is None else p for p in parts]
        return ast.Subscript(value=value, slice=ast.Slice(*parts), ctx=ast.Load())
    if head.startswith("."):
        receiver = read(depth)
        args = [read(depth) for _ in range(width - 1)]
        return ast.Call(func=ast.Attribute(value=receiver, attr=head[1:], ctx=ast.Load()), args=args, keywords=[])
    module, _, attr = head.partition(".")
    if module in MODULES and attr:
        return ast.Call(func=ast.Attribute(value=_name(module), attr=attr, ctx=ast.Load()),
                        args=[read(depth) for _ in range(width)], keywords=[])
    name, *flags = head.split(":")
    if name not in BUILTINS:
        raise ValueError(f"unknown token {token}")
    keywords, body = [], None
    if "key" in flags:
        body = read(depth + 1)
    positional = width - ("key" in flags) - ("default" in flags)
    args = [read(depth) for _ in range(positional)]
    if "key" in flags:
        keywords.append(ast.keyword(arg="key", value=ast.Lambda(args=_lambda_args(f"v{depth}"), body=body)))
    if "reverse" in flags:
        keywords.append(ast.keyword(arg="reverse", value=ast.Constant(True)))
    if "default" in flags:
        keywords.append(ast.keyword(arg="default", value=read(depth)))
    return ast.Call(func=_name(name), args=args, keywords=keywords)


def _lambda_args(name):
    return ast.arguments(posonlyargs=[], args=[ast.arg(arg=name)], vararg=None, kwonlyargs=[], kw_defaults=[],
                         kwarg=None, defaults=[])


def _comprehension(head, depth, read):
    _, kind, pattern = head.split(":")
    shapes = [tuple(int(x) for x in part.split("i")) for part in pattern.split(",")]
    total = sum(count for count, _ in shapes)
    if kind == "dict":
        key, value = read(depth + total), read(depth + total)
    else:
        element = read(depth + total)
    generators, bound = [], depth
    for count, n_ifs in shapes:
        iterable = read(bound)
        names = [f"v{bound + i}" for i in range(count)]
        bound += count
        target = ast.Name(id=names[0], ctx=ast.Store()) if count == 1 else \
            ast.Tuple(elts=[ast.Name(id=n, ctx=ast.Store()) for n in names], ctx=ast.Store())
        generators.append(ast.comprehension(target=target, iter=iterable, ifs=[read(bound) for _ in range(n_ifs)],
                                            is_async=0))
    if kind == "dict":
        return ast.DictComp(key=key, value=value, generators=generators)
    return {"list": ast.ListComp, "gen": ast.GeneratorExp, "set": ast.SetComp}[kind](elt=element, generators=generators)


def _leaf(token):
    if re.fullmatch(r"[acv]\d", token):
        return _name(token)
    if token.startswith("#"):
        text = token[1:]
        return ast.Constant(float(text) if "." in text else int(text))
    if token.startswith("$"):
        return ast.Constant(token[1:])
    if token == "o":
        return _name("o")
    if token == "o!":
        return _name("o_not")
    if token in ("None", "True", "False"):
        return ast.Constant({"None": None, "True": True, "False": False}[token])
    if token in MODULE_CONSTANTS:
        module, attr = token.split(".")
        return ast.Attribute(value=_name(module), attr=attr, ctx=ast.Load())
    if token.startswith("T:"):
        return _name(token[2:])
    raise ValueError(f"unknown leaf {token}")


# ---------------------------------------------------------------- running and emitting code
# A beam proposes thousands of programs, some absurd: 99 ** 99 ** 99 or 'ab' * 10 ** 9 runs in C,
# where a timer signal cannot stop it. Execution routes the few operators that can explode through
# guards; emitted code does not need them (the sandbox that runs it has its own limits).
MAX_BITS = 1 << 14
MAX_ITEMS = 1 << 16


def _guard_pow(base, exponent):
    if isinstance(base, int) and isinstance(exponent, int) and exponent > 0 and abs(base) > 1 \
            and exponent * max(1, abs(base).bit_length()) > MAX_BITS:
        raise OverflowError("power too large")
    return base ** exponent


def _guard_mul(left, right):
    for sequence, count in ((left, right), (right, left)):
        if isinstance(sequence, (str, list, tuple, bytes)) and isinstance(count, int) \
                and len(sequence) * max(count, 0) > MAX_ITEMS:
            raise OverflowError("repetition too long")
    if isinstance(left, int) and isinstance(right, int) and left.bit_length() + right.bit_length() > MAX_BITS:
        raise OverflowError("product too large")
    return left * right


def _guard_lshift(value, shift):
    if isinstance(shift, int) and shift > MAX_BITS:
        raise OverflowError("shift too large")
    return value << shift


class _SafeMath:
    """math, with the functions whose cost grows with their argument bounded."""

    def __getattr__(self, name):
        function = getattr(math, name)
        if name in ("factorial", "comb", "perm", "prod", "pow", "lcm", "gcd"):
            def bounded(*args):
                if any(isinstance(a, int) and a > 5000 for a in args if name in ("factorial", "comb", "perm")):
                    raise OverflowError("argument too large")
                return function(*args)
            return bounded
        return function


GUARDS = {ast.Pow: "_guard_pow", ast.Mult: "_guard_mul", ast.LShift: "_guard_lshift"}


class _Guarded(ast.NodeTransformer):
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if type(node.op) in GUARDS:
            return ast.copy_location(ast.Call(func=_name(GUARDS[type(node.op)]), args=[node.left, node.right],
                                              keywords=[]), node)
        return node


def environment():
    import builtins
    env = {name: getattr(builtins, name) for name in BUILTINS + ["map", "filter"]}
    env.update({callable_name(key): op.fn for key, op in bi.OPERATORS.items()})
    env.update(MODULES)
    env.update(math=_SafeMath(), _guard_pow=_guard_pow, _guard_mul=_guard_mul, _guard_lshift=_guard_lshift)
    return {"__builtins__": {}, **env}


_ENV = None


def execute(tokens, args, literals=(), output=None):
    """Value of the program on the arguments (exceptions propagate)."""
    global _ENV
    if _ENV is None:
        _ENV = environment()
    node = ast.fix_missing_locations(ast.Expression(_Guarded().visit(deserialize(tokens))))
    scope = dict(_ENV)
    scope.update({f"a{i}": value for i, value in enumerate(args)})
    scope.update({f"c{i}": value for i, value in enumerate(literals)})
    if isinstance(output, str):
        scope.update(o=output, o_not=complement(output))
    return eval(compile(node, "<pyx>", "eval"), scope)


class _Inline(ast.NodeTransformer):
    """Literal and output names become constants in emitted code."""

    def __init__(self, literals, output):
        self.values = {f"c{i}": v for i, v in enumerate(literals)}
        if isinstance(output, str):
            self.values.update(o=output, o_not=complement(output))

    def visit_Name(self, node):
        if node.id in self.values:
            return ast.copy_location(ast.Constant(self.values[node.id]), node)
        return node


def to_code(entry_point, n_args, tokens, literals=(), output=None):
    """A standalone function: wrappers for the library operators used, then `return <expression>`."""
    node = _Inline(literals, output).visit(copy.deepcopy(deserialize(tokens)))
    used = [t for t in dict.fromkeys(tokens) if t in bi.OPERATORS]
    helpers, wrappers = [], []
    for key in used:
        op = bi.OPERATORS[key]
        if op.helper:
            helpers.append(op.helper.rstrip("\n") + "\n\n\n")
        params = [f"p{i}" for i in range(op.arity)]
        wrappers.append(f"def {callable_name(key)}({', '.join(params)}):\n    return {bi.render(op, params)}\n\n\n")
    params = ", ".join(f"a{i}" for i in range(n_args))
    body = ast.unparse(ast.fix_missing_locations(node))
    source = "".join(dict.fromkeys(helpers)) + "".join(wrappers) + f"def {entry_point}({params}):\n    return {body}\n"
    return bi.add_missing_imports(source)
