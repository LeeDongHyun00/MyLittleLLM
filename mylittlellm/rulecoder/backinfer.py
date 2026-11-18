"""Example back-inference: search backwards from the expected output of the visible example.

Prompt rules and repair feedback both ask a weak model to *follow text*, which the 1.5B
student could not do. This module needs no model. It takes the one example a task shows
(arguments and expected value) and enumerates compositions of small operators until one
reproduces the expected value. The operator library is the rule base: when the search
fails on a development task the teacher adds an operator, not an instruction, and a rule
is applied only if executing it reproduces the example.

One example under-determines the program, so matches are ranked (operators named by the
task description first, learned operator reliability next) and carry a confidence flag.
"""
import ast
from collections import Counter
import functools
import heapq
import importlib
import itertools
import json
import math
import os
import re
import sys
import time

from .codemods import add_missing_imports
from .tasks import ROOT

WEIGHTS = ROOT / "data/rule_pilot/operator_weights.json"
PACKS = ROOT / "data/rule_pilot/operator_packs.json"
MAX_REPR = 600           # skip values too large to be a plausible intermediate result
MAX_LEVEL_VALUES = 6000  # cap per level keeps the search around a second
MAX_MATCHES = 40
MAX_COLLECTED = 4000     # with "keep_best": matches gathered before ranking cuts them to MAX_MATCHES
MAX_PAIR_OPERANDS = 40   # with "named_pairs": derived values that may be combined with each other


class Op:
    """One rule: an executable operator, its code template and the words that name it."""

    __slots__ = ("name", "arity", "fn", "template", "alternatives", "gated", "cost", "kind", "helper", "pack")

    def __init__(self, name, arity, fn, template, keywords="", *, gated=False, cost=1.0, kind="generic",
                 helper=None):
        self.name, self.arity, self.fn, self.template = name, arity, fn, template
        # "a&b|c" means (a and b) or c, matched as substrings of the lower-cased description.
        self.alternatives = [tuple(alt.split("&")) for alt in keywords.split("|") if alt]
        self.gated, self.cost, self.kind, self.helper = gated, cost, kind, helper
        self.pack = "core"      # which teacher round added the rule; set by load_packs for later rounds

    def named_in(self, text):
        return any(all(part in text for part in alternative) for alternative in self.alternatives)


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def is_seq(v):
    return isinstance(v, (list, tuple))


def numseq(v):
    return is_seq(v) and len(v) > 0 and all(is_num(x) for x in v)


def intseq(v):
    return is_seq(v) and len(v) > 0 and all(is_int(x) for x in v)


def seqseq(v):
    return is_seq(v) and len(v) > 0 and all(is_seq(x) for x in v)


def strseq(v):
    return is_seq(v) and len(v) > 0 and all(isinstance(x, str) for x in v)


def need(condition):
    if not condition:
        raise TypeError("operator does not apply")


def like(source, items):
    """Keep the container type of the input (tuple in, tuple out)."""
    return tuple(items) if isinstance(source, tuple) else list(items)


def small(n, limit=2000):
    need(is_int(n) and 0 <= n <= limit)
    return n


def helper(source):
    """Compile a helper once; the same source is pasted into generated answers."""
    scope = {"math": math, "functools": functools}
    exec(source, scope)
    name = re.search(r"^def (\w+)", source, re.MULTILINE).group(1)       # a decorator may come first
    return scope[name], source


U, B, T = [], [], []


def unary(name, fn, template, keywords="", **options):
    U.append(Op(name, 1, fn, template, keywords, **options))


def binary(name, fn, template, keywords="", **options):
    B.append(Op(name, 2, fn, template, keywords, **options))


def ternary(name, fn, template, keywords="", **options):
    T.append(Op(name, 3, fn, template, keywords, **options))


# ---------------------------------------------------------------- generic sequence operators
SEQ_OR_STR = lambda x: is_seq(x) or isinstance(x, str)  # noqa: E731

unary("len", lambda x: (need(is_seq(x) or isinstance(x, (str, dict, set))), len(x))[1], "len({0})",
      "length|count|number of|size|how many")
unary("sum", lambda x: (need(numseq(x)), sum(x))[1], "sum({0})", "sum|total|add")
unary("max", lambda x: (need(numseq(x) or strseq(x)), max(x))[1], "max({0})", "max|largest|biggest|highest|greatest")
unary("min", lambda x: (need(numseq(x) or strseq(x)), min(x))[1], "min({0})", "min|smallest|lowest|least")
unary("prod", lambda x: (need(numseq(x)), math.prod(x))[1], "math.prod({0})", "product|multipl")
unary("mean", lambda x: (need(numseq(x)), sum(x) / len(x))[1], "sum({0}) / len({0})", "average|mean")
unary("range_width", lambda x: (need(numseq(x)), max(x) - min(x))[1], "max({0}) - min({0})",
      "difference&largest|difference&max|range")
unary("max_plus_min", lambda x: (need(numseq(x)), max(x) + min(x))[1], "max({0}) + min({0})",
      "sum&largest&smallest|sum&max&min")
unary("second_largest", lambda x: (need(numseq(x) and len(set(x)) > 1), sorted(set(x))[-2])[1],
      "sorted(set({0}))[-2]", "second&largest|second&max|second&highest")
unary("second_smallest", lambda x: (need(numseq(x) and len(set(x)) > 1), sorted(set(x))[1])[1],
      "sorted(set({0}))[1]", "second&smallest|second&min|second&lowest")
unary("sorted", lambda x: (need((is_seq(x) and len(x) > 1) or isinstance(x, set)), sorted(x))[1], "sorted({0})",
      "sort|ascending|order|arrange")
unary("sorted_desc", lambda x: (need(is_seq(x) and len(x) > 1), sorted(x, reverse=True))[1],
      "sorted({0}, reverse=True)", "descending|reverse&sort|decreasing")
unary("reverse", lambda x: (need(SEQ_OR_STR(x) and len(x) > 1), x[::-1])[1], "{0}[::-1]", "revers")
unary("dedupe", lambda x: (need(is_seq(x) and len(x) > 0), like(x, dict.fromkeys(x)))[1],
      "type({0})(dict.fromkeys({0}))", "unique|duplicate|distinct")
unary("sorted_set", lambda x: (need(is_seq(x) and len(x) > 0), sorted(set(x)))[1], "sorted(set({0}))",
      "unique|distinct|duplicate")
unary("once_only", lambda x: (need(is_seq(x) and len(x) > 0), like(x, [v for v in x if x.count(v) == 1]))[1],
      "type({0})(v for v in {0} if {0}.count(v) == 1)", "unique|once|duplicate|non-repeated|not repeated")
unary("repeated", lambda x: (need(is_seq(x) and len(x) > 0),
                             like(x, dict.fromkeys(v for v in x if x.count(v) > 1)))[1],
      "type({0})(dict.fromkeys(v for v in {0} if {0}.count(v) > 1))", "duplicate|repeated|more than once")
unary("first", lambda x: (need(SEQ_OR_STR(x) and len(x) > 0), x[0])[1], "{0}[0]", "first|initial")
unary("last", lambda x: (need(SEQ_OR_STR(x) and len(x) > 0), x[-1])[1], "{0}[-1]", "last|rear|final")
unary("drop_first", lambda x: (need(SEQ_OR_STR(x) and len(x) > 1), x[1:])[1], "{0}[1:]", "remove&first|except&first")
unary("drop_last", lambda x: (need(SEQ_OR_STR(x) and len(x) > 1), x[:-1])[1], "{0}[:-1]", "remove&last|except&last")
unary("drop_ends", lambda x: (need(SEQ_OR_STR(x) and len(x) > 2), x[1:-1])[1], "{0}[1:-1]",
      "first&last|both ends|middle")
unary("even_index", lambda x: (need(SEQ_OR_STR(x) and len(x) > 1), x[::2])[1], "{0}[::2]",
      "even ind|even pos|alternate|odd char|odd ind|odd pos")
unary("odd_index", lambda x: (need(SEQ_OR_STR(x) and len(x) > 1), x[1::2])[1], "{0}[1::2]",
      "odd ind|odd pos|alternate|even char|odd char|even ind|even pos")
unary("evens", lambda x: (need(intseq(x)), like(x, [v for v in x if v % 2 == 0]))[1],
      "type({0})(v for v in {0} if v % 2 == 0)", "even|odd")
unary("odds", lambda x: (need(intseq(x)), like(x, [v for v in x if v % 2 == 1]))[1],
      "type({0})(v for v in {0} if v % 2 == 1)", "odd|even|uneven")
unary("positives", lambda x: (need(numseq(x)), like(x, [v for v in x if v > 0]))[1],
      "type({0})(v for v in {0} if v > 0)", "positive|negative")
unary("negatives", lambda x: (need(numseq(x)), like(x, [v for v in x if v < 0]))[1],
      "type({0})(v for v in {0} if v < 0)", "negative")
unary("ints_only", lambda x: (need(is_seq(x) and len(x) > 0), [v for v in x if is_int(v)])[1],
      "[v for v in {0} if isinstance(v, int)]", "heterogeneous|mixed")
unary("strs_only", lambda x: (need(is_seq(x) and len(x) > 0), [v for v in x if isinstance(v, str)])[1],
      "[v for v in {0} if isinstance(v, str)]", "heterogeneous|mixed")
unary("argmax", lambda x: (need(numseq(x)), x.index(max(x)))[1], "{0}.index(max({0}))",
      "index&max|position&max|index&largest|position&largest")
unary("argmin", lambda x: (need(numseq(x)), x.index(min(x)))[1], "{0}.index(min({0}))",
      "index&min|position&min|index&smallest|position&smallest")
# Ties go to the value that occurs first. Iterating a set of strings instead would pick a different
# winner in every process, so neither the search nor the emitted code would be reproducible.
unary("most_common", lambda x: (need(is_seq(x) and len(x) > 0), max(dict.fromkeys(x), key=x.count))[1],
      "max(dict.fromkeys({0}), key={0}.count)",
      "most&frequen|most&occurr|maximum&occurr|maximum&frequen|most common")
unary("counter", lambda x: (need(SEQ_OR_STR(x) and len(x) > 0), dict(Counter(x)))[1],
      "dict(collections.Counter({0}))", "frequen|count|occurr|times")
unary("move_zeros", lambda x: (need(numseq(x) and 0 in x), [v for v in x if v != 0] + [0] * x.count(0))[1],
      "[v for v in {0} if v != 0] + [0] * {0}.count(0)", "zero&end|move&zero")
unary("swap_ends", lambda x: (need(isinstance(x, list) and len(x) > 1), [x[-1]] + x[1:-1] + [x[0]])[1],
      "[{0}[-1]] + {0}[1:-1] + [{0}[0]]", "interchange|swap|exchange")
unary("drop_consecutive", lambda x: (need(is_seq(x) and len(x) > 1), [k for k, _ in itertools.groupby(x)])[1],
      "[k for k, _ in itertools.groupby({0})]", "consecutive&duplicate")
unary("pack_consecutive", lambda x: (need(is_seq(x) and len(x) > 1),
                                     [list(g) for _, g in itertools.groupby(x)])[1],
      "[list(g) for _, g in itertools.groupby({0})]", "pack|consecutive&sublist")
unary("run_lengths", lambda x: (need(SEQ_OR_STR(x) and len(x) > 1),
                                [[len(list(g)), k] for k, g in itertools.groupby(x)])[1],
      "[[len(list(g)), k] for k, g in itertools.groupby({0})]", "run-length|run length|encod")
# element-wise maps


def _each(check, fn):
    def run(x):
        need(check(x))
        return like(x, [fn(v) for v in x])
    return run


unary("abs_each", _each(numseq, abs), "type({0})(abs(v) for v in {0})", "absolute")
unary("square_each", _each(numseq, lambda v: v * v), "type({0})(v * v for v in {0})", "square")
unary("cube_each", _each(numseq, lambda v: v ** 3), "type({0})(v ** 3 for v in {0})", "cube")
unary("negate_each", _each(numseq, lambda v: -v), "type({0})(-v for v in {0})", "negat")
unary("str_each", _each(lambda x: is_seq(x) and len(x) > 0, str), "type({0})(str(v) for v in {0})", "string", cost=0.5)
unary("int_each", lambda x: (need(strseq(x)), like(x, [int(v) for v in x]))[1], "type({0})(int(v) for v in {0})",
      "integer|convert|number", cost=0.5)
unary("len_each", lambda x: (need(strseq(x) or seqseq(x)), like(x, [len(v) for v in x]))[1],
      "type({0})(len(v) for v in {0})", "length")
unary("reverse_each", lambda x: (need(strseq(x)), like(x, [v[::-1] for v in x]))[1],
      "type({0})(v[::-1] for v in {0})", "revers")
unary("upper_each", lambda x: (need(strseq(x)), like(x, [v.upper() for v in x]))[1],
      "type({0})(v.upper() for v in {0})", "upper")
unary("adjacent_diff", lambda x: (need(numseq(x) and len(x) > 1), like(x, [b - a for a, b in zip(x, x[1:])]))[1],
      "type({0})(b - a for a, b in zip({0}, {0}[1:]))", "consecutive&difference|adjacent&difference")
unary("adjacent_prod", lambda x: (need(numseq(x) and len(x) > 1), like(x, [a * b for a, b in zip(x, x[1:])]))[1],
      "type({0})(a * b for a, b in zip({0}, {0}[1:]))",
      "consecutive&multipl|adjacent&multipl|consecutive&product|adjacent&product")
unary("adjacent_sum", lambda x: (need(numseq(x) and len(x) > 1), like(x, [a + b for a, b in zip(x, x[1:])]))[1],
      "type({0})(a + b for a, b in zip({0}, {0}[1:]))", "pairwise&addition|consecutive&sum|adjacent&sum|neighbor")
unary("adjacent_pairs", lambda x: (need(is_seq(x) and len(x) > 1), list(zip(x, x[1:])))[1],
      "list(zip({0}, {0}[1:]))", "consecutive&pair|adjacent&pair|pair&consecutive")
unary("prefix_sums", lambda x: (need(numseq(x)), like(x, itertools.accumulate(x)))[1],
      "type({0})(itertools.accumulate({0}))", "cumulative|running|prefix")
# nested sequences


def _rows(fn, guard=lambda row: True):
    def run(x):
        need(seqseq(x) and all(len(r) > 0 and guard(r) for r in x))
        return like(x, [fn(r) for r in x])
    return run


def _columns(fn):
    def run(x):
        need(seqseq(x) and len({len(r) for r in x}) == 1 and all(numseq(r) for r in x))
        return like(x, [fn(c) for c in zip(*x)])
    return run


PAIR = lambda r: len(r) == 2 and numseq(r)  # noqa: E731
unary("flatten", lambda x: (need(seqseq(x)), [v for row in x for v in row])[1], "[v for row in {0} for v in row]",
      "flatten|nested|single|all")
unary("transpose", lambda x: (need(seqseq(x) and len({len(r) for r in x}) == 1), [list(c) for c in zip(*x)])[1],
      "[list(c) for c in zip(*{0})]", "transpose|column|merge|zip")
unary("unzip", lambda x: (need(seqseq(x) and len({len(r) for r in x}) == 1), list(zip(*x)))[1], "list(zip(*{0}))",
      "transpose|column|unzip|zip")
unary("row_sum", _rows(sum, numseq), "type({0})(sum(row) for row in {0})", "sum|total")
unary("row_max", _rows(max, numseq), "type({0})(max(row) for row in {0})", "max|largest")
unary("row_min", _rows(min, numseq), "type({0})(min(row) for row in {0})", "min|smallest")
unary("row_prod", _rows(math.prod, numseq), "type({0})(math.prod(row) for row in {0})", "product|multipl")
unary("row_mean", _rows(lambda r: sum(r) / len(r), numseq), "type({0})(sum(row) / len(row) for row in {0})",
      "average|mean")
unary("row_sorted", _rows(sorted), "type({0})(sorted(row) for row in {0})", "sort")
unary("row_first", _rows(lambda r: r[0]), "type({0})(row[0] for row in {0})", "first|initial")
unary("row_last", _rows(lambda r: r[-1]), "type({0})(row[-1] for row in {0})", "last|rear")
unary("row_reverse", _rows(lambda r: r[::-1]), "type({0})(row[::-1] for row in {0})", "revers")
unary("pair_absdiff", _rows(lambda r: abs(r[0] - r[1]), PAIR), "type({0})(abs(a - b) for a, b in {0})",
      "difference")
unary("pair_diff", _rows(lambda r: r[0] - r[1], PAIR), "type({0})(a - b for a, b in {0})", "subtract|minus")
unary("row_sorted_tuple", _rows(lambda r: tuple(sorted(r))), "type({0})(tuple(sorted(row)) for row in {0})",
      "sort|order|occur")
unary("col_sum", _columns(sum), "type({0})(sum(c) for c in zip(*{0}))", "sum&column|sum&index|sum&each|sum&position")
unary("col_max", _columns(max), "type({0})(max(c) for c in zip(*{0}))", "max&column|max&index|max&each")
unary("col_min", _columns(min), "type({0})(min(c) for c in zip(*{0}))", "min&column|min&index|min&each")
unary("col_mean", _columns(lambda c: sum(c) / len(c)), "type({0})(sum(c) / len(c) for c in zip(*{0}))",
      "average|mean")
unary("longest", lambda x: (need(seqseq(x) or strseq(x)), max(x, key=len))[1], "max({0}, key=len)",
      "longest|maximum length|max length|largest")
unary("shortest", lambda x: (need(seqseq(x) or strseq(x)), min(x, key=len))[1], "min({0}, key=len)",
      "shortest|minimum length|min length|smallest")
unary("sort_by_len", lambda x: (need((seqseq(x) or strseq(x)) and len(x) > 1), sorted(x, key=len))[1],
      "sorted({0}, key=len)", "sort&length")
unary("sort_by_last", lambda x: (need(seqseq(x) and len(x) > 1 and all(len(r) > 0 for r in x)),
                                 sorted(x, key=lambda r: r[-1]))[1],
      "sorted({0}, key=lambda row: row[-1])", "sort&last|sort&second")
unary("common_all", lambda x: (need(seqseq(x) and len(x) > 1), sorted(set.intersection(*map(set, x))))[1],
      "sorted(set.intersection(*map(set, {0})))", "common|intersection|shared")
unary("union_all", lambda x: (need(seqseq(x) and len(x) > 1), sorted(set().union(*x)))[1],
      "sorted(set().union(*{0}))", "union|unique|distinct")
unary("pair_sums", lambda x: (need(seqseq(x) and len(x) > 1 and all(numseq(r) for r in x)
                                   and len({len(r) for r in x}) == 1),
                              [tuple(a + b for a, b in zip(p, q)) for p, q in itertools.combinations(x, 2)])[1],
      "[tuple(a + b for a, b in zip(p, q)) for p, q in itertools.combinations({0}, 2)]", "combination&sum")
unary("best_pair_product", lambda x: (need(numseq(x) and len(x) > 1),
                                      max(itertools.combinations(x, 2), key=lambda p: p[0] * p[1]))[1],
      "max(itertools.combinations({0}, 2), key=lambda p: p[0] * p[1])", "pair&product&high|pair&product&max")
# conversions are cheap: they change the container, not the content
# Sets are excluded on purpose: their iteration order differs between processes, so a program
# that turns a set into a sequence could match the example in one run and not in the next.
unary("tuple", lambda x: (need(isinstance(x, (list, str)) and len(x) > 0), tuple(x))[1], "tuple({0})",
      "tuple", cost=0.5)
unary("list", lambda x: (need(isinstance(x, (tuple, str)) and len(x) > 0), list(x))[1], "list({0})",
      "list", cost=0.5)
unary("set", lambda x: (need(is_seq(x) and len(x) > 0), set(x))[1], "set({0})", "set|unique|distinct", cost=0.5)
unary("join_space", lambda x: (need(strseq(x)), " ".join(x))[1], "' '.join({0})", "join|concatenat|sentence|string")
unary("join", lambda x: (need(strseq(x)), "".join(x))[1], "''.join({0})", "join|concatenat|string")
unary("join_digits", lambda x: (need(intseq(x) and all(v >= 0 for v in x)), int("".join(map(str, x))))[1],
      "int(''.join(map(str, {0})))", "single integer|join|concatenat|convert")

# ---------------------------------------------------------------- strings
STR = lambda x: isinstance(x, str)  # noqa: E731
unary("upper", lambda x: (need(STR(x)), x.upper())[1], "{0}.upper()", "upper")
unary("lower", lambda x: (need(STR(x)), x.lower())[1], "{0}.lower()", "lower")
unary("swapcase", lambda x: (need(STR(x)), x.swapcase())[1], "{0}.swapcase()", "toggle|swap&case|invert&case")
unary("title", lambda x: (need(STR(x)), x.title())[1], "{0}.title()", "title|capital")
unary("strip", lambda x: (need(STR(x)), x.strip())[1], "{0}.strip()", "strip|whitespace|space|trim")
unary("split", lambda x: (need(STR(x)), x.split())[1], "{0}.split()", "split|word")
unary("split_space", lambda x: (need(STR(x) and " " in x), x.split(" "))[1], "{0}.split(' ')", "split&space")
unary("split_comma", lambda x: (need(STR(x) and "," in x), [p.strip() for p in x.split(",")])[1],
      "[p.strip() for p in {0}.split(',')]", "split|comma")
unary("no_spaces", lambda x: (need(STR(x) and " " in x), x.replace(" ", ""))[1], "{0}.replace(' ', '')",
      "remove&space|remove&whitespace")
unary("squeeze_spaces", lambda x: (need(STR(x) and "  " in x), " ".join(x.split()))[1], "' '.join({0}.split())",
      "multiple space|extra space")
unary("spaces_to_underscore", lambda x: (need(STR(x) and " " in x), x.replace(" ", "_"))[1],
      "{0}.replace(' ', '_')", "underscore")
unary("underscore_to_space", lambda x: (need(STR(x) and "_" in x), x.replace("_", " "))[1],
      "{0}.replace('_', ' ')", "underscore")
unary("sorted_chars", lambda x: (need(STR(x) and len(x) > 1), "".join(sorted(x)))[1], "''.join(sorted({0}))", "sort")
unary("digits_only", lambda x: (need(STR(x)), "".join(c for c in x if c.isdigit()))[1],
      "''.join(c for c in {0} if c.isdigit())", "digit|number")
unary("no_digits", lambda x: (need(STR(x)), "".join(c for c in x if not c.isdigit()))[1],
      "''.join(c for c in {0} if not c.isdigit())", "remove&digit|remove&number")
unary("letters_only", lambda x: (need(STR(x)), "".join(c for c in x if c.isalpha()))[1],
      "''.join(c for c in {0} if c.isalpha())", "letter|alphabet")
unary("alnum_only", lambda x: (need(STR(x)), "".join(c for c in x if c.isalnum()))[1],
      "''.join(c for c in {0} if c.isalnum())", "alphanumeric|special")
unary("upper_only", lambda x: (need(STR(x)), "".join(c for c in x if c.isupper()))[1],
      "''.join(c for c in {0} if c.isupper())", "upper")
unary("no_upper", lambda x: (need(STR(x)), "".join(c for c in x if not c.isupper()))[1],
      "''.join(c for c in {0} if not c.isupper())", "remove&upper")
unary("no_lower", lambda x: (need(STR(x)), "".join(c for c in x if not c.islower()))[1],
      "''.join(c for c in {0} if not c.islower())", "remove&lower")
unary("no_vowels", lambda x: (need(STR(x)), "".join(c for c in x if c.lower() not in "aeiou"))[1],
      "''.join(c for c in {0} if c.lower() not in 'aeiou')", "vowel")
unary("numbers_in", lambda x: (need(STR(x)), [int(v) for v in re.findall(r"\d+", x)])[1],
      "[int(v) for v in re.findall(r'\\d+', {0})]", "number|digit|integer")
unary("quoted", lambda x: (need(STR(x) and '"' in x), re.findall(r'"(.*?)"', x))[1],
      "re.findall(r'\"(.*?)\"', {0})", "quot")
unary("words", lambda x: (need(STR(x)), re.findall(r"\w+", x))[1], "re.findall(r'\\w+', {0})", "word")
unary("reverse_words", lambda x: (need(STR(x) and " " in x), " ".join(x.split()[::-1]))[1],
      "' '.join({0}.split()[::-1])", "revers&word")
unary("reverse_each_word", lambda x: (need(STR(x) and " " in x), " ".join(w[::-1] for w in x.split()))[1],
      "' '.join(w[::-1] for w in {0}.split())", "revers&word")
unary("snake_to_camel", lambda x: (need(STR(x) and "_" in x), "".join(w.title() for w in x.split("_")))[1],
      "''.join(w.title() for w in {0}.split('_'))", "camel")
unary("camel_to_snake", lambda x: (need(STR(x) and any(c.isupper() for c in x)),
                                   re.sub(r"(?<!^)(?=[A-Z])", "_", x).lower())[1],
      "re.sub(r'(?<!^)(?=[A-Z])', '_', {0}).lower()", "snake")
unary("parse_int", lambda x: (need(STR(x) and x.lstrip("-").isdigit()), int(x))[1], "int({0})", "integer|number",
      cost=0.5)
unary("digit_sum_str", lambda x: (need(STR(x)), sum(int(c) for c in x if c.isdigit()))[1],
      "sum(int(c) for c in {0} if c.isdigit())", "sum&digit")
unary("char_count", lambda x: (need(STR(x)), sum(c.isalpha() for c in x))[1], "sum(c.isalpha() for c in {0})",
      "count&letter|count&alphabet|number of&letter")
unary("upper_count", lambda x: (need(STR(x)), sum(c.isupper() for c in x))[1], "sum(c.isupper() for c in {0})",
      "count&upper|number of&upper")
unary("vowel_count", lambda x: (need(STR(x)), sum(c.lower() in "aeiou" for c in x))[1],
      "sum(c.lower() in 'aeiou' for c in {0})", "count&vowel|number of&vowel")
unary("digit_count_str", lambda x: (need(STR(x)), sum(c.isdigit() for c in x))[1], "sum(c.isdigit() for c in {0})",
      "count&digit|number of&digit")

# ---------------------------------------------------------------- integers
unary("abs", lambda x: (need(is_num(x) and x < 0), abs(x))[1], "abs({0})", "absolute")
unary("square", lambda x: (need(is_num(x) and abs(x) < 10 ** 6), x * x)[1], "{0} * {0}", "square")
unary("cube", lambda x: (need(is_num(x) and abs(x) < 10 ** 4), x ** 3)[1], "{0} ** 3", "cube")
unary("str", lambda x: (need(is_int(x)), str(x))[1], "str({0})", "string", cost=0.5)
unary("digits", lambda x: (need(is_int(x) and x >= 0), [int(d) for d in str(x)])[1], "[int(d) for d in str({0})]",
      "digit")
unary("digit_sum", lambda x: (need(is_int(x) and x >= 0), sum(int(d) for d in str(x)))[1],
      "sum(int(d) for d in str({0}))", "sum&digit")
unary("reverse_digits", lambda x: (need(is_int(x) and x >= 10), int(str(x)[::-1]))[1], "int(str({0})[::-1])",
      "revers")
unary("digit_count", lambda x: (need(is_int(x) and x >= 0), len(str(x)))[1], "len(str({0}))",
      "count&digit|number of&digit")
unary("binary", lambda x: (need(is_int(x) and x >= 0), bin(x)[2:])[1], "bin({0})[2:]", "binary")
unary("popcount", lambda x: (need(is_int(x) and x >= 0), bin(x).count("1"))[1], "bin({0}).count('1')",
      "set bit|number of&bit")
unary("range", lambda x: list(range(small(x))), "list(range({0}))", "range")
unary("range1", lambda x: list(range(1, small(x) + 1)), "list(range(1, {0} + 1))", "range|first&natural")
unary("factorial", lambda x: math.factorial(small(x, 20)), "math.factorial({0})", "factorial")
unary("float", lambda x: (need(is_int(x)), float(x))[1], "float({0})", "float", cost=0.5)
unary("int_part", lambda x: (need(isinstance(x, float) and abs(x) < 10 ** 12), int(x))[1], "int({0})",
      "integer|round|floor", cost=0.5)
for digits_kept in (1, 2):
    unary(f"round{digits_kept}", lambda x, d=digits_kept: (need(isinstance(x, float)), round(x, d))[1],
          f"round({{0}}, {digits_kept})", "round|decimal|precision", cost=0.5)

# ---------------------------------------------------------------- dictionaries
DICT = lambda x: isinstance(x, dict) and len(x) > 0  # noqa: E731
unary("keys", lambda x: (need(DICT(x)), list(x.keys()))[1], "list({0}.keys())", "key")
unary("values", lambda x: (need(DICT(x)), list(x.values()))[1], "list({0}.values())", "value")
unary("sorted_items", lambda x: (need(DICT(x) and len(x) > 1), sorted(x.items()))[1], "sorted({0}.items())",
      "sort&key|sort")
unary("items_by_value", lambda x: (need(DICT(x) and len(x) > 1), sorted(x.items(), key=lambda kv: kv[1]))[1],
      "sorted({0}.items(), key=lambda kv: kv[1])", "sort&value")
unary("items_by_value_desc", lambda x: (need(DICT(x) and len(x) > 1),
                                        sorted(x.items(), key=lambda kv: kv[1], reverse=True))[1],
      "sorted({0}.items(), key=lambda kv: kv[1], reverse=True)", "sort&value")
unary("sum_values", lambda x: (need(DICT(x) and numseq(list(x.values()))), sum(x.values()))[1],
      "sum({0}.values())", "sum|total")
unary("pairs_to_dict", lambda x: (need(isinstance(x, list) and len(x) > 0
                                       and all(is_seq(p) and len(p) == 2 for p in x)), dict(x))[1],
      "dict({0})", "dictionary", cost=0.5)

# ---------------------------------------------------------------- binary: numbers


def _floor_div(a, b):
    need(b != 0)
    return a // b


def _true_div(a, b):
    need(b != 0)
    return a / b


def _mod(a, b):
    need(b != 0)
    return a % b


def _power(a, b):
    need(is_num(a) and is_int(b) and 0 <= b <= 12 and abs(a) <= 10 ** 6)
    return a ** b


NUMS = lambda x, y: is_num(x) and is_num(y)  # noqa: E731
INTS = lambda x, y: is_int(x) and is_int(y)  # noqa: E731
binary("add", lambda x, y: (need(NUMS(x, y)), x + y)[1], "{0} + {1}", "sum|add|plus|total")
binary("sub", lambda x, y: (need(NUMS(x, y)), x - y)[1], "{0} - {1}", "difference|subtract|minus")
binary("mul", lambda x, y: (need(NUMS(x, y)), x * y)[1], "{0} * {1}", "product|multipl|times")
binary("floordiv", lambda x, y: (need(NUMS(x, y)), _floor_div(x, y))[1], "{0} // {1}", "quotient|divi|floor")
binary("truediv", lambda x, y: (need(NUMS(x, y)), _true_div(x, y))[1], "{0} / {1}", "divi|ratio|average|mean")
binary("mod", lambda x, y: (need(NUMS(x, y)), _mod(x, y))[1], "{0} % {1}", "remainder|modulo")
binary("pow", _power, "{0} ** {1}", "power|exponent|raise")
binary("max2", lambda x, y: (need(NUMS(x, y)), max(x, y))[1], "max({0}, {1})", "max|larger|greater|bigger")
binary("min2", lambda x, y: (need(NUMS(x, y)), min(x, y))[1], "min({0}, {1})", "min|smaller|lesser")
binary("gcd", lambda x, y: (need(INTS(x, y)), math.gcd(x, y))[1], "math.gcd({0}, {1})", "gcd|common divisor|hcf")
binary("lcm", lambda x, y: (need(INTS(x, y) and x > 0 and y > 0), x * y // math.gcd(x, y))[1],
       "{0} * {1} // math.gcd({0}, {1})", "lcm|common multiple")
binary("xor", lambda x, y: (need(INTS(x, y)), x ^ y)[1], "{0} ^ {1}", "xor")
binary("bit_and", lambda x, y: (need(INTS(x, y)), x & y)[1], "{0} & {1}", "bitwise and")
# element-wise on two sequences (one or two levels deep)
ELEMENTWISE = ("type({0})(type(p)({expr} for a, b in zip(p, q)) if isinstance(p, (list, tuple)) else "
               "(lambda a, b: {expr})(p, q) for p, q in zip({0}, {1}))")


def _pairwise(fn):
    def run(x, y):
        need(is_seq(x) and is_seq(y) and len(x) == len(y) and len(x) > 0)
        if numseq(x) and numseq(y):
            return like(x, [fn(a, b) for a, b in zip(x, y)])
        need(all(numseq(a) and numseq(b) and len(a) == len(b) for a, b in zip(x, y)))
        return like(x, [like(a, [fn(p, q) for p, q in zip(a, b)]) for a, b in zip(x, y)])
    return run


def _ints(fn):
    return lambda a, b: (need(is_int(a) and is_int(b)), fn(a, b))[1]


for _name, _fn, _expr, _words in [
        ("add", lambda a, b: a + b, "a + b", "add|sum|addition"),
        ("sub", lambda a, b: a - b, "a - b", "subtract|substract|difference|minus"),
        ("mul", lambda a, b: a * b, "a * b", "multipl|product"),
        ("floordiv", _floor_div, "a // b", "divi|quotient"),
        ("truediv", _true_div, "a / b", "divi"),
        ("mod", _mod, "a % b", "modulo|mod|remainder"),
        ("pow", _power, "a ** b", "power|exponent"),
        ("and", _ints(lambda a, b: a & b), "a & b", "and"),
        ("xor", _ints(lambda a, b: a ^ b), "a ^ b", "xor"),
        ("or", _ints(lambda a, b: a | b), "a | b", "bitwise or"),
        ("max", max, "max(a, b)", "max|larger"),
        ("min", min, "min(a, b)", "min|smaller")]:
    binary(f"each_{_name}", _pairwise(_fn), "ELEMENTWISE:" + _expr, _words)
# two sequences
SEQS = lambda x, y: is_seq(x) and is_seq(y) and len(x) > 0 and len(y) > 0  # noqa: E731
binary("concat", lambda x, y: (need(type(x) is type(y) and SEQ_OR_STR(x) and len(x) and len(y)), x + y)[1],
       "{0} + {1}", "concatenat|append|join|merge|combine|add")
binary("replace_last", lambda x, y: (need(type(x) is type(y) and is_seq(x) and len(x) > 1 and len(y)),
                                     x[:-1] + y)[1], "{0}[:-1] + {1}", "replace&last")
binary("keep_common", lambda x, y: (need(SEQS(x, y)), like(x, [v for v in x if v in y]))[1],
       "type({0})(v for v in {0} if v in {1})", "common|intersection|shared|similar|both")
binary("remove_present", lambda x, y: (need(SEQS(x, y)), like(x, [v for v in x if v not in y]))[1],
       "type({0})(v for v in {0} if v not in {1})", "remove|not in|missing|except|not present")
binary("set_and", lambda x, y: (need(SEQS(x, y)), sorted(set(x) & set(y)))[1], "sorted(set({0}) & set({1}))",
       "common|intersection|shared|similar")
binary("set_or", lambda x, y: (need(SEQS(x, y)), sorted(set(x) | set(y)))[1], "sorted(set({0}) | set({1}))",
       "union")
binary("set_xor", lambda x, y: (need(SEQS(x, y)), sorted(set(x) ^ set(y)))[1], "sorted(set({0}) ^ set({1}))",
       "symmetric|dissimilar|not common")
binary("zip", lambda x, y: (need(SEQS(x, y) and len(x) == len(y)), list(zip(x, y)))[1], "list(zip({0}, {1}))",
       "zip|pair")
binary("zip_dict", lambda x, y: (need(SEQS(x, y) and len(x) == len(y)), dict(zip(x, y)))[1],
       "dict(zip({0}, {1}))", "dictionary|map")
binary("interleave", lambda x, y: (need(SEQS(x, y) and len(x) == len(y)), [v for p in zip(x, y) for v in p])[1],
       "[v for pair in zip({0}, {1}) for v in pair]", "interleav|alternat")
binary("merge_dicts", lambda x, y: (need(isinstance(x, dict) and isinstance(y, dict)), {**x, **y})[1],
       "{{**{0}, **{1}}}", "merge")
# sequence with an integer
SLICE = lambda x, n: SEQ_OR_STR(x) and is_int(n) and 0 < n < len(x)  # noqa: E731
binary("take", lambda x, n: (need(SLICE(x, n)), x[:n])[1], "{0}[:{1}]", "first|initial|prefix")
binary("drop", lambda x, n: (need(SLICE(x, n)), x[n:])[1], "{0}[{1}:]", "remove&first|after|rest")
binary("take_last", lambda x, n: (need(SLICE(x, n)), x[-n:])[1], "{0}[-{1}:]", "last|end|rear")
binary("at", lambda x, n: (need(is_seq(x) and is_int(n) and -len(x) <= n < len(x)), x[n])[1], "{0}[{1}]",
       "index|position|element")
binary("kth", lambda x, n: (need(is_seq(x) and is_int(n) and 0 < n <= len(x)), x[n - 1])[1], "{0}[{1} - 1]",
       "kth|nth|k'th|n'th")
binary("rotate_left", lambda x, n: (need(SLICE(x, n) and is_seq(x)), x[n:] + x[:n])[1], "{0}[{1}:] + {0}[:{1}]",
       "rotate&left")
binary("rotate_right", lambda x, n: (need(SLICE(x, n) and is_seq(x)), x[-n:] + x[:-n])[1],
       "{0}[-{1}:] + {0}[:-{1}]", "rotate&right")
binary("remove_kth", lambda x, n: (need(is_seq(x) and is_int(n) and 0 < n <= len(x)), x[:n - 1] + x[n:])[1],
       "{0}[:{1} - 1] + {0}[{1}:]", "remove&k|delete&k")
binary("chunks", lambda x, n: (need(SLICE(x, n) and is_seq(x)), [x[i:i + n] for i in range(0, len(x), n)])[1],
       "[{0}[i:i + {1}] for i in range(0, len({0}), {1})]", "chunk|split|every|group")
binary("every_nth", lambda x, n: (need(is_seq(x) and is_int(n) and 1 < n <= len(x)), [x[i::n] for i in range(n)])[1],
       "[{0}[i::{1}] for i in range({1})]", "split&every|split&nth")
binary("split_at", lambda x, n: (need(SLICE(x, n) and is_seq(x)), (x[:n], x[n:]))[1], "({0}[:{1}], {0}[{1}:])",
       "split&two|two parts")


def _filter(compare):
    def run(x, n):
        need(numseq(x) and is_num(n))
        return like(x, [v for v in x if compare(v, n)])
    return run


binary("gt", _filter(lambda v, n: v > n), "type({0})(v for v in {0} if v > {1})", "greater|larger|more than|above")
binary("ge", _filter(lambda v, n: v >= n), "type({0})(v for v in {0} if v >= {1})", "at least|not less")
binary("lt", _filter(lambda v, n: v < n), "type({0})(v for v in {0} if v < {1})", "less|smaller|below")
binary("ne", _filter(lambda v, n: v != n), "type({0})(v for v in {0} if v != {1})", "remove|delete|except")
binary("scale", lambda x, n: (need(numseq(x) and is_num(n)), like(x, [v * n for v in x]))[1],
       "type({0})(v * {1} for v in {0})", "multipl|times")
binary("shift", lambda x, n: (need(numseq(x) and is_num(n)), like(x, [v + n for v in x]))[1],
       "type({0})(v + {1} for v in {0})", "add|increment")
binary("power_each", lambda x, n: (need(numseq(x) and is_int(n) and 0 <= n <= 8), like(x, [v ** n for v in x]))[1],
       "type({0})(v ** {1} for v in {0})", "power|n-th|nth")
binary("nlargest", lambda x, n: (need(numseq(x) and is_int(n) and 0 < n <= len(x)), heapq.nlargest(n, x))[1],
       "heapq.nlargest({1}, {0})", "largest|biggest|maximum|top|expensive|highest")
binary("nsmallest", lambda x, n: (need(numseq(x) and is_int(n) and 0 < n <= len(x)), heapq.nsmallest(n, x))[1],
       "heapq.nsmallest({1}, {0})", "smallest|minimum|cheap|lowest")
binary("kth_smallest", lambda x, n: (need(numseq(x) and is_int(n) and 0 < n <= len(x)), sorted(x)[n - 1])[1],
       "sorted({0})[{1} - 1]", "smallest")
binary("kth_largest", lambda x, n: (need(numseq(x) and is_int(n) and 0 < n <= len(x)),
                                    sorted(x, reverse=True)[n - 1])[1], "sorted({0}, reverse=True)[{1} - 1]", "largest")
COLUMN = lambda x, n: seqseq(x) and is_int(n) and all(-len(r) <= n < len(r) for r in x)  # noqa: E731
binary("column", lambda x, n: (need(COLUMN(x, n)), [r[n] for r in x])[1], "[row[{1}] for row in {0}]",
       "column|nth|n'th|index|specified")
binary("sort_by_column", lambda x, n: (need(COLUMN(x, n) and len(x) > 1), sorted(x, key=lambda r: r[n]))[1],
       "sorted({0}, key=lambda row: row[{1}])", "sort")
binary("column_max", lambda x, n: (need(COLUMN(x, n) and all(numseq(r) for r in x)), max(r[n] for r in x))[1],
       "max(row[{1}] for row in {0})", "max")
binary("column_sum", lambda x, n: (need(COLUMN(x, n) and all(numseq(r) for r in x)), sum(r[n] for r in x))[1],
       "sum(row[{1}] for row in {0})", "sum")


def _by_length(compare):
    def run(x, n):
        need(is_int(n))
        words = x.split() if isinstance(x, str) else x
        need(is_seq(words) and len(words) > 0 and all(isinstance(w, (str, list, tuple)) for w in words))
        return [w for w in words if compare(len(w), n)]
    return run


WORDS = "[w for w in ({0}.split() if isinstance({0}, str) else {0}) if len(w) %s {1}]"
binary("longer_than", _by_length(lambda a, n: a > n), WORDS % ">", "longer|greater|more than")
binary("at_least_long", _by_length(lambda a, n: a >= n), WORDS % ">=", "at least")
binary("of_length", _by_length(lambda a, n: a == n), WORDS % "==", "length|exactly|of size")
binary("shorter_than", _by_length(lambda a, n: a < n), WORDS % "<", "shorter|less|smaller")
binary("repeat", lambda x, n: (need(isinstance(x, (str, list, tuple)) and len(x) and is_int(n) and 0 < n <= 20),
                               x * n)[1], "{0} * {1}", "repeat|times")
# sequence or string with a value
binary("count", lambda x, v: (need(SEQ_OR_STR(x) and len(x) and not is_seq(v)
                                   and (not isinstance(x, str) or isinstance(v, str))), x.count(v))[1],
       "{0}.count({1})", "count|occurr|frequen|times|number of")
binary("index", lambda x, v: (need(is_seq(x) and v in x), x.index(v))[1], "{0}.index({1})",
       "index|position|find|search")
binary("remove_value", lambda x, v: (need(is_seq(x) and not is_seq(v) and v in x),
                                     like(x, [e for e in x if e != v]))[1],
       "type({0})(e for e in {0} if e != {1})", "remove|delete|except")
binary("append", lambda x, v: (need(isinstance(x, list) and not is_seq(v)), x + [v])[1], "{0} + [{1}]",
       "append|add|insert")
binary("append_tuple", lambda x, v: (need(isinstance(x, tuple) and not is_seq(v)), x + (v,))[1], "{0} + ({1},)",
       "append|add")
STRS = lambda x, y: isinstance(x, str) and isinstance(y, str) and len(y) > 0  # noqa: E731
binary("remove_sub", lambda x, y: (need(STRS(x, y) and y in x), x.replace(y, ""))[1], "{0}.replace({1}, '')",
       "remove|delete")
binary("remove_chars", lambda x, y: (need(STRS(x, y)), "".join(c for c in x if c not in y))[1],
       "''.join(c for c in {0} if c not in {1})", "remove&character|remove&present")
binary("split_on", lambda x, y: (need(STRS(x, y) and y in x), x.split(y))[1], "{0}.split({1})", "split")
binary("find", lambda x, y: (need(STRS(x, y) and y in x), x.find(y))[1], "{0}.find({1})",
       "position|index|find|occurr")
binary("span", lambda x, y: (need(STRS(x, y) and y in x), (y, x.find(y), x.find(y) + len(y)))[1],
       "({1}, {0}.find({1}), {0}.find({1}) + len({1}))", "position|occurr")
binary("join_with", lambda x, y: (need(strseq(x) and isinstance(y, str) and len(y) <= 2), y.join(x))[1],
       "{1}.join({0})", "join|concatenat|delimiter|separator")
binary("lookup", lambda x, y: (need(isinstance(x, dict) and not is_seq(y) and y in x), x[y])[1], "{0}[{1}]",
       "value|key|access")
binary("filter_dict", lambda x, n: (need(DICT(x) and is_num(n) and numseq(list(x.values()))),
                                    {k: v for k, v in x.items() if v >= n})[1],
       "{{k: v for k, v in {0}.items() if v >= {1}}}", "filter")
# ternary
ternary("replace", lambda x, y, z: (need(STRS(x, y) and isinstance(z, str) and y in x), x.replace(y, z))[1],
        "{0}.replace({1}, {2})", "replace")
ternary("slice", lambda x, i, j: (need(SEQ_OR_STR(x) and INTS(i, j) and 0 <= i < j <= len(x)), x[i:j])[1],
        "{0}[{1}:{2}]", "slice|between|range|from")
ternary("merge3", lambda x, y, z: (need(all(isinstance(d, dict) for d in (x, y, z))), {**x, **y, **z})[1],
        "{{**{0}, **{1}, **{2}}}", "merge")
ternary("median3", lambda x, y, z: (need(all(is_num(v) for v in (x, y, z))), sorted([x, y, z])[1])[1],
        "sorted([{0}, {1}, {2}])[1]", "median")

# ---------------------------------------------------------------- knowledge operators
# The teacher's formula rules in executable form. Each is tried only when the description
# names the concept, and it counts only if it reproduces the visible example.


def know(arity, name, fn, template, keywords, **options):
    {1: unary, 2: binary, 3: ternary}[arity](name, fn, template, keywords, gated=True, kind="knowledge", **options)


def _n(fn, limit=10 ** 6):
    return lambda n: fn(small(n, limit))


for _name, _formula, _words in [
        ("triangular", "{0} * ({0} + 1) // 2", "triangular num"),
        ("pentagonal", "{0} * (3 * {0} - 1) // 2", "pentagonal num"),
        ("hexagonal", "{0} * (2 * {0} - 1)", "hexagonal num"),
        ("heptagonal", "{0} * (5 * {0} - 3) // 2", "heptagonal num"),
        ("octagonal", "{0} * (3 * {0} - 2)", "octagonal num"),
        ("nonagonal", "{0} * (7 * {0} - 5) // 2", "nonagonal num"),
        ("decagonal", "{0} * (4 * {0} - 3)", "decagonal num"),
        ("dodecagonal", "{0} * (5 * {0} - 4)", "dodecagonal num"),
        ("centered_triangular", "3 * {0} * ({0} - 1) // 2 + 1", "centered triangular"),
        ("centered_square", "2 * {0} * ({0} - 1) + 1", "centered square"),
        ("centered_pentagonal", "5 * {0} * ({0} - 1) // 2 + 1", "centered pentagonal"),
        ("centered_hexagonal", "3 * {0} * ({0} - 1) + 1", "centered hexagonal"),
        ("star", "6 * {0} * ({0} - 1) + 1", "star num"),
        ("tetrahedral", "{0} * ({0} + 1) * ({0} + 2) // 6", "tetrahedral"),
        ("square_pyramidal", "{0} * ({0} + 1) * (2 * {0} + 1) // 6", "pyramidal num"),
        ("octahedral", "{0} * (2 * {0} * {0} + 1) // 3", "octahedral"),
        ("pronic", "{0} * ({0} + 1)", "pronic|oblong|rectangular num"),
        ("woodall", "{0} * 2 ** {0} - 1", "woodall"),
        ("cullen", "{0} * 2 ** {0} + 1", "cullen"),
        ("mersenne", "2 ** {0} - 1", "mersenne"),
        ("carol", "(2 ** {0} - 1) ** 2 - 2", "carol num"),
        ("kynea", "(2 ** {0} + 1) ** 2 - 2", "kynea"),
        ("catalan", "math.comb(2 * {0}, {0}) // ({0} + 1)", "catalan"),
        ("even_binomial_sum", "2 ** ({0} - 1)", "even ind&binomial"),
        ("binomial_row_sum", "2 ** {0}", "sum&binomial"),
        ("binomial_product_sum", "math.comb(2 * {0}, {0} - 1)", "product&consecutive&binomial"),
        ("central_binomial", "math.comb(2 * {0}, {0})", "central binomial|sum&square&binomial"),
        ("natural_sum", "{0} * ({0} + 1) // 2", "sum&natural")]:
    know(1, _name, _n(eval("lambda n: " + _formula.format("n"), {"math": math}), 10 ** 4), _formula, _words)

for _name, _formula, _words in [
        ("cube_surface", "6 * {0} * {0}", "cube&surface"),
        ("cube_lateral", "4 * {0} * {0}", "cube&lateral"),
        ("cube_volume", "{0} ** 3", "cube&volume"),
        ("sphere_surface", "4 * math.pi * {0} * {0}", "sphere&surface|sphere&area"),
        ("sphere_volume", "4 / 3 * math.pi * {0} ** 3", "sphere&volume"),
        ("circle_area", "math.pi * {0} * {0}", "circle&area"),
        ("circumference", "2 * math.pi * {0}", "circumference|circle&perimeter"),
        ("tetrahedron_area", "math.sqrt(3) * {0} * {0}", "tetrahedron&area|tetrahedron&surface"),
        ("tetrahedron_volume", "{0} ** 3 / (6 * math.sqrt(2))", "tetrahedron&volume"),
        ("square_perimeter", "4 * {0}", "perimeter&square|perimeter&rhombus"),
        ("pentagon_perimeter", "5 * {0}", "perimeter&pentagon")]:
    know(1, _name, lambda x, f=eval("lambda x: " + _formula.format("x"), {"math": math}):
         (need(is_num(x) and 0 < x < 10 ** 6), f(x))[1], _formula, _words)

for _kind, _term in (("natural", "i"), ("even", "(2 * i)"), ("odd", "(2 * i - 1)")):
    for _power_value, _word in ((1, ""), (2, "&square"), (3, "&cube"), (4, "&fourth"), (5, "&fifth")):
        if _kind == "natural" and _power_value == 1:
            continue
        _source = f"sum({_term} ** {_power_value} for i in range(1, {{0}} + 1))"
        know(1, f"{_kind}_power{_power_value}_sum",
             _n(eval("lambda n: " + _source.format("n")), 5000), _source, f"sum&{_kind}{_word}")

_SEQUENCES = [
    ("lucas", "lucas", "def _lucas(n):\n    a, b = 2, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n"),
    ("pell", "pell", "def _pell(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, 2 * b + a\n    return a\n"),
    ("jacobsthal_lucas", "jacobsthal&lucas",
     "def _jacobsthal_lucas(n):\n    a, b = 2, 1\n    for _ in range(n):\n        a, b = b, b + 2 * a\n    return a\n"),
    ("jacobsthal", "jacobsthal",
     "def _jacobsthal(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, b + 2 * a\n    return a\n"),
    ("perrin", "perrin",
     "def _perrin(n):\n    a, b, c = 3, 0, 2\n    for _ in range(n):\n        a, b, c = b, c, a + b\n    return a\n"),
    ("padovan", "padovan",
     "def _padovan(n):\n    a, b, c = 1, 1, 1\n    for _ in range(n):\n        a, b, c = b, c, a + b\n    return a\n"),
    ("fibonacci", "fibonacci",
     "def _fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n"),
    ("nsw", "shanks|newman&prime",
     "def _nsw(n):\n    a, b = 1, 1\n    for _ in range(n):\n        a, b = b, 2 * b + a\n    return a\n"),
    ("newman_conway", "newman&conway",
     "def _newman_conway(n):\n    p = [0, 1, 1]\n    for i in range(3, n + 1):\n"
     "        p.append(p[p[i - 1]] + p[i - p[i - 1]])\n    return p[n]\n"),
    ("bell", "bell num",
     "def _bell(n):\n    row = [1]\n    for _ in range(n):\n        new = [row[-1]]\n        for value in row:\n"
     "            new.append(new[-1] + value)\n        row = new\n    return row[0]\n"),
]
for _name, _words, _source in _SEQUENCES:
    _fn, _text = helper(_source)
    # The Bell triangle is quadratic in n, so it gets a smaller range than the linear recurrences.
    know(1, _name, _n(_fn, 400 if _name == "bell" else 3000), f"_{_name}({{0}})", _words, helper=_text)
    # Sum of the first n terms is a frequent variant of the same definition.
    _sum_source = _text + f"\n\ndef _{_name}_sum(n):\n    return sum(_{_name}(i) for i in range(1, n + 1))\n"
    _scope = {}
    exec(_sum_source, _scope)
    know(1, _name + "_sum", _n(_scope[f"_{_name}_sum"], 300), f"_{_name}_sum({{0}})", "sum&" + _words.split("|")[0],
         helper=_sum_source)

# Doubly recursive definitions are memoised: unmemoised, delannoy(12, 10) alone takes seconds, both
# in the search and in the answer the engine ships.
_MEMO = "@functools.lru_cache(maxsize=None)\n"
_TABLES = [
    ("eulerian", "eulerian",
     _MEMO + "def _eulerian(n, m):\n    if m >= n or n == 0:\n        return 0\n    if m == 0:\n        return 1\n"
     "    return (n - m) * _eulerian(n - 1, m - 1) + (m + 1) * _eulerian(n - 1, m)\n"),
    ("stirling2", "stirling",
     _MEMO + "def _stirling2(n, k):\n    if n == k:\n        return 1\n    if k == 0 or k > n:\n        return 0\n"
     "    return k * _stirling2(n - 1, k) + _stirling2(n - 1, k - 1)\n"),
    ("delannoy", "delannoy",
     _MEMO + "def _delannoy(m, n):\n    if m == 0 or n == 0:\n        return 1\n"
     "    return _delannoy(m - 1, n) + _delannoy(m - 1, n - 1) + _delannoy(m, n - 1)\n"),
]
for _name, _words, _source in _TABLES:
    _fn, _text = helper(_source)
    know(2, _name, lambda a, b, f=_fn: f(small(a, 12), small(b, 12)), f"_{_name}({{0}}, {{1}})", _words, helper=_text)

know(2, "comb", lambda n, k: math.comb(small(n, 200), small(k, 200)), "math.comb({0}, {1})",
     "binomial|combination|ncr")
know(2, "perm", lambda n, k: math.perm(small(n, 60), small(k, 60)), "math.perm({0}, {1})", "permutation")
for _name, _formula, _words in [
        ("cylinder_lateral", "2 * math.pi * {0} * {1}", "cylinder&lateral"),
        ("cylinder_surface", "2 * math.pi * {0} * ({0} + {1})", "cylinder&surface"),
        ("cylinder_volume", "math.pi * {0} * {0} * {1}", "cylinder&volume"),
        ("cone_lateral", "math.pi * {0} * math.sqrt({0} * {0} + {1} * {1})", "cone&lateral"),
        ("cone_surface", "math.pi * {0} * ({0} + math.sqrt({0} * {0} + {1} * {1}))", "cone&surface"),
        ("cone_volume", "math.pi * {0} * {0} * {1} / 3", "cone&volume"),
        ("pyramid_surface", "{0} * {0} + 2 * {0} * {1}", "pyramid&surface"),
        ("pyramid_volume", "{0} * {0} * {1} / 3", "pyramid&volume"),
        ("rhombus_area", "{0} * {1} / 2", "rhombus&area"),
        ("triangle_area", "{0} * {1} / 2", "triangle&area"),
        ("ellipse_area", "math.pi * {0} * {1}", "ellipse&area"),
        ("rectangle_perimeter", "2 * ({0} + {1})", "perimeter&rectangle|perimeter&parallelogram"),
        ("sector_area", "math.pi * {0} * {0} * {1} / 360", "sector&area"),
        ("arc_length", "2 * math.pi * {0} * {1} / 360", "arc length"),
        ("polygon_area", "{0} * {1} * {1} / (4 * math.tan(math.pi / {0}))", "polygon&area"),
        ("trapezium_median", "({0} + {1}) / 2", "trapez&median"),
        ("wind_chill", "round(13.12 + 0.6215 * {1} - 11.37 * {0} ** 0.16 + 0.3965 * {1} * {0} ** 0.16)",
         "wind chill")]:
    know(2, _name, lambda a, b, f=eval("lambda a, b: " + _formula.format("a", "b"), {"math": math}):
         (need(is_num(a) and is_num(b) and 0 < abs(a) < 10 ** 6 and 0 < abs(b) < 10 ** 6), f(a, b))[1],
         _formula, _words)
for _name, _formula, _words in [
        ("cuboid_surface", "2 * ({0} * {1} + {0} * {2} + {1} * {2})", "cuboid&surface"),
        ("cuboid_lateral", "2 * {2} * ({0} + {1})", "cuboid&lateral"),
        ("cuboid_volume", "{0} * {1} * {2}", "cuboid&volume|volume&prism"),
        ("trapezium_area", "({0} + {1}) / 2 * {2}", "trapez&area"),
        ("prism_volume", "{0} * {1} * {2} / 2", "triangular prism|volume&prism")]:
    know(3, _name, lambda a, b, c, f=eval("lambda a, b, c: " + _formula.format("a", "b", "c"), {"math": math}):
         (need(all(is_num(v) and 0 < abs(v) < 10 ** 6 for v in (a, b, c))), f(a, b, c))[1], _formula, _words)



# ---------------------------------------------------------------- operator packs from later rounds
def also_named(key, keywords):
    """Let a pack teach an existing operator more of the words that name it ("a&b|c" syntax)."""
    for op in U + B + T:
        if op.name + "/" + str(op.arity) == key:
            op.alternatives += [tuple(alt.split("&")) for alt in keywords.split("|") if alt]
            return
    raise KeyError(key)


def load_packs():
    """Add the operators of later teacher rounds, each kept in its own file under oppacks/.

    Separate files let several teachers write rules at the same time and let a whole round be
    switched off, so its effect can be measured against the library it was added to.
    RULE_PACKS overrides the manifest: "none" for the core library alone, or names joined by commas.
    """
    manifest = json.loads(PACKS.read_text()) if PACKS.exists() else {}
    override = os.environ.get("RULE_PACKS")
    names = manifest.get("active", []) if override is None else [n for n in override.split(",") if n and n != "none"]
    known = {op.name + "/" + str(op.arity) for op in U + B + T}
    for name in names:
        seen = {id(op) for op in U + B + T}
        importlib.import_module(f"{__package__}.oppacks.{name}").register(sys.modules[__name__])
        for op in U + B + T:
            if id(op) in seen:
                continue
            key = op.name + "/" + str(op.arity)
            if key in known:
                raise ValueError(f"operator pack {name} redefines {key}")
            known.add(key)
            op.pack = name
    # Retired rules stay in their files, so the record of what was tried and dropped is kept.
    # An override names packs as written, without retirement: that is what attribution needs.
    retired = set(manifest.get("retired", {})) if override is None else set()
    retired |= {key for key in os.environ.get("RULE_RETIRE", "").split(",") if key}
    for bucket in (U, B, T):
        bucket[:] = [op for op in bucket if op.name + "/" + str(op.arity) not in retired]
    # Search behaviour added after the first library was frozen is switched on by the manifest as
    # well, so "RULE_PACKS=none" still reproduces the frozen engine. RULE_ENGINE adds features by name.
    features = set(manifest.get("engine", [])) if override is None else set()
    features |= {name for name in os.environ.get("RULE_ENGINE", "").split(",") if name}
    return names, frozenset(features)


ACTIVE_PACKS, ENGINE = load_packs()
OPERATORS = {op.name + "/" + str(op.arity): op for op in U + B + T}


def render(op, parts):
    # Sub-expressions are parenthesised so templates such as `{0}[-{1}:]` keep their meaning.
    parts = [part if part.isidentifier() else f"({part})" for part in parts]
    if op.template.startswith("ELEMENTWISE:"):
        return ELEMENTWISE.replace("{expr}", op.template.split(":", 1)[1]).format(*parts)
    return op.template.format(*parts)


def parse_example(asserts, entry_point):
    """Read (arguments, expected value, comparison mode) from the visible assert, or None."""
    try:
        node = ast.parse(asserts).body[0]
    except (SyntaxError, IndexError):
        return None
    if not isinstance(node, ast.Assert):
        return None

    def call_args(expr):
        if (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == entry_point
                and not expr.keywords):
            return [ast.literal_eval(a) for a in expr.args]
        return None

    def literal(expr):
        try:
            return ast.literal_eval(expr)
        except (ValueError, SyntaxError):
            constructors = {"set": set, "tuple": tuple, "list": list, "sorted": sorted, "frozenset": frozenset}
            if (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and len(expr.args) == 1
                    and expr.func.id in constructors):
                return constructors[expr.func.id](ast.literal_eval(expr.args[0]))
            raise

    test = node.test
    try:
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
            left, right = test.left, test.comparators[0]
            args = call_args(left)
            if args is not None:
                return {"args": args, "expected": literal(right), "mode": "eq"}
            if (isinstance(left, ast.Call) and isinstance(left.func, ast.Name) and left.func.id == "set"
                    and len(left.args) == 1 and call_args(left.args[0]) is not None):
                return {"args": call_args(left.args[0]), "expected": set(literal(right)), "mode": "set"}
        if (isinstance(test, ast.Call) and isinstance(test.func, ast.Attribute) and test.func.attr == "isclose"
                and len(test.args) >= 2 and call_args(test.args[0]) is not None):
            return {"args": call_args(test.args[0]), "expected": float(literal(test.args[1])), "mode": "close"}
        # `assert f(x)` and `assert not f(x)` state a truth value without a comparison.
        if call_args(test) is not None:
            return {"args": call_args(test), "expected": True, "mode": "eq"}
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not) and call_args(test.operand) is not None:
            return {"args": call_args(test.operand), "expected": False, "mode": "eq"}
    except (ValueError, SyntaxError, TypeError):
        return None
    return None


def _matches(value, example):
    target, mode = example["expected"], example["mode"]
    try:
        if mode == "eq":
            return type(value) is type(target) and value == target
        if mode == "set":
            return isinstance(value, (list, tuple, set)) and set(value) == target
        if mode == "close":
            return is_num(value) and math.isclose(value, target, rel_tol=1e-3)
    except TypeError:
        return False
    return False


def _key(value):
    text = repr(value)
    return None if len(text) > MAX_REPR else (type(value).__name__, text)


def informative(expected):
    """A single example pins the program down only if the expected value carries enough detail."""
    if isinstance(expected, bool) or expected is None:
        return False
    if isinstance(expected, float):
        return len(repr(expected).replace(".", "").replace("-", "").strip("0")) >= 3
    if is_int(expected):
        return abs(expected) >= 100
    return len(expected) >= 2 if hasattr(expected, "__len__") else False


NAMED_CHARACTERS = {"space": " ", "comma": ",", "dot": ".", "period": ".", "underscore": "_", "colon": ":",
                    "semicolon": ";", "hyphen": "-", "dash": "-", "slash": "/", "asterisk": "*", "hash": "#",
                    "newline": "\n", "tab": "\t"}
MAX_LITERALS = 6


def description_literals(description):
    """Values a description states outright: quoted strings, integers, characters called by name.

    Operators see only the example's arguments, so a task such as "remove every 'std'" or
    "numbers divisible by 7" is out of reach unless the stated value joins the search.
    Number words are left out on purpose: "two lists" names the arguments, not a constant.
    """
    found = []
    for quoted in re.findall(r"""(?<![\w'"])(?:'([^'\n]{1,20})'|"([^"\n]{1,20})")(?![\w'"])""", description):
        found.append(quoted[0] or quoted[1])
    for number in re.findall(r"(?<![\w.])\d{1,6}(?!\w|\.\d)", description):
        found.append(int(number))
    lowered = description.lower()
    for name, character in NAMED_CHARACTERS.items():
        if re.search(rf"\b{name}s?\b", lowered):
            found.append(character)
    return list(dict.fromkeys(found))[:MAX_LITERALS]


def load_weights(path=WEIGHTS):
    """Operator reliabilities learned from development outcomes (see scripts/rule_pilot.py learn-operators)."""
    return json.loads(path.read_text())["operators"] if path.exists() else {}


def search(example, description="", budget_seconds=2.0, weights=None, extra_named=()):
    """Return programs that reproduce the example, best first.

    `extra_named` lists operator keys to treat as named by the description although no keyword
    matched, so another reader of the description (a learned model) can open their gates.
    """
    text = description.lower()
    weights = load_weights() if weights is None else weights
    deadline = time.monotonic() + budget_seconds
    extra_named = set(extra_named)
    named = {id(op): op.named_in(text) or (op.name + "/" + str(op.arity)) in extra_named for op in U + B + T}
    # How specifically the description names an operator: the length of the longest keyword
    # alternative that matched. "regex&position" says more than "position" alone.
    specific = {id(op): max((sum(map(len, alt)) for alt in op.alternatives if all(part in text for part in alt)),
                            default=0) for op in U + B + T} if "tie_break" in ENGINE else {}
    argument_names = [re.compile(rf"\ba{i}\b") for i in range(len(example["args"]))]
    unary_ops = [op for op in U if named[id(op)] or not op.gated]
    binary_ops = [op for op in B if named[id(op)] or not op.gated]
    ternary_ops = [op for op in T if named[id(op)] or not op.gated]
    # An item is (value, expression, operators used). Equal values are merged, keeping the
    # expression the task description supports best, so a coincidence cannot hide the real reading.
    base = [(arg, f"a{i}", ()) for i, arg in enumerate(example["args"])]
    seen = {_key(v): None for v, _, _ in base}
    # The prefix-token form of every expression built, so a match can be learned from (SeedLM).
    programs = {f"a{i}": (f"a{i}",) for i in range(len(base))}
    # Values stated in the description may be the second operand of an operator, never its subject.
    constants = []
    if "description_constants" in ENGINE:
        for index, literal in enumerate(description_literals(description)):
            if _key(literal) not in seen:
                seen[_key(literal)] = None
                constants.append((literal, repr(literal), ()))
                programs[repr(literal)] = (f"c{index}",)
    found = []

    def build(op, child_exprs):
        expr = render(op, child_exprs)
        if expr not in programs:
            programs[expr] = (op.name + "/" + str(op.arity),) + tuple(
                token for child in child_exprs for token in programs.get(child, ("?",)))
        return expr

    def quality(ops):
        semantic = [op for op in ops if op.cost >= 1]
        hits = sum(named[id(op)] for op in semantic)
        return (2 * hits - len(semantic), -sum(op.cost for op in ops))

    # The first library kept the first MAX_MATCHES matches in enumeration order, so a well-supported
    # program found late was lost behind early coincidences. "keep_best" ranks before it cuts.
    room = MAX_COLLECTED if "keep_best" in ENGINE else MAX_MATCHES

    def record(expr, ops):
        if len(found) < room:
            semantic = [op for op in ops if op.cost >= 1]
            hits = sum(named[id(op)] for op in semantic)
            support = hits / len(semantic) if semantic else (1.0 if any(named[id(op)] for op in ops) else 0.5)
            learned = sum(weights.get(op.name + "/" + str(op.arity), 0.0) for op in ops)
            found.append({"expr": expr, "ops": [op.name + "/" + str(op.arity) for op in ops],
                          "size": sum(op.cost for op in ops), "support": support, "learned": learned,
                          "knowledge": any(op.kind == "knowledge" for op in ops),
                          "specific": sum(specific.get(id(op), 0) for op in ops),
                          "all_args": all(name.search(expr) for name in argument_names),
                          "program": list(programs.get(expr, ())),
                          "helpers": [op.helper for op in ops if op.helper]})

    def consider(value, expr, ops, bucket):
        if _matches(value, example):
            record(expr, ops)
            return
        key = _key(value)
        if key is None:
            return
        if key in seen:
            slot = seen[key]
            # Only a value first produced in this same level may be re-labelled.
            if slot is not None and slot[0] is bucket and quality(ops) > slot[2]:
                bucket[slot[1]] = (value, expr, ops)
                seen[key] = (bucket, slot[1], quality(ops))
            return
        if len(bucket) >= MAX_LEVEL_VALUES:
            return
        seen[key] = (bucket, len(bucket), quality(ops))
        bucket.append((value, expr, ops))

    def expand(items, partners, bucket, both_orders):
        for value, expr, ops in items:
            for op in unary_ops:
                try:
                    consider(op.fn(value), build(op, [expr]), ops + (op,), bucket)
                except Exception:
                    pass
            for other, other_expr, other_ops in partners:
                if other_expr == expr:
                    continue  # x op x is either degenerate or already a unary operator
                orders = [((value, expr), (other, other_expr))]
                # A constant is never an item of its own, so its other order has to be tried here.
                if both_orders or not other_expr.isidentifier():
                    orders.append(((other, other_expr), (value, expr)))
                for op in binary_ops:
                    for (first, first_expr), (second, second_expr) in orders:
                        try:
                            consider(op.fn(first, second), build(op, [first_expr, second_expr]),
                                     ops + other_ops + (op,), bucket)
                        except Exception:
                            pass
            if time.monotonic() > deadline:
                return

    level1 = []
    for triple in itertools.permutations(base + constants, 3):
        if all(item in constants for item in triple):
            continue
        for op in ternary_ops:
            try:
                consider(op.fn(*(item[0] for item in triple)), build(op, [item[1] for item in triple]), (op,), level1)
            except Exception:
                pass
    expand(base, base + constants, level1, both_orders=False)
    if time.monotonic() < deadline:
        expand(level1, base + constants, [], both_orders=True)
    if "named_pairs" in ENGINE and time.monotonic() < deadline:
        # Arithmetic on two derived numbers ("sum of the cubes minus the sum") is not reachable by
        # the levels above, which never combine two derived values. Only numbers produced by an
        # operator the description names take part: that keeps the pairs few and an accidental
        # match unlikely.
        derived = [item for item in level1 if is_num(item[0]) and all(named[id(op)] for op in item[2])]
        for (x, x_expr, x_ops), (y, y_expr, y_ops) in itertools.permutations(derived[:MAX_PAIR_OPERANDS], 2):
            for op in binary_ops:
                try:
                    value = op.fn(x, y)
                except Exception:
                    continue
                if _matches(value, example):
                    record(build(op, [x_expr, y_expr]), x_ops + y_ops + (op,))
    # Named by the task first, then learned reliability, then the smaller program. Equal scores used
    # to be decided by the order of discovery, which always favoured the oldest operators and unary
    # programs; three teachers found correct programs lost that way. With "tie_break" a tie goes to
    # the program that uses every argument (an argument is given for a reason), then to one with a
    # named definition, then to the more specifically named one.
    def tie(match):
        if "tie_break" not in ENGINE:
            return (0, 0, 0)
        return (-match["all_args"], -match["knowledge"], -match["specific"])

    found.sort(key=lambda m: (-(2.0 * m["support"] + m["learned"] - 0.3 * m["size"]), tie(m), m["size"]))
    return found[:MAX_MATCHES]


def to_code(entry_point, arity, match):
    params = ", ".join(f"a{i}" for i in range(arity))
    helpers = "".join(dict.fromkeys(source.rstrip("\n") + "\n\n\n" for source in match["helpers"]))
    return add_missing_imports(f"{helpers}def {entry_point}({params}):\n    return {match['expr']}\n")


def synthesize(description, asserts, entry_point, budget_seconds=2.0, weights=None, keep=1, extra_named=()):
    """Best program(s) for the visible example with a confidence label, or None."""
    example = parse_example(asserts, entry_point)
    if example is None or not example["args"]:
        return None
    matches = search(example, description, budget_seconds, weights, extra_named)
    if not matches:
        return None
    best = matches[0]
    # Trust a match only when the example is detailed enough, or the description names every step.
    confident = best["knowledge"] or informative(example["expected"]) or \
        (best["size"] >= 1 and best["support"] == 1.0)
    result = {"code": to_code(entry_point, len(example["args"]), best), "expr": best["expr"], "ops": best["ops"],
              "program": best["program"],
              "support": round(best["support"], 2), "matches": len(matches), "confident": confident,
              "knowledge": best["knowledge"], "mode": example["mode"]}
    if keep > 1:
        result["alternatives"] = [dict(m, code=to_code(entry_point, len(example["args"]), m)) for m in matches[:keep]]
    return result
