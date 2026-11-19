"""Descriptions for pyx programs that have none: the words a task statement uses for each piece.

SeedLM-2 composed descriptions for synthetic library programs from teacher-written operator phrases
(seedlm/language.py). SeedLM-6's grafted programs also hold Python nodes (comprehensions, arithmetic,
comparisons, methods). Each node here gets the plain words MBPP-style statements use for it ("sum",
"product", "each", "greater than", "split into words"), library operators keep their teacher phrases,
and arguments are named by their type ("list", "string"). Words come outer node first, the order a
statement reads ("the sum of the squares of the odd numbers"), with a little noise so the coder does
not depend on exact wording. The result is a bag of the right words in roughly the right order, not
fluent English.
"""
import re

from . import language

BINARY = {
    "Add": ["add", "plus", "sum of", "total"], "Sub": ["subtract", "minus", "difference"],
    "Mult": ["multiply", "product", "times"], "Div": ["divide", "ratio", "divided by"],
    "FloorDiv": ["integer division", "divide"], "Mod": ["remainder", "modulo", "divisible"],
    "Pow": ["power", "raised to", "square"], "BitAnd": ["bitwise and"], "BitOr": ["bitwise or"],
    "BitXor": ["xor"], "LShift": ["shift left"], "RShift": ["shift right"],
    "USub": ["negative"], "Not": ["not"], "Invert": ["invert"],
    "And": ["and", "both"], "Or": ["or", "either"],
    "Eq": ["equal", "same as"], "NotEq": ["different", "not equal"], "Lt": ["less than", "smaller than"],
    "LtE": ["at most", "not more than"], "Gt": ["greater than", "larger than"], "GtE": ["at least"],
    "In": ["contains", "in"], "NotIn": ["not in", "missing"], "Is": ["is"], "IsNot": ["is not"],
    "IfExp": ["if", "otherwise", "when"], "Index": ["element at", "index", "position"],
    "Slice": ["slice", "part", "first", "last"], "Slice3": ["every second", "step", "slice"],
    "Tuple": ["tuple", "pair"], "List": ["list"], "Set": ["set"],
}
BUILTIN = {
    "abs": ["absolute value"], "all": ["all", "every"], "any": ["any", "some"], "bool": ["true or false"],
    "chr": ["character"], "dict": ["dictionary"], "divmod": ["quotient and remainder"],
    "enumerate": ["index", "position"], "float": ["float", "decimal"], "int": ["integer", "number"],
    "len": ["length", "number of", "count"], "list": ["list"], "max": ["maximum", "largest", "highest"],
    "min": ["minimum", "smallest", "lowest"], "ord": ["character code", "ascii value"], "pow": ["power"],
    "range": ["numbers from", "range", "up to"], "reversed": ["reverse"], "round": ["round"],
    "set": ["unique", "distinct"], "sorted": ["sort", "sorted", "in order"], "str": ["string", "digits"],
    "sum": ["sum", "total", "add up"], "tuple": ["tuple"], "zip": ["pairs", "corresponding elements"],
    "isinstance": ["type"], "frozenset": ["set"],
}
METHOD = {
    "split": ["split", "words"], "join": ["join", "concatenate"], "replace": ["replace"],
    "upper": ["uppercase"], "lower": ["lowercase"], "count": ["count", "occurrences"],
    "index": ["index", "position"], "find": ["find", "position"], "rfind": ["last position"],
    "startswith": ["starts with", "begins with"], "endswith": ["ends with"],
    "strip": ["strip", "remove spaces"], "rstrip": ["trailing"], "lstrip": ["leading"],
    "isdigit": ["digits", "numeric"], "isalpha": ["letters", "alphabetic"], "isalnum": ["alphanumeric"],
    "isupper": ["uppercase"], "islower": ["lowercase"], "title": ["title case"], "capitalize": ["capitalize"],
    "items": ["items", "key value pairs"], "keys": ["keys"], "values": ["values"], "get": ["value of key"],
    "zfill": ["pad with zeros"], "center": ["center"], "ljust": ["pad"], "partition": ["split once"],
    "rsplit": ["split from the right"], "bit_length": ["number of bits"], "fromkeys": ["dictionary"],
    "translate": ["translate"], "maketrans": ["translate"], "encode": ["bytes"], "istitle": ["title case"],
    "removesuffix": ["remove suffix"],
}
COMPREHENSION = {"list": ["list of", "each", "every"], "gen": ["each", "every", "for all"],
                 "set": ["unique", "set of"], "dict": ["dictionary", "map each"]}
CONDITION = ["that", "where", "if", "whose", "which"]
TYPES = {int: "number", float: "number", str: "string", list: "list", tuple: "tuple", dict: "dictionary",
         set: "set", bool: "flag"}


def node_words(token):
    """Phrases for a Python node token, or None for leaves and library operators."""
    if not token.startswith("@"):
        return None
    head = token[1:].rsplit("/", 1)[0]
    if head.startswith("Comp:"):
        return COMPREHENSION[head.split(":")[1]]
    if head.startswith("."):
        return METHOD.get(head[1:], [head[1:].replace("_", " ")])
    if head in BINARY:
        return BINARY[head]
    if head.startswith(("math.", "re.")):
        return [head.split(".", 1)[1].replace("_", " ")]
    return BUILTIN.get(head.split(":")[0], [head.split(":")[0]])


def describe(rng, tokens, literals=(), args=(), phrases=None, describer=None):
    """Description words for a prefix pyx program, outer pieces first."""
    parts = []
    for token in tokens:
        if describer is not None and token in describer.operators:
            parts.append(describer.phrase(rng, token))
        elif (words := node_words(token)) is not None:
            if token.startswith("@Comp:") and token.split(":")[2].split("/")[0].count("i") and \
                    any(int(p.split("i")[1]) for p in token.split(":")[2].split("/")[0].split(",")):
                parts.append(rng.choice(words) + " " + rng.choice(CONDITION))
            else:
                parts.append(rng.choice(words))
        elif re.fullmatch(r"c\d", token) and int(token[1:]) < len(literals):
            value = literals[int(token[1:])]
            parts.append(repr(value) if isinstance(value, str) else str(value))
        elif re.fullmatch(r"#-?\d+", token) and rng.random() < 0.5:
            parts.append(token[1:])
        elif token.startswith("$") and len(token) > 1:
            parts.append(token[1:])
        elif re.fullmatch(r"a\d", token) and int(token[1:]) < len(args):
            parts.append("the " + TYPES.get(type(args[int(token[1:])]), "value"))
    text = (rng.choice(language.OPENERS) + " " + " ".join(parts)).strip()
    noisy = []
    for word in language.words(text):
        if rng.random() < 0.04:
            continue
        noisy.append(word)
        if rng.random() < 0.03:
            noisy.append(rng.choice(language.FILLER))
    return noisy
