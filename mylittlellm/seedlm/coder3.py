"""SeedLM-3's coder: the SeedLM-2 sequence layout with pyx programs (mylittlellm/seedlm/pyx.py).

The vocabulary keeps SeedLM-2's layout (bytes, markers, argument and literal slots, words) and then
lists every program token: the library operators, the Python node tokens and the extra leaves
(bound variables, small integers, short strings, the visible output and its opposite). Decoding is a
beam search that tracks, for every open slot, how many variables are bound there, so a hypothesis
can only use a variable a comprehension or key function around it has bound.
"""
import re

import torch

from ..rulecoder import backinfer as bi
from ..rulecoder.limits import Timeout, time_limit
from . import data2, pyx

MAX_PROGRAM = 32
MAX_LENGTH = data2.MAX_LENGTH - data2.MAX_PROGRAM + MAX_PROGRAM
LEAVES = ([f"v{i}" for i in range(pyx.MAX_VARS)] + [f"#{i}" for i in pyx.SMALL_INTS] + ["#0.5", "#0.0", "#1.0", "#2.0"]
          + ["o", "o!", "None", "True", "False"] + sorted(pyx.MODULE_CONSTANTS) + [f"T:{t}" for t in pyx.TYPES])
EXECUTION_SECONDS = 1.0


class Vocabulary(data2.Vocabulary):
    """data2.Vocabulary whose program tokens are any pyx token, not only library operators."""

    def token(self, symbol):
        if symbol in self.index:
            return self.index[symbol]
        if re.fullmatch(r"a\d", symbol):
            return data2.ARG0 + int(symbol[1:])
        if re.fullmatch(r"c\d", symbol):
            return data2.CONST0 + int(symbol[1:])
        raise KeyError(symbol)

    def encode(self, description_words, literals, args, expected, named, program):
        prefix = self.condition(description_words, literals, args, expected, named)
        return prefix + [self.token(s) for s in program] + [data2.EOS], len(prefix)


def program_tokens(extra=()):
    """Library operators, then Python node tokens and leaves seen in data (sorted for stability)."""
    nodes = sorted({t for t in extra if t not in bi.OPERATORS and not re.fullmatch(r"[ac]\d", t)} | set(LEAVES))
    return list(bi.OPERATORS) + nodes


def child_depths(token, depth):
    """Binding depth of each child slot, in order."""
    width = pyx.arity(token)
    if token.startswith("@Comp:"):
        _, kind, pattern = token[1:].rsplit("/", 1)[0].split(":")
        shapes = [tuple(int(x) for x in part.split("i")) for part in pattern.split(",")]
        total = sum(count for count, _ in shapes)
        out = [depth + total] * (2 if kind == "dict" else 1)
        bound = depth
        for count, n_ifs in shapes:
            out.append(bound)
            bound += count
            out += [bound] * n_ifs
        return out
    head = token[1:].rsplit("/", 1)[0] if token.startswith("@") else ""
    if ":key" in head:
        return [depth + 1] + [depth] * (width - 1)
    return [depth] * width


def bindings(token):
    """How many variables a token binds for its children (comprehension targets, key parameters)."""
    if token.startswith("@Comp:"):
        pattern = token[1:].rsplit("/", 1)[0].split(":")[2]
        return sum(int(part.split("i")[0]) for part in pattern.split(","))
    return 1 if token.startswith("@") and ":key" in token else 0


class Grammar:
    """Static per-token tables for constrained decoding."""

    def __init__(self, vocabulary):
        size = len(vocabulary)
        self.vocabulary = vocabulary
        self.arity = torch.full((size,), -1, dtype=torch.long)
        self.variable = torch.full((size,), -1, dtype=torch.long)
        self.binds = torch.zeros(size, dtype=torch.long)
        self.output_leaf = torch.zeros(size, dtype=torch.bool)
        self.arity[data2.ARG0:data2.FIRST_WORD] = 0
        for symbol, index in vocabulary.index.items():
            self.arity[index] = pyx.arity(symbol)
            if re.fullmatch(r"v\d", symbol):
                self.variable[index] = int(symbol[1:])
            self.binds[index] = bindings(symbol)
            self.output_leaf[index] = symbol in ("o", "o!")

    def base_mask(self, n_args, n_literals, string_output):
        valid = self.arity >= 0
        valid[data2.ARG0 + n_args:data2.CONST0] = False
        valid[data2.CONST0 + n_literals:data2.FIRST_WORD] = False
        if not string_output:
            valid &= ~self.output_leaf
        return valid


@torch.no_grad()
def beam_search(net, vocabulary, grammar, prefix, n_args, n_literals, string_output, width=16, max_tokens=MAX_PROGRAM):
    """Well-formed pyx programs with log-probabilities, best first."""
    base = grammar.base_mask(n_args, n_literals, string_output)
    beams, finished = [(0.0, [], [0])], []      # (score, tokens, stack of open-slot depths; top is last)
    for _ in range(max_tokens):
        if not beams:
            break
        batch = torch.tensor([prefix + tokens for _, tokens, _ in beams])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = net(batch)[:, -1]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        candidates = []
        for (score, tokens, stack), row in zip(beams, log_probs):
            depth = stack[-1]
            room = max_tokens - len(tokens) - 1
            allowed = base & (len(stack) - 1 + grammar.arity <= room)
            allowed &= ~((grammar.variable >= 0) & (grammar.variable >= depth))
            allowed &= depth + grammar.binds <= pyx.MAX_VARS
            row = row.masked_fill(~allowed, float("-inf"))
            values, indices = row.topk(min(width, int(allowed.sum())))
            for value, index in zip(values.tolist(), indices.tolist()):
                symbol = vocabulary.symbol(index)
                rest = stack[:-1] + list(reversed(child_depths(symbol, depth)))
                entry = (score + value, tokens + [index], rest)
                (finished if not rest else candidates).append(entry)
        beams = sorted(candidates, key=lambda e: -e[0])[:width]
    finished.sort(key=lambda e: -e[0])
    return [([vocabulary.symbol(t) for t in tokens], score) for score, tokens, _ in finished[:width]]


def reproduces(program, args, literals, expected, mode="eq"):
    try:
        with time_limit(EXECUTION_SECONDS):
            value = pyx.execute(program, args, literals, expected if isinstance(expected, str) else None)
    except (Exception, Timeout):
        return False
    return bi._matches(value, {"expected": expected, "mode": mode})


@torch.no_grad()
def program_log_prob(net, vocabulary, prefix, program):
    tokens = prefix + [vocabulary.token(s) for s in program] + [data2.EOS]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = net(torch.tensor([tokens]))[0]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return float(sum(log_probs[i - 1, tokens[i]] for i in range(len(prefix), len(tokens))))


def load(path):
    from . import model as seedmodel
    checkpoint = torch.load(path)
    vocabulary = Vocabulary(checkpoint["vocabulary"]["words"], checkpoint["vocabulary"]["operators"])
    net = seedmodel.SeedLM(seedmodel.Config(**checkpoint["config"]))
    net.load_state_dict(checkpoint["state"])
    net.eval()
    return net, vocabulary
