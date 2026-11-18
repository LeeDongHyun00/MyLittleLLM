"""SeedLM-2 data: programs over the whole operator library, with constants and written descriptions.

A sequence is
  BOS DESC <description words> LIT <bytes of the literal list> ARGS <bytes of the arguments>
  OUT <bytes of the expected value> NAMED <operator tokens the keyword table found> PROG <program> EOS
and only the program is trained on. Programs may use the example's arguments (a0..a3) and the
description's literals (c0..c5). The NAMED channel is dropped in part of the training data so the
model learns to read the words rather than lean on the keyword table.
"""
import random

from ..rulecoder import backinfer as bi
from . import language, synth

PAD, BOS, DESC, LIT, ARGS, OUT, NAMED, PROG, EOS, UNK = range(256, 266)
ARG0, CONST0 = 266, 270
FIRST_WORD = 276
MAX_ARGS, MAX_CONSTS = 4, 6
MAX_DESC, MAX_LIT, MAX_NAMED = 48, 40, 20
MAX_PROGRAM = synth.MAX_PROGRAM
MAX_LENGTH = 1 + 1 + MAX_DESC + 1 + MAX_LIT + 1 + synth.MAX_IN + 1 + synth.MAX_OUT + 1 + MAX_NAMED + 1 + MAX_PROGRAM + 1


class Vocabulary:
    """Bytes, markers, argument and constant slots, words, then operators (new operators append)."""

    def __init__(self, words, operators):
        self.words, self.operators = list(words), list(operators)
        self.word_index = {w: FIRST_WORD + i for i, w in enumerate(self.words)}
        self.first_op = FIRST_WORD + len(self.words)
        self.index = {name: self.first_op + i for i, name in enumerate(self.operators)}

    def __len__(self):
        return self.first_op + len(self.operators)

    def to_dict(self):
        return {"words": self.words, "operators": self.operators}

    def token(self, symbol):
        if "/" in symbol:
            return self.index[symbol]
        return (ARG0 if symbol[0] == "a" else CONST0) + int(symbol[1:])

    def symbol(self, token):
        if ARG0 <= token < CONST0:
            return f"a{token - ARG0}"
        if CONST0 <= token < FIRST_WORD:
            return f"c{token - CONST0}"
        return self.operators[token - self.first_op]

    def condition(self, description_words, literals, args, expected, named):
        desc = [self.word_index.get(w, UNK) for w in description_words][:MAX_DESC]
        named = [self.index[n] for n in named if n in self.index][:MAX_NAMED]
        lit = list(repr(list(literals)).encode())[:MAX_LIT]
        return ([BOS, DESC] + desc + [LIT] + lit + [ARGS] + list(repr(list(args))[1:-1].encode())
                + [OUT] + list(repr(expected).encode()) + [NAMED] + named + [PROG])

    def encode(self, description_words, literals, args, expected, named, program):
        prefix = self.condition(description_words, literals, args, expected, named)
        return prefix + [self.token(s) for s in program] + [EOS], len(prefix)


# ---------------------------------------------------------------- synthetic programs with constants
def _constant(rng, args):
    """A literal a description might state: a small number, or a piece of a string argument."""
    strings = [a for a in args if isinstance(a, str) and a]
    if strings and rng.random() < 0.6:
        text = rng.choice(strings)
        start = rng.randrange(len(text))
        return text[start:start + rng.choice([1, 1, 2, 3])]
    return rng.choice([2, 3, 4, 5, 7, 10, 11, 12, 0, 1])


def _apply(rng, pool, operators, tries=40):
    """Like synth._apply, but a True/False result is allowed: predicates are a third of real tasks,
    and with a description to read, a boolean example is no longer uninformative."""
    for _ in range(tries):
        op = rng.choice(operators)
        if op.arity > len(pool):
            continue
        chosen = rng.sample(pool, k=op.arity)
        try:
            value = op.fn(*(item[0] for item in chosen))
            if value is None or isinstance(value, (set, frozenset)) or len(repr(value)) > synth.MAX_OUT:
                continue
        except Exception:
            continue
        return value, (op.name + "/" + str(op.arity), [item[1] for item in chosen])
    return None


def sample(rng, operators, max_depth=3, constant_rate=0.25):
    """One example: {'args', 'literals', 'expected', 'program', 'ops'} or None."""
    kinds = rng.choices([s for s, _ in synth.SIGNATURES], weights=[w for _, w in synth.SIGNATURES])[0]
    args = [synth.random_value(kind, rng) for kind in kinds]
    if len(repr(args)) > synth.MAX_IN:
        return None
    base = [(arg, f"a{i}") for i, arg in enumerate(args)]
    literals = []
    if rng.random() < constant_rate:
        literals = [_constant(rng, args)]
        base.append((literals[0], "c0"))
    depth = rng.choices([1, 2, 3][:max_depth], weights=[5, 4, 2][:max_depth])[0]
    current = None
    for level in range(depth):
        pool = base if current is None else [current] + base
        for _ in range(8):
            made = _apply(rng, pool, operators)
            if made is None:
                return None
            if current is None or synth._uses(made[1], current[1]):
                break
        else:
            return None
        if made[0] in args and level == depth - 1:
            return None
        current = made
    program = synth._prefix(current[1])
    if len(program) > MAX_PROGRAM or not synth._uses_argument(program):
        return None
    return {"args": args, "literals": literals, "expected": current[0], "program": program,
            "ops": sorted({t for t in program if "/" in t})}


def execute(program, args, literals=()):
    position = 0

    def read():
        nonlocal position
        token = program[position]
        position += 1
        if "/" not in token:
            return (args if token[0] == "a" else literals)[int(token[1:])]
        op = bi.OPERATORS[token]
        return op.fn(*[read() for _ in range(op.arity)])

    value = read()
    if position != len(program):
        raise ValueError("trailing tokens")
    return value


def to_expression(program, literals=()):
    """The program as the Python expression the rule engine would emit (literals inlined)."""
    position = 0
    helpers = []

    def read():
        nonlocal position
        token = program[position]
        position += 1
        if "/" not in token:
            return token if token[0] == "a" else repr(literals[int(token[1:])])
        op = bi.OPERATORS[token]
        if op.helper:
            helpers.append(op.helper)
        return bi.render(op, [read() for _ in range(op.arity)])

    return read(), helpers


def named_for_training(rng, ops, vocabulary, drop_rate=0.5):
    """The keyword table's view, imitated: most true operators plus distractors, or nothing at all."""
    if rng.random() < drop_rate:
        return []
    kept = [name for name in ops if rng.random() < 0.8]
    distractors = rng.sample(vocabulary.operators, k=rng.randint(0, 10))
    names = list(dict.fromkeys(kept + distractors))
    rng.shuffle(names)
    return names


def build_synthetic(count, seed, vocabulary, describer, operators, max_depth=3):
    rng = random.Random(seed)
    rows = []
    while len(rows) < count:
        example = sample(rng, operators, max_depth)
        if example is None:
            continue
        text = describer.describe(rng, example["program"], example["literals"])
        named = named_for_training(rng, example["ops"], vocabulary)
        tokens, start = vocabulary.encode(text, example["literals"], example["args"], example["expected"], named,
                                          example["program"])
        if len(tokens) <= MAX_LENGTH:
            rows.append((tokens, start, dict(example, words=text)))
    return rows


def collate(rows, torch):
    width = max(len(tokens) for tokens, _, _ in rows)
    inputs = torch.full((len(rows), width), PAD, dtype=torch.long)
    labels = torch.full((len(rows), width), -100, dtype=torch.long)
    for i, (tokens, start, _) in enumerate(rows):
        inputs[i, :len(tokens)] = torch.tensor(tokens)
        labels[i, start - 1:len(tokens) - 1] = torch.tensor(tokens[start:])
    return inputs, labels
