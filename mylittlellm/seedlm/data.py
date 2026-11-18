"""Token layout and batching for SeedLM.

A sequence is: BOS, bytes of the argument list, OUT, bytes of the expected value, NAMED,
tokens of the operators the task description names, PROG, the program in prefix form, EOS.
Only the program part is trained on; everything before it is the condition.
"""
import random

from ..rulecoder import backinfer as bi
from . import synth

PAD, BOS, OUT, NAMED, PROG, EOS = 256, 257, 258, 259, 260, 261
ARG0 = 262                      # a0..a3
FIRST_OP = ARG0 + 4
MAX_NAMED = 20
MAX_LENGTH = 1 + synth.MAX_IN + 1 + synth.MAX_OUT + 1 + MAX_NAMED + 1 + synth.MAX_PROGRAM + 1


class Vocabulary:
    """Operator tokens are appended in a fixed order so that new rules never renumber old ones."""

    def __init__(self, operator_names=None):
        self.operators = list(operator_names) if operator_names else sorted(bi.OPERATORS)
        self.index = {name: FIRST_OP + i for i, name in enumerate(self.operators)}

    def __len__(self):
        return FIRST_OP + len(self.operators)

    def token(self, symbol):
        return self.index[symbol] if "/" in symbol else ARG0 + int(symbol[1:])

    def symbol(self, token):
        return f"a{token - ARG0}" if token < FIRST_OP else self.operators[token - FIRST_OP]

    def condition(self, args, expected, named):
        named = [self.index[n] for n in named if n in self.index][:MAX_NAMED]
        return ([BOS] + list(repr(args)[1:-1].encode()) + [OUT] + list(repr(expected).encode())
                + [NAMED] + named + [PROG])

    def encode(self, args, expected, named, program):
        prefix = self.condition(args, expected, named)
        return prefix + [self.token(symbol) for symbol in program] + [EOS], len(prefix)


def named_for_training(example, vocabulary, rng):
    """Imitate a real description: most true operators are named, plus unrelated ones."""
    kept = [name for name in example["ops"] if rng.random() < 0.8]
    distractors = rng.sample(vocabulary.operators, k=rng.randint(0, 12))
    names = list(dict.fromkeys(kept + distractors))
    rng.shuffle(names)
    return names


def build(count, seed, vocabulary, operators=None, max_depth=3):
    """Generate `count` encoded sequences as (tokens, program_start) pairs."""
    rng = random.Random(seed)
    rows = []
    while len(rows) < count:
        example = synth.sample(rng, operators, max_depth)
        if example is None:
            continue
        tokens, start = vocabulary.encode(example["args"], example["expected"],
                                          named_for_training(example, vocabulary, rng), example["program"])
        if len(tokens) <= MAX_LENGTH:
            rows.append((tokens, start, example))
    return rows


def collate(rows, torch):
    """Pad a batch; labels are -100 outside the program so only program tokens are learned."""
    width = max(len(tokens) for tokens, _, _ in rows)
    inputs = torch.full((len(rows), width), PAD, dtype=torch.long)
    labels = torch.full((len(rows), width), -100, dtype=torch.long)
    for i, (tokens, start, _) in enumerate(rows):
        inputs[i, :len(tokens)] = torch.tensor(tokens)
        labels[i, start - 1:len(tokens) - 1] = torch.tensor(tokens[start:])
    return inputs, labels
