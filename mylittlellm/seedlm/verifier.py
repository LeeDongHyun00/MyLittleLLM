"""SeedLM-24's verifier: does a candidate program do what the description asks, judged by what it does?

The coder ranks the programs that reproduce the visible example by probability, and the first is often a
wrong one that fits the example by accident (SeedLM-21, 23). Following execution-based verifiers for
language-to-code generation (LEVER, Ni et al. 2023), a second small transformer, trained from scratch,
reads the task as the coder does (description, literals, visible example), one candidate program, and the
candidate's outputs on probe inputs (type-preserving mutations of the example's arguments), and says whether
the candidate is right:

  BOS DESC <words> LIT .. ARGS .. OUT .. NAMED .. PROG <program> EOS BEH <args> SEP <output> SEP ... VERDICT

The label is predicted at VERDICT (RIGHT or WRONG); nothing else is trained on.
"""
import torch

from . import coder4, data2

PROBES = 5
PROBE_ARGS_BYTES = 20
PROBE_OUT_BYTES = 12
CONTEXT = 512


class Tokens:
    """The verifier's extra tokens, after the coder vocabulary's own."""

    def __init__(self, vocabulary):
        base = len(vocabulary)
        self.BEH, self.SEP, self.VERDICT, self.RIGHT, self.WRONG = range(base, base + 5)
        self.size = base + 5


def _bytes(text, limit):
    return list(text.encode("utf-8", "replace")[:limit])


def behaviour_tokens(tokens, probes, results):
    """Each probe's arguments and the candidate's output (or the exception's name), as bytes."""
    out = [tokens.BEH]
    for args, (status, value) in zip(probes[:PROBES], results[:PROBES]):
        out += _bytes(repr(args)[1:-1], PROBE_ARGS_BYTES) + [tokens.SEP]
        out += _bytes(value if status == "ok" else "!" + value, PROBE_OUT_BYTES) + [tokens.SEP]
    return out


def encode(vocabulary, tokens, words, literals, args, expected, named, program, probes, results):
    """(sequence, verdict position) or None if it does not fit."""
    sequence, _ = vocabulary.encode(words, literals, args, expected, named, program)
    sequence = sequence + behaviour_tokens(tokens, probes, results) + [tokens.VERDICT]
    if len(sequence) > CONTEXT:
        return None
    return sequence, len(sequence) - 1


def collate(rows, tokens, labels=None):
    """rows: (sequence, verdict position); labels: True/False per row, or None to score."""
    width = max(len(s) for s, _ in rows)
    inputs = torch.full((len(rows), width), data2.PAD, dtype=torch.long)
    target = torch.full((len(rows), width), -100, dtype=torch.long)
    for i, (sequence, position) in enumerate(rows):
        inputs[i, :len(sequence)] = torch.tensor(sequence)
        if labels is not None:
            target[i, position] = tokens.RIGHT if labels[i] else tokens.WRONG
    return inputs, target


@torch.no_grad()
def right_probability(net, tokens, rows, batch=64):
    """P(RIGHT) against WRONG at each row's verdict position."""
    out = []
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        inputs, _ = collate(chunk, tokens)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = net(inputs)
        positions = torch.tensor([p for _, p in chunk])
        pair = logits[torch.arange(len(chunk)), positions][:, [tokens.RIGHT, tokens.WRONG]].float()
        out += torch.softmax(pair, dim=-1)[:, 0].tolist()
    return out


def load(path):
    from . import model as seedmodel
    checkpoint = torch.load(path)
    vocabulary = coder4.Vocabulary(checkpoint["vocabulary"]["words"], checkpoint["vocabulary"]["operators"])
    net = seedmodel.SeedLM(seedmodel.Config(**checkpoint["config"]))
    net.load_state_dict(checkpoint["state"])
    net.eval()
    return net, vocabulary, Tokens(vocabulary)
