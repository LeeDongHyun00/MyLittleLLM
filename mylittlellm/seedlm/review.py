"""SeedLM-3's reviewer: reads a program and the example, then scores the task description.

A coder ranks candidate programs by P(program | description). On real tasks it fails in a typical
way: it puts almost all its mass on an operator whose phrase shares words with the description
(`binomial_product_sum` for "the sum of squares of binomial coefficients") and the right program,
which reproduces the example, falls outside the beam. A reviewer asks the reverse question,
P(description | program): among programs that all reproduce the visible example, which one
explains every word of the description? This is coder-reviewer reranking (Zhang et al., 2022) for
programs over a rule library. Every candidate is scored on the same word sequence, so the scores
are directly comparable, whatever the programs' lengths.

A sequence is
  BOS PROG <program> LIT <literal bytes> ARGS <argument bytes> OUT <expected bytes> DESC <words> EOS
and only the words (and EOS) are trained on.
"""
import torch

from . import data2, synth


def condition(vocabulary, program, literals, args, expected):
    lit = list(repr(list(literals)).encode())[:data2.MAX_LIT]
    return ([data2.BOS, data2.PROG] + [vocabulary.token(s) for s in program] + [data2.LIT] + lit + [data2.ARGS]
            + list(repr(list(args))[1:-1].encode())[:synth.MAX_IN] + [data2.OUT]
            + list(repr(expected).encode())[:synth.MAX_OUT] + [data2.DESC])


def encode(vocabulary, program, literals, args, expected, words):
    prefix = condition(vocabulary, program, literals, args, expected)
    desc = [vocabulary.word_index.get(w, data2.UNK) for w in words][:data2.MAX_DESC]
    return prefix + desc + [data2.EOS], len(prefix)


def known(vocabulary, program):
    return bool(program) and all(("/" not in s and s[0] in "ac") or s in vocabulary.index for s in program)


@torch.no_grad()
def scores(net, vocabulary, programs, literals, args, expected, words, batch_size=32):
    """log P(words | program, example) for each program (None where the program has unknown tokens)."""
    out = [None] * len(programs)
    rows = [(i, encode(vocabulary, p, literals, args, expected, words)) for i, p in enumerate(programs)
            if known(vocabulary, p)]
    for begin in range(0, len(rows), batch_size):
        chunk = rows[begin:begin + batch_size]
        width = max(len(tokens) for _, (tokens, _) in chunk)
        batch = torch.full((len(chunk), width), data2.PAD, dtype=torch.long)
        for j, (_, (tokens, _)) in enumerate(chunk):
            batch[j, :len(tokens)] = torch.tensor(tokens)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = net(batch)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        for j, (i, (tokens, start)) in enumerate(chunk):
            positions = torch.arange(start, len(tokens))
            targets = torch.tensor(tokens[start:])
            out[i] = float(log_probs[j, positions - 1, targets].sum())
    return out
