"""Program proposals from SeedLM: grammar-constrained beam search, then execution as the judge."""
from contextlib import contextmanager
import signal
import threading

import torch

from ..rulecoder import backinfer as bi
from . import data, model as seedmodel, synth

EXECUTION_SECONDS = 1.0     # generous: a program that fits the example runs in microseconds


class Timeout(BaseException):
    """Not an Exception on purpose: operators that catch Exception must not swallow it."""


@contextmanager
def time_limit(seconds):
    """Stop a proposed program that runs away.

    Exhaustive search drops oversized intermediate values before composing further; a proposed
    program is executed as a whole, so nothing else bounds what a three-step composition costs.
    """
    if threading.current_thread() is not threading.main_thread():
        yield                       # signals only reach the main thread; run unguarded elsewhere
        return

    def interrupt(signum, frame):
        raise Timeout

    previous = signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def load(path):
    """Model and vocabulary from a checkpoint written by the training script."""
    checkpoint = torch.load(path)
    vocabulary = data.Vocabulary(checkpoint["operators"])
    net = seedmodel.SeedLM(seedmodel.Config(**checkpoint["config"]))
    net.load_state_dict(checkpoint["state"])
    net.eval()
    return net, vocabulary


def arity_table(vocabulary):
    """Arity per token id: 0 for arguments, the operator arity for operators, -1 otherwise."""
    table = torch.full((len(vocabulary),), -1, dtype=torch.long)
    table[data.ARG0:data.FIRST_OP] = 0
    for name, token in vocabulary.index.items():
        table[token] = bi.OPERATORS[name].arity
    return table


@torch.no_grad()
def beam_search(model, vocabulary, prefix, n_args, width=16, max_tokens=synth.MAX_PROGRAM, arities=None):
    """Return complete prefix programs (token lists) with log-probabilities, best first.

    The operator grammar is enforced while decoding: a token is allowed only if the program can
    still be completed within the length budget, and only arguments that exist may be named.
    """
    arities = arity_table(vocabulary) if arities is None else arities
    valid = arities >= 0
    valid[data.ARG0 + n_args:data.FIRST_OP] = False
    beams, finished = [(0.0, [], 1)], []
    for _ in range(max_tokens):
        if not beams:
            break
        batch = torch.tensor([prefix + tokens for _, tokens, _ in beams])
        log_probs = torch.log_softmax(model(batch)[:, -1], dim=-1)
        candidates = []
        for (score, tokens, pending), row in zip(beams, log_probs):
            room = max_tokens - len(tokens) - 1          # tokens left after this one
            allowed = valid & (pending - 1 + arities <= room)
            row = row.masked_fill(~allowed, float("-inf"))
            values, indices = row.topk(min(width, int(allowed.sum())))
            for value, index in zip(values.tolist(), indices.tolist()):
                entry = (score + value, tokens + [index], pending - 1 + int(arities[index]))
                (finished if entry[2] == 0 else candidates).append(entry)
        beams = sorted(candidates, key=lambda e: -e[0])[:width]
    finished.sort(key=lambda e: -e[0])
    return [([vocabulary.symbol(t) for t in tokens], score) for score, tokens, _ in finished[:width]]


def propose(model, vocabulary, args, expected, named, width=16, arities=None):
    """Programs proposed for one example that actually reproduce the expected value, best first."""
    prefix = vocabulary.condition(args, expected, named)
    if len(prefix) + synth.MAX_PROGRAM + 1 > model.config.context or len(args) > 4:
        return []
    example = {"args": args, "expected": expected, "mode": "eq"}
    verified = []
    for program, score in beam_search(model, vocabulary, prefix, len(args), width, arities=arities):
        try:
            with time_limit(EXECUTION_SECONDS):
                value = synth.execute(program, args)
        except (Exception, Timeout):
            continue
        if bi._matches(value, example):
            verified.append({"program": program, "log_prob": round(score, 3)})
    return verified


def proposer(path, width=32):
    """The pipeline's view of a checkpoint: (args, expected, named operators) -> verified programs."""
    net, vocabulary = load(path)
    arities = arity_table(vocabulary)
    return lambda args, expected, named: propose(net, vocabulary, args, expected, named, width, arities)
