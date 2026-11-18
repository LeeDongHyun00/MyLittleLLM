"""SeedLM-2 proposals: grammar-constrained beam search over arguments, literals and operators."""
import torch

from ..rulecoder import backinfer as bi
from ..rulecoder.limits import Timeout, time_limit
from . import data2, model as seedmodel

EXECUTION_SECONDS = 1.0


def load(path):
    checkpoint = torch.load(path)
    vocabulary = data2.Vocabulary(checkpoint["vocabulary"]["words"], checkpoint["vocabulary"]["operators"])
    net = seedmodel.SeedLM(seedmodel.Config(**checkpoint["config"]))
    net.load_state_dict(checkpoint["state"])
    net.eval()
    return net, vocabulary


def arity_table(vocabulary):
    table = torch.full((len(vocabulary),), -1, dtype=torch.long)
    table[data2.ARG0:data2.FIRST_WORD] = 0
    for name, token in vocabulary.index.items():
        table[token] = bi.OPERATORS[name].arity if name in bi.OPERATORS else -1
    return table


@torch.no_grad()
def beam_search(net, vocabulary, prefix, n_args, n_literals, width=16, max_tokens=data2.MAX_PROGRAM, arities=None):
    """Complete prefix programs with log-probabilities, best first; only well-formed programs."""
    arities = arity_table(vocabulary) if arities is None else arities
    valid = arities >= 0
    valid[data2.ARG0 + n_args:data2.CONST0] = False
    valid[data2.CONST0 + n_literals:data2.FIRST_WORD] = False
    beams, finished = [(0.0, [], 1)], []
    for _ in range(max_tokens):
        if not beams:
            break
        batch = torch.tensor([prefix + tokens for _, tokens, _ in beams])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = net(batch)[:, -1]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        candidates = []
        for (score, tokens, pending), row in zip(beams, log_probs):
            room = max_tokens - len(tokens) - 1
            allowed = valid & (pending - 1 + arities <= room)
            row = row.masked_fill(~allowed, float("-inf"))
            values, indices = row.topk(min(width, int(allowed.sum())))
            for value, index in zip(values.tolist(), indices.tolist()):
                entry = (score + value, tokens + [index], pending - 1 + int(arities[index]))
                (finished if entry[2] == 0 else candidates).append(entry)
        beams = sorted(candidates, key=lambda e: -e[0])[:width]
    finished.sort(key=lambda e: -e[0])
    return [([vocabulary.symbol(t) for t in tokens], score) for score, tokens, _ in finished[:width]]


def reproduces(program, args, literals, expected, mode="eq"):
    try:
        with time_limit(EXECUTION_SECONDS):
            value = data2.execute(program, args, literals)
    except (Exception, Timeout):
        return False
    return bi._matches(value, {"expected": expected, "mode": mode})


def propose(net, vocabulary, words, literals, args, expected, named, width=16, arities=None, mode="eq"):
    """Programs that reproduce the example, best first, each with its log-probability."""
    if len(args) > data2.MAX_ARGS or len(literals) > data2.MAX_CONSTS:
        return []
    prefix = vocabulary.condition(words, literals, args, expected, named)
    if len(prefix) + data2.MAX_PROGRAM + 1 > net.config.context:
        return []
    found = beam_search(net, vocabulary, prefix, len(args), len(literals), width, arities=arities)
    return [{"program": p, "log_prob": round(s, 3)} for p, s in found if reproduces(p, args, literals, expected, mode)]


@torch.no_grad()
def program_log_prob(net, vocabulary, prefix, program):
    """log P(program | condition): how much the model believes a given program, e.g. a search match."""
    tokens = prefix + [vocabulary.token(s) for s in program] + [data2.EOS]
    batch = torch.tensor([tokens])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = net(batch)[0]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    start = len(prefix)
    return float(sum(log_probs[i - 1, tokens[i]] for i in range(start, len(tokens))))
