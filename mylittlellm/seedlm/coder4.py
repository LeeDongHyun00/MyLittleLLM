"""SeedLM-4's coder: SeedLM-3's sequence layout with annotated postfix programs (seedlm/post.py).

  BOS DESC <words> LIT <bytes> ARGS <bytes> OUT <bytes> NAMED <operators> PROG t1 t2 ... EOS

The program is in postfix order. Each program token's position also carries three notes, added to
its embedding: the kind, the relation to the expected output and the size of what the partial
program computes on the visible example once that token is written. So the prediction of the next
token sees every value built so far, and the sequence is no longer than SeedLM-3's. Notes are
computed by running the partial program, both when the data is built and while decoding.

Decoding is a beam search over postfix programs: a token is allowed only if it can follow (an
operator needs its operands on the stack of the current scope, a variable must be bound, a closing
token must match the scopes it closes) and the program can still be finished within the length limit.
"""
import math
import re

import torch

from . import coder3, data2, post, pyx

MAX_LENGTH = data2.MAX_LENGTH - data2.MAX_PROGRAM + post.MAX_POST
MAX_ITEMS = 7                 # items a scope may hold at once
NOTE_INDEX = {symbol: i + 1 for i, symbol in enumerate(post.ANNOTATIONS)}     # 0 is "no note"
NO_NOTE = (0, 0, 0)


class Vocabulary(coder3.Vocabulary):
    """coder3.Vocabulary whose program tokens include the scope openers."""


def program_tokens(extra=()):
    tokens = coder3.program_tokens(extra)
    return tokens + [t for t in post.FOR_TOKENS if t not in tokens]


def note_ids(annotation):
    return tuple(NOTE_INDEX[symbol] for symbol in annotation)


def encode(vocabulary, prefix, notes):
    """(tokens, notes per position) for a conditioning prefix and an annotated postfix program."""
    tokens = list(prefix) + [vocabulary.token(token) for token, _ in notes] + [data2.EOS]
    side = [NO_NOTE] * len(prefix) + [note_ids(annotation) for _, annotation in notes] + [NO_NOTE]
    return tokens, side


def collate(rows):
    """Rows are (tokens, start, example) with the notes in example["notes"]; only the program is trained on."""
    width = max(len(tokens) for tokens, _, _ in rows)
    inputs = torch.full((len(rows), width), data2.PAD, dtype=torch.long)
    notes = torch.zeros((len(rows), width, 3), dtype=torch.long)
    labels = torch.full((len(rows), width), -100, dtype=torch.long)
    for i, (tokens, start, ex) in enumerate(rows):
        inputs[i, :len(tokens)] = torch.tensor(tokens)
        notes[i, :len(tokens)] = torch.tensor(ex["notes"])
        labels[i, start - 1:len(tokens) - 1] = torch.tensor(tokens[start:])
    return inputs, notes, labels


def need(counts):
    """A lower bound on the tokens still needed to finish, from the item counts of the open scopes."""
    total, top = 0, counts[0]
    if len(counts) > 1:
        total += (counts[-1] == 0) + 1      # an element or body in the innermost scope, then a closing token
        top += 1
    return total + (1 if top == 0 else math.ceil((top - 1) / 5))


class Grammar:
    """Static per-token tables and the per-hypothesis mask of tokens that may come next."""

    LEAF, OPERATOR, FOR, COMPREHENSION, KEY = range(1, 6)

    def __init__(self, vocabulary):
        size = len(vocabulary)
        self.vocabulary = vocabulary
        self.category = torch.zeros(size, dtype=torch.long)
        self.arity = torch.zeros(size, dtype=torch.long)
        self.variable = torch.full((size,), -1, dtype=torch.long)
        self.output_leaf = torch.zeros(size, dtype=torch.bool)
        self.closers = []
        self.category[data2.ARG0:data2.FIRST_WORD] = self.LEAF
        for symbol, index in vocabulary.index.items():
            if symbol.startswith(post.FOR):
                self.category[index] = self.FOR
            elif post.is_comprehension(symbol):
                self.category[index] = self.COMPREHENSION
                self.closers.append((index, symbol))
            elif post.is_key(symbol):
                self.category[index] = self.KEY
                self.closers.append((index, symbol))
            elif pyx.arity(symbol) == 0:
                self.category[index] = self.LEAF
                if re.fullmatch(r"v\d", symbol):
                    self.variable[index] = int(symbol[1:])
                self.output_leaf[index] = symbol in ("o", "o!")
            else:
                self.category[index] = self.OPERATOR
                self.arity[index] = pyx.arity(symbol)
        self.leaf = self.category == self.LEAF
        self.operator = self.category == self.OPERATOR
        self.openers = [(index, int(symbol[len(post.FOR):])) for symbol, index in vocabulary.index.items()
                        if symbol.startswith(post.FOR)]

    def base_mask(self, n_args, n_literals, string_output):
        valid = self.category > 0
        valid[data2.ARG0 + n_args:data2.CONST0] = False
        valid[data2.CONST0 + n_literals:data2.FIRST_WORD] = False
        if not string_output:
            valid &= ~self.output_leaf
        return valid

    def allowed(self, base, frames, room):
        """Tokens that may follow, given that at most `room` more tokens may come after the next one."""
        counts = [len(f.items) for f in frames]
        current, variables = counts[-1], post.bound(frames)
        by_count = torch.tensor([need(counts[:-1] + [c]) for c in range(MAX_ITEMS + 1)])
        mask = torch.zeros_like(base)
        if current < MAX_ITEMS and need(counts[:-1] + [current + 1]) <= room:
            mask |= self.leaf & (self.variable < variables)
        after = (current - self.arity + 1).clamp(0, MAX_ITEMS)
        mask |= self.operator & (self.arity <= current) & (by_count[after] <= room)
        if current >= 1:
            for index, count in self.openers:
                if variables + count <= pyx.MAX_VARS and need(counts[:-1] + [current - 1, 0]) <= room:
                    mask[index] = True
        if len(frames) > 1:
            for index, symbol in self.closers:
                try:
                    closed = post.step(frames, symbol)
                except ValueError:
                    continue
                if need([len(f.items) for f in closed]) <= room:
                    mask[index] = True
        return mask & base


@torch.no_grad()
def beam_search(net, vocabulary, grammar, prefix, args, literals, expected, mode="eq", width=16,
                max_tokens=post.MAX_POST, keep=None):
    """Postfix programs that can be finished, as ([postfix symbols], log-probability), best first."""
    string_output = isinstance(expected, str)
    scope = post._scope(args, literals, expected if string_output else None)
    base = grammar.base_mask(len(args), len(literals), string_output)
    beams = [(0.0, [], post.start())]          # (score, postfix symbols, frames)
    finished = []
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits, cache = net.prefill(torch.tensor([prefix]), torch.zeros((1, len(prefix), 3), dtype=torch.long))
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    for position in range(max_tokens + 1):
        for row, (score, symbols, frames) in enumerate(beams):
            if symbols and post.finished(frames):
                finished.append((score + float(log_probs[row, data2.EOS]), symbols))
        if position == max_tokens:
            break
        room = max_tokens - position - 1
        masks = torch.stack([grammar.allowed(base, frames, room) for _, _, frames in beams])
        scores = torch.tensor([score for score, _, _ in beams])
        total = (scores.unsqueeze(1) + log_probs).masked_fill(~masks, float("-inf")).flatten()
        count = min(2 * width, int(torch.isfinite(total).sum()))
        if count == 0:
            break
        values, flat = total.topk(count)
        size = log_probs.size(1)
        chosen, parents, tokens, notes = [], [], [], []
        for value, position_flat in zip(values.tolist(), flat.tolist()):
            row, index = divmod(position_flat, size)
            _, symbols, frames = beams[row]
            symbol = vocabulary.symbol(index)
            try:
                frames = post.step(frames, symbol)
            except ValueError:
                continue
            chosen.append((value, symbols + [symbol], frames))
            parents.append(row)
            tokens.append(index)
            notes.append(NO_NOTE if getattr(net, "blind", False)
                         else note_ids(post.annotation(frames, symbol, scope, expected, mode)))
            if len(chosen) == width:
                break
        if not chosen:
            break
        rows = torch.tensor(parents)
        cache = [(k[rows], v[rows]) for k, v in cache]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits, cache = net.step(torch.tensor(tokens), cache, torch.tensor(notes))
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        beams = chosen
    finished.sort(key=lambda e: -e[0])
    return [(symbols, score) for score, symbols in finished[:keep or width]]


@torch.no_grad()
def program_log_prob(net, vocabulary, prefix, postfix, args, literals, expected, mode="eq"):
    """log P(program) under the coder, with the program's own notes in context."""
    tokens, side = encode(vocabulary, prefix, post.annotated(postfix, args, literals, expected, mode))
    if getattr(net, "blind", False):
        side = [NO_NOTE] * len(side)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = net(torch.tensor([tokens]), notes=torch.tensor([side]))[0]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return float(sum(log_probs[i - 1, tokens[i]] for i in range(len(prefix), len(tokens))))


def reproduces(postfix, args, literals, expected, mode="eq"):
    try:
        program = post.to_prefix(postfix)
    except ValueError:
        return False
    return coder3.reproduces(program, args, literals, expected, mode)


def load(path):
    from . import model as seedmodel
    checkpoint = torch.load(path)
    vocabulary = Vocabulary(checkpoint["vocabulary"]["words"], checkpoint["vocabulary"]["operators"])
    net = seedmodel.SeedLM(seedmodel.Config(**checkpoint["config"]))
    net.load_state_dict(checkpoint["state"])
    net.blind = bool(checkpoint.get("blind"))     # the ablation trained without notes decodes without them
    net.eval()
    return net, vocabulary
