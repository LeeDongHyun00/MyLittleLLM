"""Program sketches: the structure of a program with its small parts left as holes (SeedLM-30, 31).

A sketch keeps a program's prefix tokens down to the subtrees of at most S tokens, each of which becomes a
hole "?" (Nye et al. 2019, "Learning to infer program sketches"). On held-out teacher programs, sketches
seen at least twice among the teacher-written training programs, with holes of up to 5 tokens, express 118
of 139 programs, against 59 to 62 that a single coder writes (scripts/seedlm30_sketches.py). The novel part
of an unseen program is mostly inside its holes, which a coder fills well.

This module holds the sketch library and a naive Bayes ranker that guesses, from a task's description words,
its number of arguments and the kind of its visible output, which sketches to try first.
"""
import collections
import math

from . import language, pyx

HOLE = "?"
STOP = set("write a an the function to python of in and given for from with that is are by list string number "
           "numbers find check whether return returns its it as be or each all if".split())


def tree(tokens):
    position = 0

    def read():
        nonlocal position
        token = tokens[position]
        position += 1
        return token, [read() for _ in range(pyx.arity(token))]

    out = read()
    if position != len(tokens):
        raise ValueError("trailing tokens")
    return out


def _size(node):
    return 1 + sum(_size(child) for child in node[1])


def sketch_of(tokens, limit=5):
    """(sketch as a tuple of prefix tokens with HOLE for every maximal subtree of at most `limit` tokens,
    number of holes)."""
    out, holes = [], 0

    def walk(node):
        nonlocal holes
        if _size(node) <= limit:
            out.append(HOLE)
            holes += 1
            return
        out.append(node[0])
        for child in node[1]:
            walk(child)

    walk(tree(list(tokens)))
    return tuple(out), holes


def content_words(text_or_words):
    words = language.words(text_or_words) if isinstance(text_or_words, str) else text_or_words
    return [w for w in words if w not in STOP and not w.isdigit()]


def value_kind(value):
    if isinstance(value, bool):
        return "bool"
    for kind in (int, float, str, list, tuple, dict, set):
        if isinstance(value, kind):
            return kind.__name__
    return "other"


class Library:
    """Sketches seen at least `min_count` times, with naive Bayes statistics for ranking them.

    examples: (tokens, description words, number of arguments, visible output) per training program.
    The whole-program sketch (a single hole) is left out: a plain beam already covers short programs.
    """

    def __init__(self, examples, limit=5, min_count=2, alpha=0.5):
        self.limit, self.alpha = limit, alpha
        seen = collections.defaultdict(list)
        for tokens, words, n_args, output in examples:
            try:
                found, _ = sketch_of(tokens, limit)
            except (ValueError, IndexError, KeyError):
                continue
            if found != (HOLE,):
                seen[found].append((content_words(words), n_args, value_kind(output)))
        self.sketches = [s for s, rows in seen.items() if len(rows) >= min_count]
        self.count = {s: len(seen[s]) for s in self.sketches}
        self.words = {s: collections.Counter(w for row in seen[s] for w in row[0]) for s in self.sketches}
        self.word_total = {s: sum(self.words[s].values()) for s in self.sketches}
        self.args = {s: collections.Counter(row[1] for row in seen[s]) for s in self.sketches}
        self.kinds = {s: collections.Counter(row[2] for row in seen[s]) for s in self.sketches}
        self.vocabulary = {w for s in self.sketches for w in self.words[s]}
        self.total = sum(self.count.values())

    def score(self, sketch, words, n_args, output):
        a, n = self.alpha, self.count[sketch]
        value = math.log(n / self.total)
        value += math.log((self.args[sketch][n_args] + a) / (n + 5 * a))
        value += math.log((self.kinds[sketch][value_kind(output)] + a) / (n + 8 * a))
        size = len(self.vocabulary) + 1
        for word in set(content_words(words)):
            value += math.log((self.words[sketch][word] + a) / (self.word_total[sketch] + a * size))
        return value

    def rank(self, words, n_args, output):
        return sorted(self.sketches, key=lambda s: -self.score(s, words, n_args, output))


def sketch_search(net, vocabulary, grammar, prefix, n_args, n_literals, string_output, sketches, per_sketch=4,
                  hole_size=5, max_tokens=None):
    """Prefix-coder beam search held to each sketch: a sketch's fixed tokens are forced, each hole is filled
    by the coder with a subtree of at most `hole_size` tokens. All sketches decode in one batch, each keeping
    its own `per_sketch` best hypotheses. Returns (program symbols, log-probability, sketch index), best first.

    Forcing the fixed tokens is the point: the coder gives a new parent -> child link almost no probability
    (SeedLM-5), so its own beam never builds the structure; the small parts inside the holes it fills well.
    """
    import torch
    from . import coder3
    max_tokens = max_tokens or coder3.MAX_PROGRAM
    base = grammar.base_mask(n_args, n_literals, string_output)
    arity = grammar.arity
    variable, has_variable = grammar.variable, grammar.variable >= 0
    items = []
    for sketch in sketches:
        try:
            items.append([None if t == HOLE else vocabulary.token(t) for t in sketch])
        except (KeyError, ValueError):
            items.append(None)          # a token this coder's vocabulary lacks
    # A row: [sketch index, stack of slot depths, next item, hole start (stack length) or None, hole tokens used]
    rows = [[g, [0], 0, None, 0] for g, it in enumerate(items) if it is not None]
    if not rows:
        return []
    scores = torch.zeros(len(rows))
    sequences = [[] for _ in rows]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits, cache = net.prefill(torch.tensor([prefix]))
    cache = [(k.expand(len(rows), *k.shape[1:]).contiguous(), v.expand(len(rows), *v.shape[1:]).contiguous())
             for k, v in cache]
    log_probs = torch.log_softmax(logits.float(), dim=-1).expand(len(rows), -1)
    finished = []
    for step in range(max_tokens):
        room = max_tokens - step - 1
        masks = []
        for g, stack, position, hole, used in rows:
            depth = stack[-1]
            allowed = base & (len(stack) - 1 + arity <= room)
            allowed &= ~(has_variable & (variable >= depth))
            allowed &= depth + grammar.binds <= pyx.MAX_VARS
            if hole is None and items[g][position] is not None:
                fixed = torch.zeros_like(allowed)
                fixed[items[g][position]] = True
                allowed &= fixed
            else:
                open_in_hole = 1 if hole is None else len(stack) - (hole - 1)
                used_now = 0 if hole is None else used
                allowed &= arity <= hole_size - used_now - open_in_hole
            masks.append(allowed)
        total = (scores.unsqueeze(1) + log_probs).masked_fill(~torch.stack(masks), float("-inf"))
        values, indices = total.topk(min(per_sketch, total.shape[1]), dim=1)
        candidates = {}
        for r, (g, stack, position, hole, used) in enumerate(rows):
            for value, index in zip(values[r].tolist(), indices[r].tolist()):
                if value == float("-inf"):
                    break
                symbol = vocabulary.symbol(index)
                new_stack = stack[:-1] + list(reversed(coder3.child_depths(symbol, stack[-1])))
                start, count, next_position = (len(stack), 0, position) if hole is None and items[g][position] is None \
                    else (hole, used, position)
                if start is None:                    # a forced token: the item is consumed
                    next_position, state = position + 1, (None, 0)
                else:
                    count += 1
                    if len(new_stack) - (start - 1) == 0:    # the hole's subtree is complete
                        next_position, state = position + 1, (None, 0)
                    else:
                        state = (start, count)
                if not new_stack:
                    if next_position == len(items[g]):
                        finished.append((value, g, sequences[r] + [index]))
                    continue
                if next_position >= len(items[g]):
                    continue
                candidates.setdefault(g, []).append((value, r, index, new_stack, next_position, state))
        new_rows, parents, chosen, new_scores, new_sequences = [], [], [], [], []
        for g, options in candidates.items():
            options.sort(key=lambda o: -o[0])
            for value, r, index, new_stack, next_position, (start, count) in options[:per_sketch]:
                new_rows.append([g, new_stack, next_position, start, count])
                parents.append(r)
                chosen.append(index)
                new_scores.append(value)
                new_sequences.append(sequences[r] + [index])
        if not new_rows:
            break
        rows, sequences, scores = new_rows, new_sequences, torch.tensor(new_scores)
        index_rows = torch.tensor(parents)
        cache = [(k[index_rows], v[index_rows]) for k, v in cache]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits, cache = net.step(torch.tensor(chosen), cache)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
    finished.sort(key=lambda e: -e[0])
    return [([vocabulary.symbol(t) for t in tokens], score, g) for score, g, tokens in finished]
