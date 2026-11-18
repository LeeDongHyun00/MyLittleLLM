"""Consensus of independent hypothesis sources: compare candidates on inputs nobody has answers for.

One visible example cannot tell a correct answer from one that only reproduces it. The model and
the rule engine fail for different reasons: the model misreads the task or gets the algorithm
wrong, the engine matches the example by coincidence. Two sources with different failure modes
rarely make the same mistake on fresh inputs, so

  - agreement on fresh inputs between independent sources is evidence that an answer is right
    (corroboration), and
  - among many candidates, the behaviour supported by the most independent weight is the best
    guess (execution-based minimum Bayes risk, extended to mixed symbolic and neural sources).

Fresh inputs come from a probe grammar: type-preserving mutations of the visible example's
arguments. No probe has a known answer; probes only separate behaviours.
"""
import copy
import hashlib
import math
import random

from .limits import Timeout, time_limit
from .sandbox import CANONICAL, run_calls

_scope = {}
exec(CANONICAL, _scope)     # the same canonical form the sandbox uses
canonical = _scope["canonical"]

LETTERS = "abcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"


# ---------------------------------------------------------------- probe grammar
def _seed(args):
    return int(hashlib.sha256(repr(args).encode()).hexdigest()[:12], 16)


def _number_like(rng, x):
    """A nearby value of the same type and sign class: probes should stay inside the task's domain."""
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        options = [x + 1, x + 2, x - 1, 2 * x, x // 2, x + rng.randint(3, 9), x * 3 + 1]
        if x >= 0:
            options = [v for v in options if v >= 0] or [x + 1]
        return rng.choice(options)
    if isinstance(x, float):
        options = [x * 1.5, x + 0.5, x / 2 if x else 0.75, round(x + rng.uniform(1, 5), 2), x * 2]
        if x >= 0:
            options = [v for v in options if v >= 0]
        return rng.choice(options)
    return x


def _char_like(rng, c):
    if c.isdigit():
        return rng.choice(DIGITS)
    if c.isalpha():
        letter = rng.choice(LETTERS)
        return letter.upper() if c.isupper() else letter
    return c


def _string_like(rng, s):
    if not s:
        return rng.choice(["a", "ab", " "])
    words = s.split(" ")
    choices = ["swap", "drop", "dup", "replace", "reverse", "insert"]
    if len(words) > 1:
        choices += ["shuffle_words", "drop_word", "dup_word"]
    move = rng.choice(choices)
    i = rng.randrange(len(s))
    if move == "swap" and len(s) > 1:
        j = rng.randrange(len(s))
        chars = list(s)
        chars[i], chars[j] = chars[j], chars[i]
        return "".join(chars)
    if move == "drop" and len(s) > 1:
        return s[:i] + s[i + 1:]
    if move == "dup":
        return s[:i] + s[i] + s[i:]
    if move == "replace":
        return s[:i] + _char_like(rng, s[i]) + s[i + 1:]
    if move == "reverse":
        return s[::-1]
    if move == "insert":
        return s[:i] + _char_like(rng, s[i]) + s[i:]
    if move == "shuffle_words":
        rng.shuffle(words)
        return " ".join(words)
    if move == "drop_word":
        words.pop(rng.randrange(len(words)))
        return " ".join(words)
    if move == "dup_word":
        k = rng.randrange(len(words))
        words.insert(k, words[k])
        return " ".join(words)
    return s


def _sequence_like(rng, seq, depth):
    items = list(seq)
    if not items:
        return seq
    # Inside a container a sequence is usually a record or a row: keep its length and order.
    moves = ["change"] if depth else ["shuffle", "reverse", "drop", "append", "change", "dup", "change", "sort"]
    move = rng.choice(moves)
    if move == "shuffle":
        rng.shuffle(items)
    elif move == "reverse":
        items.reverse()
    elif move == "drop" and len(items) > 1:
        items.pop(rng.randrange(len(items)))
    elif move == "append":
        items.insert(rng.randrange(len(items) + 1), mutate_value(rng, rng.choice(items), depth + 1))
    elif move == "change":
        k = rng.randrange(len(items))
        items[k] = mutate_value(rng, items[k], depth + 1)
    elif move == "dup":
        k = rng.randrange(len(items))
        items.insert(k, copy.deepcopy(items[k]))
    elif move == "sort":
        try:
            items.sort()
        except TypeError:
            rng.shuffle(items)
    if isinstance(seq, tuple):
        return tuple(items)
    if isinstance(seq, set):
        return set(items)
    return items


def _dict_like(rng, d, depth):
    if not d:
        return d
    out = dict(d)
    key = rng.choice(list(out))
    move = rng.choice(["value", "value", "drop"])
    if move == "drop" and len(out) > 1 and not depth:
        del out[key]
    else:
        out[key] = mutate_value(rng, out[key], depth + 1)
    return out


def mutate_value(rng, value, depth=0):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return _number_like(rng, value)
    if isinstance(value, str):
        return _string_like(rng, value)
    if isinstance(value, (list, tuple, set)):
        return _sequence_like(rng, value, depth)
    if isinstance(value, dict):
        return _dict_like(rng, value, depth)
    return value


def _keep_relations(original, mutated):
    """An integer argument that equalled a sequence's length, or indexed into it, keeps doing so."""
    out = list(mutated)
    for i, value in enumerate(original):
        if not isinstance(value, int) or isinstance(value, bool):
            continue
        for j, other in enumerate(original):
            if j == i or not isinstance(other, (list, tuple, str)):
                continue
            new_length = len(out[j]) if isinstance(out[j], (list, tuple, str)) else None
            if new_length is None:
                continue
            if value == len(other):
                out[i] = new_length
            elif 0 <= value < len(other) and not 0 <= out[i] < max(new_length, 1):
                out[i] = min(value, max(new_length - 1, 0))
    return out


def probe_inputs(args, count=24, tries=200):
    """Distinct type-preserving variants of the example's argument list, deterministic per example."""
    rng = random.Random(_seed(args))
    seen, probes = {canonical(list(args))}, []
    for _ in range(tries):
        if len(probes) >= count:
            break
        mutated = copy.deepcopy(list(args))
        positions = [rng.randrange(len(mutated))] if mutated else []
        if len(mutated) > 1 and rng.random() < 0.3:
            positions = list(range(len(mutated)))
        for position in positions:
            mutated[position] = mutate_value(rng, mutated[position])
        mutated = _keep_relations(args, mutated)
        key = canonical(mutated)
        if key not in seen:
            seen.add(key)
            probes.append(mutated)
    return probes


# ---------------------------------------------------------------- running candidates
def run_model_code(code, entry_point, probes, per_call=0.5):
    """Untrusted model code runs in the sandbox; returns [(status, canonical output)] per probe."""
    outcome = run_calls(code, entry_point, probes, per_call)
    if outcome.get("status") != "ok":
        return [("exc", "load")] * len(probes)
    return [tuple(item) for item in outcome["results"]]


def run_engine_code(code, entry_point, probes, per_call=0.5):
    """Engine code is composed from the library's own templates, so it runs in this process."""
    scope = {}
    try:
        exec(code, scope)
        fn = scope[entry_point]
    except Exception:
        return [("exc", "load")] * len(probes)
    results = []
    for args in probes:
        try:
            with time_limit(per_call):
                results.append(("ok", canonical(fn(*copy.deepcopy(args)))))
        except (Exception, Timeout) as error:
            results.append(("exc", type(error).__name__))
    return results


# ---------------------------------------------------------------- agreement and selection
def compare(left, right, minimum=3):
    """'conflict' if some probe gives both a value and the values differ; 'agree' if at least
    `minimum` probes give both a value and none differ; otherwise 'unknown'.

    A probe on which one side raises says nothing: the probe may be outside the task's domain,
    where a correct program is allowed to fail.
    """
    both = [(a, b) for (sa, a), (sb, b) in zip(left, right) if sa == "ok" and sb == "ok"]
    if any(a != b for a, b in both):
        return "conflict"
    return "agree" if len(both) >= minimum else "unknown"


def mbr_select(candidates, minimum=3):
    """Pick the candidate whose behaviour has the most independent support.

    `candidates` is a list of dicts with "results" (from run_*_code), "weight" and "source".
    A candidate's score is its own weight plus the weight of every candidate that agrees with it.
    Ties go to the earlier candidate, so listing the current answer first keeps it unless another
    behaviour is strictly better supported.
    """
    best, best_score = None, -math.inf
    for i, candidate in enumerate(candidates):
        score = candidate["weight"] + sum(other["weight"] for j, other in enumerate(candidates)
                                          if j != i and compare(candidate["results"], other["results"], minimum) == "agree")
        if score > best_score:
            best, best_score = i, score
    return best, best_score
