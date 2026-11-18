"""SeedLM-2's language side: a word vocabulary and descriptions written for synthetic programs.

SeedLM-1 never saw a task description; it saw the operators a keyword table found in it. The
keyword table is brittle: "find" names `index`, and a description that says "aggregate" instead of
"sum" names nothing. SeedLM-2 reads the words. Its training descriptions are composed from phrases
that teacher models wrote for every operator (data/seedlm/operator_phrases.json), so the language it
learns is distilled from the teachers, not taken from the benchmark.
"""
import json
import re

from ..rulecoder.tasks import ROOT

PHRASES = ROOT / "data/seedlm/operator_phrases.json"
WORD = re.compile(r"[a-z]+|\d+")
OPENERS = ["write a function to", "write a python function to", "write a function that", "create a function to",
           "write a program to", "", "", "implement a function to"]
JOINERS = [" and then ", " and ", ", then ", " after that ", " and return ", " then "]
OF_THE = [" of ", " of the result of ", " for ", " applied to "]
FILLER = "given the a an of in to and for with from each all its that is".split()


def words(text):
    return WORD.findall(text.lower())


def load_phrases(path=PHRASES):
    """Teacher-written phrases per operator key; operators without phrases fall back to their keywords."""
    return json.loads(path.read_text()) if path.exists() else {}


def keyword_phrases(op):
    """Fallback: the words of the operator's keyword alternatives, e.g. "sum&square" -> "sum square"."""
    out = [" ".join(alt) for alt in op.alternatives if alt]
    return out or [op.name.replace("_", " ")]


class Describer:
    """Turns a prefix program into a plausible task description, with deliberate variety and noise."""

    def __init__(self, operators, phrases=None):
        self.operators = operators
        self.phrases = phrases if phrases is not None else load_phrases()

    def phrase(self, rng, key):
        options = self.phrases.get(key) or keyword_phrases(self.operators[key])
        return rng.choice(options)

    def describe(self, rng, program, literals=()):
        """Inner operations first (in reading order), joined; literals mentioned next to their user."""
        position = 0
        clauses = []

        def walk():
            nonlocal position
            token = program[position]
            position += 1
            if "/" not in token:
                return
            op = self.operators[token]
            children = []
            for _ in range(op.arity):
                child_start = position
                walk()
                children.append(program[child_start:position])
            text = self.phrase(rng, token)
            for child in children:
                if len(child) == 1 and child[0].startswith("c"):
                    value = literals[int(child[0][1:])]
                    text += " " + (repr(value) if isinstance(value, str) else str(value))
            clauses.append(text)

        walk()
        if len(clauses) > 1 and rng.random() < 0.35:
            # Outer operation first: "find the sum of the k smallest numbers ..."
            text = clauses[-1]
            for inner in reversed(clauses[:-1]):
                text += rng.choice(OF_THE) + _strip_verb(inner)
        else:
            text = clauses[0]
            for clause in clauses[1:]:
                text += rng.choice(JOINERS) + clause
        opener = rng.choice(OPENERS)
        text = (opener + " " + text).strip()
        tokens = words(text)
        # Noise, so the reader does not depend on exact wording: drop or insert a few small words.
        noisy = []
        for token in tokens:
            if rng.random() < 0.04:
                continue
            noisy.append(token)
            if rng.random() < 0.03:
                noisy.append(rng.choice(FILLER))
        return noisy


def _strip_verb(clause):
    parts = clause.split(" ", 1)
    return parts[1] if len(parts) == 2 and parts[0] in {"find", "get", "return", "compute", "calculate"} else clause


def build_vocabulary(extra_texts=(), phrases=None, minimum=1):
    """Words from the teacher phrases, the openers and extra development texts (never held-out ones)."""
    phrases = load_phrases() if phrases is None else phrases
    counts = {}
    for texts in [sum(phrases.values(), []), OPENERS, JOINERS, OF_THE, FILLER, list(extra_texts)]:
        for text in texts:
            for token in words(text):
                counts[token] = counts.get(token, 0) + 1
    return sorted(token for token, count in counts.items() if count >= minimum)
