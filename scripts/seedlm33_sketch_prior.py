"""SeedLM-33: score a candidate as structure times filling, not as one token sequence. Counts only.

The sketch search forces a known sketch's fixed tokens because the coder gives a new parent -> child link
almost no probability (SeedLM-5, 31); scoring the finished program by the coder's whole-sequence probability
then charges those forced tokens again, so on real tasks the right sketch programs lose (SeedLM-32). Here a
candidate's score is split the way it was made:

  structure  log P(sketch | description words, argument count, output kind) from the naive Bayes sketch
             ranker, with the whole-program sketch (programs of at most 5 tokens) counted as a sketch too
  filling    the generator coder's log-probability of the tokens inside the holes only

Rules (on the gathered candidates of scripts/seedlm32_candidates.py): structure + filling ("factored");
structure + filling + the six-coder mean ("factored_ensemble", the six coders' whole-sequence score as a
third term). Each is reported alone and as the system (the engine where it is confident).

  python scripts/seedlm33_sketch_prior.py --set teacher          # exploring
  python scripts/seedlm33_sketch_prior.py --set development --rule <registered rule>
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import coder3, data2, language, sketch as sk
from seedlm3_train import named_in
from seedlm31_sketch_search import training_examples
from seedlm32_candidates import OUT, development_tasks, teacher_tasks
from seedlm32_select import ensemble

RULES = ["factored", "factored_ensemble"]
UNSEEN = -40.0      # log-score of a sketch the library has never seen


class WholeLibrary(sk.Library):
    """The sketch library with the single-hole sketch (programs of at most 5 tokens) kept as a sketch."""

    def __init__(self, examples):
        super().__init__(examples)
        short = [(sk.content_words(w), n, sk.value_kind(o)) for t, w, n, o in examples
                 if sk.sketch_of(t)[0] == (sk.HOLE,)]
        whole = (sk.HOLE,)
        self.sketches.append(whole)
        self.count[whole] = len(short)
        self.words[whole] = __import__("collections").Counter(w for row in short for w in row[0])
        self.word_total[whole] = sum(self.words[whole].values())
        self.args[whole] = __import__("collections").Counter(row[1] for row in short)
        self.kinds[whole] = __import__("collections").Counter(row[2] for row in short)
        self.vocabulary |= set(self.words[whole])
        self.total += len(short)


@torch.no_grad()
def hole_log_prob(net, vocabulary, prefix, program, sketch):
    """The coder's log-probability of the program's tokens that fall inside the sketch's holes."""
    tokens = prefix + [vocabulary.token(s) for s in program] + [data2.EOS]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = net(torch.tensor([tokens]))[0]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    fixed = _fixed_positions(program, sketch)
    return float(sum(log_probs[len(prefix) + i - 1, tokens[len(prefix) + i]]
                     for i in range(len(program)) if i not in fixed))


def _fixed_positions(program, sketch):
    """Positions of the program's tokens that are the sketch's fixed tokens (the rest are inside holes)."""
    fixed, position = set(), 0

    def walk(node, item_iter):
        nonlocal position
        item = next(item_iter)
        if item == sk.HOLE:
            position += _count(node)
            return
        fixed.add(position)
        position += 1
        for child in node[1]:
            walk(child, item_iter)

    def _count(node):
        return 1 + sum(_count(c) for c in node[1])

    walk(sk.tree(list(program)), iter(sketch))
    return fixed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=["teacher", "development"], required=True)
    parser.add_argument("--rule", choices=RULES)
    parser.add_argument("--model", default="s20-p5")
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.set_num_threads(4)
    path = OUT / f"candidates_{args.set}.json"
    rows = json.loads(path.read_text())
    tasks = teacher_tasks() if args.set == "teacher" else development_tasks()
    library = WholeLibrary(training_examples())
    known = set(library.sketches)
    net, vocabulary = coder3.load(ROOT / "outputs/seedlm" / f"{args.model}.pt")
    for row, (case, description, _, _, _) in zip(rows, tasks):
        if row is None:
            continue
        words = language.words(description)
        prefix = vocabulary.condition(words, case["literals"], case["args"], case["expected"], named_in(description))
        items = row["candidates"] + ([row["engine"]] if row["engine"] and row["engine"].get("program")
                                     and "behaviour" in row["engine"] else [])
        for item in items:
            try:
                sketch, _ = sk.sketch_of(item["program"])
                item["structure"] = library.score(sketch, words, case["args"].__len__(), case["expected"]) \
                    if sketch in known else UNSEEN
                item["filling"] = hole_log_prob(net, vocabulary, prefix, item["program"], sketch)
            except (KeyError, ValueError, IndexError):
                item["structure"], item["filling"] = UNSEEN, -100.0
    path.write_text(json.dumps(rows) + "\n")
    rows = [r for r in rows if r is not None]

    def value(rule, item):
        base = item["structure"] + item["filling"]
        return base if rule == "factored" else base + ensemble(item)

    report = {"set": args.set, "tasks": len(rows), "rules": {}}
    for rule in ([args.rule] if args.rule else RULES):
        alone = system = fixed = broken = 0
        for r in rows:
            pick = max(r["candidates"], key=lambda c: value(rule, c)) if r["candidates"] else None
            alone += bool(pick and pick["right"])
            engine = r["engine"]
            if engine and engine["confident"]:
                answer = engine["right"]
            else:
                pool = r["candidates"] + ([engine] if engine and "structure" in engine else [])
                choice = max(pool, key=lambda c: value(rule, c)) if pool else None
                answer = bool(choice["right"]) if choice else bool(engine and engine["right"])
            system += answer
            fixed += answer and not (engine and engine["right"])
            broken += bool(engine and engine["right"]) and not answer
        report["rules"][rule] = {"alone": alone, "system": system, "system_versus_engine": [fixed, broken]}
    text = json.dumps(report, indent=1)
    print(text)
    if args.output:
        (ROOT / args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
