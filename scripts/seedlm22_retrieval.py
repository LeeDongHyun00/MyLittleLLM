"""Is a held-out program near a training program that a search by description would find? Counts only.

The coder writes the combinations it has seen (SeedLM-5) and, on the development tasks beyond the
engine, its first verified program is usually six or more tokens away from the teacher's (SeedLM-5's
near-miss diagnostic). MBPP-style tasks come in families, so a held-out task may have a relative in
the training data whose program differs in a few tokens. If such relatives can be found from the
description alone, a coder that sees a retrieved program while it writes (retrieval-augmented
generation) could reach programs it would not compose by itself.

For every held-out teacher task and held-out development one-liner, training programs (teacher tasks,
variants, composed tasks, development one-liners, the engine's programs on training development
tasks; training parts only) are ranked by the content words their description shares with the
held-out description, ties broken by the number of arguments matching. Reported per set (the held-out
teacher tasks also for those whose program holds a new combination, SeedLM-21's 77): the smallest
token edit distance from the held-out program to a program among the top 1, 5 and 20, and the
distance to the nearest training program of all (what retrieval could reach at best), as
histograms with SeedLM-5's bins; and how often a top-k program has the held-out program's shape
(the same tokens once leaves are reduced to their kinds).

  python scripts/seedlm22_retrieval.py [--output reports/seedlm/retrieval_distance.json]
"""
import argparse
import collections
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import language
from seedlm3_train import oneliner_items, teacher_held, teacher_task_items
from seedlm4_train import variant_held
from seedlm5_fragments import first_unknown, kind, pieces

STOP = set("write a an the function to python of in and given for from with that is are by list string number "
           "numbers find check whether return returns its it as be or each all if".split())
BINS = ["0", "1", "2", "3", "4-5", "6+"]
TOP = (1, 5, 20)


def content(text):
    return {w for w in language.words(text) if w not in STOP and not w.isdigit()}


def distance(a, b):
    """Token edit distance (insert, delete, substitute)."""
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def bin_of(d):
    return str(d) if d <= 3 else "4-5" if d <= 5 else "6+"


def n_args(tokens):
    return len({t for t in tokens if len(t) == 2 and t[0] == "a" and t[1].isdigit()})


def corpus():
    out = []
    for task in teacher_task_items():
        if not teacher_held(task["id"]):
            out.append((task["description"], task["tokens"]))
    for task in teacher_task_items("variant_tasks"):
        if not variant_held(task):
            out.append((task["description"], task["tokens"]))
    for task in teacher_task_items("compose_tasks"):
        if not teacher_held(task["id"]):
            out.append((task["description"], task["tokens"]))
    for entry in oneliner_items():
        if entry["part"] == "train":
            out.append((entry["item"]["description"], entry["tokens"]))
    for row in json.loads((ROOT / "outputs/seedlm/v2/real_dev.json").read_text()):
        if row["part"] == "train":
            out.append((" ".join(row["words"]), row["program"]))
    return [{"words": content(d), "tokens": list(t), "shape": tuple(kind(x) for x in t), "args": n_args(t)}
            for d, t in out]


def held_sets():
    from seedlm21_coverage import known_pieces
    known = known_pieces()
    teacher = [(t["description"], t["tokens"]) for t in teacher_task_items() if teacher_held(t["id"])]
    novel = [(d, t) for d, t in teacher if first_unknown(pieces(t), known) in ("tokens", "edges", "twigs")]
    return {"teacher_holdout": teacher, "teacher_new_combination": novel,
            "beyond": [(e["item"]["description"], e["tokens"]) for e in oneliner_items() if e["part"] == "holdout"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/seedlm/retrieval_distance.json")
    args = parser.parse_args()
    pool = corpus()
    report = {"training_programs": len(pool)}
    for name, items in held_sets().items():
        hist = {k: collections.Counter() for k in [f"top{k}" for k in TOP] + ["nearest_any"]}
        shape = collections.Counter()
        for description, tokens in items:
            words, arity, gold_shape = content(description), n_args(tokens), tuple(kind(x) for x in tokens)
            ranked = sorted(pool, key=lambda p: (-len(words & p["words"]), p["args"] != arity))
            distances = [distance(tokens, p["tokens"]) for p in ranked[:max(TOP)]]
            for k in TOP:
                hist[f"top{k}"][bin_of(min(distances[:k]))] += 1
                shape[f"top{k}"] += any(p["shape"] == gold_shape for p in ranked[:k])
            hist["nearest_any"][bin_of(min(distance(tokens, p["tokens"]) for p in pool))] += 1
            shape["any"] += any(p["shape"] == gold_shape for p in pool)
        report[name] = {"programs": len(items),
                        "distance": {k: {b: v.get(b, 0) for b in BINS} for k, v in hist.items()},
                        "same_shape": dict(shape)}
    text = json.dumps(report, indent=1)
    print(text)
    (ROOT / args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
