"""Do the teacher's programs share skeletons that programs it has not written yet would reuse? Counts only.

Round 4 found that operators written for single failing tasks do not carry over to unseen tasks, and
the held-out one-liners beyond the engine use the library's existing operators in new arrangements.
So the question for the next kind of rule is whether the arrangements recur. A program's sketch is its
tree with every maximal subtree of at most S tokens replaced by a hole: the part a small exhaustive
search could fill (S = 1: leaves only; S = 3: things like `x % 2` or `len(a0)`; S = 5: about one more
step). A sketch library is the set of sketches seen in at least m training programs.

Proxy (training data only, for choosing S and m): the teacher-written training programs (teacher tasks,
variants, composed tasks, development one-liners; training parts) are cut into five parts by task (a
variant goes with its seed); a program is covered if its sketch is in the library built from the other
four parts. Also reported: coverage with only the N most frequent sketches, coverage as the number of
library programs grows (what more teacher programs would buy), and holes per covered program.

With --held-out (once, for the chosen settings): the held-out teacher tasks and the held-out development
one-liners against the library from all training programs.

  python scripts/seedlm30_sketches.py [--held-out --sizes 3 --min-count 2]
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylittlellm.rulecoder.tasks import ROOT
from seedlm3_train import oneliner_items, teacher_held, teacher_task_items
from seedlm4_train import variant_held
from seedlm5_fragments import tree


def size(node):
    return 1 + sum(size(child) for child in node[1])


def sketch(tokens, limit):
    """(prefix tokens with ? for holes, number of holes); a program within the limit is one hole."""
    out, holes = [], 0

    def walk(node):
        nonlocal holes
        if size(node) <= limit:
            out.append("?")
            holes += 1
            return
        out.append(node[0])
        for child in node[1]:
            walk(child)

    walk(tree(list(tokens)))
    return tuple(out), holes


def training():
    """(group id, tokens) of every teacher-written training program."""
    rows = [(t["id"], t["tokens"]) for t in teacher_task_items() if not teacher_held(t["id"])]
    rows += [(t.get("seed", t["id"]), t["tokens"]) for t in teacher_task_items("variant_tasks") if not variant_held(t)]
    rows += [(t["id"], t["tokens"]) for t in teacher_task_items("compose_tasks") if not teacher_held(t["id"])]
    rows += [(e["task_id"], e["tokens"]) for e in oneliner_items() if e["part"] == "train"]
    return rows


def part(group):
    return int(hashlib.sha256(("sketch:" + group).encode()).hexdigest(), 16) % 5


def library(programs, limit):
    counts = collections.Counter()
    for tokens in programs:
        try:
            counts[sketch(tokens, limit)[0]] += 1
        except (ValueError, IndexError, KeyError):
            pass
    return counts


def covered(programs, counts, limit, min_count, top=None):
    allowed = {s for s, c in counts.most_common(top) if c >= min_count} if top else \
        {s for s, c in counts.items() if c >= min_count}
    hit, holes, small = 0, [], 0
    for tokens in programs:
        try:
            s, k = sketch(tokens, limit)
        except (ValueError, IndexError, KeyError):
            continue
        if s == ("?",):
            small += 1          # the whole program is within the filler size
            hit += 1
        elif s in allowed:
            hit += 1
            holes.append(k)
    return hit, small, holes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--min-count", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--held-out", action="store_true")
    parser.add_argument("--output", default="reports/seedlm/sketch_coverage.json")
    args = parser.parse_args()
    rows = training()
    report = {"training_programs": len(rows), "proxy": {}}
    rng = random.Random(0)
    for limit in args.sizes:
        for min_count in args.min_count:
            total = hit = small = 0
            holes, by_top, by_n = [], collections.Counter(), collections.Counter()
            distinct = 0
            for fold in range(5):
                inside = [t for g, t in rows if part(g) == fold]
                outside = [t for g, t in rows if part(g) != fold]
                counts = library(outside, limit)
                distinct += sum(c >= min_count for c in counts.values())
                h, s, k = covered(inside, counts, limit, min_count)
                total += len(inside)
                hit += h
                small += s
                holes += k
                for top in (50, 200, 1000):
                    by_top[top] += covered(inside, counts, limit, min_count, top)[0]
                for n in (500, 1000, 2000):
                    sample = rng.sample(outside, min(n, len(outside)))
                    by_n[n] += covered(inside, library(sample, limit), limit, min_count)[0]
            holes.sort()
            report["proxy"][f"S{limit}_m{min_count}"] = {
                "programs": total, "covered": hit, "of_which_whole_program_within_S": small,
                "library_sketches_mean": round(distinct / 5),
                "covered_with_top_sketches": dict(by_top), "covered_with_library_programs": dict(by_n),
                "holes_median": holes[len(holes) // 2] if holes else None,
                "holes_max": holes[-1] if holes else None}
            print(f"S{limit}_m{min_count}", json.dumps(report["proxy"][f"S{limit}_m{min_count}"]), flush=True)
    if args.held_out:
        everything = [t for _, t in rows]
        sets = {"teacher_holdout": [t["tokens"] for t in teacher_task_items() if teacher_held(t["id"])],
                "beyond": [e["tokens"] for e in oneliner_items() if e["part"] == "holdout"]}
        report["held_out"] = {}
        for limit in args.sizes:
            counts = library(everything, limit)
            for min_count in args.min_count:
                for name, programs in sets.items():
                    h, s, k = covered(programs, counts, limit, min_count)
                    report["held_out"][f"{name}_S{limit}_m{min_count}"] = {
                        "programs": len(programs), "covered": h, "of_which_whole_program_within_S": s}
        # Where would sketches add something? Held-out teacher programs by SeedLM-21's label and by
        # whether any cached coder already writes a right program (SeedLM-15's cache), for the first setting.
        from seedlm15_ensemble_scaling import CACHE
        from seedlm21_compare import NEW, labels
        limit, min_count = args.sizes[0], args.min_count[0]
        counts = library(everything, limit)
        allowed = {s for s, c in counts.items() if c >= min_count}
        flags = []
        for tokens in sets["teacher_holdout"]:
            found = sketch(tokens, limit)[0]
            flags.append(found == ("?",) or found in allowed)
        task_labels = labels()
        cached = [json.loads(path.read_text())["teacher"] for path in sorted(CACHE.glob("*.json"))]
        some_right = [any(rows[i] and rows[i]["correct"] for rows in cached) for i in range(len(flags))]
        report["held_out"]["teacher_holdout_detail"] = {
            "setting": f"S{limit}_m{min_count}", "cached_coders": len(cached),
            "new_combination": {"programs": sum(l in NEW for l in task_labels),
                                "covered": sum(f for f, l in zip(flags, task_labels) if l in NEW)},
            "no_cached_coder_right": {"programs": sum(not r for r in some_right),
                                      "covered": sum(f for f, r in zip(flags, some_right) if not r)},
            "some_cached_coder_right": {"programs": sum(some_right),
                                        "covered": sum(f for f, r in zip(flags, some_right) if r)}}
        print(json.dumps(report["held_out"]))
    (ROOT / args.output).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
