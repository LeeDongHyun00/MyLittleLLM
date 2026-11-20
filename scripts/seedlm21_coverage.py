"""How much of what the coder has never seen do generated programs cover? (SeedLM-21)

A program's edges are its parent -> child links over token kinds (scripts/seedlm5_fragments.py). An
edge is new for a program if neither the synthetic library programs (100,000, as the coder's training
draws them) nor any training program has it. This script measures how many new edges a set of
generated programs (mylittlellm/seedlm/pysynth.py) contains, in two ways:

  proxy    - training data only, for choosing the generator's settings: the training teacher tasks
             and variants are cut into five parts; for each part, its programs' edges that neither the
             synthetic programs nor the other training programs have are its "new" edges
  held-out - once, for the chosen settings: the new edges of the held-out teacher tasks and of the
             held-out development one-liners (counts only)

Reported: new edges covered, and programs whose every new edge is covered, by class of parent and child
(library operator, Python node, builtin, method, comprehension, leaf).

  python scripts/seedlm21_coverage.py --count 30000 [--seed 0] [--held-out] [--output ...]
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import pysynth
from seedlm3_train import oneliner_items, teacher_held, teacher_task_items
from seedlm4_train import variant_held
from seedlm5_fragments import evaluation_sets, pieces, synthetic_programs
from seedlm6_grafts import programs as host_programs

CACHE = ROOT / "outputs/seedlm/v21/synthetic_pieces.json"


def edge_class(token_kind):
    if token_kind in ("ARG", "LIT", "VAR", "INT", "STR"):
        return "leaf"
    if token_kind.startswith("@Comp"):
        return "comprehension"
    if token_kind.startswith("@."):
        return "method"
    if token_kind.startswith("@"):
        return "python" if token_kind[1:2].isupper() else "builtin"
    return "library" if "/" in token_kind else "leaf"


def synthetic_pieces():
    """Tokens, edges, twigs and whole programs (by kinds) of the 100,000 synthetic library programs that
    SeedLM-5's diagnostics count as seen (the same draw), cached."""
    if CACHE.exists():
        return {name: {tuple(x) if isinstance(x, list) else x for x in values}
                for name, values in json.loads(CACHE.read_text()).items()}
    known = {"tokens": set(), "edges": set(), "twigs": set(), "whole": set()}
    for program in synthetic_programs(100000):
        for name, found in pieces(program).items():
            known[name] |= found
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps({name: sorted(list(x) if isinstance(x, tuple) else x for x in values)
                                 for name, values in known.items()}))
    return known


def known_pieces():
    """SeedLM-5's seen pieces: the synthetic programs and every training program."""
    from seedlm5_fragments import training_programs
    known = synthetic_pieces()
    for program in training_programs():
        for name, found in pieces(program).items():
            known[name] |= found
    return known


def synthetic_edges():
    return synthetic_pieces()["edges"]


def training_parts():
    """Training teacher tasks and variants, each with its part (0-4), and the other training programs."""
    parted = []
    for task in teacher_task_items():
        if not teacher_held(task["id"]):
            parted.append(task)
    for task in teacher_task_items("variant_tasks"):
        if not variant_held(task):
            parted.append(task)
    rest = [e["tokens"] for e in oneliner_items() if e["part"] == "train"]
    rest += [row["program"] for row in json.loads((ROOT / "outputs/seedlm/v2/real_dev.json").read_text())
             if row["part"] == "train"]

    def part(task):
        key = task.get("seed", task["id"])     # a variant goes with its seed's other variants
        return int(hashlib.sha256(("part:" + key).encode()).hexdigest(), 16) % 5
    return [(part(t), t["tokens"]) for t in parted], rest


def coverage(targets, generated):
    """targets: list of sets of new edges (one per program with any). Counts by edge class."""
    by_class = collections.defaultdict(lambda: [0, 0])
    edges_total = edges_covered = programs_covered = 0
    for new in targets:
        for edge in new:
            key = f"{edge_class(edge[0])}->{edge_class(edge[2])}"
            by_class[key][0] += 1
            by_class[key][1] += edge in generated
            edges_total += 1
            edges_covered += edge in generated
        programs_covered += new <= generated
    return {"programs": len(targets), "programs_all_new_edges_covered": programs_covered,
            "new_edges": edges_total, "new_edges_covered": edges_covered,
            "by_class": {k: {"edges": v[0], "covered": v[1]} for k, v in sorted(by_class.items(), key=lambda kv: -kv[1][0])}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--library", default="all", choices=["all", "training", "sqrt"])
    parser.add_argument("--families", help="JSON weights of the step families (default pysynth.FAMILIES)")
    parser.add_argument("--held-out", action="store_true", help="also the held-out sets (once, for the chosen settings)")
    parser.add_argument("--output")
    args = parser.parse_args()
    started = time.time()
    synthetic = synthetic_edges()
    parted, rest = training_parts()
    rest_edges = set()
    for program in rest:
        rest_edges |= pieces(program)["edges"]
    part_edges = [set() for _ in range(5)]
    for p, program in parted:
        part_edges[p] |= pieces(program)["edges"]
    from seedlm5_fragments import training_programs
    families = json.loads(args.families) if args.families else None
    made = pysynth.generate(host_programs(), args.count, args.seed, families,
                            pysynth.library_from(training_programs(), args.library))
    generated = set()
    for row in made:
        generated |= pieces(row["tokens"])["edges"]
    proxy_targets = []
    for p, program in parted:
        known = synthetic | rest_edges | set().union(*(part_edges[q] for q in range(5) if q != p))
        new = pieces(program)["edges"] - known
        if new:
            proxy_targets.append(new)
    report = {"count": len(made), "seed": args.seed, "library": args.library, "families": families or pysynth.FAMILIES,
              "generated_edges": len(generated),
              "generated_edges_new_to_training": len(generated - synthetic - rest_edges - set().union(*part_edges)),
              "proxy": coverage(proxy_targets, generated)}
    if args.held_out:
        known = synthetic | rest_edges | set().union(*part_edges)
        for name, programs in evaluation_sets().items():
            targets = [pieces(p)["edges"] - known for p in programs]
            report[name] = coverage([t for t in targets if t], generated)
            report[name]["programs_in_set"] = len(programs)
    report["seconds"] = round(time.time() - started, 1)
    text = json.dumps(report, indent=1)
    print(text)
    if args.output:
        (ROOT / args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
