"""Are the programs the coder cannot find new compositions of known pieces, or do they need new pieces?

Programs are prefix pyx trees. For each evaluation program (a teacher's program the coder never saw)
this script asks which of its pieces occur somewhere in the coder's training programs (every source,
training parts only):

  tokens   every token (operator, Python node, leaf kind) occurs in training
  edges    every parent -> child link (parent token, child slot, child token) occurs in training
  twigs    every depth-two fragment (a node with its children's tokens) occurs in training
  whole    the program itself occurs in training

Leaves are reduced to their kind (argument, literal slot, bound variable, small integer, string), so a
piece is about structure, not about which argument it reads. If most programs have known tokens and
edges but unknown twigs or wholes, what is missing is composition (pieces never seen together);
if their tokens or edges are unseen, what is missing is knowledge (pieces never seen at all).

Sets: held-out teacher tasks, variants of held-out seeds, and the held-out development tasks the engine
fails that a teacher solved in one line (counts only: that set belongs to the held-out development
fifth).

  python scripts/seedlm5_fragments.py
"""
import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import pyx
from seedlm3_train import oneliner_items, teacher_held, teacher_task_items
from seedlm4_train import variant_held


def kind(token):
    """A leaf reduced to its kind; any other token as it is."""
    if re.fullmatch(r"a\d", token):
        return "ARG"
    if re.fullmatch(r"c\d", token):
        return "LIT"
    if re.fullmatch(r"v\d", token):
        return "VAR"
    if re.fullmatch(r"#-?\d+", token):
        return "INT"
    if token.startswith("$"):
        return "STR"
    return token


def tree(tokens):
    """(token, [children]) from prefix tokens."""
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


def pieces(tokens):
    """The program's token kinds, edges and twigs, and the whole program by kinds."""
    out = {"tokens": set(), "edges": set(), "twigs": set()}

    def walk(node):
        token, children = node
        out["tokens"].add(kind(token))
        for slot, child in enumerate(children):
            out["edges"].add((kind(token), slot, kind(child[0])))
        if children:
            out["twigs"].add((kind(token),) + tuple(kind(child[0]) for child in children))
        for child in children:
            walk(child)

    walk(tree(list(tokens)))
    out["whole"] = {tuple(kind(t) for t in tokens)}
    return out


def training_programs():
    programs = [t["tokens"] for t in teacher_task_items() if not teacher_held(t["id"])]
    programs += [t["tokens"] for t in teacher_task_items("variant_tasks") if not variant_held(t)]
    programs += [e["tokens"] for e in oneliner_items() if e["part"] == "train"]
    programs += [row["program"] for row in json.loads((ROOT / "outputs/seedlm/v2/real_dev.json").read_text())
                 if row["part"] == "train"]
    return programs


def evaluation_sets():
    return {"teacher_holdout": [t["tokens"] for t in teacher_task_items() if teacher_held(t["id"])],
            "variant_holdout": [t["tokens"] for t in teacher_task_items("variant_tasks") if variant_held(t)],
            "beyond": [e["tokens"] for e in oneliner_items() if e["part"] == "holdout"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/seedlm/fragments_holdout.json")
    args = parser.parse_args()
    known = {"tokens": set(), "edges": set(), "twigs": set(), "whole": set()}
    skipped = 0
    for program in training_programs():
        try:
            for name, found in pieces(program).items():
                known[name] |= found
        except (ValueError, IndexError, KeyError):
            skipped += 1
    report = {"training_programs_skipped": skipped, "known": {name: len(v) for name, v in known.items()}}
    for set_name, programs in evaluation_sets().items():
        entry = {"programs": 0, "all_known": {name: 0 for name in known}, "first_unknown": {}}
        for program in programs:
            try:
                found = pieces(program)
            except (ValueError, IndexError, KeyError):
                continue
            entry["programs"] += 1
            first = None
            for name in ("tokens", "edges", "twigs", "whole"):
                complete = found[name] <= known[name]
                entry["all_known"][name] += complete
                if not complete and first is None:
                    first = name
            # The coarsest level at which the program is new: a new token, else a new edge, else ...
            label = first or "nothing (seen whole)"
            entry["first_unknown"][label] = entry["first_unknown"].get(label, 0) + 1
        report[set_name] = entry
    (ROOT / args.output).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
