"""Are the programs the coder cannot find new compositions of known pieces, or do they need new pieces?

Programs are prefix pyx trees. For each evaluation program (a teacher's program the coder never saw)
this script asks which of its pieces occur somewhere in the coder's training programs (every source,
training parts only):

  tokens   every token (operator, Python node, leaf kind) occurs in training
  edges    every parent -> child link (parent token, child slot, child token) occurs in training
  twigs    every depth-two fragment (a node with its children's tokens) occurs in training
  whole    the program itself occurs in training

Leaves are reduced to their kind (argument, literal slot, bound variable, small integer, string), so a
piece is about structure, not about which argument it reads. The coder also trained on synthetic
library programs; a sample of them (the same sampler, operator-first rows included) adds their pieces
to what counts as seen. If most programs have known tokens and
edges but unknown twigs or wholes, what is missing is composition (pieces never seen together);
if their tokens or edges are unseen, what is missing is knowledge (pieces never seen at all).

Sets: held-out teacher tasks, variants of held-out seeds, and the held-out development tasks the engine
fails that a teacher solved in one line (counts only: that set belongs to the held-out development
fifth).

With --model, the coder is also run (beam width 32) on the held-out teacher tasks and the beyond set,
and its first verified program is judged, by the level at which each teacher program is new: does
the coder find programs made of known pieces, and fail on programs with new ones?

  python scripts/seedlm5_fragments.py [--model s5-var]
"""
import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylittlellm.rulecoder.limits import cap_memory
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


def synthetic_programs(count, seed=1, bank_rate=0.3):
    """Synthetic library programs as the coder's training draws them (scripts/seedlm4_train.py)."""
    import random
    from mylittlellm.rulecoder import backinfer as bi
    from mylittlellm.seedlm import argbank, data2
    rng, bank, operators = random.Random(seed), argbank.load(), list(bi.OPERATORS.values())
    out = []
    while len(out) < count:
        example = argbank.sample(rng, bank, operators) if rng.random() < bank_rate else \
            data2.sample(rng, operators, 3, clean=True)
        if example is not None:
            out.append(example["program"])
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


def first_unknown(found, known):
    for name in ("tokens", "edges", "twigs", "whole"):
        if not found[name] <= known[name]:
            return name
    return "nothing (seen whole)"


def outgoing_edges(tokens):
    """The edges from each prefix position to its children (empty for a leaf)."""
    edges = [[] for _ in tokens]
    position = 0

    def read():
        nonlocal position
        index, token = position, tokens[position]
        position += 1
        for slot in range(pyx.arity(token)):
            edges[index].append((kind(token), slot, kind(tokens[position])))
            read()

    read()
    return edges


def postfix_order(tokens):
    """The prefix position of each postfix token (None for a scope opener), as post.to_postfix orders them."""
    from mylittlellm.seedlm import post
    order, position = [], 0

    def read():
        nonlocal position
        index, token = position, tokens[position]
        position += 1
        return token, index, [read() for _ in range(pyx.arity(token))]

    def emit(node):
        token, index, children = node
        if post.is_comprehension(token):
            kind_, shapes = post.comprehension_shape(token)
            elements = 2 if kind_ == "dict" else 1
            rest, i = children[elements:], 0
            for _, conditions in shapes:
                emit(rest[i])
                i += 1
                order.append(None)
                for _ in range(conditions):
                    emit(rest[i])
                    i += 1
            for element in children[:elements]:
                emit(element)
        elif post.is_key(token):
            for argument in children[2:]:
                emit(argument)
            emit(children[1])
            order.append(None)
            emit(children[0])
        else:
            for child in children:
                emit(child)
        order.append(index)

    emit(read())
    postfix = post.to_postfix(list(tokens))
    if [tokens[i] for i in order if i is not None] != [t for t in postfix if not t.startswith(post.FOR)]:
        raise ValueError("order mismatch")
    return order


def coder_by_level(model, known, width=32):
    """{set: {level: {"programs", "first_correct"}}} for the coder's first verified program, and the
    teacher program's per-token surprisal by kind of decision. The coder writes postfix: an operator
    comes after its operands, so writing it decides its links to its children (its outgoing edges).
    Its token is a known composition if all those links occur in training, a new one otherwise."""
    import torch
    from mylittlellm.seedlm import coder4, language, post
    from seedlm3_discovery import beyond_cases, teacher_cases
    from seedlm3_train import named_in
    torch.set_num_threads(4)
    cap_memory()      # a program's runaway allocation fails instead of killing the run
    net, vocabulary = coder4.load(ROOT / "outputs/seedlm" / f"{model}.pt")
    grammar = coder4.Grammar(vocabulary)
    out, surprisal = {}, {}
    for set_name, tasks in (("teacher_holdout", teacher_cases()), ("beyond", beyond_cases())):
        table = {}
        split = {"leaf": [0, 0.0, 0], "known_edges": [0, 0.0, 0], "new_edge": [0, 0.0, 0]}  # tokens, nats, first
        for task in tasks:
            try:
                level = first_unknown(pieces(task["gold"]), known)
            except (ValueError, IndexError, KeyError):
                continue
            prefix = vocabulary.condition(language.words(task["description"]), task["literals"], task["args"],
                                          task["expected"], named_in(task["description"]))
            beam = coder4.beam_search(net, vocabulary, grammar, prefix, task["args"], task["literals"],
                                      task["expected"], task["mode"], width)
            verified = [post.to_prefix(p) for p, _ in beam
                        if coder4.reproduces(p, task["args"], task["literals"], task["expected"], task["mode"])]
            row = table.setdefault(level, {"programs": 0, "first_correct": 0})
            row["programs"] += 1
            row["first_correct"] += bool(verified) and task["judge"](verified[0])
            try:
                gold = list(task["gold"])
                order, edges = postfix_order(gold), outgoing_edges(gold)
                tokens, side = coder4.encode(vocabulary, prefix, post.annotated(
                    post.to_postfix(gold), task["args"], task["literals"], task["expected"], task["mode"]))
            except (KeyError, ValueError):
                continue
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                logits = net(torch.tensor([tokens]), notes=torch.tensor([side]))[0]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            for j, index in enumerate(order):
                if index is None:
                    continue
                at = len(prefix) + j
                links = edges[index]
                entry = split["leaf" if not links else "known_edges" if all(e in known["edges"] for e in links)
                              else "new_edge"]
                entry[0] += 1
                entry[1] += -float(log_probs[at - 1, tokens[at]])
                entry[2] += int(log_probs[at - 1].argmax()) == tokens[at]
        out[set_name] = table
        surprisal[set_name] = {name: {"tokens": n, "mean_nats": round(total / n, 2) if n else None,
                                      "ranked_first": round(first / n, 3) if n else None}
                               for name, (n, total, first) in split.items()}
    return out, surprisal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--synthetic", type=int, default=100000, help="synthetic programs counted as seen")
    parser.add_argument("--output", default="reports/seedlm/fragments_holdout.json")
    args = parser.parse_args()
    known = {"tokens": set(), "edges": set(), "twigs": set(), "whole": set()}
    skipped = 0
    for program in training_programs() + synthetic_programs(args.synthetic):
        try:
            for name, found in pieces(program).items():
                known[name] |= found
        except (ValueError, IndexError, KeyError):
            skipped += 1
    report = {"synthetic_programs": args.synthetic, "training_programs_skipped": skipped,
              "known": {name: len(v) for name, v in known.items()}}
    for set_name, programs in evaluation_sets().items():
        entry = {"programs": 0, "all_known": {name: 0 for name in known}, "first_unknown": {}}
        for program in programs:
            try:
                found = pieces(program)
            except (ValueError, IndexError, KeyError):
                continue
            entry["programs"] += 1
            for name in ("tokens", "edges", "twigs", "whole"):
                entry["all_known"][name] += found[name] <= known[name]
            # The coarsest level at which the program is new: a new token, else a new edge, else ...
            label = first_unknown(found, known)
            entry["first_unknown"][label] = entry["first_unknown"].get(label, 0) + 1
        report[set_name] = entry
    if args.model:
        levels, surprisal = coder_by_level(args.model, known)
        report["coder"] = {"model": args.model, "width": 32, "by_first_unknown": levels,
                           "teacher_program_tokens": surprisal}
    (ROOT / args.output).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
