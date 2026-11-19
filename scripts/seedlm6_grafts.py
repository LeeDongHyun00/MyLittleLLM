"""Grafted programs: training programs with one subtree replaced by a subtree of another training program,
so that the junction is a parent -> child link no training program has.

SeedLM-5's diagnostics (scripts/seedlm5_fragments.py) found that about half of the programs the coder
never saw hold such a new link, and that the programs the coder misses are mostly those. A graft
keeps both pieces as the teachers wrote them and only joins them in a new way. It is kept only if
it runs on all three of the host task's inputs, its output changes with the input, it differs from
the host's own output, and it does not just return an argument. Grafts have no description: they
teach which compositions exist and what they compute, from the example and the coder's value notes.

Hosts and donors come from the training parts of the teacher tasks and SeedLM-5's variants only.

  python scripts/seedlm6_grafts.py --count 2000 [--seed 0]     prints counts and a few samples
"""
import argparse
import ast
import json
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylittlellm.rulecoder import backinfer as bi
from mylittlellm.rulecoder.limits import Timeout, time_limit
from mylittlellm.seedlm import coder3, post, pyx
from seedlm3_train import teacher_held, teacher_task_items
from seedlm4_train import variant_held
from seedlm5_fragments import kind, pieces

SECONDS = 0.2
MAX_REPR = 120
VARIABLE = re.compile(r"v(\d)")


def nodes(tokens):
    """(start, end, depth, parent position, slot) for every node of a prefix program."""
    out, position = [], 0

    def read(depth, parent, slot):
        nonlocal position
        start, token = position, tokens[position]
        position += 1
        index = len(out)
        out.append(None)
        for child_slot, child_depth in enumerate(coder3.child_depths(token, depth)):
            read(child_depth, start, child_slot)
        out[index] = (start, position, depth, parent, slot)

    read(0, None, None)
    return out


def free_variables(span, depth):
    """Variable indices a subtree uses but does not bind (those below the depth it sits at)."""
    return {int(m.group(1)) for token in span for m in [VARIABLE.fullmatch(token)] if m and int(m.group(1)) < depth}


def uses_its_variables(tokens):
    """Every comprehension reads a variable it binds (in its element, a condition or a later iterable),
    every comprehension condition reads one, and every key function reads its argument:
    `[a0 for v0 in a0]` or `if 1` compute nothing per element, while `sum(1 for v in a0 if v > 0)`
    counts."""
    position, ok = 0, True

    def read(depth):
        """Reads one subtree; returns the set of variable indices it mentions."""
        nonlocal position, ok
        token = tokens[position]
        position += 1
        if m := VARIABLE.fullmatch(token):
            return {int(m.group(1))}
        depths = coder3.child_depths(token, depth)
        used = [read(d) for d in depths]
        bound = coder3.bindings(token)
        if bound and token.startswith("@Comp:"):
            kind_, shapes = post.comprehension_shape(token)
            elements = 2 if kind_ == "dict" else 1
            mine = set(range(depth, depth + bound))
            if not any(u & mine for u in used[:elements] + used[elements + 1:]):
                ok = False                  # the first iterable is outside the binder's scope
            if kind_ != "gen" and not all(u & mine for u in used[:elements]):
                ok = False                  # a list of the same value per element; a generator may count
            rest, binder, i = used[elements:], depth, 0
            for count, conditions in shapes:
                i += 1                      # the iterable
                binder += count
                for _ in range(conditions):
                    if not rest[i] & set(range(depth, binder)):
                        ok = False
                    i += 1
        elif bound and depth not in used[0]:
            ok = False                      # a key function that ignores its argument
        return set().union(*used) if used else set()

    read(0)
    return ok


def repeats_operand(tokens):
    """An operation applied to the same computed operand twice (`a0 >> a0`, `v0.endswith(v0)`); sums,
    products, slices (`a0[::-1]`) and comprehensions over the same range twice are fine."""
    for start, end, _, _, _ in nodes(tokens):
        token = tokens[start]
        if pyx.arity(token) < 2 or token in ("@Add/2", "@Mult/2") or token.startswith(("@Comp:", "@Slice")):
            continue
        children, position = [], start + 1
        for _ in range(pyx.arity(token)):
            child_end = next(e for s_, e, _, _, _ in nodes(tokens) if s_ == position)
            children.append(tuple(tokens[position:child_end]))
            position = child_end
        computed = [c for c in children if computes(c)]
        if len(set(computed)) < len(computed):
            return True
    return False


def leaves_fit(span, n_args, n_literals):
    for token in span:
        if re.fullmatch(r"a\d", token) and int(token[1:]) >= n_args:
            return False
        if re.fullmatch(r"c\d", token) and int(token[1:]) >= n_literals:
            return False
    return True


def computes(span):
    """A subtree that reads an argument or a variable, so it brings a computation, not a constant."""
    return any(re.fullmatch(r"[av]\d", token) for token in span)


def same_kind(old, new, case, literals):
    """For closed subtrees: both evaluate, on the host's first case, to values of the same type."""
    values = run(old, [case], literals, None), run(new, [case], literals, None)
    return all(v is not None for v in values) and type(values[0][0]) is type(values[1][0])


def run(tokens, cases, literals, output):
    values = []
    for args in cases:
        try:
            with time_limit(SECONDS):
                value = pyx.execute(list(tokens), args, literals, output)
            text = repr(value)
        except (Exception, Timeout):
            return None
        if len(text) > MAX_REPR or isinstance(value, (set, frozenset)) or value is None:
            return None
        values.append(value)
    return values


def programs():
    """Training-part teacher tasks and variants as hosts/donors: tokens, input cases, literals, outputs."""
    out = []
    items = [t for t in teacher_task_items() if not teacher_held(t["id"])]
    items += [t for t in teacher_task_items("variant_tasks") if not variant_held(t)]
    for task in items:
        try:
            cases = [list(ast.literal_eval(text)) for text in task["inputs"]]
            outputs = [ast.literal_eval(text) for text in task["outputs"]]
        except (ValueError, SyntaxError):
            continue
        if len(cases) < 3 or any(len(c) != len(cases[0]) for c in cases):
            continue
        if any(token in ("o", "o!") for token in task["tokens"]):
            continue        # "o" names the row's own output string, which a graft changes
        literals = bi.description_literals(task["description"])
        out.append({"id": task["id"], "tokens": list(task["tokens"]), "cases": cases, "outputs": outputs,
                    "literals": literals})
    return out


def known_edges(items):
    edges = set()
    for item in items:
        edges |= pieces(item["tokens"])["edges"]
    return edges


def grafts(count, seed=0, items=None, tries_per_graft=40):
    """Up to `count` distinct grafts, each with its host's cases and outputs on them."""
    rng = random.Random(seed)
    items = items if items is not None else programs()
    edges = known_edges(items)
    spans = [(item, nodes(item["tokens"])) for item in items]
    donors = [(item, item["tokens"][s:e], d) for item, ns in spans for s, e, d, _, _ in ns]
    found, seen = [], set()
    attempts = 0
    while len(found) < count and attempts < count * tries_per_graft:
        attempts += 1
        host, host_nodes = rng.choice(spans)
        start, end, depth, parent, slot = rng.choice(host_nodes)
        if parent is None:
            continue
        donor, span, donor_depth = rng.choice(donors)
        if donor is host or span == host["tokens"][start:end]:
            continue
        free = free_variables(span, donor_depth)
        if free and donor_depth != depth:
            continue        # its free variables would point at other binders
        if not free and donor_depth != depth:
            span = [f"v{int(m.group(1)) - donor_depth + depth}" if (m := VARIABLE.fullmatch(t)) else t for t in span]
        if any(int(m.group(1)) > 3 for t in span if (m := VARIABLE.fullmatch(t))):
            continue
        if not leaves_fit(span, len(host["cases"][0]), len(host["literals"])) or not computes(span):
            continue
        old = host["tokens"][start:end]
        if not free and not free_variables(old, depth) and not same_kind(old, span, host["cases"][0],
                                                                          host["literals"]):
            continue        # a closed subtree is replaced only by one of the same type
        parent_token = host["tokens"][parent]
        junction = (kind(parent_token), slot, kind(span[0]))
        if junction in edges:
            continue
        tokens = host["tokens"][:start] + list(span) + host["tokens"][end:]
        key = tuple(tokens)
        if key in seen:
            continue
        seen.add(key)
        try:
            postfix = post.to_postfix(tokens)
        except (ValueError, IndexError):
            continue
        if not uses_its_variables(tokens) or repeats_operand(tokens):
            continue
        if len(postfix) > post.MAX_POST:
            continue
        values = run(tokens, host["cases"], host["literals"], None)
        if values is None:
            continue
        texts = [repr(v) for v in values]
        if len(set(texts)) < 2 or texts == [repr(v) for v in host["outputs"]]:
            continue
        if all(any(repr(arg) == text for arg in case) for case, text in zip(host["cases"], texts)):
            continue
        found.append({"host": host["id"], "donor": donor["id"], "tokens": tokens, "cases": host["cases"],
                      "outputs": values, "literals": host["literals"], "junction": list(junction)})
    return found, attempts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    items = programs()
    found, attempts = grafts(args.count, args.seed, items)
    junctions = {tuple(g["junction"]) for g in found}
    print(json.dumps({"hosts_and_donors": len(items), "grafts": len(found), "attempts": attempts,
                      "distinct_new_edges": len(junctions)}))
    rng = random.Random(1)
    for g in rng.sample(found, min(8, len(found))):
        n_args = len(g["cases"][0])
        print(pyx.to_code("f", n_args, g["tokens"], g["literals"]).splitlines()[-1].strip(),
              "|", repr(g["cases"][0])[:60], "->", repr(g["outputs"][0])[:40])


if __name__ == "__main__":
    main()
