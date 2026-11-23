"""Input contracts inferred from a task's example calls, so that probe inputs stay inside the task's domain.

The strict judge (SeedLM-36) compares a program with the task's reference solution on probe inputs made by
mutating the visible example. Reading its conflicts showed that about half were artifacts: the mutated input
had left the domain the task assumes (a sorted list that is no longer sorted, a binary array holding a 4, a
list of primes holding a 1), and a correct program then differed from the reference where neither was meant
to be defined. EvalPlus met the same problem and wrote input contracts by hand. Here a contract is inferred:
it records every property below that holds for an argument in all of the task's example calls, and a probe is
kept only if it keeps them all.

Properties are deliberately few and cheap to check. A property that holds by coincidence in three calls
(all numbers positive, say) narrows the probes more than the task does; that makes the judge more lenient,
never wrongly harsher, which is the safer direction for a judge of correctness.
"""

SEQ = (list, tuple)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numbers(values):
    return bool(values) and all(_is_number(v) for v in values)


def _value_properties(values):
    """Properties shared by one argument position across all example calls."""
    props = {"type": type(values[0]).__name__} if len({type(v) for v in values}) == 1 else {}
    if _numbers(values):
        if all(v >= 0 for v in values):
            props["nonnegative"] = True
        if all(v > 0 for v in values):
            props["positive"] = True
    elif all(isinstance(v, str) for v in values):
        text = "".join(values)
        if text:
            for name, test in (("digits", str.isdigit), ("alpha", str.isalpha)):
                if all(test(v) for v in values if v):
                    props[name] = True
            if any(c.isalpha() for c in text):
                if text == text.lower():
                    props["lower"] = True
                elif text == text.upper():
                    props["upper"] = True
            if " " not in text:
                props["no_space"] = True
            if all(len(v) == 1 for v in values):
                props["single_char"] = True     # "a character" argument: a probe of length 2 is another task
            if set(text) <= {"0", "1"}:
                props["binary_text"] = True
    elif all(isinstance(v, SEQ) for v in values):
        items = [x for v in values for x in v]
        longest = max((len(v) for v in values), default=0)
        if items and len({type(x) for x in items}) == 1:
            props["item_type"] = type(items[0]).__name__
        if _numbers(items):
            if all(x >= 0 for x in items):
                props["items_nonnegative"] = True
            if len(items) >= 3 and set(items) <= {0, 1}:
                props["items_binary"] = True
            elif len(items) >= 6 and len(set(items)) <= 3:
                # "an array of only 0s, 1s and 2s": a few values repeated many times are the domain itself
                props["items_in"] = sorted(set(items))
            if longest >= 3 and all(list(v) == sorted(v) for v in values):
                props["ascending"] = True
            elif longest >= 3 and all(list(v) == sorted(v, reverse=True) for v in values):
                props["descending"] = True
        if longest >= 3 and all(_hashable(v) and len(set(v)) == len(v) for v in values):
            props["distinct"] = True
        if all(v and all(isinstance(row, SEQ) for row in v) for v in values):
            if all(len({len(row) for row in v}) == 1 for v in values) and max(len(v) for v in values) >= 2:
                props["rectangular"] = True
                if all(len(v) == len(v[0]) for v in values):
                    props["square"] = True
    return props


def _hashable(sequence):
    try:
        set(sequence)
        return True
    except TypeError:
        return False


def infer(examples):
    """A contract from example argument lists (every assert of the task, the visible one included)."""
    examples = [list(e) for e in examples if e is not None]
    if not examples or len({len(e) for e in examples}) != 1:
        return {"positions": [], "relations": []}
    width = len(examples[0])
    positions = [_value_properties([e[i] for e in examples]) for i in range(width)]
    relations = []
    for i in range(width):
        for j in range(i + 1, width):
            if all(isinstance(e[i], SEQ + (str,)) and isinstance(e[j], SEQ + (str,)) and len(e[i]) == len(e[j])
                   for e in examples):
                relations.append(("same_length", i, j))
    return {"positions": positions, "relations": relations}


def _holds(props, value):
    kind = props.get("type")
    if kind and type(value).__name__ != kind:
        return False
    if props.get("nonnegative") and not (_is_number(value) and value >= 0):
        return False
    if props.get("positive") and not (_is_number(value) and value > 0):
        return False
    if isinstance(value, str):
        if props.get("digits") and value and not value.isdigit():
            return False
        if props.get("alpha") and value and not value.isalpha():
            return False
        if props.get("lower") and value != value.lower():
            return False
        if props.get("upper") and value != value.upper():
            return False
        if props.get("no_space") and " " in value:
            return False
        if props.get("binary_text") and not set(value) <= {"0", "1"}:
            return False
        if props.get("single_char") and len(value) != 1:
            return False
    if isinstance(value, SEQ):
        items = list(value)
        if props.get("item_type") and any(type(x).__name__ != props["item_type"] for x in items):
            return False
        if props.get("items_nonnegative") and not all(_is_number(x) and x >= 0 for x in items):
            return False
        if props.get("items_binary") and not set(items) <= {0, 1}:
            return False
        if props.get("items_in") and not set(items) <= set(props["items_in"]):
            return False
        if props.get("ascending") and items != sorted(items):
            return False
        if props.get("descending") and items != sorted(items, reverse=True):
            return False
        if props.get("distinct") and (not _hashable(items) or len(set(items)) != len(items)):
            return False
        if props.get("rectangular"):
            if not items or not all(isinstance(row, SEQ) for row in items) or len({len(r) for r in items}) != 1:
                return False
            if props.get("square") and len(items) != len(items[0]):
                return False
    return True


def satisfies(contract, args):
    """True if the argument list keeps every property and relation of the contract."""
    positions = contract.get("positions", [])
    if positions and len(args) != len(positions):
        return False
    if not all(_holds(props, value) for props, value in zip(positions, args)):
        return False
    for name, i, j in contract.get("relations", []):
        if name == "same_length":
            if not (hasattr(args[i], "__len__") and hasattr(args[j], "__len__") and len(args[i]) == len(args[j])):
                return False
    return True


def in_domain_probes(contract, candidates, count=24):
    """The first `count` candidate probes that satisfy the contract."""
    return [p for p in candidates if satisfies(contract, p)][:count]
