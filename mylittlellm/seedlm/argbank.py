"""Real-shaped arguments for every operator, so synthetic programs run on the inputs real tasks use.

Synthetic arguments come from a fixed menu of shapes (lists of small integers, short words, ...).
An operator that needs a month name, a list of dictionaries or a tuple of string pairs never runs on
them, so it never appears in training: 43 of the 780 operators did not occur once in 30,000
synthetic programs, and such specialised operators are what real tasks call. The bank stores, for
every operator, argument tuples it accepts, taken from the visible examples of development tasks
(the train split and the extra pool; never validation or test) and from mutations of them. Sampling
from the bank picks the operator first, uniformly, so every operator is seen as often as any other.
"""
import ast
import copy
import json
import random
import time

from ..rulecoder import backinfer as bi
from ..rulecoder.consensus import _keep_relations, mutate_value
from ..rulecoder.limits import Timeout, time_limit
from ..rulecoder.tasks import ROOT
from . import synth

BANK_PATH = ROOT / "outputs/seedlm/v3/argbank.json"
TASKS = ROOT / "outputs/rule_pilot/teacher/round3/all_tasks.json"
PER_OPERATOR = 16
SECONDS_PER_OPERATOR = 3.0


def example_args():
    """Argument tuples of the development tasks' visible examples (deduplicated)."""
    seen, out = set(), []
    for item in json.loads(TASKS.read_text()):
        example = bi.parse_example(item["visible_assert"], item["entry_point"])
        if not example or not example["args"] or len(repr(example["args"])) > synth.MAX_IN:
            continue
        key = repr(example["args"])
        if key not in seen:
            seen.add(key)
            out.append(list(example["args"]))
    return out


def _acceptable(value):
    if value is None or isinstance(value, (set, frozenset)):
        return False
    try:
        return len(repr(value)) <= synth.MAX_OUT
    except ValueError:       # integers too long to print
        return False


def _call(op, args, seconds=0.2):
    try:
        with time_limit(seconds):
            return op.fn(*copy.deepcopy(list(args)))
    except (Exception, Timeout):
        return None


def candidates(arity, tuples, singles):
    """Whole example tuples of the right length, in-order windows of longer ones, and for binary
    operators a single value with a small integer (the shape of "list, k")."""
    out = [list(t) for t in tuples if len(t) == arity]
    for t in tuples:
        for start in range(len(t) - arity + 1) if len(t) > arity else ():
            out.append(list(t[start:start + arity]))
    if arity == 1:
        out += [[v] for v in singles]
    if arity == 2:
        out += [[v, k] for v in singles for k in (1, 2, 3)]
    return out


def build(seed=0):
    """For each operator, up to PER_OPERATOR accepted argument tuples (repr strings), varied outputs first."""
    rng = random.Random(seed)
    tuples = example_args()
    singles = list({repr(a): a for t in tuples for a in t}.values())
    bank = {}
    for key, op in bi.OPERATORS.items():
        pool = candidates(op.arity, tuples, singles)
        rng.shuffle(pool)
        accepted, outputs = [], set()
        started = time.monotonic()
        for args in pool:
            if len(accepted) >= PER_OPERATOR or time.monotonic() - started > SECONDS_PER_OPERATOR:
                break
            value = _call(op, args)
            if not _acceptable(value):
                continue
            label = repr(value)
            # Prefer arguments that give a new output: one operator seen with one output teaches little.
            if label in outputs and len(outputs) < PER_OPERATOR // 2 and rng.random() < 0.7:
                continue
            outputs.add(label)
            accepted.append(repr(args))
        if accepted:
            bank[key] = accepted
    BANK_PATH.parent.mkdir(parents=True, exist_ok=True)
    BANK_PATH.write_text(json.dumps(bank))
    return bank


def load(path=BANK_PATH):
    if not path.exists():
        return {}
    return {key: [ast.literal_eval(a) for a in tuples] for key, tuples in json.loads(path.read_text()).items()}


def sample(rng, bank, operators, compose_rate=0.4, mutate_rate=0.5):
    """One example in data2.sample's shape, operator first: {'args', 'literals', 'expected', 'program', 'ops'}."""
    key = rng.choice(sorted(bank))
    op = bi.OPERATORS[key]
    args = copy.deepcopy(rng.choice(bank[key]))
    if rng.random() < mutate_rate:
        args = _keep_relations(args, [mutate_value(rng, copy.deepcopy(a)) for a in args])
    if len(repr(args)) > synth.MAX_IN:
        return None
    value = _call(op, args)
    if not _acceptable(value):
        return None
    program = [key] + [f"a{i}" for i in range(len(args))]
    if rng.random() < compose_rate:
        # An outer operation on the result, sometimes with one of the arguments as well.
        for _ in range(20):
            outer = rng.choice(operators)
            if outer.arity == 1:
                inputs, tail = [value], []
            elif outer.arity == 2:
                index = rng.randrange(len(args))
                inputs, tail = [value, args[index]], [f"a{index}"]
            else:
                continue
            result = _call(outer, inputs)
            if _acceptable(result) and repr(result) != repr(value):
                value, program = result, [f"{outer.name}/{outer.arity}"] + program + tail
                break
    if len(program) > synth.MAX_PROGRAM:
        return None
    return {"args": args, "literals": [], "expected": value, "program": program,
            "ops": sorted({t for t in program if "/" in t})}
