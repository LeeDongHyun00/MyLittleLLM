"""Rule bank utilities: triggers, rendering, HypoGeniC-style reward, RuleFit-style attribution."""
import json
import math
import random
import re

from .tasks import ROOT

BANK = ROOT / "data/rule_pilot/rule_bank.json"
HEADER = "Follow these rules when writing the solution:"

# Control condition, written before any student failure was inspected. It is the kind of
# best-practice list a person would write without data, and is about as long as the rule block.
GENERIC_ADVICE = [
    "Read the problem statement carefully and make sure you understand what is being asked before writing code.",
    "Write clean, readable and well-structured code with meaningful names.",
    "Make sure the function name and parameters match what the example expects.",
    "Think about edge cases such as empty inputs, single elements, zero and negative numbers.",
    "Prefer simple, correct logic over clever tricks; use built-in functions and the standard library where appropriate.",
    "Return the result from the function instead of printing it.",
    "Double-check the code against the given example before answering.",
    "Make sure the code is efficient enough for large inputs.",
    "Handle different input types correctly (lists, tuples, strings, numbers).",
    "Do not include unnecessary input/output code or extra explanations.",
]


def load_rules(path=BANK, ids=None):
    """Active guidance rules in bank order, or exactly the requested ids."""
    rules = json.loads(path.read_text())["rules"]
    if ids is None:
        return [r for r in rules if r.get("status", "active") == "active"]
    index = {r["id"]: r for r in rules}
    return [index[i] for i in ids]


def applies(rule, task):
    """Deterministic trigger: a rule fires only for tasks whose prompt matches its pattern.

    This is the classic production-rule form (IF condition THEN advice); keeping the condition
    executable means rule selection costs no model call and never fills the prompt with
    rules that are irrelevant to the task at hand.
    """
    trigger = rule.get("trigger") or {"type": "always"}
    if trigger["type"] == "always":
        return True
    if trigger["type"] == "prompt_regex":
        return re.search(trigger["pattern"], task["prompt"], re.I | re.S) is not None
    raise ValueError(f"Unknown trigger type: {trigger['type']}")


def render(texts, header=HEADER):
    return "\n".join([header] + [f"{i}. {text}" for i, text in enumerate(texts, 1)])


def ucb_reward(successes, trials, step, alpha=0.5):
    """HypoGeniC Eq. 1: accuracy of a rule on the examples it was tried on plus an exploration bonus."""
    if trials == 0:
        return math.inf
    return successes / trials + alpha * math.sqrt(math.log(max(step, 1)) / trials)


def random_subset(rule_ids, key, probability=0.5):
    """Seeded inclusion draw so every (task, draw) pair gets a reproducible rule subset."""
    rng = random.Random(key)
    return [rule_id for rule_id in rule_ids if rng.random() < probability]


def lasso(rows, targets, penalty, sweeps=1000, tolerance=1e-10):
    """Coordinate-descent lasso with an unpenalized intercept (squared loss, RuleFit Eq. 10).

    Features stay unnormalized on purpose: like RuleFit, this penalizes rules whose inclusion
    indicator has very small or very large support more heavily.
    """
    n, p = len(rows), len(rows[0])
    coef, intercept = [0.0] * p, sum(targets) / n
    resid = [y - intercept for y in targets]
    scale = [sum(row[j] * row[j] for row in rows) / n for j in range(p)]
    for _ in range(sweeps):
        shift = sum(resid) / n
        intercept += shift
        resid = [e - shift for e in resid]
        change = abs(shift)
        for j in range(p):
            if scale[j] == 0:
                continue
            rho = sum(rows[i][j] * resid[i] for i in range(n)) / n + scale[j] * coef[j]
            new = math.copysign(max(abs(rho) - penalty, 0.0), rho) / scale[j]
            if new != coef[j]:
                step = new - coef[j]
                resid = [resid[i] - step * rows[i][j] for i in range(n)]
                change, coef[j] = max(change, abs(step)), new
        if change < tolerance:
            break
    return intercept, coef


def cross_validated_penalty(rows, targets, penalties, folds=5, seed=0):
    """Pick the penalty with the lowest held-out squared error (ties go to the sparser model)."""
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    errors = []
    for penalty in penalties:
        total = 0.0
        for fold in range(folds):
            held = set(order[fold::folds])
            train = [i for i in order if i not in held]
            intercept, coef = lasso([rows[i] for i in train], [targets[i] for i in train], penalty)
            total += sum((targets[i] - intercept - sum(c * x for c, x in zip(coef, rows[i]))) ** 2 for i in held)
        errors.append(total / len(rows))
    best = min(errors)
    # Largest penalty within numerical noise of the best error: prefer fewer rules.
    return max(p for p, e in zip(penalties, errors) if e <= best + 1e-12), dict(zip(penalties, errors))


def rule_importance(coefficient, support):
    """RuleFit importance: |coefficient| times the standard deviation of the rule indicator."""
    return abs(coefficient) * math.sqrt(support * (1 - support))
