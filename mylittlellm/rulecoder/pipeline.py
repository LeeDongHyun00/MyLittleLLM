"""One student run per task under a named condition, with an optional verify-and-repair loop."""
import copy
import math
import time

from .extract import extract_code, keep_definitions
from .limits import Timeout, time_limit
from .llm import chat
from .sandbox import run_public_asserts, syntax_error
from .tasks import parse_prompt

# EvalPlus instruction format, so the no-rule baseline is comparable with published MBPP+ numbers.
INSTRUCTION = ("Please provide a self-contained Python script that solves the following problem "
               "in a markdown code block:\n```\n{prompt}\n```\n")
# Default system prompt of the Qwen2.5 chat template, restated so a rule block can be appended.
DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
REPAIR = "Fix the problem and reply with the complete corrected Python script in one markdown code block."
LLM_FIELDS = ("finish_reason", "prompt_tokens", "completion_tokens", "prompt_tokens_processed",
              "prompt_ms", "predicted_ms", "elapsed_seconds")


def system_prompt(rule_block=None):
    return DEFAULT_SYSTEM if not rule_block else DEFAULT_SYSTEM + "\n\n" + rule_block


def public_check(code, asserts):
    """Gate a candidate on what a user could verify: it parses and the visible example holds."""
    error = syntax_error(code)
    if error:
        return {"ok": False, "kind": "syntax", "message": f"The code does not parse: {error}"}
    if not asserts:
        return {"ok": True, "kind": "no_example", "message": ""}
    outcome = run_public_asserts(code, asserts)
    status = outcome["status"]
    if status == "pass":
        return {"ok": True, "kind": "pass", "message": ""}
    if status == "fail" and "got" in outcome:
        message = (f"This example check failed:\n{outcome['assert']}\n"
                   f"`{outcome['left']}` evaluated to: {outcome['got']}\nExpected: {outcome['expected']}")
    elif status == "fail":
        message = f"This example check failed:\n{outcome['assert']}"
    elif status == "timeout":
        message = f"The example did not finish ({outcome['detail']}); the code may loop forever."
    elif outcome.get("stage") == "assert":
        message = f"Running the example raised an error:\n{outcome['assert']}\n{outcome['detail']}"
    else:
        message = f"The script raised an error when it was loaded: {outcome['detail']}"
    return {"ok": False, "kind": status, "message": message}


def feedback(check, hints):
    parts = [check["message"]] if check["message"] else []
    if hints:
        parts.append("Review these points:\n" + "\n".join(f"- {hint}" for hint in hints))
    return "\n\n".join(parts + [REPAIR])


def choose(attempts):
    """Prefer candidates that pass the visible example; among them the fewest checker flags, earliest first.

    An unresolved flag means the repair did not address the concern, so the smaller change wins.
    """
    passing = [a for a in attempts if a["check"]["ok"]]
    if not passing:
        return attempts[-1]
    return min(passing, key=lambda a: (len(a["hints"]), a["round"]))


ZERO_COST = {"finish_reason": "backinfer", "prompt_tokens": 0, "completion_tokens": 0,
             "prompt_tokens_processed": 0, "prompt_ms": 0, "predicted_ms": 0, "elapsed_seconds": 0}


def same_output(result, expected):
    if isinstance(result, float) or isinstance(expected, float):
        return (isinstance(result, (int, float)) and isinstance(expected, (int, float))
                and math.isclose(result, expected, rel_tol=1e-6, abs_tol=1e-6))
    return result == expected


def agrees(code, entry_point, examples, seconds=1.0):
    """Cheap filter for engine-written code: does it reproduce every public example?

    Runs in this process, so it is only for code the rule engine composed from its own
    templates. Model answers always go through the sandbox.
    """
    scope = {}
    try:
        with time_limit(seconds):
            exec(code, scope)
            return all(same_output(scope[entry_point](*copy.deepcopy(list(args))), expected)
                       for args, expected in examples)
    except (Exception, Timeout):
        return False


def back_infer(task, asserts, round_index, accept=None, examples=(), tries=5):
    """Rule-only fallback: search the operator library for a program that reproduces the example.

    With one public example the best-ranked match is the answer. When more is public, pass
    `accept` (the authoritative check of a candidate) and `examples` ((arguments, expected)
    pairs): the search still starts from the first example, and its matches are taken best
    first until one agrees with every example and is accepted.
    """
    from .backinfer import synthesize
    description, _ = parse_prompt(task["prompt"])
    found = synthesize(description, asserts, task["entry_point"], keep=10 ** 6 if accept else 1)
    if not found:
        return None
    if accept:
        agreeing = [(rank, m) for rank, m in enumerate(found["alternatives"])
                    if agrees(m["code"], task["entry_point"], examples)]
        chosen = next(((rank, m) for rank, m in agreeing[:tries] if accept(m["code"])), None)
        if chosen is None:
            return None
        rank, match = chosen
        found = dict(found, code=match["code"], expr=match["expr"], ops=match["ops"], rank=rank,
                     support=round(match["support"], 2), knowledge=match["knowledge"], confident=True)
        check = {"ok": True, "kind": "public_tests", "message": ""}
    else:
        check = public_check(found["code"], asserts)
    inference = {k: found[k] for k in ("expr", "ops", "support", "matches", "confident", "knowledge")}
    inference["rank"] = found.get("rank", 0)
    return {"round": round_index, "code": found["code"], "raw_code": found["code"], "autofixed": False,
            "source": "backinfer", "llm": dict(ZERO_COST), "check": check, "hints": [], "inference": inference}


def seed_infer(task, asserts, round_index, proposer):
    """Learned fallback: a program the from-scratch model proposes for the visible example.

    `proposer(args, expected, named_operators)` returns programs that already reproduce the
    example, best first. Taking it as a callable keeps this module free of the network code.
    """
    from . import backinfer as bi
    from ..seedlm import synth
    description, _ = parse_prompt(task["prompt"])
    example = bi.parse_example(asserts, task["entry_point"])
    # The proposer was trained on exact-match examples whose result is not a boolean.
    if example is None or example["mode"] != "eq" or not example["args"] \
            or example["expected"] is None or isinstance(example["expected"], bool):
        return None
    named = [name for name, op in bi.OPERATORS.items() if op.named_in(description.lower())]
    started = time.monotonic()
    proposals = proposer(example["args"], example["expected"], named)
    seconds = round(time.monotonic() - started, 3)
    if not proposals:
        return None
    best = proposals[0]
    expr, helpers, knowledge = synth.to_expression(best["program"])
    code = bi.to_code(task["entry_point"], len(example["args"]), {"expr": expr, "helpers": helpers})
    return {"round": round_index, "code": code, "raw_code": code, "autofixed": False, "source": "seedlm",
            "llm": dict(ZERO_COST), "check": public_check(code, asserts), "hints": [],
            "inference": {"expr": expr, "program": best["program"], "log_prob": best["log_prob"],
                          "knowledge": knowledge, "seconds": seconds}}


def solve(task, *, rule_block=None, verify=False, checkers=None, autofix=None, infer=False, max_repairs=2,
          max_tokens=768, client=chat):
    """Return the final code plus every attempt, so cost and repair behaviour can be audited."""
    _, asserts = parse_prompt(task["prompt"])
    messages = [{"role": "system", "content": system_prompt(rule_block)},
                {"role": "user", "content": INSTRUCTION.format(prompt=task["prompt"].strip())}]
    attempts = []
    for round_index in range(max_repairs + 1 if verify else 1):
        reply = client(messages, max_tokens=max_tokens)
        raw_code = keep_definitions(extract_code(reply["text"], task["entry_point"]))
        # Deterministic repairs run before any check, so they can save a model call.
        code = autofix(raw_code, task["entry_point"]) if autofix else raw_code
        # raw_code is what a run without codemods and without the repair loop would have returned.
        attempt = {"round": round_index, "code": code, "raw_code": raw_code, "autofixed": code != raw_code,
                   "llm": {k: reply[k] for k in LLM_FIELDS}}
        attempts.append(attempt)
        if not verify:
            return {"final_code": code, "attempts": attempts}
        attempt["check"] = public_check(code, asserts)
        attempt["hints"] = checkers(task, code, asserts, attempt["check"]) if checkers else []
        if attempt["check"]["ok"] and not attempt["hints"]:
            break
        if infer and round_index == 0:
            # The answer already fails the example it was shown, so a program found by search
            # can only replace a wrong answer. It costs no model call, so it goes before repair.
            inferred = back_infer(task, asserts, len(attempts))
            if inferred and inferred["check"]["ok"]:
                attempts.append(inferred)
                break
        if round_index == max_repairs:
            break
        messages += [{"role": "assistant", "content": reply["text"]},
                     {"role": "user", "content": feedback(attempt["check"], attempt["hints"])}]
    return {"final_code": choose(attempts)["code"], "attempts": attempts}


def solve_branching(task, *, rule_block=None, max_tokens=768, client=chat, proposer=None):
    """Branch and select: rules are alternatives chosen by execution, never obligations.

    1. Answer with the fired rules in the prompt (or plainly when none fired).
    2. If that answer fails the visible example and rules were used, answer again without them.
    3. If every model answer fails the example, search the operator library (no model call).
    4. If the search finds nothing and a proposer is given, take its best verified program.
    The first candidate that reproduces the visible example wins; otherwise the first answer
    is returned unchanged, so a branch can replace only an answer already known to be wrong.
    """
    _, asserts = parse_prompt(task["prompt"])
    user = {"role": "user", "content": INSTRUCTION.format(prompt=task["prompt"].strip())}
    attempts = []
    for branch, block in (("rules", rule_block), ("plain", None)):
        if branch == "plain" and (not rule_block or attempts[-1]["check"]["ok"]):
            break
        reply = client([{"role": "system", "content": system_prompt(block)}, user], max_tokens=max_tokens)
        code = keep_definitions(extract_code(reply["text"], task["entry_point"]))
        attempts.append({"round": len(attempts), "code": code, "raw_code": code, "autofixed": False,
                         "source": branch if rule_block else "plain", "llm": {k: reply[k] for k in LLM_FIELDS},
                         "check": public_check(code, asserts), "hints": []})
    if not any(a["check"]["ok"] for a in attempts):
        inferred = back_infer(task, asserts, len(attempts))
        if inferred and inferred["check"]["ok"]:
            attempts.append(inferred)
    if proposer and not any(a["check"]["ok"] for a in attempts):
        proposed = seed_infer(task, asserts, len(attempts), proposer)
        if proposed and proposed["check"]["ok"]:
            attempts.append(proposed)
    chosen = next((a for a in attempts if a["check"]["ok"]), attempts[0])
    return {"final_code": chosen["code"], "attempts": attempts, "chosen": chosen["source"]}
