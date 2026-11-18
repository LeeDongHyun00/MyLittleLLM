"""One student run per task under a named condition, with an optional verify-and-repair loop."""
from .extract import extract_code, keep_definitions
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


def solve(task, *, rule_block=None, verify=False, checkers=None, max_repairs=2, max_tokens=768, client=chat):
    """Return the final code plus every attempt, so cost and repair behaviour can be audited."""
    _, asserts = parse_prompt(task["prompt"])
    messages = [{"role": "system", "content": system_prompt(rule_block)},
                {"role": "user", "content": INSTRUCTION.format(prompt=task["prompt"].strip())}]
    attempts = []
    for round_index in range(max_repairs + 1 if verify else 1):
        reply = client(messages, max_tokens=max_tokens)
        code = keep_definitions(extract_code(reply["text"], task["entry_point"]))
        attempt = {"round": round_index, "code": code, "llm": {k: reply[k] for k in LLM_FIELDS}}
        attempts.append(attempt)
        if not verify:
            return {"final_code": code, "attempts": attempts}
        attempt["check"] = public_check(code, asserts)
        attempt["hints"] = checkers(task, code, asserts, attempt["check"]) if checkers else []
        if (attempt["check"]["ok"] and not attempt["hints"]) or round_index == max_repairs:
            break
        messages += [{"role": "assistant", "content": reply["text"]},
                     {"role": "user", "content": feedback(attempt["check"], attempt["hints"])}]
    return {"final_code": choose(attempts)["code"], "attempts": attempts}
