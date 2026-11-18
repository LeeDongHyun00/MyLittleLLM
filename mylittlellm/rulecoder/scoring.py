"""Hidden-test scoring with the EvalPlus primitives (the oracle behind the public MBPP+ leaderboard).

Requires the scorer environment from requirements-rule-pilot.txt. The dataset file is the
pinned copy from config/rule_pilot.lock.json, not whatever EvalPlus would download.
"""
import hashlib
import os
import pickle
import sys
import types

from .sandbox import run_call
from .tasks import DATASET, ROOT

CACHE = ROOT / "outputs/rule_pilot/groundtruth"
PASS = "pass"


def _evalplus():
    # evalplus.data imports `datasets` only for EvalPerf; a stub keeps the scorer dependency-light.
    sys.modules.setdefault("datasets", types.SimpleNamespace(load_dataset=None))
    os.environ["MBPP_OVERRIDE_PATH"] = str(DATASET)  # read by evalplus at import time
    from evalplus.data.mbpp import get_mbpp_plus
    from evalplus.eval import untrusted_check
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    from evalplus.gen.util import trusted_exec
    return get_mbpp_plus, untrusted_check, MBPP_OUTPUT_NOT_NONE_TASKS, trusted_exec


def load_problems():
    """Tasks with inputs deserialized exactly as EvalPlus does (tuples, sets, ...)."""
    return _evalplus()[0]()


def groundtruth(problems):
    """Expected outputs from the canonical solutions; mirrors evalplus.evaluate.get_groundtruth."""
    _, _, not_none, trusted_exec = _evalplus()
    path = CACHE / (hashlib.sha256(DATASET.read_bytes()).hexdigest()[:16] + ".pkl")
    if path.exists():
        with path.open("rb") as stream:
            return pickle.load(stream)
    expected = {}
    for task_id, problem in problems.items():
        code = problem["prompt"] + problem["canonical_solution"]
        flag = problem["entry_point"] in not_none
        oracle = {}
        oracle["base"], oracle["base_time"] = trusted_exec(code, problem["base_input"], problem["entry_point"],
                                                           record_time=True, output_not_none=flag)
        oracle["plus"], oracle["plus_time"] = trusted_exec(code, problem["plus_input"], problem["entry_point"],
                                                           record_time=True, output_not_none=flag)
        expected[task_id] = oracle
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(expected, stream)
    return expected


def score(problem, solution, oracle):
    """Return base/plus status plus per-input pass flags; mirrors evalplus.evaluate.check_correctness."""
    untrusted_check = _evalplus()[1]
    result = {}
    for part in ("base", "plus"):
        status, details = untrusted_check("mbpp", solution, problem[part + "_input"], problem["entry_point"],
                                          expected=oracle[part], atol=problem["atol"],
                                          ref_time=oracle[part + "_time"], fast_check=False)
        result[part] = status
        result[part + "_details"] = [bool(x) for x in details]
    # MBPP+ counts a task as solved only when both the original and the extra tests pass.
    result["solved"] = result["base"] == PASS and result["plus"] == PASS
    return result


def base_passes(problem, solution, oracle):
    """Whether the task's original tests pass, for the setting where those tests are public."""
    untrusted_check = _evalplus()[1]
    status, _ = untrusted_check("mbpp", solution, problem["base_input"], problem["entry_point"],
                                expected=oracle["base"], atol=problem["atol"], ref_time=oracle["base_time"],
                                fast_check=True)
    return status == PASS


def diagnose(problem, solution, oracle, result):
    """Describe the first failing hidden test. For teacher analysis on development splits only."""
    for part in ("base", "plus"):
        if result[part] == PASS:
            continue
        details, inputs = result[part + "_details"], problem[part + "_input"]
        # A shorter details list means execution stopped (timeout/crash) at that input.
        index = next((i for i, ok in enumerate(details) if not ok), len(details))
        if index >= len(inputs):
            return {"part": part, "kind": result[part], "note": "no failing input identified"}
        outcome = run_call(solution, problem["entry_point"], inputs[index])
        text = repr(oracle[part][index])
        return {"part": part, "kind": outcome["status"], "input": repr(inputs[index])[:400],
                "expected": text if len(text) <= 300 else text[:300] + "...(truncated)",
                "actual": outcome.get("repr") or outcome.get("detail"),
                "failed_inputs": sum(1 for ok in details if not ok), "total_inputs": len(inputs)}
    return None
