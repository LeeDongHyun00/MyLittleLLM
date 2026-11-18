"""Draw extra model samples only where no independent source corroborates the pipeline's answer.

Corroboration-gated compute: a task gets k samples if its pipeline answer
  - fails the visible example, or
  - came from the rule engine (whose coincidental matches are often wrong), or
  - is a model answer that no rule-engine program agrees with on probe inputs.
Every other answer is kept and costs nothing more. Samples use the plain prompt at a fixed
temperature with fixed seeds; the run can be stopped and resumed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mylittlellm.rulecoder import backinfer as bi, consensus as cs
from mylittlellm.rulecoder.extract import extract_code, keep_definitions
from mylittlellm.rulecoder.llm import LLM_FIELDS_SAFE, chat, server_model
from mylittlellm.rulecoder.pipeline import DEFAULT_SYSTEM, INSTRUCTION, public_check
from mylittlellm.rulecoder.tasks import ROOT, parse_prompt, read_tasks

OUT = ROOT / "outputs/rule_pilot"
RUNS = {"train": "s15-branch-dev-r3", "val": "s15-branch-dev-r3", "test": "s15-branch-test-r3"}


def read_jsonl(path):
    if not Path(path).exists():
        return []
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def engine_programs(task, asserts, keep):
    description, _ = parse_prompt(task["prompt"])
    found = bi.synthesize(description, asserts, task["entry_point"], weights={}, keep=keep)
    if not found:
        return []
    return found.get("alternatives") or [{"code": found["code"]}]


def target_reason(row, task, keep, probes, minimum):
    """Why this task needs more samples, or None when its answer is corroborated."""
    if not row["visible_ok"]:
        return "fails_example"
    if row["chosen"] == "backinfer":
        return "engine_answer"
    _, asserts = parse_prompt(task["prompt"])
    example = bi.parse_example(asserts, task["entry_point"])
    if example is None or not example["args"]:
        return "no_probes"
    inputs = cs.probe_inputs(example["args"], probes)
    model = cs.run_model_code(row["final_code"], task["entry_point"], inputs)
    for program in engine_programs(task, asserts, keep):
        if cs.compare(model, cs.run_engine_code(program["code"], task["entry_point"], inputs), minimum) == "agree":
            return None
    return "uncorroborated"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--splits", nargs="+", default=["val", "test", "train"])
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--keep", type=int, default=10)
    parser.add_argument("--probes", type=int, default=24)
    parser.add_argument("--minimum", type=int, default=3)
    args = parser.parse_args()
    tasks = read_tasks()
    model = server_model()["id"]
    folder = OUT / "samples"
    folder.mkdir(parents=True, exist_ok=True)
    targets_path, samples_path = folder / f"{args.name}_targets.json", folder / f"{args.name}.jsonl"
    targets = json.loads(targets_path.read_text()) if targets_path.exists() else {}
    done = {(r["task_id"], r["index"]) for r in read_jsonl(samples_path)}
    lock = threading.Lock()

    def draw(job):
        task_id, split, reason, index = job
        task = tasks[task_id]
        _, asserts = parse_prompt(task["prompt"])
        messages = [{"role": "system", "content": DEFAULT_SYSTEM},
                    {"role": "user", "content": INSTRUCTION.format(prompt=task["prompt"].strip())}]
        reply = chat(messages, max_tokens=768, temperature=args.temperature, seed=1000 + index)
        code = keep_definitions(extract_code(reply["text"], task["entry_point"]))
        record = {"task_id": task_id, "split": split, "index": index, "seed": 1000 + index, "reason": reason,
                  "model": model, "temperature": args.temperature, "code": code,
                  "visible_ok": public_check(code, asserts)["ok"], "llm": {k: reply[k] for k in LLM_FIELDS_SAFE}}
        with lock, samples_path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    for split in args.splits:
        rows = {r["task_id"]: r for r in read_jsonl(OUT / "gen" / f"{RUNS[split]}.jsonl") if r["split"] == split}
        if split not in targets:
            targets[split] = {}
            for task_id, row in rows.items():
                reason = target_reason(row, tasks[task_id], args.keep, args.probes, args.minimum)
                if reason:
                    targets[split][task_id] = reason
            targets_path.write_text(json.dumps(targets, indent=1) + "\n")
        jobs = [(task_id, split, reason, i) for task_id, reason in targets[split].items() for i in range(args.k)
                if (task_id, i) not in done]
        counts = {}
        for reason in targets[split].values():
            counts[reason] = counts.get(reason, 0) + 1
        print(split, "tasks", len(rows), "targets", len(targets[split]), counts, "samples to draw", len(jobs), flush=True)
        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(draw, jobs))
        print(split, "done", flush=True)


if __name__ == "__main__":
    main()
