"""Student answers for HumanEval+ (greedy, plain EvalPlus prompt), plus optional samples for consensus.

Writes outputs/rule_pilot/gen/<name>.jsonl in the same shape as the MBPP+ runs, with split "he".
With --samples k it also draws k temperature samples per task listed in --targets.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mylittlellm.rulecoder import humaneval as he
from mylittlellm.rulecoder.extract import extract_code, keep_definitions
from mylittlellm.rulecoder.llm import LLM_FIELDS_SAFE, chat, server_model
from mylittlellm.rulecoder.pipeline import DEFAULT_SYSTEM, INSTRUCTION, public_check
from mylittlellm.rulecoder.tasks import ROOT

OUT = ROOT / "outputs/rule_pilot"


def read_jsonl(path):
    if not Path(path).exists():
        return []
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--targets", help="JSON file with task ids to sample (default: all)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    tasks = he.read_tasks()
    model = server_model()["id"]
    lock = threading.Lock()

    def messages(task):
        return [{"role": "system", "content": DEFAULT_SYSTEM},
                {"role": "user", "content": INSTRUCTION.format(prompt=task["prompt"].strip())}]

    if not args.samples:
        path = OUT / "gen" / f"{args.name}.jsonl"
        done = {r["task_id"] for r in read_jsonl(path)}

        def greedy(task_id):
            task = tasks[task_id]
            reply = chat(messages(task), max_tokens=768)
            code = keep_definitions(extract_code(reply["text"], task["entry_point"]))
            no_example = {"ok": True, "kind": "no_example", "message": ""}
            check = public_check(code, task["visible_assert"]) if task["visible_assert"] else no_example
            attempt = {"round": 0, "code": code, "raw_code": code, "source": "plain", "check": check, "hints": [],
                       "llm": {k: reply[k] for k in LLM_FIELDS_SAFE}}
            row = {"name": args.name, "task_id": task_id, "split": "he", "draw": 0, "condition": "base", "model": model,
                   "final_code": code, "attempts": [attempt], "chosen": "plain", "visible_ok": check["ok"],
                   "wall_seconds": reply["elapsed_seconds"]}
            with lock, path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")

        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(greedy, [t for t in tasks if t not in done]))
        return

    path = OUT / "samples" / f"{args.name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    wanted = json.loads(Path(args.targets).read_text()) if args.targets else list(tasks)
    done = {(r["task_id"], r["index"]) for r in read_jsonl(path)}

    def sample(job):
        task_id, index = job
        task = tasks[task_id]
        reply = chat(messages(task), max_tokens=768, temperature=args.temperature, seed=1000 + index)
        code = keep_definitions(extract_code(reply["text"], task["entry_point"]))
        ok = public_check(code, task["visible_assert"])["ok"] if task["visible_assert"] else True
        record = {"task_id": task_id, "split": "he", "index": index, "seed": 1000 + index, "model": model,
                  "temperature": args.temperature, "code": code, "visible_ok": ok,
                  "llm": {k: reply[k] for k in LLM_FIELDS_SAFE}}
        with lock, path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    jobs = [(t, i) for t in wanted for i in range(args.samples) if (t, i) not in done]
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(sample, jobs))


if __name__ == "__main__":
    main()
