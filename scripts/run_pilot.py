"""Development smoke comparison, NOT a held-out accuracy benchmark or training."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mylittlellm.assistant import ask, SYSTEM
from mylittlellm.retrieval import ROOT, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/pilot.jsonl")
    args = parser.parse_args()
    questions = read_jsonl(ROOT / "data/examples/qa_candidates.jsonl")
    if not args.allow_unreviewed and any(q["review_status"] != "approved" for q in questions):
        parser.error("Public candidates are unreviewed. Pass --allow-unreviewed only for a development smoke run")
    if args.output.exists():
        parser.error("Output already exists; choose a new path to preserve prior results")
    index = read_jsonl(ROOT / "outputs/index.jsonl")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for question in questions:
            for mode in ["closed", "rag"]:
                # Only question + user-visible scope reach the model. Never
                # pass answer, required_facts, evidence, or explanation labels.
                response = ask(question["question"], year=question["academic_year"],
                               task="housing" if question["semester"] else "admissions",
                               institution=question["institution"], semester=question["semester"],
                               mode=mode, index=index)
                gold_pages = {(e["source_id"], e["pdf_page"]) for e in question["evidence"]}
                found_pages = {(r["source_id"], r["pdf_page"]) for r in response["retrieved"]}
                row = {"id": question["id"], "question": question["question"],
                       "label_status": question["review_status"], "evaluation_kind": "development_smoke",
                       "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                       "prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
                       "index_sha256": hashlib.sha256((ROOT / "outputs/index.jsonl").read_bytes()).hexdigest(),
                       "candidate_page_hit": bool(gold_pages & found_pages) if mode == "rag" else None,
                       "response": response}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n"); stream.flush()
                print(question["id"], mode, response["status"], response["elapsed_seconds"], flush=True)


if __name__ == "__main__": main()
