"""Build local evidence/answer/explanation training examples, never a test split."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mylittlellm.distillation_data import build_examples, sha256_file
from mylittlellm.retrieval import ROOT, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", type=Path, default=ROOT / "data/examples/qa_candidates.jsonl")
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/examples/distillation_annotations.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-unreviewed-smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a fresh experiment directory")
    examples = build_examples(read_jsonl(args.qa), read_jsonl(args.annotations), smoke=args.allow_unreviewed_smoke)
    args.output.mkdir(parents=True)
    path = args.output / "examples.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in examples))
    summary = {"examples": len(examples), "supervision_tasks": ["answer", "explanation"],
               "supervision_rows": 2 * len(examples),
               "groups": len({e['leakage_group'] for e in examples}),
               "independently_reviewed": not args.allow_unreviewed_smoke,
               "context_protocol": "oracle_plus_distractor_not_retrieval_evaluation",
               "qa_sha256": sha256_file(args.qa), "annotations_sha256": sha256_file(args.annotations),
               "examples_sha256": sha256_file(path), "final_test_set_created": False}
    (args.output / "preparation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
