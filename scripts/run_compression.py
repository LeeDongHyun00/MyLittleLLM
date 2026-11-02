"""Measure actual LLMLingua-2 compression on public development evidence pages."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mylittlellm.compression import guarded_compress, protected_spans, load_compressor
from mylittlellm.distillation_data import sha256_file
from mylittlellm.retrieval import ROOT, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rate", type=float, default=.5)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists")
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT / "models/.tiktoken-cache"))
    import torch
    from transformers import AutoTokenizer
    config = json.loads((ROOT / "config/distillation.json").read_text())
    torch.set_num_threads(config["threads"])
    tokenizer = AutoTokenizer.from_pretrained(ROOT / config["local_model"], local_files_only=True)
    started = time.monotonic()
    compressor = load_compressor(ROOT / config["local_compressor"], config['compressor'], tokenizer)
    load_seconds = time.monotonic() - started
    documents = {d["id"]: d for e in read_jsonl(args.examples) for d in e["documents"]}
    results = []
    for doc_id, doc in sorted(documents.items()):
        full = doc["full_text"]
        start = time.monotonic()
        raw = compressor.compress_prompt_llmlingua2([full], rate=args.rate,
                                                    force_reserve_digit=True)["compressed_prompt"]
        raw_seconds = time.monotonic() - start
        guarded = guarded_compress(full, compressor, args.rate)
        spans = protected_spans(full)
        row = {"id": doc_id, "sha256": doc["sha256"],
               "original_qwen_tokens": len(tokenizer.encode(full, add_special_tokens=False)),
               "raw_qwen_tokens": len(tokenizer.encode(raw, add_special_tokens=False)),
               "guarded_qwen_tokens": len(tokenizer.encode(guarded['text'], add_special_tokens=False)),
               "raw_seconds": round(raw_seconds, 4),
               "raw_protected_spans_retained": sum(s in raw for s in spans),
               "raw_text": raw, "guarded": guarded}
        results.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in {"raw_text", "guarded"}}, ensure_ascii=False), flush=True)
    args.output.mkdir(parents=True)
    (args.output / "compressed_documents.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in results))
    total_original = sum(r["original_qwen_tokens"] for r in results)
    summary = {"compressor": config["compressor"], "revision": config["compressor_revision"],
               "budget_tokenizer": "Qwen3-4B (local; replaces llmlingua default GPT tokenizer accounting)",
               "examples_sha256": sha256_file(args.examples), "requested_rate": args.rate,
               "documents": len(results), "load_seconds": round(load_seconds, 4),
               "original_qwen_tokens": total_original,
               "raw_qwen_tokens": sum(r['raw_qwen_tokens'] for r in results),
               "guarded_qwen_tokens": sum(r['guarded_qwen_tokens'] for r in results),
               "guarded_retention_fraction": sum(r['guarded_qwen_tokens'] for r in results)/total_original,
               "protected_spans": sum(r['guarded']['protected_spans'] for r in results),
               "raw_protected_spans_retained": sum(r['raw_protected_spans_retained'] for r in results),
               "guarded_protected_spans_retained": sum(r['guarded']['protected_spans_retained'] for r in results),
               "fallback_documents": sum(r['guarded']['fallback_to_original'] for r in results),
               "accuracy_measured": False,
               "limitation": "Conservative lexical guards are not a proof of semantic completeness; no end-to-end speed claim."}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
