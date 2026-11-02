"""LLMLingua-2 with conservative, answer-independent policy-span protection."""
import re
import time
from functools import lru_cache

CRITICAL = re.compile(r"\d|이내|미만|이상|이하|제외|불가|없음|않|아니|단[,，\s]|정액|한도|예외|학교장|지원\s*자격")


def load_compressor(local_path, model_name, qwen_tokenizer):
    """Use local Qwen token accounting; no unrelated GPT tokenizer download.

    llmlingua 0.2.2 hardcodes a tiktoken download in __init__. This small pinned
    constructor adapter retains its real token classifier and compression code,
    replacing only the budget tokenizer with the downstream model's tokenizer.
    """
    from llmlingua import PromptCompressor

    class Counter:
        def encode(self, text):
            return qwen_tokenizer.encode(text, add_special_tokens=False)

    class LocalCompressor(PromptCompressor):
        def __init__(self):
            self.model_name = model_name
            self.use_llmlingua2 = True
            self.retrieval_model = None
            self.retrieval_model_name = None
            self.open_api_config = {}
            self.cache_bos_num = 10
            self.prefix_bos_num = 100
            self.oai_tokenizer = Counter()
            self.load_model(str(local_path), 'cpu', {'local_files_only': True, 'trust_remote_code': False})
            self.init_llmlingua2(max_batch_size=8)

    return LocalCompressor()


def policy_segments(text):
    # Do not split decimal numbers, dates or negative amounts at a hyphen/dot.
    return [s for s in re.split(r"(?=[※●□])|(?<=[.!?])(?=\s+[가-힣A-Za-z])|\n+", text) if s]


def protected_spans(text):
    return [s for s in policy_segments(text) if CRITICAL.search(s)]


def guarded_compress(text, compressor, rate=.5):
    if not 0 < rate <= 1:
        raise ValueError("Compression rate must be in (0, 1]")
    start = time.monotonic()
    parts = []
    for segment in policy_segments(text):
        if CRITICAL.search(segment) or len(segment.strip()) < 40:
            parts.append(segment)
        else:
            parts.append(compressor.compress_prompt_llmlingua2(
                [segment], rate=rate, force_reserve_digit=True,
                force_tokens=["아니", "않", "제외", "불가", "없음"],
            )["compressed_prompt"])
    output = "\n".join(parts)
    required = protected_spans(text)
    missing = [s for s in required if s not in output]
    fallback = bool(missing) or not output.strip() or len(output) > len(text)
    if fallback:
        output = text
    return {"text": output, "protected_spans": len(required),
            "protected_spans_retained": sum(s in output for s in required),
            "fallback_to_original": fallback,
            "elapsed_seconds": round(time.monotonic() - start, 4)}


@lru_cache(maxsize=1)
def runtime_compressor():
    import json
    import torch
    from transformers import AutoTokenizer
    from .retrieval import ROOT
    config = json.loads((ROOT / 'config/distillation.json').read_text())
    torch.set_num_threads(config['threads'])
    tokenizer = AutoTokenizer.from_pretrained(ROOT / config['local_model'], local_files_only=True)
    return load_compressor(ROOT / config['local_compressor'], config['compressor'], tokenizer), tokenizer


def compress_documents(documents, rate=.5, *, compressor=None, tokenizer=None):
    started = time.monotonic()
    if compressor is None or tokenizer is None:
        compressor, tokenizer = runtime_compressor()
    results, output = [], []
    for doc in documents:
        result = guarded_compress(doc['text'], compressor, rate)
        result['original_tokens'] = len(tokenizer.encode(doc['text'], add_special_tokens=False))
        result['compressed_tokens'] = len(tokenizer.encode(result['text'], add_special_tokens=False))
        # Character length alone does not bound the downstream token budget.
        if result['compressed_tokens'] > result['original_tokens']:
            result.update(text=doc['text'], compressed_tokens=result['original_tokens'], fallback_to_original=True)
        output.append(dict(doc, text=result.pop('text')))
        results.append(dict(result, chunk_id=doc['chunk_id']))
    return output, {'method': 'llmlingua2_protected_policy_spans', 'requested_rate': rate,
                    'elapsed_seconds': round(time.monotonic()-started, 4),
                    'original_context_tokens': sum(r['original_tokens'] for r in results),
                    'compressed_context_tokens': sum(r['compressed_tokens'] for r in results),
                    'documents': results}
