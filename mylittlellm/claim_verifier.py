"""Local evidence/claim checks. Scores are model signals, not truth guarantees.

MiniCheck's prefix, separator and decoder label IDs follow the official
Liyan06/MiniCheck minicheck/inference.py (blob 074d1b71d887ee3d6dd58f83c1125f3c87b8f8e7).
We do not execute downloaded Python code, truncate conditions, or translate Korean.
"""
import re
import time
from pathlib import Path

from .retrieval import ROOT


class ClaimVerifier:
    def __init__(self, backend='multilingual_nli'):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
        self.torch = torch
        torch.set_num_threads(4)
        self.backend = backend
        if backend == 'minicheck':
            path = ROOT/'models/MiniCheck-Flan-T5-Large'
            cls = AutoModelForSeq2SeqLM
            self.limit = 2048
        elif backend == 'multilingual_nli':
            path = ROOT/'models/mDeBERTa-nli'
            cls = AutoModelForSequenceClassification
            self.limit = 512
        else:
            raise ValueError('Unknown verifier backend')
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = cls.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                        dtype=torch.float32, low_cpu_mem_usage=True).eval()
        self.labels = {str(v).lower(): int(k) for k,v in self.model.config.id2label.items()}

    def score(self, document, claim):
        start = time.monotonic()
        if self.backend == 'minicheck':
            encoded = self.tokenizer('predict: '+document+self.tokenizer.eos_token+claim,
                                     return_tensors='pt', truncation=False)
        else:
            encoded = self.tokenizer(document, claim, return_tensors='pt', truncation=False)
        ids = encoded['input_ids']
        metadata = dict(tokens=ids.shape[1], unknown_tokens=int((ids == self.tokenizer.unk_token_id).sum()),
                        backend=self.backend)
        if ids.shape[1] > self.limit:
            return dict(metadata, usable=False, reason='whole_evidence_exceeds_limit', support=None, contradiction=None)
        with self.torch.inference_mode():
            if self.backend == 'minicheck':
                output = self.model(**encoded, decoder_input_ids=self.torch.zeros((1,1),dtype=self.torch.long))
                prob = self.torch.softmax(output.logits[0,0,[3,209]],dim=-1)
                support = float(prob[1]); contradiction = None
            else:
                prob = self.torch.softmax(self.model(**encoded).logits[0],dim=-1)
                support = float(prob[self.labels['entailment']])
                contradiction = float(prob[self.labels['contradiction']])
        return dict(metadata, usable=True, support=support, contradiction=contradiction,
                    seconds=round(time.monotonic()-start,4))


def split_claims(answer):
    # Preserve numbers such as 3.5 and dates. This is sentence splitting, not a
    # full proposition/negation-scope parser. Conjoined facts remain together.
    return [x.strip() for x in re.split(r'(?<=[.!?。])\s+|\n+', answer) if x.strip()]


def verify_claims(answer, evidence, verifier, *, threshold, contradiction_threshold=.8):
    claims = split_claims(answer)
    checks = []
    for claim in claims:
        scores = [dict(chunk_id=d['chunk_id'], **verifier.score(d['text'],claim)) for d in evidence]
        usable = [s for s in scores if s['usable']]
        supported = any(s['support'] >= threshold for s in usable)
        contradicted = any((s['contradiction'] or 0) >= contradiction_threshold for s in usable)
        checks.append(dict(claim=claim,supported=supported,contradicted=contradicted,scores=scores))
    return dict(passed=bool(checks) and all(c['supported'] and not c['contradicted'] for c in checks),
                threshold=threshold, checks=checks)
