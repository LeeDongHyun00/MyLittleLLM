"""Page/section/leaf retrieval with real Qwen3 embeddings and reranking.

HiChunk-inspired hierarchy and evidence merging; not its trained chunker.
No QA labels or reference answers are used to construct or search the index.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import time

from .retrieval import ROOT, ambiguous_comparison, read_jsonl, terms


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_config():
    return json.loads((ROOT/'config/rag_v2.json').read_text())


def token_windows(text, tokenizer, budget, overlap):
    """Cut at Unicode character offsets, never inside a multi-token character."""
    if not 0<=overlap<budget:
        raise ValueError('Overlap must be smaller than the token budget')
    offsets = tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
    seen = set()
    for start in range(0,len(offsets),budget-overlap):
        end = min(start+budget,len(offsets))
        left,right = offsets[start][0],offsets[end-1][1]
        while end>start and len(tokenizer.encode(text[left:right],add_special_tokens=False))>budget:
            end -= 1
            right = offsets[end-1][1] if end>start else left
        if right>left and (left,right) not in seen:
            seen.add((left,right))
            yield text[left:right],left,right


def section_texts(text, tokenizer, budget):
    """Keep clauses/notes together where possible, with overlap for long blocks."""
    units = [s.strip() for s in re.split(r'(?=[□◦])|\n\s*\n', text) if s.strip()]
    blocks, current = [], ''
    for unit in units:
        unit = re.sub(r'\s+', ' ', unit).strip()
        ids = tokenizer.encode(unit, add_special_tokens=False)
        if len(ids) > budget:
            if current:
                blocks.append(current)
                current = ''
            blocks.extend(part for part,_,_ in token_windows(unit,tokenizer,budget,48))
        elif current and len(tokenizer.encode(current+'\n'+unit, add_special_tokens=False)) > budget:
            blocks.append(current)
            current = unit
        else:
            current = (current+'\n'+unit).strip()
    if current:
        blocks.append(current)
    return blocks


class QwenRetrievalModels:
    def __init__(self, config=None, reranker=True):
        import torch
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
        self.config = config or load_config()
        torch.set_num_threads(self.config['threads'])
        self.torch = torch
        spec = self.config['embedding']
        self.tokenizer = AutoTokenizer.from_pretrained(ROOT/spec['path'], local_files_only=True, padding_side='left')
        self.encoder = AutoModel.from_pretrained(ROOT/spec['path'], local_files_only=True,
                                                torch_dtype=torch.float32, trust_remote_code=False).eval()
        if self.config.get('runtime_quantization') == 'torch_dynamic_int8_linear':
            self.encoder = torch.ao.quantization.quantize_dynamic(self.encoder, {torch.nn.Linear}, dtype=torch.qint8)
        self.reranker = None
        if reranker:
            spec = self.config['reranker']
            self.rerank_tokenizer = AutoTokenizer.from_pretrained(ROOT/spec['path'], local_files_only=True, padding_side='left')
            self.reranker = AutoModelForCausalLM.from_pretrained(ROOT/spec['path'], local_files_only=True,
                                                               torch_dtype=torch.float32, trust_remote_code=False).eval()
            if self.config.get('runtime_quantization') == 'torch_dynamic_int8_linear':
                self.reranker = torch.ao.quantization.quantize_dynamic(self.reranker, {torch.nn.Linear}, dtype=torch.qint8)
            self.yes_id = self.rerank_tokenizer.convert_tokens_to_ids('yes')
            self.no_id = self.rerank_tokenizer.convert_tokens_to_ids('no')

    def encode(self, texts, query=False):
        import numpy as np
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)
        if query:
            texts = [f"Instruct: {self.config['instruction']}\nQuery:{t}" for t in texts]
        batches = []
        batch_size = self.config['embedding_batch_size']
        with self.torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = self.tokenizer(texts[start:start+batch_size], padding=True, return_tensors='pt', truncation=False)
                if batch['input_ids'].shape[1] > 8192:
                    raise ValueError('Embedding input exceeds explicit budget; never silently truncate')
                output = self.encoder(**batch, use_cache=False).last_hidden_state[:, -1].float()
                output = self.torch.nn.functional.normalize(output, p=2, dim=1)
                batches.append(output.cpu().numpy())
        return np.concatenate(batches)

    def rerank(self, question, documents):
        if self.reranker is None:
            raise ValueError('Reranker was not loaded')
        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        texts = [prefix+f"<Instruct>: {self.config['instruction']}\n<Query>: {question}\n<Document>: {d}"+suffix for d in documents]
        scores = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), 4):
                inputs = self.rerank_tokenizer(texts[start:start+4], padding=True, return_tensors='pt', truncation=False)
                if inputs['input_ids'].shape[1] > 8192:
                    raise ValueError('Reranking input exceeds explicit budget')
                logits = self.reranker(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
                scores.extend(self.torch.softmax(logits[:, [self.no_id, self.yes_id]], dim=-1)[:, 1].tolist())
        return scores


def build_hierarchy(models, output, root=ROOT, reuse_index=None):
    import numpy as np
    output = Path(output)
    if output.exists():
        raise ValueError('Index output already exists; version the index rather than overwrite it')
    config = models.config
    sources = {s['source_id']:s for s in read_jsonl(root/'data/sources.jsonl')}
    snapshots = {s['source_id']:s for s in read_jsonl(root/'data/retrieval_snapshot.jsonl')}
    corpus = json.loads((root/'config/corpus.json').read_text())['sources']
    parents, leaves, quarantined = [], [], []
    for sid, policy in corpus.items():
        source, snapshot = sources[sid], snapshots[sid]
        if sha(root/snapshot['raw_path']) != snapshot['sha256']:
            raise ValueError('Original document changed: '+sid)
        for page in read_jsonl(root/snapshot['extracted_path']):
            if page['source_id'] != sid or page['sha256'] != snapshot['sha256']:
                raise ValueError('Extraction provenance mismatch')
            if not any(a<=page['pdf_page']<=b for a,b in policy.get('page_ranges', [[1,snapshot['pdf_pages']]])):
                continue
            if ambiguous_comparison(page['text']):
                quarantined.append([sid,page['pdf_page']])
                continue
            if len(page['text'].strip()) < 40:
                continue
            meta = {'source_id':sid, 'pdf_page':page['pdf_page'], 'year':source['academic_year'],
                    'semester':source['semester'], 'task':policy['task'], 'institution':policy['institution'],
                    'sha256':snapshot['sha256'], 'url':source['landing_url']}
            for j, text in enumerate(section_texts(page['text'], models.tokenizer, config['parent_tokens'])):
                parent_id = f"{sid}:p{page['pdf_page']}:h{j}"
                parent = dict(meta, chunk_id=parent_id, text=text)
                parents.append(parent)
                for part,start,end in token_windows(text,models.tokenizer,config['leaf_tokens'],32):
                    if len(part.strip()) < 12:
                        continue
                    leaves.append(dict(meta, chunk_id=f'{parent_id}:l{start}', parent_id=parent_id,
                                       text=part, char_start=start, char_end=end))
    output.mkdir(parents=True)
    for name, rows in [('parents.jsonl',parents), ('leaves.jsonl',leaves)]:
        (output/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    cache, reuse_manifest_hash = {}, None
    def embedding_text(row):
        return f"{row['year']}학년도 {row['institution']}\n{row['text']}"
    if reuse_index is not None:
        reuse_index = Path(reuse_index)
        previous = json.loads((reuse_index/'manifest.json').read_text())
        if previous['config']!=config:
            raise ValueError('Cannot reuse vectors from different retrieval model settings')
        for name,digest in previous['hashes'].items():
            if sha(reuse_index/name)!=digest:raise ValueError('Reuse index integrity failure')
        old_rows = read_jsonl(reuse_index/'leaves.jsonl')
        old_vectors = np.load(reuse_index/'vectors.npy',allow_pickle=False)
        if len(old_rows)!=len(old_vectors):raise ValueError('Reuse vector alignment failure')
        cache = {embedding_text(row):vec for row,vec in zip(old_rows,old_vectors)}
        reuse_manifest_hash = sha(reuse_index/'manifest.json')
    vectors, reused, computed = [], 0, 0
    started = time.monotonic()
    for start in range(0, len(leaves), 64):
        texts = [embedding_text(r) for r in leaves[start:start+64]]
        missing = list(dict.fromkeys(text for text in texts if text not in cache))
        reused += sum(text in cache for text in texts)
        if missing:
            cache.update(zip(missing,models.encode(missing)))
            computed += len(missing)
        vectors.append(np.stack([cache[text] for text in texts]))
        print(json.dumps({'embedded':min(start+64,len(leaves)), 'total':len(leaves),
                          'reused':reused,'computed':computed,'seconds':round(time.monotonic()-started,2)}), flush=True)
    array = np.concatenate(vectors)
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite embeddings')
    np.save(output/'vectors.npy', array, allow_pickle=False)
    manifest = {'parents':len(parents), 'leaves':len(leaves), 'sources':list(corpus), 'config':config,
                'quarantined_pages':quarantined, 'embedding_seconds':time.monotonic()-started,
                'unicode_safe_boundaries':True,'reused_vectors':reused,'computed_vectors':computed,
                'reuse_manifest_sha256':reuse_manifest_hash,
                'hashes':{name:sha(output/name) for name in ['parents.jsonl','leaves.jsonl','vectors.npy']}}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest


def bm25_ranks(rows, question):
    counters = [Counter(terms(r['text'])) for r in rows]
    if not counters:
        return []
    df = Counter(t for c in counters for t in c)
    avg = max(1, sum(sum(c.values()) for c in counters)/len(counters))
    query = set(terms(question))
    scores = []
    for c in counters:
        length = sum(c.values())
        scores.append(sum(math.log(1+(len(rows)-df[t]+.5)/(df[t]+.5))*c[t]*2.5/
                          (c[t]+1.5*(.25+.75*length/avg)) for t in query if c[t]))
    return sorted((i for i in range(len(rows)) if scores[i]>0), key=lambda i:(-scores[i],rows[i]['chunk_id']))


def scope_matches(row, year, task, institution=None, semester=None):
    return row['year']==year and row['task']==task and (institution is None or row['institution']==institution) and (semester is None or row['semester']==semester)


class HierarchicalRetriever:
    def __init__(self, path=None, models=None):
        import numpy as np
        path = Path(path or ROOT/'outputs/rag-v2')
        self.manifest = json.loads((path/'manifest.json').read_text())
        for name,digest in self.manifest['hashes'].items():
            if sha(path/name)!=digest:
                raise ValueError('Index integrity failure')
        self.config = self.manifest['config']
        self.models = models or QwenRetrievalModels(self.config)
        if self.models.config != self.config:
            raise ValueError('Retrieval models/config differ from index')
        self.parents = {r['chunk_id']:r for r in read_jsonl(path/'parents.jsonl')}
        self.leaves = read_jsonl(path/'leaves.jsonl')
        self.vectors = np.load(path/'vectors.npy', allow_pickle=False)
        if len(self.vectors) != len(self.leaves):
            raise ValueError('Embedding row alignment failure')

    def retrieve(self, question, *, year, task, institution=None, semester=None):
        started = time.monotonic()
        if task=='admissions' and not institution:
            raise ValueError('Admissions queries require an institution')
        if task=='housing' and semester is None:
            raise ValueError('Housing queries require a semester')
        indices = [i for i,r in enumerate(self.leaves) if scope_matches(r,year,task,institution,semester)]
        if not indices:
            return [], {'seconds':time.monotonic()-started, 'candidates':[], 'reason':'no_scope'}
        rows = [self.leaves[i] for i in indices]
        query = f'{year}학년도 {institution or ""} {question}'
        qvec = self.models.encode([query], query=True)[0]
        scores = self.vectors[indices] @ qvec
        dense = sorted(range(len(rows)), key=lambda i:(-float(scores[i]),rows[i]['chunk_id']))[:32]
        lexical = bm25_ranks(rows, question)[:32]
        fusion = Counter()
        for ranks in [dense,lexical]:
            for rank,i in enumerate(ranks,1):
                fusion[i] += 1/(self.config['rrf_constant']+rank)
        chosen = sorted(fusion,key=lambda i:(-fusion[i],rows[i]['chunk_id']))[:self.config['rerank_candidates']]
        reranked = self.models.rerank(query,[rows[i]['text'] for i in chosen])
        if len(reranked)!=len(chosen) or not all(math.isfinite(s) and 0<=s<=1 for s in reranked):
            raise ValueError('Invalid reranker scores')
        candidates = [dict(rows[i],dense_score=float(scores[i]),rrf_score=fusion[i],reranker_score=score) for i,score in zip(chosen,reranked)]
        candidates.sort(key=lambda r:(-r['reranker_score'],r['chunk_id']))
        counts = Counter(r['parent_id'] for r in candidates)
        evidence, used, budget = [], set(), self.config['context_tokens']
        for row in candidates:
            pid = row['parent_id']
            if pid in used:
                continue
            parent = self.parents[pid]
            full_tokens = len(self.models.tokenizer.encode(parent['text'],add_special_tokens=False))
            # Parent expansion keeps the query's clause with conditions/notes. Multiple
            # selected siblings are merged into one citation rather than duplicated.
            if full_tokens <= budget:
                selected = dict(parent, retrieval_method='auto_merge_parent' if counts[pid]>1 else 'parent_expansion')
            else:
                selected = {k:v for k,v in row.items() if k not in ['dense_score','rrf_score','reranker_score']}
                selected['retrieval_method'] = 'leaf_budget_fallback'
            size = len(self.models.tokenizer.encode(selected['text'],add_special_tokens=False))
            if size > budget:
                continue
            selected['score'] = row['reranker_score']
            evidence.append(selected)
            budget -= size
            used.add(pid)
            if len(evidence) == self.config['top_k']:
                break
        return evidence, {'seconds':round(time.monotonic()-started,4), 'context_tokens':self.config['context_tokens']-budget,
                          'candidates':[{k:r[k] for k in ['chunk_id','parent_id','dense_score','rrf_score','reranker_score']} for r in candidates]}
