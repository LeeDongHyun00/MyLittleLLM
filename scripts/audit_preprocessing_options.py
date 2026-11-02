"""Post-hoc retrieval diagnostic on exposed RAG-v2 questions, not a new test.

No model inference, training, or production setting changes. Full-page and
candidate-union variants ignore the old context budget and are coverage probes.
Labels are used only by the frozen scorer after selecting each evidence set.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mylittlellm.evaluation_v2 import supported
from mylittlellm.hierarchical import sha
from mylittlellm.retrieval import ROOT, read_jsonl, retrieve


def unique(rows, key):
    return list({key(row): row for row in rows}.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/preprocessing_audit.json')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output path; never silently replace an audit')
    paths = [
        'outputs/rag-v2-inputs/test.jsonl', 'outputs/index.jsonl',
        'outputs/rag-v2/parents.jsonl', 'outputs/rag-v2/leaves.jsonl',
        'data/evaluation/rag_v2_gold.jsonl', 'data/retrieval_snapshot.jsonl',
        'mylittlellm/retrieval.py', 'mylittlellm/evaluation_v2.py',
        'mylittlellm/hierarchical.py', 'scripts/audit_preprocessing_options.py',
    ]
    hashes = {p: sha(ROOT/p) for p in paths}
    lock = json.loads((ROOT/'reports/rag_v2_input_lock.json').read_text())
    protocol = json.loads((ROOT/'data/evaluation/rag_v2_protocol.json').read_text())
    expected = {
        'outputs/rag-v2-inputs/test.jsonl': lock['files']['test.jsonl'],
        'outputs/index.jsonl': lock['legacy_index_sha256'],
        'data/evaluation/rag_v2_gold.jsonl': protocol['gold_sha256'],
    }
    manifest = json.loads((ROOT/'outputs/rag-v2/manifest.json').read_text())
    for name in ['parents.jsonl', 'leaves.jsonl']:
        expected['outputs/rag-v2/'+name] = manifest['hashes'][name]
    for path, digest in expected.items():
        if hashes[path] != digest:
            raise ValueError('Frozen input changed: '+path)

    inputs = read_jsonl(ROOT/paths[0])
    legacy = read_jsonl(ROOT/paths[1])
    parents = {r['chunk_id']: r for r in read_jsonl(ROOT/paths[2])}
    leaves = {r['chunk_id']: r for r in read_jsonl(ROOT/paths[3])}
    # Read labels into a separate lookup; never pass labels to retrieval.
    gold = {r['id']: r for r in read_jsonl(ROOT/paths[4])}
    snapshots = {r['source_id']: r for r in read_jsonl(ROOT/paths[5])}
    pages = {}
    for source_id in {r['source_id'] for r in parents.values()}:
        path = snapshots[source_id]['extracted_path']
        hashes[path] = sha(ROOT/path)
        for page in read_jsonl(ROOT/path):
            if page['sha256'] != snapshots[source_id]['sha256']:
                raise ValueError('Page provenance mismatch: '+source_id)
            pages[(source_id, page['pdf_page'])] = page['text']

    def expand_pages(docs):
        return [dict(d, chunk_id=f"{d['source_id']}:p{d['pdf_page']}:full",
                     text=pages[(d['source_id'], d['pdf_page'])])
                for d in unique(docs, lambda x: (x['source_id'], x['pdf_page']))]

    measures = defaultdict(list)
    details = []
    for row in inputs:
        candidates = row['improved']['candidates']
        candidate_parents = unique([parents[c['parent_id']] for c in candidates],
                                   lambda x: x['chunk_id'])
        scope = {k: row[k] for k in ['year', 'task', 'institution', 'semester']}
        ranked = retrieve(legacy, row['question'], k=8, **scope)
        # Verify we reproduce the prior BM25 selection before comparing larger k.
        if [d['chunk_id'] for d in ranked[:2]] != [d['chunk_id'] for d in row['legacy']['evidence']]:
            raise ValueError('BM25 baseline no longer reproduces: '+row['id'])
        variants = {
            'legacy_bm25_k2': ranked[:2],
            'legacy_bm25_k4': ranked[:4],
            'legacy_bm25_k8': ranked,
            'hierarchical_selected': row['improved']['evidence'],
            'hierarchical_all8_candidate_leaves': [leaves[c['chunk_id']] for c in candidates],
            'hierarchical_all8_candidate_parents': candidate_parents,
            'hierarchical_selected_full_pages': expand_pages(row['improved']['evidence']),
            'hierarchical_all8_candidate_full_pages': expand_pages(candidate_parents),
            'union_bm25_k2_and_hierarchical_selected': unique(
                ranked[:2]+row['improved']['evidence'], lambda x: x['chunk_id']),
        }
        item = {'id': row['id'], 'variants': {}}
        for name, docs in variants.items():
            result = {'clause_pattern_hit': supported(docs, gold[row['id']]),
                      'characters': sum(len(d['text']) for d in docs),
                      'documents': len(docs),
                      'chunk_ids': [d['chunk_id'] for d in docs]}
            measures[name].append(result)
            item['variants'][name] = result
        details.append(item)

    baseline = {r['id']: r['variants']['hierarchical_selected']['clause_pattern_hit'] for r in details}
    summary = {name: {
        'n': len(rows), 'clause_pattern_hits': sum(r['clause_pattern_hit'] for r in rows),
        'median_characters': statistics.median(r['characters'] for r in rows),
        'max_characters': max(r['characters'] for r in rows),
        'median_documents': statistics.median(r['documents'] for r in rows),
        'miss_ids': [r['id'] for r in details if not r['variants'][name]['clause_pattern_hit']],
        'additional_hits_vs_hierarchical_selected': [r['id'] for r in details
            if not baseline[r['id']] and r['variants'][name]['clause_pattern_hit']],
    } for name, rows in measures.items()}
    audit = {
        'purpose': 'post_hoc_development_diagnostic_not_fresh_test',
        'model_generations': 0, 'production_settings_changed': False,
        'metric': 'original_frozen_support_patterns_not_semantic_or_answer_accuracy',
        'scope_limit': 'document_metadata_only; section/row applicability not verified',
        'budget_limit': 'expanded variants are NOT capped at 900 tokens; characters are not tokens',
        'selection': 'question/scope or previously frozen candidates; no gold-driven document selection',
        'data_access': 'full retrieved text remains ignored locally; report contains IDs, hashes, counts only',
        'input_sha256': hashes, 'summary': summary, 'items': details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
