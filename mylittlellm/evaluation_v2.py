"""Frozen fact-pattern and cited-clause checks, not human semantic accuracy."""
import json
import re
import statistics
import unicodedata

from .assistant import validate_answer
from .evaluation import cluster_interval, paired_comparison
from .hierarchical import scope_matches

ARMS = ['legacy', 'retrieval', 'prompt_schema', 'triplet']


def compact(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text))


def supported(documents, gold):
    scoped = [d for d in documents if scope_matches(d, gold['year'], gold['task'],
                                                    gold['institution'], gold.get('semester'))]
    return all(any(re.search(pattern, compact(d['text'])) for d in scoped)
               for pattern in gold['support_patterns'])


def score(response, gold, evidence):
    result = {'valid':False, 'facts_pass':False, 'citation_clause_pass':False,
              'strict_pass':False, 'retrieval_clause_hit':supported(evidence, gold)}
    if response.get('error') or response.get('finish_reason') == 'length':
        return dict(result, reason='request_error_or_truncation')
    try:
        value = validate_answer(json.loads(response['content']), evidence, 'rag')
    except (ValueError, TypeError, KeyError):
        return dict(result, reason='invalid_json_schema_or_citation')
    required = [bool(re.search(p, compact(value['answer']))) for p in gold['required_patterns']]
    cites = set(value['citations'])
    cited = [d for d in evidence if d['chunk_id'] in cites]
    citation_pass = bool(cited) and all(scope_matches(d, gold['year'], gold['task'],
                                                     gold['institution'], gold.get('semester')) for d in cited) and supported(cited, gold)
    action = value['status'] == gold['expected_status']
    return dict(result, valid=True, required_matched=required, facts_pass=bool(action and all(required)),
                citation_clause_pass=bool(citation_pass), strict_pass=bool(action and all(required) and citation_pass),
                reason='frozen_pattern_proxy_not_expert_semantic_review')


def summarize(responses, gold, inputs):
    lookup = {r['id']:r for r in gold}
    scored = []
    for response in responses:
        g = lookup[response['id']]
        source = inputs[response['id']]
        mode = 'legacy' if response['arm']=='legacy' else 'improved'
        scored.append(dict(response, group=g['leakage_group'], task=g['task'],
                           score=score(response,g,source[mode]['evidence'])))
    arms = {}
    for arm in ARMS:
        rows = [r for r in scored if r['arm']==arm]
        if not rows:
            continue
        totals = {key:sum(r['score'][key] for r in rows) for key in
                  ['valid','facts_pass','citation_clause_pass','strict_pass','retrieval_clause_hit']}
        mode = 'legacy' if arm=='legacy' else 'improved'
        retrieval_seconds = [inputs[r['id']][mode]['seconds'] for r in rows]
        example_seconds = [inputs[r['id']]['example_seconds'] if arm=='triplet' else 0 for r in rows]
        generation_seconds = [r['elapsed_seconds'] for r in rows]
        arms[arm] = dict(totals, n=len(rows), strict_rate=totals['strict_pass']/len(rows),
                         strict_cluster_95_ci=cluster_interval([{'group':r['group'],'correct':r['score']['strict_pass']} for r in rows]),
                         truncated=sum(r.get('finish_reason')=='length' for r in rows),
                         errors=sum(bool(r.get('error')) for r in rows),
                         median_retrieval_seconds=statistics.median(retrieval_seconds),
                         median_generation_seconds=statistics.median(generation_seconds),
                         median_estimated_warm_total_seconds=statistics.median(a+b+c for a,b,c in zip(retrieval_seconds,example_seconds,generation_seconds)),
                         mean_prompt_tokens=statistics.mean((r.get('usage') or {}).get('prompt_tokens',0) for r in rows),
                         mean_completion_tokens=statistics.mean((r.get('usage') or {}).get('completion_tokens',0) for r in rows),
                         by_task={task:{'n':sum(r['task']==task for r in rows),
                                        'strict_pass':sum(r['score']['strict_pass'] for r in rows if r['task']==task)}
                                  for task in sorted({r['task'] for r in rows})})
    comparisons = {}
    for left,right in [('legacy','retrieval'),('retrieval','prompt_schema'),('prompt_schema','triplet'),('legacy','triplet')]:
        a = {r['id']:r['score']['strict_pass'] for r in scored if r['arm']==left}
        b = {r['id']:r['score']['strict_pass'] for r in scored if r['arm']==right}
        if a and b:
            comparisons[left+'_vs_'+right] = dict(paired_comparison(a,b),
                paired_delta_cluster_95_ci=cluster_interval([{'group':lookup[k]['leakage_group'],'correct':int(b[k])-int(a[k])} for k in a]))
    return {'arms':arms,'paired_comparisons':comparisons,'responses':len(scored),
            'independent_human_review':False,'metric_is_pattern_proxy':True,
            'latency_note':'Warm retrieval and example selection measured once, plus separate serial generation; excludes cold model/index loads and preprocessing.'}, scored
