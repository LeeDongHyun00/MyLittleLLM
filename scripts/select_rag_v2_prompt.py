"""Select two error-informed prompt candidates on OLD development questions only."""
import json
from pathlib import Path
import statistics
import sys
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mylittlellm.evaluation import score
from mylittlellm.hierarchical import sha
from mylittlellm.rag_v2 import PROMPTS, complete, request_body
from mylittlellm.retrieval import ROOT, read_jsonl


def main():
    output = ROOT/'outputs/rag-v2-prompt-selection'
    selected = ROOT/'config/rag_v2_selected.json'
    if output.exists() or selected.exists():
        raise ValueError('Prompt selection already started; never silently rerun')
    inputs = read_jsonl(ROOT/'outputs/rag-v2-inputs/dev.jsonl')
    gold = {r['id']:r for r in read_jsonl(ROOT/'data/evaluation/rag_v2_development.jsonl')}
    if any(not gold[r['id']].get('required') for r in inputs):
        raise ValueError('Development rubric must contain the original required fact patterns')
    output.mkdir()
    jobs = [dict(id=row['id'],candidate=name,payload=request_body(row,row['improved']['evidence'],prompt_name=name))
            for row in inputs for name in PROMPTS]
    (output/'requests.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in jobs))
    (output/'protocol.json').write_text(json.dumps({'created_at':datetime.now(timezone.utc).isoformat(),
        'requests_sha256':sha(output/'requests.jsonl'),'development_ids':[r['id'] for r in inputs],
        'development_gold_sha256':sha(ROOT/'data/evaluation/rag_v2_development.jsonl'),
        'selection':'maximize old frozen strict rubric passes, then minimize mean completion tokens, then candidate name',
        'feedback_basis':'Old holdout-v1 truncation, redundant follow-up and long answers; two assistant-authored candidates, not an external optimizer model.',
        'new_test_responses_seen':False},ensure_ascii=False,indent=2)+'\n')
    lookup = {r['id']:r for r in inputs}
    responses = []
    for i,job in enumerate(jobs,1):
        response = dict(id=job['id'],candidate=job['candidate'],mode='free',**complete(job['payload']))
        response['score'] = score(response,gold[job['id']],{'retrieved_documents':lookup[job['id']]['improved']['evidence']})
        responses.append(response)
        with (output/'responses.jsonl').open('a') as stream:stream.write(json.dumps(response,ensure_ascii=False)+'\n')
        print(json.dumps({'completed':i,'total':len(jobs),'id':job['id'],'candidate':job['candidate'],'error':response.get('error')},ensure_ascii=False),flush=True)
    if any(r.get('error') for r in responses):
        raise RuntimeError('Development request error: no selection or test launch')
    metrics = {name:{'strict_pass':sum(r['score']['strict_pass'] for r in responses if r['candidate']==name),
                     'mean_completion_tokens':statistics.mean((r['usage'] or {}).get('completion_tokens',0) for r in responses if r['candidate']==name)} for name in PROMPTS}
    winner = min(PROMPTS,key=lambda name:(-metrics[name]['strict_pass'],metrics[name]['mean_completion_tokens'],name))
    selected.write_text(json.dumps({'prompt_name':winner,'selected_at':datetime.now(timezone.utc).isoformat(),
                                   'metrics':metrics,'development_ids':[r['id'] for r in inputs],
                                   'selection_protocol_sha256':sha(output/'protocol.json'),
                                   'prompt_module_sha256':sha(ROOT/'mylittlellm/rag_v2.py'),
                                   'new_test_used':False},ensure_ascii=False,indent=2)+'\n')
    print(selected.read_text(),flush=True)


if __name__=='__main__':main()
