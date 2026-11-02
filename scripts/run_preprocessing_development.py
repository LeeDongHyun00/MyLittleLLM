"""Paired output-budget/prompt development study on eight exposed questions."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mylittlellm.evaluation_v2 import score
from mylittlellm.hierarchical import sha
from mylittlellm.rag_v2 import complete, request_body
from mylittlellm.retrieval import ROOT, read_jsonl

IDS = ['E01', 'E04', 'E06', 'E07', 'H01', 'H02', 'H05', 'H07']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = ROOT/'outputs/preprocessing-development'
    inputs = {r['id']: r for r in read_jsonl(ROOT/'outputs/rag-v2-inputs/test.jsonl')}
    old_requests = {(r['id'], r['arm']): r['payload'] for r in read_jsonl(ROOT/'outputs/rag-v2-evaluation/requests.jsonl')}
    old_responses = {r['id']: r for r in read_jsonl(ROOT/'reports/rag_v2_responses.jsonl') if r['arm']=='legacy'}
    jobs, reused = [], []
    rng = random.Random(42)
    for id in IDS:
        row = inputs[id]
        evidence = row['legacy']['evidence']
        if request_body(row, evidence, prompt_name='legacy', max_tokens=160) != old_requests[id, 'legacy']:
            raise ValueError('Cannot reuse a nonidentical baseline request')
        reused.append(dict(old_responses[id], arm='legacy160', reused=True))
        pair = [('legacy384','legacy'), ('minimal384','minimal')]
        rng.shuffle(pair)
        for arm,prompt in pair:
            jobs.append(dict(id=id,arm=arm,payload=request_body(row,evidence,prompt_name=prompt,max_tokens=384)))
    dependencies = ['mylittlellm/rag_v2.py','mylittlellm/assistant.py','mylittlellm/evaluation_v2.py',
                    'mylittlellm/hierarchical.py','mylittlellm/retrieval.py',
                    'scripts/run_preprocessing_development.py','outputs/rag-v2-inputs/test.jsonl',
                    'data/evaluation/rag_v2_gold.jsonl','reports/rag_v2_responses.jsonl',
                    'config/runtime.lock.json','serve.sh']
    hashes = {p:sha(ROOT/p) for p in dependencies}
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def state(path):
        with opener.open('http://127.0.0.1:8080'+path,timeout=10) as response:return json.load(response)
    props=state('/props')
    if state('/lora-adapters') or state('/health').get('status')!='ok':
        raise ValueError('Healthy base server required')
    model=ROOT/'models/Qwen3-4B-Q4_K_M.gguf'
    if Path(props['model_path']).resolve()!=model.resolve():raise ValueError('Wrong server model')
    if sha(model)!=json.loads((ROOT/'config/runtime.lock.json').read_text())['model']['sha256']:
        raise ValueError('Model hash differs')
    if out.exists():
        if not args.resume:raise ValueError('Use --resume; refusing overwrite')
        lock=json.loads((out/'lock.json').read_text())
        if hashes!=lock['hashes'] or jobs!=read_jsonl(out/'requests.jsonl'):raise ValueError('Frozen study changed')
    else:
        if args.resume:raise ValueError('No run to resume')
        out.mkdir()
        lock={'created_at':datetime.now(timezone.utc).isoformat(),'ids':IDS,'hashes':hashes,
              'phase':'development_previously_exposed_questions','new_calls':len(jobs),'reused_calls':len(reused),
              'selection_rule':'strict pattern passes, then valid outputs, then fewer mean completion tokens; development proxy only',
              'scope':'output budget, then prompt/schema; unchanged BM25 evidence; no examples',
              'server':props,'model_sha256':sha(model)}
        (out/'lock.json').write_text(json.dumps(lock,ensure_ascii=False,indent=2)+'\n')
        (out/'requests.jsonl').write_text(''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in jobs))
        (out/'responses.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in reused))
    responses=read_jsonl(out/'responses.jsonl')
    seen={(r['id'],r['arm']) for r in responses}
    if len(seen)!=len(responses):raise ValueError('Duplicate responses')
    for job in jobs:
        if (job['id'],job['arm']) in seen:continue
        r=dict(id=job['id'],arm=job['arm'],reused=False,**complete(job['payload']))
        with (out/'responses.jsonl').open('a') as stream:stream.write(json.dumps(r,ensure_ascii=False)+'\n')
        responses.append(r)
        print(json.dumps({'completed_new':sum(not r['reused'] for r in responses),'total_new':len(jobs),
                          'id':r['id'],'arm':r['arm'],'seconds':r['elapsed_seconds'],'error':r.get('error')},ensure_ascii=False),flush=True)
    if hashes!={p:sha(ROOT/p) for p in dependencies}:raise ValueError('Dependency changed during run')
    gold={r['id']:r for r in read_jsonl(ROOT/'data/evaluation/rag_v2_gold.jsonl')}
    scored=[dict(r,score=score(r,gold[r['id']],inputs[r['id']]['legacy']['evidence'])) for r in responses]
    summary={}
    for arm in ['legacy160','legacy384','minimal384']:
        rows=[r for r in scored if r['arm']==arm]
        summary[arm]={'n':len(rows),**{k:sum(r['score'][k] for r in rows) for k in ['valid','facts_pass','citation_clause_pass','strict_pass']},
                      'truncated':sum(r.get('finish_reason')=='length' for r in rows),
                      'median_seconds':statistics.median(r['elapsed_seconds'] for r in rows),
                      'mean_completion_tokens':statistics.mean((r.get('usage') or {}).get('completion_tokens',0) for r in rows)}
    selected=max(['legacy384','minimal384'],key=lambda a:(summary[a]['strict_pass'],summary[a]['valid'],-summary[a]['mean_completion_tokens']))
    result={'phase':'development_not_holdout','arms':summary,'selected_arm':selected,
            'selected_prompt':'legacy' if selected=='legacy384' else 'minimal',
            'max_tokens':384,'independent_human_review':False}
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (out/'scored.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in scored))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
