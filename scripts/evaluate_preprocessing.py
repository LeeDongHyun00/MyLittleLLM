"""Prepare then run a frozen three-arm holdout; no labels enter requests."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
import sys
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mylittlellm.assistant import validate_answer
from mylittlellm.evaluation_v2 import score
from mylittlellm.hierarchical import sha
from mylittlellm.preprocessed_assistant import make_request
from mylittlellm.preprocessing import PreprocessedRetriever,token_count
from mylittlellm.rag_v2 import complete
from mylittlellm.retrieval import ROOT,read_jsonl,retrieve

OUT=ROOT/'outputs/preprocessing-holdout'
ARMS=['concise_k2','bm25_budget','preprocessed']
DEPENDENCIES=['mylittlellm/preprocessing.py','mylittlellm/preprocessed_assistant.py',
              'mylittlellm/rag_v2.py','mylittlellm/assistant.py','mylittlellm/evaluation_v2.py',
              'mylittlellm/hierarchical.py','mylittlellm/retrieval_runtime.py','mylittlellm/retrieval.py',
              'scripts/evaluate_preprocessing.py','config/preprocessing.json','config/rag_v2.json',
              'config/runtime.lock.json','serve.sh','data/evaluation/preprocessing_holdout.jsonl',
              'data/evaluation/preprocessing_protocol.json','outputs/preprocessed/index.jsonl',
              'outputs/preprocessed/manifest.json','outputs/index.jsonl','outputs/rag-v2/manifest.json',
              'outputs/preprocessing-development/summary.json','config/preprocessing_selected.json']


def state(path):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://127.0.0.1:8080'+path,timeout=10) as response:return json.load(response)


def verify_server():
    props=state('/props');model=ROOT/'models/Qwen3-4B-Q4_K_M.gguf'
    if state('/health').get('status')!='ok' or state('/lora-adapters')!=[]:raise ValueError('Healthy base server required')
    if Path(props['model_path']).resolve()!=model.resolve() or props.get('model_alias')!='Qwen3-4B':raise ValueError('Wrong model')
    digest=sha(model)
    if digest!=json.loads((ROOT/'config/runtime.lock.json').read_text())['model']['sha256']:raise ValueError('Wrong weights')
    return dict(properties=props,model_sha256=digest,adapters=[])


def prepare():
    if OUT.exists():raise ValueError('Frozen inputs already exist')
    protocol=json.loads((ROOT/'data/evaluation/preprocessing_protocol.json').read_text())
    if sha(ROOT/'data/evaluation/preprocessing_holdout.jsonl')!=protocol['gold_sha256']:raise ValueError('Gold changed')
    # Only these fields may pass to the retrieval and generation pipeline.
    fields=['id','question','year','institution','task','semester']
    queries=[{k:r[k] for k in fields} for r in read_jsonl(ROOT/'data/evaluation/preprocessing_holdout.jsonl')]
    selection=json.loads((ROOT/'config/preprocessing_selected.json').read_text())
    server=verify_server()
    from mylittlellm.hierarchical import HierarchicalRetriever
    from mylittlellm.retrieval_runtime import LowMemoryQwenRetrievalModels
    retriever=PreprocessedRetriever(semantic=HierarchicalRetriever(models=LowMemoryQwenRetrievalModels()))
    inputs=[];jobs=[];rng=random.Random(42)
    for row in queries:
        scope={k:row[k] for k in ['year','institution','task','semester']}
        start=__import__('time').monotonic()
        baseline=retrieve(retriever.legacy,row['question'],k=8,**scope)
        baseline_seconds=__import__('time').monotonic()-start
        budgeted=[];remaining=1800
        for doc in baseline:
            size=token_count(doc['text'],retriever.tok)
            if size<=remaining:
                budgeted.append(doc);remaining-=size
            if len(budgeted)==4:break
        prepared,metadata=retriever.retrieve(row['question'],**scope)
        arms={
            'concise_k2':{'evidence':baseline[:2],'seconds':baseline_seconds},
            'bm25_budget':{'evidence':budgeted,'seconds':baseline_seconds},
            'preprocessed':dict(metadata,evidence=prepared),
        }
        for arm,value in arms.items():
            value['context_tokens']=sum(token_count(d['text'],retriever.tok) for d in value['evidence'])
            if value['context_tokens']>1800:raise ValueError('Evidence budget exceeded')
        inputs.append(dict(row,arms=arms))
        order=ARMS.copy();rng.shuffle(order)
        for arm in order:
            evidence=arms[arm]['evidence']
            jobs.append(dict(id=row['id'],arm=arm,llm_called=bool(evidence),
                             payload=make_request(row,evidence,prompt_name=selection['prompt_name'],max_tokens=384) if evidence else None))
        print(json.dumps({'prepared':row['id'],'n':len(inputs)},ensure_ascii=False),flush=True)
    OUT.mkdir()
    for name,rows in [('inputs.jsonl',inputs),('requests.jsonl',jobs)]:
        (OUT/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    lock={'created_at':datetime.now(timezone.utc).isoformat(),'hashes':{p:sha(ROOT/p) for p in DEPENDENCIES},
          'inputs_sha256':sha(OUT/'inputs.jsonl'),'requests_sha256':sha(OUT/'requests.jsonl'),
          'server':server,'jobs':len(jobs),'planned_model_calls':sum(j['llm_called'] for j in jobs),
          'prompt':selection['prompt_name'],'max_tokens':384,'temperature':0,'seed':42,'thinking':False,
          'responses_seen_before_lock':False,'answer_labels_passed_to_retrieval':False,
          'arms':{'concise_k2':'unchanged BM25 k=2; selected prompt/schema; 384 output tokens',
                  'bm25_budget':'unchanged BM25 candidates; max 4 whole passages / 1800 evidence tokens; same generation',
                  'preprocessed':'structured PDF units + reviewed scopes + lexical retention + semantic candidate pages; max 4 / 1800 tokens; same generation'},
          'limits':['Same source documents as development; new rule groups, not independent institutions.',
                    'No independent expert gold review; frozen pattern metric, manual checks reported separately.',
                    'Preprocessing arm bundles parsing, scope, query normalization and selection; not isolated effects.',
                    'Equal maximum evidence budget does not imply identical actual input length.']}
    (OUT/'lock.json').write_text(json.dumps(lock,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'prepared':len(jobs),'planned_model_calls':lock['planned_model_calls']},ensure_ascii=False),flush=True)


def verify_lock():
    lock=json.loads((OUT/'lock.json').read_text())
    for p,digest in lock['hashes'].items():
        if sha(ROOT/p)!=digest:raise ValueError('Frozen dependency changed: '+p)
    for name in ['inputs','requests']:
        if sha(OUT/(name+'.jsonl'))!=lock[name+'_sha256']:raise ValueError('Frozen '+name+' changed')
    return lock


def run(resume):
    lock=verify_lock();verify_server()
    path=OUT/'responses.jsonl'
    if path.exists() and not resume:raise ValueError('Use --resume; refusing repeat requests')
    responses=read_jsonl(path) if path.exists() else []
    jobs=read_jsonl(OUT/'requests.jsonl');seen={(r['id'],r['arm']) for r in responses}
    if len(seen)!=len(responses) or not seen<={(r['id'],r['arm']) for r in jobs}:raise ValueError('Unexpected responses')
    for job in jobs:
        if (job['id'],job['arm']) in seen:continue
        if job['llm_called']:
            response=complete(job['payload'])
        else:
            value=dict(status='cannot_determine',answer='해당 대학·연도·학기의 근거를 찾지 못했습니다.',
                       citations=[],follow_up=['해당 범위의 공식 안내 자료를 확인해 주세요.'])
            response=dict(content=json.dumps(value,ensure_ascii=False),finish_reason='no_evidence',elapsed_seconds=0,usage=None)
        response.update(id=job['id'],arm=job['arm'],llm_called=job['llm_called'])
        with path.open('a') as stream:stream.write(json.dumps(response,ensure_ascii=False)+'\n')
        responses.append(response)
        print(json.dumps({'completed':len(responses),'total':len(jobs),'id':job['id'],'arm':job['arm'],
                          'seconds':response['elapsed_seconds'],'error':response.get('error')},ensure_ascii=False),flush=True)
    verify_lock()
    gold={r['id']:r for r in read_jsonl(ROOT/'data/evaluation/preprocessing_holdout.jsonl')}
    inputs={r['id']:r for r in read_jsonl(OUT/'inputs.jsonl')}
    scored=[]
    for response in responses:
        g=gold[response['id']];docs=inputs[response['id']]['arms'][response['arm']]['evidence']
        result=score(response,g,docs)
        answerable=g['expected_status']=='answer'
        if not answerable:
            try:
                value=validate_answer(json.loads(response['content']),docs,'rag')
                valid=not response.get('error') and response.get('finish_reason')!='length'
                abstained=valid and value['status']=='cannot_determine'
                result.update(valid=bool(valid),facts_pass=bool(abstained),citation_clause_pass=bool(abstained and not value['citations']),
                              strict_pass=bool(abstained and not value['citations']),retrieval_clause_hit=None,
                              unsafe_assertion=bool(valid and value['status']=='answer'))
            except (ValueError,TypeError):result.update(strict_pass=False,retrieval_clause_hit=None,unsafe_assertion=False)
        scored.append(dict(response,answerable=answerable,score=result))
    summary={}
    for arm in ARMS:
        rows=[r for r in scored if r['arm']==arm];answerable=[r for r in rows if r['answerable']]
        summary[arm]={'n':len(rows),'model_calls':sum(r['llm_called'] for r in rows),
                      **{k:sum(bool(r['score'][k]) for r in rows) for k in ['valid','facts_pass','citation_clause_pass','strict_pass']},
                      'answerable_n':len(answerable),'answerable_strict':sum(r['score']['strict_pass'] for r in answerable),
                      'retrieval_clause_hits':sum(bool(r['score']['retrieval_clause_hit']) for r in answerable),
                      'unanswerable_n':len(rows)-len(answerable),
                      'unanswerable_correct_abstentions':sum(r['score']['strict_pass'] for r in rows if not r['answerable']),
                      'unsafe_assertions':sum(r['score'].get('unsafe_assertion',False) for r in rows),
                      'truncated':sum(r.get('finish_reason')=='length' for r in rows),
                      'errors':sum(bool(r.get('error')) for r in rows),
                      'median_generation_seconds_called_only':statistics.median(r['elapsed_seconds'] for r in rows if r['llm_called']),
                      'median_evidence_tokens':statistics.median(inputs[r['id']]['arms'][arm]['context_tokens'] for r in rows),
                      'mean_completion_tokens_called_only':statistics.mean((r.get('usage') or {}).get('completion_tokens',0) for r in rows if r['llm_called'])}
    result={'completed_at':datetime.now(timezone.utc).isoformat(),'arms':summary,
            'metric':'frozen_pattern_proxy_not_expert_accuracy','lock_sha256':sha(OUT/'lock.json'),
            'new_model_calls':sum(r['llm_called'] for r in responses),'total_decisions':len(responses)}
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (OUT/'scored.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in scored))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.prepare==args.run:parser.error('Choose exactly one of --prepare or --run')
    if args.prepare:prepare()
    else:run(args.resume)
