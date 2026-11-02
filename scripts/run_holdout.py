"""Evaluate frozen held-out inputs with the same Q4 deployment base and adapters."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mylittlellm.distillation_data import sha256_file
from mylittlellm.evaluation import prompt,summarize
from mylittlellm.retrieval import ROOT,read_jsonl


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ROOT/'outputs/evaluation-v1')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/evaluation-run-v1')
    parser.add_argument('--base-url',default='http://127.0.0.1:8080')
    args=parser.parse_args()
    if args.output.exists():parser.error('Run output exists; never silently rerun or overwrite a test')
    protocol=json.loads((args.dataset/'protocol.json').read_text())
    for name,digest in protocol['hashes'].items():
        if sha256_file(args.dataset/name)!=digest:raise ValueError('Frozen test changed')
    for name,digest in protocol['model_files'].items():
        if sha256_file(ROOT/name)!=digest:raise ValueError('Frozen model changed')
    inputs=read_jsonl(args.dataset/'inputs.jsonl')
    gold=read_jsonl(args.dataset/'gold.jsonl'); by_id={r['id']:r for r in inputs}
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def request(path,body=None):
        data=json.dumps(body).encode() if body is not None else None
        req=urllib.request.Request(args.base_url.rstrip('/')+path,data=data,headers={'Content-Type':'application/json'})
        with opener.open(req,timeout=360) as response:return json.load(response)
    def activate(arm):
        scales=[{'id':0,'scale':1. if arm=='sft' else 0.},{'id':1,'scale':1. if arm=='opsd' else 0.}]
        request('/lora-adapters',scales)
        state=request('/lora-adapters')
        if len(state)!=2 or {s['id']:s['scale'] for s in state}!={s['id']:s['scale'] for s in scales}:
            raise RuntimeError('Server adapter activation verification failed')
        if 'sft-adapter-f32.gguf' not in state[0]['path'] or 'opsd-adapter-f32.gguf' not in state[1]['path']:
            raise RuntimeError('Server loaded unexpected adapters')
        return state
    args.output.mkdir(parents=True)
    rng=random.Random(protocol['seed']); jobs=[]
    for mode in ['mcq','free']:
        for row in inputs:
            if mode=='free' and 'retrieved_documents' not in row:continue
            arms=['base','sft','opsd'];rng.shuffle(arms)
            for arm in arms:
                payload={'model':'Qwen3-4B','temperature':0,'seed':42,'stream':False,'cache_prompt':False,
                         'max_tokens':4 if mode=='mcq' else 160,
                         'chat_template_kwargs':{'enable_thinking':False},'messages':prompt(row,mode)}
                if mode=='mcq':payload['grammar']='root ::= [ABCD]'
                jobs.append({'id':row['id'],'mode':mode,'arm':arm,'payload':payload})
    (args.output/'requests.jsonl').write_text(''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in jobs))
    lock={'created_at':datetime.now(timezone.utc).isoformat(),'requests_sha256':sha256_file(args.output/'requests.jsonl'),
          'protocol_sha256':sha256_file(args.dataset/'protocol.json'),'jobs':len(jobs),
          'evaluation_code_sha256':sha256_file(ROOT/'mylittlellm/evaluation.py')}
    (args.output/'run_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    results=[]
    try:
        for i,job in enumerate(jobs,1):
            state=activate(job['arm']); start=time.monotonic()
            result={k:job[k] for k in ['id','mode','arm']};result['adapter_state']=state
            try:
                response=request('/v1/chat/completions',job['payload'])
                choice=response['choices'][0]
                result.update(content=choice['message'].get('content') or '',finish_reason=choice['finish_reason'],
                              usage=response.get('usage'),model=response.get('model'))
            except Exception as error:
                result.update(error=f'{type(error).__name__}: {error}',content='')
            result['elapsed_seconds']=round(time.monotonic()-start,4)
            results.append(result)
            with (args.output/'responses.jsonl').open('a') as stream:stream.write(json.dumps(result,ensure_ascii=False)+'\n')
            # No correctness feedback until every prespecified request is done.
            print(json.dumps({'completed':i,'total':len(jobs),'id':job['id'],'mode':job['mode'],'arm':job['arm'],
                              'seconds':result['elapsed_seconds'],'error':result.get('error')},ensure_ascii=False),flush=True)
    finally:
        activate('base')
    summary,scored=summarize(results,gold,by_id)
    summary['run_lock']=lock
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (args.output/'scored.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in scored))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
