"""Freeze, generate shared drafts, then evaluate rule/gate/verifier ablations."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mylittlellm.grounded_assistant import GroundedAssistant
from mylittlellm.hierarchical import sha
from mylittlellm.retrieval import ROOT,read_jsonl
from scripts.evaluate_preprocessing import state,verify_server

OUT=ROOT/'outputs/grounding-evaluation'
DEPS=['mylittlellm/claim_verifier.py','mylittlellm/policy_rules.py','mylittlellm/grounded_assistant.py',
      'mylittlellm/preprocessed_assistant.py','mylittlellm/costs.py','mylittlellm/preprocessing.py',
      'mylittlellm/rag_v2.py','mylittlellm/retrieval.py','mylittlellm/assistant.py',
      'config/grounding_rules.json','config/grounding.json','config/runtime.lock.json',
      'scripts/evaluate_grounding.py','data/evaluation/grounding_scenarios.jsonl',
      'data/evaluation/grounding_protocol.json','outputs/preprocessed/index.jsonl','outputs/index.jsonl']

def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def write_rows(path,rows):path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def append(path,value):
    with path.open('a') as stream:stream.write(json.dumps(value,ensure_ascii=False)+'\n')

def prepare():
    if OUT.exists():raise ValueError('Refusing to replace a frozen run')
    server=verify_server();assistant=GroundedAssistant();plans=[]
    fields=['id','question','year','semester','task','institution']
    for row in read_jsonl(ROOT/'data/evaluation/grounding_scenarios.jsonl'):
        query={k:row.get(k) for k in fields}
        plans.append(dict(id=row['id'],**assistant.plan(query,row.get('facts'))))
    order=[r['id'] for r in plans];random.Random(42).shuffle(order)
    OUT.mkdir();write_rows(OUT/'plans.jsonl',plans)
    model_files={str(p.relative_to(ROOT)):sha(p) for dirname in ['MiniCheck-Flan-T5-Large','mDeBERTa-nli']
                 for p in (ROOT/'models'/dirname).iterdir() if p.suffix in ['.json','.bin','.safetensors','.model']}
    lock=dict(created_at=datetime.now(timezone.utc).isoformat(),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              hashes={p:sha(ROOT/p) for p in DEPS},plan_sha256=sha(OUT/'plans.jsonl'),model_hashes=model_files,
              server=server,order=order,labels_in_requests=False,responses_seen_before_lock=False,
              planned_calls=sum(p['payload'] is not None for p in plans),
              known_limits=json.loads((ROOT/'data/evaluation/grounding_protocol.json').read_text()))
    write(OUT/'lock.json',lock);print(json.dumps({'prepared':len(plans),'planned_calls':lock['planned_calls']}),flush=True)

def verify_lock(models=False):
    lock=json.loads((OUT/'lock.json').read_text())
    for p,digest in lock['hashes'].items():
        if sha(ROOT/p)!=digest:raise ValueError('Frozen dependency changed: '+p)
    if sha(OUT/'plans.jsonl')!=lock['plan_sha256']:raise ValueError('Frozen plans changed')
    if models:
        for p,digest in lock['model_hashes'].items():
            if sha(ROOT/p)!=digest:raise ValueError('Verifier asset changed: '+p)
    return lock

def generate(resume):
    lock=verify_lock();verify_server();assistant=GroundedAssistant()
    path=OUT/'drafts.jsonl'
    if path.exists() and not resume:raise ValueError('Use --resume to continue missing generations')
    previous=read_jsonl(path) if path.exists() else [];seen={r['id'] for r in previous}
    if len(seen)!=len(previous):raise ValueError('Duplicate drafts')
    plans={r['id']:r for r in read_jsonl(OUT/'plans.jsonl')}
    for id in lock['order']:
        if id in seen:continue
        response=assistant.generate(plans[id]);append(path,dict(id=id,**response));seen.add(id)
        print(json.dumps(dict(completed=len(seen),total=len(plans),id=id,seconds=response['elapsed_seconds'],error=response.get('error'))),flush=True)
    verify_lock()

def score(value,gold,docs):
    status=value.get('status');answer=value.get('answer','')
    valid=status in ['answer','cannot_determine'] and bool(answer) and set(value.get('citations',[]))<={d['chunk_id'] for d in docs}
    if status=='answer':valid=valid and bool(value.get('citations'))
    answerable=gold['expected_status']=='answer'
    if not answerable:passed=valid and status=='cannot_determine' and not value.get('citations')
    else:passed=valid and status=='answer' and all(re.search(p,answer,re.I) for p in gold['required_patterns']) and not any(re.search(p,answer,re.I) for p in gold['forbidden_patterns'])
    return dict(valid=bool(valid),strict_pass=bool(passed),answered=status=='answer',answerable=answerable,
                false_assertion=not answerable and status=='answer',unnecessary_abstention=answerable and status=='cannot_determine')

def evaluate():
    verify_lock(models=True)
    if (OUT/'results.jsonl').exists():raise ValueError('Refusing to rescore/replace frozen results')
    assistant=GroundedAssistant();plans=read_jsonl(OUT/'plans.jsonl');drafts={r['id']:r for r in read_jsonl(OUT/'drafts.jsonl')}
    gold={r['id']:r for r in read_jsonl(ROOT/'data/evaluation/grounding_scenarios.jsonl')}
    if set(drafts)!=set(gold):raise ValueError('Generate all cases first')
    results=[]
    for plan in plans:
        id=plan['id'];raw=drafts[id];value=assistant.parse(raw,plan['evidence'])
        start=time.monotonic();vetoed,verification=assistant.postcheck(value,plan['evidence']);veto_seconds=time.monotonic()-start
        rule_value=plan['preflight'] or value
        full_value=plan['preflight'] or vetoed
        for arm,answer in [('qwen_shared_evidence',value),('qwen_plus_veto',vetoed),('rules_and_gate',rule_value),('full',full_value)]:
            llm_called=raw['llm_called'] and (arm.startswith('qwen') or plan['preflight'] is None)
            results.append(dict(id=id,arm=arm,value=answer,score=score(answer,gold[id],plan['evidence']),
                                llm_called=llm_called,generation_seconds=raw['elapsed_seconds'] if llm_called else 0,
                                verifier_seconds=veto_seconds if (arm=='qwen_plus_veto' or (arm=='full' and plan['preflight'] is None)) else 0,
                                verification=verification if arm=='qwen_plus_veto' or (arm=='full' and plan['preflight'] is None) else None,
                                category=gold[id]['category']))
        print(json.dumps(dict(evaluated=id,verifier_seconds=round(veto_seconds,3))),flush=True)
    arms={}
    for arm in ['qwen_shared_evidence','qwen_plus_veto','rules_and_gate','full']:
        rs=[r for r in results if r['arm']==arm];answered=sum(r['score']['answered'] for r in rs)
        correct=sum(r['score']['strict_pass'] for r in rs);correct_answered=sum(r['score']['strict_pass'] and r['score']['answered'] for r in rs)
        arms[arm]=dict(n=len(rs),strict_pass=correct,answered=answered,answer_coverage=answered/len(rs),
                       accuracy_among_answers=correct_answered/answered if answered else None,
                       false_assertions=sum(r['score']['false_assertion'] for r in rs),
                       unnecessary_abstentions=sum(r['score']['unnecessary_abstention'] for r in rs),
                       llm_calls=sum(r['llm_called'] for r in rs),
                       categories={cat:dict(n=sum(r['category']==cat for r in rs),passed=sum(r['score']['strict_pass'] for r in rs if r['category']==cat)) for cat in sorted({r['category'] for r in rs})})
    summary=dict(completed_at=datetime.now(timezone.utc).isoformat(),arms=arms,actual_model_calls=sum(r['llm_called'] for r in drafts.values()),
                 decisions=len(results),shared_identical_drafts=True,metric='frozen_pattern_proxy_not_expert_accuracy',
                 generation_errors=sum(bool(r.get('error')) for r in drafts.values()),
                 truncated=sum(r.get('finish_reason')=='length' for r in drafts.values()))
    verify_lock();write_rows(OUT/'results.jsonl',results);write(OUT/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','generate','evaluate'],required=True);p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='generate':generate(args.resume)
    else:evaluate()
