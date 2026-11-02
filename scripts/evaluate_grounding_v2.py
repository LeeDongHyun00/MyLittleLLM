"""Frozen shared-evidence comparison for expanded rules and guided-intake work."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import random
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mylittlellm.grounded_assistant import GroundedAssistant
from mylittlellm.hierarchical import sha
from mylittlellm.retrieval import ROOT,read_jsonl
from scripts.evaluate_grounding import append,score,write,write_rows
from scripts.evaluate_preprocessing import verify_server

OUT=ROOT/'outputs/grounding-v2'
DATA=ROOT/'data/evaluation/grounding_v2_scenarios.jsonl'
DEPS=['chat.py','mylittlellm/grounded_assistant.py','mylittlellm/policy_rules.py','mylittlellm/guided_facts.py',
      'mylittlellm/rag_v2.py','mylittlellm/retrieval.py','mylittlellm/assistant.py','mylittlellm/claim_verifier.py',
      'mylittlellm/preprocessed_assistant.py','mylittlellm/preprocessing.py','mylittlellm/costs.py',
      'config/grounding_rules.json','config/grounding.json','config/runtime.lock.json',
      'scripts/evaluate_grounding.py','scripts/evaluate_grounding_v2.py','scripts/evaluate_preprocessing.py',
      'data/evaluation/grounding_v2_scenarios.jsonl','data/evaluation/grounding_v2_protocol.json',
      'outputs/preprocessed/index.jsonl','outputs/index.jsonl']


def prepare():
    if OUT.exists():raise ValueError('Refusing to overwrite a frozen run')
    server=verify_server();assistant=GroundedAssistant();plans=[]
    for row in read_jsonl(DATA):
        start=time.monotonic()
        plan=assistant.plan({k:row.get(k) for k in ['question','year','task','institution','semester']},row['facts'])
        plans.append(dict(id=row['id'],planning_seconds=round(time.monotonic()-start,5),**plan))
    OUT.mkdir();write_rows(OUT/'plans.jsonl',plans)
    order=[p['id'] for p in plans];random.Random(20261102).shuffle(order)
    lock=dict(created_at=datetime.now(timezone.utc).isoformat(),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              hashes={p:sha(ROOT/p) for p in DEPS},plan_sha256=sha(OUT/'plans.jsonl'),server=server,order=order,
              labels_in_requests=False,responses_seen_before_lock=False,planned_calls=sum(p['payload'] is not None for p in plans))
    write(OUT/'lock.json',lock)
    print(json.dumps({'prepared':len(plans),'planned_calls':lock['planned_calls']}),flush=True)


def verify():
    lock=json.loads((OUT/'lock.json').read_text())
    for name,digest in lock['hashes'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen dependency changed: '+name)
    if sha(OUT/'plans.jsonl')!=lock['plan_sha256']:raise ValueError('Frozen requests changed')
    return lock


def generate(resume):
    lock=verify();verify_server();assistant=GroundedAssistant()
    path=OUT/'drafts.jsonl'
    if path.exists() and not resume:raise ValueError('Use --resume to continue')
    old=read_jsonl(path) if path.exists() else [];seen={r['id'] for r in old}
    if len(old)!=len(seen):raise ValueError('Duplicate drafts')
    plans={p['id']:p for p in read_jsonl(OUT/'plans.jsonl')}
    for id in lock['order']:
        if id in seen:continue
        response=assistant.generate(plans[id]);append(path,dict(id=id,**response));seen.add(id)
        print(json.dumps(dict(completed=len(seen),total=len(plans),id=id,seconds=response['elapsed_seconds'],error=response.get('error'))),flush=True)
    verify()


def evaluate():
    verify()
    if (OUT/'results.jsonl').exists():raise ValueError('Refusing to replace results')
    assistant=GroundedAssistant();gold={r['id']:r for r in read_jsonl(DATA)}
    drafts={r['id']:r for r in read_jsonl(OUT/'drafts.jsonl')}
    if set(drafts)!=set(gold):raise ValueError('Complete generation first')
    results=[]
    for plan in read_jsonl(OUT/'plans.jsonl'):
        raw=drafts[plan['id']];value=assistant.parse(raw,plan['evidence'])
        for arm,result in [('qwen_same_evidence',value),('expanded_rules_gate',plan['preflight'] or value)]:
            called=raw['llm_called'] and (arm=='qwen_same_evidence' or plan['preflight'] is None)
            results.append(dict(id=plan['id'],arm=arm,value=result,score=score(result,gold[plan['id']],plan['evidence']),
                                llm_called=called,generation_seconds=raw['elapsed_seconds'] if called else 0,
                                category=gold[plan['id']]['category']))
    arms={}
    for arm in ['qwen_same_evidence','expanded_rules_gate']:
        rs=[r for r in results if r['arm']==arm]
        arms[arm]=dict(n=len(rs),strict_pass=sum(r['score']['strict_pass'] for r in rs),answered=sum(r['score']['answered'] for r in rs),
                       false_assertions=sum(r['score']['false_assertion'] for r in rs),unnecessary_abstentions=sum(r['score']['unnecessary_abstention'] for r in rs),
                       llm_calls=sum(r['llm_called'] for r in rs))
    summary=dict(completed_at=datetime.now(timezone.utc).isoformat(),arms=arms,actual_model_calls=sum(r['llm_called'] for r in drafts.values()),
                 decisions=len(results),generation_errors=sum(bool(r.get('error')) for r in drafts.values()),
                 truncated=sum(r.get('finish_reason')=='length' for r in drafts.values()),independent_expert_review=False,
                 metric='frozen_pattern_proxy_with_separate_manual_review',model_training=False)
    verify();write_rows(OUT/'results.jsonl',results);write(OUT/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','generate','evaluate'],required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='generate':generate(args.resume)
    else:evaluate()
