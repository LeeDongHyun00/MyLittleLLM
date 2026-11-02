"""Freeze and run four paired arms; reveal scores only after every request ends."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import random
import sys
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from mylittlellm.evaluation_v2 import ARMS, summarize
from mylittlellm.hierarchical import sha
from mylittlellm.rag_v2 import complete, request_body
from mylittlellm.retrieval import ROOT, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true',help='Resume only missing request IDs after an interruption, verifying frozen hashes')
    args = parser.parse_args()
    output = ROOT/'outputs/rag-v2-evaluation'
    inputs_dir = ROOT/'outputs/rag-v2-inputs'
    gold_path = ROOT/'data/evaluation/rag_v2_gold.jsonl'
    protocol = json.loads((ROOT/'data/evaluation/rag_v2_protocol.json').read_text())
    if sha(gold_path)!=protocol['gold_sha256']:
        raise ValueError('Frozen gold changed')
    input_lock = json.loads((inputs_dir/'lock.json').read_text())
    for name,digest in input_lock['files'].items():
        if sha(inputs_dir/name)!=digest:
            raise ValueError('Prepared input changed: '+name)
    selection = json.loads((ROOT/'config/rag_v2_selected.json').read_text())
    if sha(ROOT/'mylittlellm/rag_v2.py')!=selection['prompt_module_sha256']:
        raise ValueError('Prompt changed since development selection')
    inputs = read_jsonl(inputs_dir/'test.jsonl')
    rng = random.Random(42)
    jobs = []
    for row in inputs:
        arms = ARMS.copy()
        rng.shuffle(arms)
        for arm in arms:
            mode = 'legacy' if arm=='legacy' else 'improved'
            prompt_name = 'legacy' if arm in ['legacy','retrieval'] else selection['prompt_name']
            jobs.append({'id':row['id'],'arm':arm,'payload':request_body(row,row[mode]['evidence'],
                         prompt_name=prompt_name,example=row['example'] if arm=='triplet' else None)})
    paths = ['data/evaluation/rag_v2_gold.jsonl','data/evaluation/rag_v2_protocol.json',
             'data/examples/rag_triplets.jsonl','outputs/rag-v2-inputs/lock.json',
             'outputs/rag-v2/manifest.json','outputs/retrieval_assets.json',
             'config/rag_v2.json','config/rag_v2_selected.json',
             'config/runtime.lock.json','serve.sh',
             'mylittlellm/hierarchical.py','mylittlellm/rag_v2.py','mylittlellm/evaluation_v2.py',
             'mylittlellm/evaluation.py','mylittlellm/assistant.py','mylittlellm/retrieval.py',
             'scripts/run_rag_v2_evaluation.py','models/Qwen3-4B-Q4_K_M.gguf']
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def state(path):
        with opener.open('http://127.0.0.1:8080'+path,timeout=10) as response:
            return json.load(response)
    if state('/lora-adapters')!=[]:
        raise ValueError('Evaluation requires the base model server with no loaded adapters')
    health = state('/health')
    if health.get('status')!='ok':
        raise ValueError('Server unhealthy')
    properties = state('/props')
    expected_model = ROOT/'models/Qwen3-4B-Q4_K_M.gguf'
    if Path(properties.get('model_path','')).resolve()!=expected_model.resolve() or properties.get('model_alias')!='Qwen3-4B':
        raise ValueError('Server is running an unexpected generator model')
    runtime = json.loads((ROOT/'config/runtime.lock.json').read_text())
    if sha(expected_model)!=runtime['model']['sha256']:
        raise ValueError('Generator weights differ from the pinned runtime lock')
    if output.exists():
        if not args.resume:
            raise ValueError('Evaluation exists; refuse silent rerun')
        lock = json.loads((output/'run_lock.json').read_text())
        for name,digest in lock['hashes'].items():
            if sha(ROOT/name)!=digest:
                raise ValueError('Frozen run dependency changed: '+name)
        if sha(output/'requests.jsonl')!=lock['requests_sha256'] or read_jsonl(output/'requests.jsonl')!=jobs:
            raise ValueError('Frozen requests changed')
    else:
        if args.resume:
            raise ValueError('No run to resume')
        output.mkdir()
        (output/'requests.jsonl').write_text(''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in jobs))
        lock = {'created_at':datetime.now(timezone.utc).isoformat(),'jobs':len(jobs),'question_count':len(inputs),
                'hashes':{name:sha(ROOT/name) for name in paths},'requests_sha256':sha(output/'requests.jsonl'),
                'seed':42,'max_output_tokens':160,'adapters':[], 'thinking':False,
                'test_responses_seen_before_lock':False,
                'arms':{'legacy':'BM25 k=2 + legacy prompt + JSON object',
                        'retrieval':'Hierarchy + Qwen dense/BM25/RRF/reranker + legacy prompt + JSON object',
                        'prompt_schema':'Improved retrieval + development-selected prompt + constrained JSON schema',
                        'triplet':'prompt_schema + one semantically selected source-checked old-data demonstration'},
                'limitations':['Hierarchy and semantic retrieval are bundled; prompt and schema are bundled.',
                               '16 answerable procedural questions / 15 rule groups / 2 institutions; no broad generalization or abstention benchmark.',
                               'Same sources as development, with different rule groups; not institution/source-disjoint or temporal holdout.',
                               'Frozen pattern grading is a proxy, not independent expert semantic accuracy.']}
        (output/'run_lock.json').write_text(json.dumps(lock,ensure_ascii=False,indent=2)+'\n')
        (output/'server_properties.json').write_text(json.dumps(properties,ensure_ascii=False,indent=2)+'\n')
    responses = read_jsonl(output/'responses.jsonl') if (output/'responses.jsonl').exists() else []
    completed = {(r['id'],r['arm']) for r in responses}
    if len(completed)!=len(responses) or not completed <= {(j['id'],j['arm']) for j in jobs}:
        raise ValueError('Duplicate or foreign response keys')
    for job in jobs:
        if (job['id'],job['arm']) in completed:
            continue
        response = dict(id=job['id'],arm=job['arm'],**complete(job['payload']))
        responses.append(response)
        with (output/'responses.jsonl').open('a') as stream:
            stream.write(json.dumps(response,ensure_ascii=False)+'\n')
        print(json.dumps({'completed':len(responses),'total':len(jobs),'id':job['id'],'arm':job['arm'],
                          'seconds':response['elapsed_seconds'],'error':response.get('error')},ensure_ascii=False),flush=True)
    # Only now load test labels and produce scores. Never adjust a prompt or rubric
    # from these results; any follow-up experiment needs a fresh holdout.
    for name,digest in lock['hashes'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError('Frozen dependency changed during evaluation: '+name)
    if state('/lora-adapters')!=[]:
        raise ValueError('Adapter state changed during evaluation')
    summary,scored = summarize(responses,read_jsonl(gold_path),{r['id']:r for r in inputs})
    summary['run_lock_sha256'] = sha(output/'run_lock.json')
    summary['completed_at'] = datetime.now(timezone.utc).isoformat()
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (output/'scored.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in scored))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
