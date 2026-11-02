"""Prespecify then evaluate guarded LLMLingua-2 on the frozen free-answer subset."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mylittlellm.distillation_data import sha256_file
from mylittlellm.evaluation import prompt, score
from mylittlellm.retrieval import ROOT, read_jsonl


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'run'])
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/evaluation-compression-v1')
    args = parser.parse_args()
    dataset = ROOT/'outputs/evaluation-v1'
    main_run = ROOT/'outputs/evaluation-run-v1'
    protocol_path = args.output/'protocol.json'
    tracked = [dataset/'inputs.jsonl', dataset/'gold.jsonl', main_run/'requests.jsonl',
               ROOT/'config/distillation.json', ROOT/'mylittlellm/compression.py',
               ROOT/'mylittlellm/evaluation.py', Path(__file__).resolve()]
    if args.phase == 'prepare':
        if args.output.exists():
            parser.error('Supplement already exists; no overwrite')
        args.output.mkdir(parents=True)
        write_json(protocol_path, {
            'frozen_at': datetime.now(timezone.utc).isoformat(),
            'model_outputs_seen': False,
            'questions': 'All 8 prespecified free-answer questions, same retrieval and rubric',
            'arm': 'opsd_guarded_llmlingua2', 'requested_rate': .5,
            'compressor_revision': json.loads((ROOT/'config/distillation.json').read_text())['compressor_revision'],
            'unchanged_input_rule': 'Reuse original OPSD response only if complete request JSON is identical; count as reused, not a new inference',
            'changed_input_rule': 'One additional inference per changed request; no retry, tuning, or rubric changes',
            'latency_rule': 'Report compression preprocessing cost and token counts; no paired speedup claim from reused or non-interleaved runs',
            'hashes': {str(p.relative_to(ROOT)): sha256_file(p) for p in tracked},
        })
        print(protocol_path)
        return
    protocol = json.loads(protocol_path.read_text())
    for name, digest in protocol['hashes'].items():
        if sha256_file(ROOT/name) != digest:
            raise ValueError('Prespecified file changed: ' + name)
    if (args.output/'run_lock.json').exists():
        parser.error('Supplement run already started; no silent rerun')
    if not (main_run/'summary.json').exists():
        parser.error('Wait for all main evaluation requests to finish first')
    baseline = {(r['id'], r['mode'], r['arm']): r for r in read_jsonl(main_run/'responses.jsonl')}
    jobs = [r for r in read_jsonl(main_run/'requests.jsonl') if r['mode']=='free' and r['arm']=='opsd']
    inputs = {r['id']: r for r in read_jsonl(dataset/'inputs.jsonl')}
    gold = {r['id']: r for r in read_jsonl(dataset/'gold.jsonl')}
    write_json(args.output/'run_lock.json', {
        'started_at': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': sha256_file(protocol_path),
        'baseline_responses_sha256': sha256_file(main_run/'responses.jsonl'),
        'questions': len(jobs),
    })
    from mylittlellm.compression import compress_documents, runtime_compressor
    started = time.monotonic()
    compressor, tokenizer = runtime_compressor()
    initialization_seconds = time.monotonic() - started
    prepared = []
    for job in jobs:
        row = copy.deepcopy(inputs[job['id']])
        documents, metrics = compress_documents(row['retrieved_documents'], .5,
                                               compressor=compressor, tokenizer=tokenizer)
        row['retrieved_documents'] = documents
        payload = dict(job['payload'], messages=prompt(row, 'free'))
        prepared.append({'id': job['id'], 'inputs': row, 'payload': payload,
                         'unchanged_request': payload == job['payload'], 'compression': metrics})
    (args.output/'prepared.jsonl').write_text(''.join(json.dumps(p, ensure_ascii=False)+'\n' for p in prepared))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def request(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request('http://127.0.0.1:8080'+path, data=data,
                                     headers={'Content-Type': 'application/json'})
        with opener.open(req, timeout=360) as response:
            return json.load(response)
    results = []
    try:
        for p in prepared:
            original = baseline[p['id'], 'free', 'opsd']
            response = dict(original, arm='opsd_guarded_llmlingua2')
            if not p['unchanged_request']:
                request('/lora-adapters', [{'id':0,'scale':0.}, {'id':1,'scale':1.}])
                state = request('/lora-adapters')
                if {s['id']:s['scale'] for s in state} != {0:0.,1:1.} or 'opsd-adapter-f32.gguf' not in state[1]['path']:
                    raise RuntimeError('Unexpected adapter state')
                response = {'id':p['id'], 'mode':'free', 'arm':'opsd_guarded_llmlingua2', 'adapter_state':state}
                start = time.monotonic()
                try:
                    value = request('/v1/chat/completions', p['payload'])
                    choice = value['choices'][0]
                    response.update(content=choice['message'].get('content') or '',
                                    finish_reason=choice['finish_reason'], usage=value.get('usage'), model=value.get('model'))
                except Exception as error:
                    response.update(error=f'{type(error).__name__}: {error}', content='')
                response['elapsed_seconds'] = round(time.monotonic()-start, 4)
            response['reused_identical_request'] = p['unchanged_request']
            response['compression'] = p['compression']
            results.append(response)
            with (args.output/'responses.jsonl').open('a') as stream:
                stream.write(json.dumps(response, ensure_ascii=False)+'\n')
            print(json.dumps({'completed':len(results), 'total':len(prepared), 'id':p['id'],
                              'reused_identical_request':p['unchanged_request'], 'error':response.get('error')}), flush=True)
    finally:
        request('/lora-adapters', [{'id':0,'scale':0.}, {'id':1,'scale':0.}])
    scored = [dict(r, score=score(r, gold[r['id']], p['inputs'])) for r,p in zip(results,prepared)]
    original_tokens = sum(r['compression']['original_context_tokens'] for r in results)
    compressed_tokens = sum(r['compression']['compressed_context_tokens'] for r in results)
    summary = {
        'n':len(results), 'new_inferences':sum(not r['reused_identical_request'] for r in results),
        'reused_identical_requests':sum(r['reused_identical_request'] for r in results),
        'original_context_tokens':original_tokens, 'compressed_context_tokens':compressed_tokens,
        'token_reduction_fraction':1-compressed_tokens/original_tokens if original_tokens else 0,
        'initialization_seconds':initialization_seconds,
        'preprocessing_seconds':sum(r['compression']['elapsed_seconds'] for r in results),
        'free_schema_valid':sum(r['score']['valid'] for r in scored),
        'free_strict_rubric_pass':sum(r['score']['strict_pass'] for r in scored),
        'free_truncated':sum(r.get('finish_reason')=='length' for r in scored),
        'latency_claim': 'No end-to-end speedup estimate; identical requests may reuse baseline responses',
    }
    (args.output/'scored.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in scored))
    write_json(args.output/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
