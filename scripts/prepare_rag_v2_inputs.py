"""Retrieve using questions/scope only; no test reference answers enter requests."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from mylittlellm.hierarchical import HierarchicalRetriever, sha
from mylittlellm.rag_v2 import TripletBank
from mylittlellm.retrieval import ROOT, read_jsonl, retrieve

DEV_IDS = ['A04','A12','S04','S06']
SCOPE_KEYS = ['id','question','year','task','institution','semester']


def main():
    output = ROOT/'outputs/rag-v2-inputs'
    if output.exists():
        raise ValueError('Input output exists; never silently replace prepared inputs')
    protocol = json.loads((ROOT/'data/evaluation/rag_v2_protocol.json').read_text())
    if sha(ROOT/'data/evaluation/rag_v2_gold.jsonl') != protocol['gold_sha256']:
        raise ValueError('Frozen test changed')
    # Project out all labels before any retrieval / example selection operation.
    test = [{k:r.get(k) for k in SCOPE_KEYS} for r in read_jsonl(ROOT/'data/evaluation/rag_v2_gold.jsonl')]
    dev = [{k:r.get(k) for k in SCOPE_KEYS} for r in read_jsonl(ROOT/'data/evaluation/rag_v2_development.jsonl') if r['id'] in DEV_IDS]
    if len(dev)!=len(DEV_IDS):
        raise ValueError('Missing development questions')
    assets = json.loads((ROOT/'outputs/retrieval_assets.json').read_text())
    for asset in assets:
        for file in asset['files']:
            if sha(ROOT/asset['path']/file['path'])!=file['sha256']:
                raise ValueError('Installed retrieval asset changed: '+asset['repo']+'/'+file['path'])
    start = time.monotonic()
    retriever = HierarchicalRetriever()
    if not retriever.manifest.get('unicode_safe_boundaries'):
        raise ValueError('Rebuild the index with Unicode-safe boundaries before evaluation')
    bank = TripletBank(retriever.models)
    load_seconds = time.monotonic()-start
    legacy_index = read_jsonl(ROOT/'outputs/index.jsonl')
    output.mkdir()
    for split,rows in [('dev',dev),('test',test)]:
        for i,row in enumerate(rows,1):
            scope = {k:row[k] for k in ['year','task','institution','semester']}
            started = time.monotonic()
            legacy = retrieve(legacy_index,row['question'],k=2,**scope)
            legacy_seconds = time.monotonic()-started
            improved,metadata = retriever.retrieve(row['question'],**scope)
            started = time.monotonic()
            example = bank.select(row['question'],row['task'],row['year'])
            result = dict(row, legacy={'evidence':legacy,'seconds':legacy_seconds},
                          improved=dict(metadata,evidence=improved), example=example,
                          example_seconds=time.monotonic()-started)
            with (output/(split+'.jsonl')).open('a') as stream:
                stream.write(json.dumps(result,ensure_ascii=False)+'\n')
            print(json.dumps({'split':split,'completed':i,'total':len(rows),'id':row['id'],
                              'retrieval_seconds':metadata['seconds']},ensure_ascii=False),flush=True)
    lock = {'files':{name:sha(output/name) for name in ['dev.jsonl','test.jsonl']},
            'index_sha256':sha(ROOT/'outputs/rag-v2/manifest.json'),
            'legacy_index_sha256':sha(ROOT/'outputs/index.jsonl'),
            'examples_sha256':sha(ROOT/'data/examples/rag_triplets.jsonl'),
            'retrieval_asset_files_verified':True,
            'model_load_and_example_embedding_seconds':load_seconds,
            'test_reference_answers_passed_to_retrieval':False, 'development_ids':DEV_IDS}
    (output/'lock.json').write_text(json.dumps(lock,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
