"""Fetch pinned verifier assets without executing repository code."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mylittlellm.retrieval import ROOT

if __name__=='__main__':
    from huggingface_hub import snapshot_download
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['minicheck','multilingual_nli','both'],default='both')
    args=p.parse_args();config=json.loads((ROOT/'config/grounding.json').read_text())
    for name in (['minicheck','multilingual_nli'] if args.model=='both' else [args.model]):
        spec=config[name]
        allow=['*.json','README.md','spiece.model','pytorch_model.bin'] if name=='minicheck' else ['*.json','README.md','spm.model','model.safetensors']
        print(snapshot_download(spec['repo'],revision=spec['revision'],local_dir=ROOT/spec['path'],allow_patterns=allow,max_workers=3))
