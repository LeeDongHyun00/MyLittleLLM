"""Download official, revision-pinned embedding/reranking weights; no remote code."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    os.environ.setdefault('HF_HOME', str(ROOT/'models/.hf-cache'))
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    from huggingface_hub import snapshot_download
    from mylittlellm.distillation_data import sha256_file
    config = json.loads((ROOT/'config/rag_v2.json').read_text())
    records = []
    for kind in ['embedding', 'reranker']:
        spec = config[kind]
        folder = Path(snapshot_download(spec['repo'], revision=spec['revision'], local_dir=ROOT/spec['path'],
                                       allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', 'README.md', 'LICENSE'],
                                       max_workers=2))
        records.append(dict(kind=kind, **spec, files=[{'path':str(f.relative_to(folder)), 'bytes':f.stat().st_size,
                        'sha256':sha256_file(f)} for f in sorted(folder.rglob('*')) if f.is_file() and '.cache' not in f.parts]))
        print(f"{kind}: downloaded and hashed", flush=True)
    (ROOT/'outputs/retrieval_assets.json').write_text(json.dumps(records, indent=2)+'\n')


if __name__ == '__main__':
    main()
