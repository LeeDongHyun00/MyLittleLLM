import json
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mylittlellm.hierarchical import QwenRetrievalModels, build_hierarchy
from mylittlellm.retrieval import ROOT

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build a Unicode-safe hierarchical retrieval index')
    parser.add_argument('--reuse-index',type=Path)
    args = parser.parse_args()
    model = QwenRetrievalModels(reranker=False)
    print(json.dumps(build_hierarchy(model, ROOT/'outputs/rag-v2',reuse_index=args.reuse_index),ensure_ascii=False,indent=2))
