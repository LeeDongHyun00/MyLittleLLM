import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mylittlellm.retrieval import build_index

if __name__ == "__main__":
    rows = build_index()
    print(f"Indexed {len(rows)} chunks from {len({r['source_id'] for r in rows})} policy documents")
