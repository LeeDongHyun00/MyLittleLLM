"""Build the experimental PDF structure index without QA data."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mylittlellm.preprocessing import build_index

if __name__=='__main__':
    result=build_index()
    print({'units':result['units'],'seconds':result['elapsed_seconds']})
