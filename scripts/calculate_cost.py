import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mylittlellm.costs import net_cost, housing_payment

parser = argparse.ArgumentParser()
parser.add_argument("kind", choices=["net", "housing"])
parser.add_argument("input", type=Path)
args = parser.parse_args()
try:
    result = {"net": net_cost, "housing": housing_payment}[args.kind](json.loads(args.input.read_text()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
except ValueError as error:
    raise SystemExit(str(error))
