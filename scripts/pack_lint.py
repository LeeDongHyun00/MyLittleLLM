"""Check operators mechanically: search function and emitted code must agree (see rulecoder/oplint.py).

  python scripts/pack_lint.py --pack r3_strings      # one pack on top of the core library
  python scripts/pack_lint.py --pack none --core     # the core library itself
"""
import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack", default="manifest", help="pack name(s), 'none', or 'manifest' for the shipped set")
    parser.add_argument("--core", action="store_true", help="also check core operators, not only pack operators")
    parser.add_argument("--trials", type=int, default=1500, help="argument tuples tried per binary/ternary operator")
    args = parser.parse_args()
    if args.pack != "manifest":
        os.environ["RULE_PACKS"] = args.pack
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mylittlellm.rulecoder import backinfer as bi, oplint
    chosen = {key: op for key, op in bi.OPERATORS.items() if args.core or op.pack != "core"}
    problems = oplint.lint(chosen, trials=args.trials)
    for key, problem, example in problems:
        print(f"{key:34s} {problem}" + (f"   on {example}" if example else ""))
    defects = [p for p in problems if not p[1].startswith("UNTESTED")]
    print(f"{len(chosen)} operators checked, {len(defects)} defects, {len(problems) - len(defects)} untested")
    return 1 if defects else 0


if __name__ == "__main__":
    sys.exit(main())
