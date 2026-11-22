"""SeedLM-32, step 3: choose one program per task from the gathered, scored candidates. Counts only.

Rules over the candidates (scripts/seedlm32_candidates.py, seedlm32_score.py):
  plain_first   the coder's own first verified program (the baseline every coder result so far used)
  generator     the highest log-probability under the coder that wrote it (sketch candidates include the
                probability of their forced tokens)
  ensemble      the highest mean log-probability under the six scoring coders (those that can write it)
  vote          the most cached coders (24) whose first verified program behaves the same; ties to ensemble
  vote_ensemble ensemble + 2 x vote
Each rule is also applied as a system: the engine's program where the engine is confident, the rule's
choice elsewhere (the engine's program counts as a candidate), the engine's program when there is no
candidate.

  python scripts/seedlm32_select.py --set teacher            # exploring rules
  python scripts/seedlm32_select.py --set development --rule <registered rule>
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylittlellm.rulecoder.tasks import ROOT
from seedlm32_candidates import OUT
from seedlm32_score import SCORERS

RULES = ["plain_first", "generator", "ensemble", "vote", "vote_ensemble"]


def ensemble(item):
    values = [v for v in (item.get("scores") or {}).values() if v is not None]
    return statistics.mean(values) if values else float("-inf")


def key(rule, item):
    if rule == "generator":
        return item.get("score", float("-inf"))       # the engine's program has no generator score
    if rule == "ensemble":
        return ensemble(item)
    if rule == "vote":
        return (item.get("voters", 0), ensemble(item))
    if rule == "vote_ensemble":
        return ensemble(item) + 2.0 * item.get("voters", 0)
    raise ValueError(rule)


def choose(rule, candidates):
    if not candidates:
        return None
    if rule == "plain_first":
        plain = [c for c in candidates if c["source"] == "plain"]
        return plain[0] if plain else None
    return max(candidates, key=lambda c: key(rule, c))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=["teacher", "development"], required=True)
    parser.add_argument("--rule", choices=RULES, help="report only this rule (the registered one)")
    parser.add_argument("--output")
    args = parser.parse_args()
    rows = [r for r in json.loads((OUT / f"candidates_{args.set}.json").read_text()) if r is not None]
    report = {"set": args.set, "tasks": len(rows), "scorers": SCORERS,
              "engine": sum(bool(r["engine"] and r["engine"]["right"]) for r in rows),
              "engine_confident": sum(bool(r["engine"] and r["engine"]["confident"]) for r in rows),
              "some_candidate_right": sum(any(c["right"] for c in r["candidates"]) for r in rows),
              "some_right_or_engine": sum(bool(r["engine"] and r["engine"]["right"]) or any(c["right"] for c in r["candidates"])
                                          for r in rows),
              "rules": {}}
    for rule in ([args.rule] if args.rule else RULES):
        alone = system = fixed = broken = 0
        for r in rows:
            pick = choose(rule, r["candidates"])
            alone += bool(pick and pick["right"])
            engine = r["engine"]
            if engine and engine["confident"]:
                answer = engine["right"]
            else:
                pool = r["candidates"] + ([engine] if engine and engine.get("program") and "behaviour" in engine else [])
                choice = choose(rule, pool) if rule != "plain_first" else (pick or engine)
                answer = bool(choice["right"]) if choice else bool(engine and engine["right"])
            system += answer
            fixed += answer and not (engine and engine["right"])
            broken += bool(engine and engine["right"]) and not answer
        report["rules"][rule] = {"alone": alone, "system": system, "system_versus_engine": [fixed, broken]}
    text = json.dumps(report, indent=1)
    print(text)
    if args.output:
        (ROOT / args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
