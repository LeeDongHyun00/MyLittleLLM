"""SeedLM-38: does the coders' committee pick truly right programs where the engine is unsure? Counts only.

The rule (reports/seedlm/seedlm38_protocol.json): the engine's program where it is confident; elsewhere the
cached first verified program (SeedLM-15) that most of SeedLM-32's 24 coders agree with on the 24 probes;
the engine's program where no coder proposes one. Judged leniently (the task's asserts) and strictly
(asserts, then no conflict with the reference solution on the probes; SeedLM-36). The same rule with
s5-var's program in place of the vote is reported as the like-for-like baseline.

  python scripts/seedlm38_consensus_strict.py --output reports/seedlm/consensus38_development.json
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylittlellm.rulecoder import consensus
from mylittlellm.rulecoder.limits import cap_memory
from mylittlellm.rulecoder.tasks import ROOT
from seedlm15_ensemble_scaling import CACHE
from seedlm32_candidates import BANK, OUT, held_out
from seedlm32_score import VOTERS
from seedlm3_tasks import reference_probe_results


def verdicts(choice, reference):
    """(lenient, strict) for a chosen {program, right/correct, behaviour} entry."""
    if not choice:
        return False, False
    lenient = bool(choice.get("right", choice.get("correct")))
    behaviour = choice.get("behaviour")
    strict = lenient and (reference is None or not behaviour
                          or consensus.compare([tuple(b) for b in behaviour], reference) != "conflict")
    return lenient, strict


def committee(proposals):
    """The proposal most others agree with (each agrees with itself); ties to the earliest."""
    present = [p for p in proposals if p is not None]
    if not present:
        return None
    support = [sum(consensus.compare(p["behaviour"], q["behaviour"]) == "agree" or p is q for q in present)
               for p in present]
    return present[support.index(max(support))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    cap_memory()
    rows = json.loads((OUT / "candidates_development.json").read_text())
    cache = {v: json.loads((CACHE / f"{v}.json").read_text())["development"] for v in VOTERS}
    items = [i for i in json.loads(BANK.read_text()) if i["source"] == "extra" and held_out(i["task_id"])]
    assert len(items) == len(rows)
    counts = {name: [0, 0] for name in ("engine", "baseline_s5var", "committee")}
    counts["unconfident_or_none"] = 0
    for i, (item, row) in enumerate(zip(items, rows)):
        engine = row["engine"] if row else None
        reference = reference_probe_results(item)
        confident = bool(engine and engine.get("program") and engine.get("confident"))
        counts["unconfident_or_none"] += not confident
        s5 = cache["s5-var"][i]
        vote = committee([cache[v][i] for v in VOTERS])
        picks = {"engine": engine if engine and engine.get("program") else None,
                 "baseline_s5var": engine if confident else (s5 or engine),
                 "committee": engine if confident else (vote or engine)}
        for name, choice in picks.items():
            lenient, strict = verdicts(choice, reference)
            counts[name][0] += lenient
            counts[name][1] += strict
    report = {"tasks": len(items), "voters": len(VOTERS),
              "lenient_strict": {k: v for k, v in counts.items() if isinstance(v, list)},
              "unconfident_or_none": counts["unconfident_or_none"]}
    text = json.dumps(report, indent=1)
    print(text)
    if args.output:
        (ROOT / args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
