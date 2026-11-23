"""SeedLM-40, part B: how noisy is a comparison between two coders that differ only in their seed? Counts only.

Every effect looked for since SeedLM-6 was 1-5 tasks. This measures the noise floor directly, for two kinds
of pairs that differ in nothing but the seed:
  scratch    coders trained from scratch with the same recipe (SeedLM-12's 40-minute A and B triples)
  finetune   400-step control fine-tunes from one base checkpoint, seed 0, 1 and 2 (batch order only;
             same synthetic draw, same data): s34-o<k>, s40-o<k>s1, s40-o<k>s2
For each pair and each judge: the difference in solved tasks and the number of discordant tasks D (solved by
exactly one of the two). Judges: held-out teacher tasks (139; lenient = the task's three inputs, strict = no
conflict with the teacher's expression on the 24 probes); development fifth (94; lenient = asserts, strict =
reference on the 24 plain probes, contract = reference on in-domain probes), each for the coder alone (its
first verified program, SeedLM-15's cache) and, where the per-task verdicts exist, for the system (engine plus
the coder, pyx_override). Reported per family: the standard deviation s of paired differences, the mean D,
and the seed pairs needed to detect 2.5 and 5 tasks at 80% power, R = (2.8 s / delta)^2.

  python scripts/seedlm40_noise.py --output reports/seedlm/seedlm40_noise.json
"""
import argparse
import itertools
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylittlellm.rulecoder import backinfer as bi, consensus
from mylittlellm.rulecoder.limits import cap_memory
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import select
from seedlm15_ensemble_scaling import CACHE
from seedlm2_real import held_out
from seedlm3_tasks import reference_contract_results, reference_probe_results

BANK = ROOT / "outputs/rule_pilot/teacher/round3/all_tasks.json"
VERDICTS = ROOT / "outputs/seedlm/verdicts"
LOCAL = ROOT / "outputs/seedlm/v40"
FAMILIES = {
    "scratch_A": ["s12-a1", "s12-a2", "s12-a3"],
    "scratch_B": ["s12-b1", "s12-b2", "s12-b3"],
    "finetune_base1": ["s34-o1", "s40-o1s1", "s40-o1s2"],
    "finetune_base2": ["s34-o2", "s40-o2s1", "s40-o2s2"],
    "finetune_base3": ["s34-o3", "s40-o3s1", "s40-o3s2"],
}


def teacher_gold():
    from seedlm3_discovery import teacher_cases
    out = []
    for case in teacher_cases():
        output = case["expected"] if isinstance(case["expected"], str) else None
        probes = consensus.probe_inputs(list(case["args"]), 24)
        out.append(select.behaviour(case["gold"], probes, case["literals"], output))
    return out


def development_references():
    """Per development task: (plain reference results, (contract probes, contract reference results)), cached."""
    path = LOCAL / "development_references.json"
    if path.exists():
        return json.loads(path.read_text())
    items = [i for i in json.loads(BANK.read_text()) if i["source"] == "extra" and held_out(i["task_id"])]
    out = []
    for item in items:
        plain = reference_probe_results(item)
        contract = reference_contract_results(item)
        out.append({"plain": plain, "contract": [contract[0], contract[1]] if contract else None,
                    "literals": bi.description_literals(item["description"]),
                    "visible": repr(bi.parse_example(item["visible_assert"], item["entry_point"])["expected"])})
    LOCAL.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out) + "\n")
    return out


def agree(behaviour, reference):
    return reference is None or not behaviour or consensus.compare(
        [tuple(b) for b in behaviour], [tuple(r) for r in reference]) != "conflict"


def coder_vectors(name, gold, references):
    cache = json.loads((CACHE / f"{name}.json").read_text())
    teacher, development = cache["teacher"], cache["development"]
    vectors = {"teacher_lenient": [], "teacher_strict": [], "dev_lenient": [], "dev_strict": [], "dev_contract": []}
    for row, g in zip(teacher, gold):
        ok = bool(row and row["correct"])
        vectors["teacher_lenient"].append(ok)
        vectors["teacher_strict"].append(ok and agree(row["behaviour"], g))
    for row, ref in zip(development, references):
        ok = bool(row and row["correct"])
        vectors["dev_lenient"].append(ok)
        vectors["dev_strict"].append(ok and agree(row["behaviour"], ref["plain"]))
        contract_ok = ok
        if ok and ref["contract"]:
            import ast
            probes, reference = ref["contract"]
            expected = ast.literal_eval(ref["visible"])
            output = expected if isinstance(expected, str) else None
            contract_ok = agree(select.behaviour(row["program"], probes, ref["literals"], output), reference)
        vectors["dev_contract"].append(contract_ok)
    for stem, suffix in (("holdout4_%s_strict.json", ""), ("holdout4_%s.json", "")):
        path = VERDICTS / (stem % name)
        if path.exists():
            per_task = json.loads(path.read_text())
            rows = list(per_task.values())
            vectors["system_lenient"] = [bool(r["pyx_override"]) for r in rows]
            if "pyx_override_strict" in rows[0]:
                vectors["system_strict"] = [bool(r["pyx_override_strict"]) for r in rows]
            if "pyx_override_contract" in rows[0]:
                vectors["system_contract"] = [bool(r["pyx_override_contract"]) for r in rows]
            break
    return vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    cap_memory()
    gold, references = teacher_gold(), development_references()
    report = {"families": {}}
    for family, names in FAMILIES.items():
        present = [n for n in names if (CACHE / f"{n}.json").exists()]
        if len(present) < 2:
            continue
        vectors = {n: coder_vectors(n, gold, references) for n in present}
        judges = sorted(set.intersection(*(set(v) for v in vectors.values())))
        entry = {"coders": present, "solved": {n: {j: sum(vectors[n][j]) for j in judges} for n in present},
                 "pairs": {}, "summary": {}}
        for judge in judges:
            diffs, discordant = [], []
            for a, b in itertools.combinations(present, 2):
                va, vb = vectors[a][judge], vectors[b][judge]
                diffs.append(sum(va) - sum(vb))
                discordant.append(sum(x != y for x, y in zip(va, vb)))
                entry["pairs"].setdefault(f"{a} - {b}", {})[judge] = {"difference": diffs[-1], "discordant": discordant[-1]}
            # The spread of one coder's count times sqrt(2) is the spread of a difference between two seeds.
            s = statistics.stdev([sum(vectors[n][judge]) for n in present]) * (2 ** 0.5)
            entry["summary"][judge] = {"paired_differences": diffs, "s_of_paired_difference": round(s, 2),
                                       "mean_discordant": round(statistics.mean(discordant), 1),
                                       "pairs_for_2.5_tasks": round((2.8 * s / 2.5) ** 2, 1) if s else 0,
                                       "pairs_for_5_tasks": round((2.8 * s / 5) ** 2, 1) if s else 0}
        report["families"][family] = entry
    text = json.dumps(report, indent=1)
    print(text)
    if args.output:
        (ROOT / args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
