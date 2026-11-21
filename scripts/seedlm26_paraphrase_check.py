"""Check teacher-written paraphrases of training task descriptions (SeedLM-26).

A paraphrase must keep the task exactly: the same function solves it. Mechanically checked here: the
literals the coder reads from a description (numbers and quoted strings, rulecoder.backinfer's
description_literals) must be the same values in the same order, since a program refers to them by
position; it must differ from the original and from the task's other paraphrase; it must be one
sentence-like statement of reasonable length.

  python scripts/seedlm26_paraphrase_check.py --part 1            # prints problems and counts
  python scripts/seedlm26_paraphrase_check.py --all --summary
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mylittlellm.rulecoder import backinfer as bi

DIR = Path(__file__).resolve().parents[1] / "outputs/paraphrase"


def problems(task, paraphrases):
    out = []
    original = task["description"]
    want = bi.description_literals(original)
    if not isinstance(paraphrases, list) or len(paraphrases) != 2:
        return ["needs exactly two paraphrases"]
    for k, text in enumerate(paraphrases):
        if not isinstance(text, str) or not text.strip():
            out.append(f"#{k + 1}: empty")
            continue
        if text.strip().lower() == original.strip().lower():
            out.append(f"#{k + 1}: same as the original")
        if not 3 <= len(text.split()) <= 80:
            out.append(f"#{k + 1}: length {len(text.split())} words")
        got = bi.description_literals(text)
        if got != want:
            out.append(f"#{k + 1}: literals {got!r} but the original has {want!r}")
    if len(paraphrases) == 2 and paraphrases[0].strip().lower() == paraphrases[1].strip().lower():
        out.append("the two paraphrases are the same")
    return out


def check(part, verbose=True):
    tasks = {t["id"]: t for t in json.loads((DIR / f"in_part{part}.json").read_text())}
    path = DIR / f"out_part{part}.json"
    written = json.loads(path.read_text()) if path.exists() else []
    seen, good, bad = set(), 0, 0
    for row in written:
        if row.get("id") not in tasks:
            print(f"unknown id {row.get('id')}")
            bad += 1
            continue
        if row["id"] in seen:
            print(f"{row['id']}: written twice")
            bad += 1
            continue
        seen.add(row["id"])
        found = problems(tasks[row["id"]], row.get("paraphrases"))
        if found:
            bad += 1
            if verbose:
                print(f"{row['id']}: " + "; ".join(found))
        else:
            good += 1
    missing = len(tasks) - len(seen)
    return {"part": part, "tasks": len(tasks), "good": good, "with_problems": bad, "missing": missing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--summary", action="store_true", help="counts only")
    args = parser.parse_args()
    parts = sorted(int(p.stem[len("in_part"):]) for p in DIR.glob("in_part*.json")) if args.all else [args.part]
    for part in parts:
        print(json.dumps(check(part, verbose=not args.summary)))


if __name__ == "__main__":
    main()
