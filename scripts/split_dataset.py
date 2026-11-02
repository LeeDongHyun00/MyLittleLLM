"""Freeze a reproducible GROUP split: target train/dev/test = 48/12/40.

Only reviewed QA is accepted. This cannot discover semantic duplicate rules:
annotators must assign connected rule/template/version families to one group.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import unicodedata


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def split(rows, group_key="leakage_group", seed=20261102):
    groups, ids, questions = {}, set(), {}
    for row in rows:
        if row.get("review_status") != "approved":
            raise ValueError(f"Unreviewed QA: {row.get('id')}")
        if not row.get("id") or row["id"] in ids:
            raise ValueError("Missing or duplicate QA ID")
        ids.add(row["id"])
        group = row.get(group_key)
        if not isinstance(group, str) or not group.strip():
            raise ValueError(f"Missing group: {row['id']}")
        question = normalized(row["question"])
        if not question:
            raise ValueError("Empty question")
        if question in questions and questions[question] != group:
            raise ValueError("Identical questions occur in different groups; merge groups or disambiguate question")
        questions[question] = group
        if not row.get("evidence") or not row.get("answer"):
            raise ValueError("Evidence and answer are required")
        groups.setdefault(group, []).append(row)
    if len(groups) < 5:
        raise ValueError("At least 5 reviewed groups are required; tiny pilots are not evaluation sets")
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    n_test = max(1, round(n * .40))
    n_dev = max(1, round(n * .12))
    assignments = {"test": keys[:n_test], "dev": keys[n_test:n_test+n_dev], "train": keys[n_test+n_dev:]}
    return {name: [row for key in members for row in sorted(groups[key], key=lambda x: x["id"])]
            for name, members in assignments.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/splits"))
    parser.add_argument("--group-key", choices=["leakage_group", "institution_group"], default="leakage_group")
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new directory to avoid overwriting a frozen test")
    body = args.input.read_bytes()
    rows = [json.loads(line) for line in body.decode().splitlines() if line.strip()]
    parts = split(rows, args.group_key, args.seed)
    args.output.mkdir(parents=True)
    report = {"seed": args.seed, "group_key": args.group_key,
              "input_sha256": hashlib.sha256(body).hexdigest(),
              "target_group_proportions": {"train": .48, "dev": .12, "test": .40}, "actual": {}}
    for name, data in parts.items():
        with (args.output / f"{name}.jsonl").open("w") as stream:
            for row in data:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        report["actual"][name] = {"rows": len(data), "row_fraction": len(data)/len(rows),
                                  "groups": len({r[args.group_key] for r in data})}
    (args.output / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
