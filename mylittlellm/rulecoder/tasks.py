"""MBPP+ task loading, prompt parsing and the frozen dev/test split."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data/raw/rule_pilot/MbppPlus.jsonl.gz"
SPLIT = ROOT / "data/rule_pilot/split.json"
SPLIT_SEED = 20261118
# Same proportions as docs/experiment_design.md: 48% train / 12% validation / 40% locked test.
FRACTIONS = {"test": 0.40, "val": 0.12}


def read_tasks(path=DATASET):
    with gzip.open(path, "rt") as stream:
        return {row["task_id"]: row for row in map(json.loads, stream)}


def parse_prompt(prompt):
    """Split an MBPP+ docstring prompt into (description, public assert source)."""
    body = prompt.strip()
    if body.startswith('"""') and body.endswith('"""'):
        body = body[3:-3]
    lines = body.strip().splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("assert ")), len(lines))
    description, asserts = "\n".join(lines[:start]).strip(), "\n".join(lines[start:]).strip()
    if asserts:
        ast.parse(asserts)  # fail loudly if the visible example is not executable Python
    return description, asserts


def make_split(task_ids, seed=SPLIT_SEED):
    """Deterministic split; the test part must never be used for rule induction or selection."""
    ids = sorted(task_ids, key=lambda t: int(t.split("/")[1]))
    random.Random(seed).shuffle(ids)
    n_test = round(len(ids) * FRACTIONS["test"])
    n_val = round(len(ids) * FRACTIONS["val"])
    return {"test": sorted(ids[:n_test]), "val": sorted(ids[n_test:n_test + n_val]),
            "train": sorted(ids[n_test + n_val:])}


def write_split(path=SPLIT):
    if path.exists():
        raise SystemExit(f"Split already frozen at {path}; refusing to overwrite the locked test set")
    split = make_split(read_tasks())
    record = {"dataset": "MBPP+ v0.2.0", "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
              "seed": SPLIT_SEED, "counts": {k: len(v) for k, v in split.items()}, "split": split}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + "\n")
    return record


def load_split(path=SPLIT):
    record = json.loads(path.read_text())
    if record["dataset_sha256"] != hashlib.sha256(DATASET.read_bytes()).hexdigest():
        raise SystemExit("Dataset file differs from the one the split was frozen on")
    return record["split"]


def split_of(task_id, split):
    for name, ids in split.items():
        if task_id in ids:
            return name
    raise KeyError(task_id)
