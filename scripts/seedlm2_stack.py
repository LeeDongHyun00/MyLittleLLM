"""A stacked ranker: combine the engine's hand-set score and SeedLM-2's log-probability, learned from data.

Neither ranker is reliable alone: the engine's keyword score picks coincidences (`a0.index(a1)` for "the
sum of the n smallest numbers"), and SeedLM-2's log-probability confuses neighbouring operators on real
descriptions. Each is evidence; a small logistic model learns how much each is worth.

  collect : for development tasks (MBPP+ train split and the extra pool's training part), gather the
            engine's top matches and the model's verified proposals, describe each by a few features,
            and label it with the task's own tests (asserts, or the original MBPP tests for train tasks)
  fit     : logistic regression on those labels; report how often the top-scored candidate is correct
            on the extra pool's held-out fifth versus the engine's and the model's own choices
Validation and test tasks are never touched.
"""
import argparse
import json
import math
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from mylittlellm.rulecoder import backinfer as bi
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import data2, language, propose2
from seedlm2_ceiling import assert_cases, quick_ok
from seedlm2_real import held_out

BANK = ROOT / "outputs/rule_pilot/teacher/round3"
FEATURES = ["engine_score", "support", "size", "knowledge", "all_args", "engine_rank", "in_engine",
            "log_prob", "log_prob_per_token", "model_rank", "in_model"]


def collect(model, keep, width):
    net, vocabulary = propose2.load(ROOT / "outputs/seedlm" / f"{model}.pt")
    arities = propose2.arity_table(vocabulary)
    rows = []
    for item in json.loads((BANK / "all_tasks.json").read_text()):
        example = bi.parse_example(item["visible_assert"], item["entry_point"])
        if example is None or not example["args"] or len(example["args"]) > data2.MAX_ARGS:
            continue
        if item["source"] == "extra":
            cases = assert_cases(item["asserts"], item["entry_point"])
        else:
            with (BANK / "scoring" / (item["task_id"].split("/")[1] + ".pkl")).open("rb") as stream:
                stored = pickle.load(stream)
            cases = [(inputs, out, "eq") for inputs, out in zip(stored["problem"]["base_input"], stored["oracle"]["base"])]
        if not cases:
            continue
        words = language.words(item["description"])
        literals = bi.description_literals(item["description"])
        named = [k for k, op in bi.OPERATORS.items() if op.named_in(item["description"].lower())]
        prefix = vocabulary.condition(words, literals, example["args"], example["expected"], named)
        found = bi.synthesize(item["description"], item["visible_assert"], item["entry_point"], weights={}, keep=keep)
        engine = (found.get("alternatives") or [found]) if found else []
        beam = propose2.beam_search(net, vocabulary, prefix, len(example["args"]), len(literals), width, arities=arities)
        verified = [p for p, _ in beam if propose2.reproduces(p, example["args"], literals, example["expected"],
                                                               example["mode"])]
        candidates = {}
        for rank, match in enumerate(engine):
            key = tuple(match["program"])
            if not key or "?" in key or key in candidates:
                continue
            candidates[key] = {"engine_score": 2.0 * match["support"] + match["learned"] - 0.3 * match["size"],
                               "support": match["support"], "size": match["size"], "knowledge": float(match["knowledge"]),
                               "all_args": float(match["all_args"]), "engine_rank": rank, "in_engine": 1.0,
                               "model_rank": 99, "in_model": 0.0}
        for rank, program in enumerate(verified):
            key = tuple(program)
            slot = candidates.setdefault(key, {"engine_score": 0.0, "support": 0.0, "size": float(len(program)),
                                               "knowledge": 0.0, "all_args": float(all(f"a{i}" in program for i in
                                                                                     range(len(example["args"])))),
                                               "engine_rank": 99, "in_engine": 0.0})
            slot["model_rank"], slot["in_model"] = rank, 1.0
        task_rows = []
        for key, features in candidates.items():
            try:
                log_prob = propose2.program_log_prob(net, vocabulary, prefix, list(key))
            except (KeyError, IndexError):
                continue
            features = dict(features, log_prob=log_prob, log_prob_per_token=log_prob / (len(key) + 1))
            features.setdefault("model_rank", 99)
            features.setdefault("in_model", 0.0)
            expr, helpers = data2.to_expression(list(key), literals)
            code = bi.to_code(item["entry_point"], len(example["args"]), {"expr": expr, "helpers": helpers})
            task_rows.append({"program": list(key), "features": features,
                              "correct": quick_ok(code, item["entry_point"], cases)})
        part = "holdout" if held_out(item["task_id"]) else "train"
        rows.append({"task_id": item["task_id"], "part": part, "candidates": task_rows})
    return rows


def vectorise(features):
    return [features["engine_score"], features["support"], features["size"], features["knowledge"],
            features["all_args"], math.exp(-features["engine_rank"] / 5.0), features["in_engine"],
            max(features["log_prob"], -50.0) / 10.0, max(features["log_prob_per_token"], -20.0),
            math.exp(-features["model_rank"] / 5.0), features["in_model"]]


def fit(rows, steps=3000, l2=1e-3):
    x = torch.tensor([vectorise(c["features"]) for r in rows if r["part"] == "train" for c in r["candidates"]])
    y = torch.tensor([float(c["correct"]) for r in rows if r["part"] == "train" for c in r["candidates"]])
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps)

    def closure():
        opt.zero_grad()
        logits = ((x - mean) / std) @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y) + l2 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return {"weights": w.detach().tolist(), "bias": float(b.detach()), "mean": mean.tolist(), "std": std.tolist()}


def score(model, features):
    v = [(a - m) / s for a, m, s in zip(vectorise(features), model["mean"], model["std"])]
    return sum(wi * vi for wi, vi in zip(model["weights"], v)) + model["bias"]


def choose(rows, model):
    picks = {}
    for row in rows:
        cands = row["candidates"]
        engine = min((c for c in cands if c["features"]["in_engine"]), key=lambda c: c["features"]["engine_rank"],
                     default=None)
        mdl = min((c for c in cands if c["features"]["in_model"]), key=lambda c: c["features"]["model_rank"], default=None)
        stack = max(cands, key=lambda c: score(model, c["features"]), default=None)
        picks[row["task_id"]] = {"engine": bool(engine and engine["correct"]), "model": bool(mdl and mdl["correct"]),
                                 "stack": bool(stack and stack["correct"]),
                                 "oracle": any(c["correct"] for c in cands)}
    return picks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["collect", "fit"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--keep", type=int, default=40)
    parser.add_argument("--width", type=int, default=32)
    args = parser.parse_args()
    path = ROOT / "outputs/seedlm" / f"stack_rows_{args.model}.json"
    if args.command == "collect":
        torch.set_num_threads(4)
        rows = collect(args.model, args.keep, args.width)
        path.write_text(json.dumps(rows))
        print(len(rows), "tasks,", sum(len(r["candidates"]) for r in rows), "candidates")
        return
    rows = json.loads(path.read_text())
    model = fit(rows)
    report = {"model": args.model, "features": FEATURES, "ranker": model, "parts": {}}
    for part in ("train", "holdout"):
        picks = choose([r for r in rows if r["part"] == part], model)
        report["parts"][part] = {name: sum(p[name] for p in picks.values()) for name in ("engine", "model", "stack", "oracle")}
        report["parts"][part]["tasks"] = len(picks)
    (ROOT / "outputs/seedlm" / f"stack_ranker_{args.model}.json").write_text(json.dumps(model, indent=1))
    (ROOT / "reports/seedlm" / f"stack_{args.model}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["parts"], indent=1))
    print("weights", dict(zip(FEATURES, [round(w, 3) for w in model["weights"]])))


if __name__ == "__main__":
    main()
