"""Train SeedLM-3's reviewer from scratch: P(description words | program, example).

Same data sources as SeedLM-2 (scripts/seedlm2_train.py), read in the other direction: synthetic
programs with descriptions composed from teacher phrases, teacher-written natural descriptions of
sampled programs, and real development tasks with a known correct program. Validation and test
tasks are never used. During training the reviewer is measured on
  - the description loss of the extra pool's held-out real rows;
  - ranking: on the held-out fifth's candidate sets (the engine's matches and SeedLM-2's verified
    proposals, labelled by each task's tests; outputs/seedlm/stack_rows_<coder>.json), how often
    the candidate the reviewer likes best is correct.
"""
import argparse
import ast
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

from mylittlellm.rulecoder import backinfer as bi
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import argbank, data2, language, model as seedmodel, review
from mylittlellm.seedlm.muon import Muon
from seedlm2_train import V2, batches, natural_texts

BANK = ROOT / "outputs/rule_pilot/teacher/round3"


def real_rows(vocabulary, part):
    rows = []
    for item in json.loads((V2 / "real_dev.json").read_text()):
        if item["part"] != part:
            continue
        literals, args, expected = (ast.literal_eval(item[k]) for k in ("literals", "args", "expected"))
        tokens, start = review.encode(vocabulary, item["program"], literals, args, expected, item["words"])
        if len(tokens) <= data2.MAX_LENGTH:
            rows.append((tokens, start, {"task_id": item["task_id"]}))
    return rows


def natural_rows(vocabulary, items):
    rows = []
    for item in items:
        if int(item["id"][1:]) % 10 == 0:      # the tenth SeedLM-2 held out stays out here too
            continue
        literals, args, expected = (ast.literal_eval(item[k]) for k in ("literals", "example_args", "example_result"))
        tokens, start = review.encode(vocabulary, item["program"], literals, args, expected,
                                      language.words(item["description"]))
        if len(tokens) <= data2.MAX_LENGTH:
            rows.append((tokens, start, {}))
    return rows


def candidate_sets(coder, part="holdout"):
    """The stacked ranker's candidate sets, with what the reviewer needs to read each task."""
    path = ROOT / "outputs/seedlm" / f"stack_rows_{coder}.json"
    if not path.exists():
        return []
    items = {i["task_id"]: i for i in json.loads((BANK / "all_tasks.json").read_text())}
    sets = []
    for row in json.loads(path.read_text()):
        if row["part"] != part or not row["candidates"]:
            continue
        item = items[row["task_id"]]
        example = bi.parse_example(item["visible_assert"], item["entry_point"])
        sets.append({"task_id": row["task_id"], "words": language.words(item["description"]),
                     "literals": bi.description_literals(item["description"]), "args": example["args"],
                     "expected": example["expected"], "candidates": row["candidates"]})
    return sets


@torch.no_grad()
def rank_eval(net, vocabulary, sets):
    """Correct picks by reviewer score, by the engine's first match and by the coder's first proposal."""
    net.eval()
    counts = {"tasks": len(sets), "review": 0, "engine": 0, "coder": 0, "oracle": 0, "coder_plus_review": 0}
    for entry in sets:
        cands = entry["candidates"]
        programs = [c["program"] for c in cands]
        found = review.scores(net, vocabulary, programs, entry["literals"], entry["args"], entry["expected"],
                              entry["words"])
        scored = [(s, c) for s, c in zip(found, cands) if s is not None]
        if scored:
            counts["review"] += max(scored, key=lambda x: x[0])[1]["correct"]
            counts["coder_plus_review"] += max(scored, key=lambda x: x[0] + x[1]["features"]["log_prob"])[1]["correct"]
        engine = [c for c in cands if c["features"]["in_engine"]]
        coder = [c for c in cands if c["features"]["in_model"]]
        counts["engine"] += bool(engine) and min(engine, key=lambda c: c["features"]["engine_rank"])["correct"]
        counts["coder"] += bool(coder) and min(coder, key=lambda c: c["features"]["model_rank"])["correct"]
        counts["oracle"] += any(c["correct"] for c in cands)
    net.train()
    return counts


@torch.no_grad()
def loss_eval(net, rows):
    net.eval()
    losses = []
    for start in range(0, len(rows), 64):
        inputs, labels = data2.collate(rows[start:start + 64], torch)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            losses.append(net(inputs, labels).item())
    net.train()
    return round(sum(losses) / max(1, len(losses)), 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--minutes", type=float, default=60)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=300000)
    parser.add_argument("--real-share", type=float, default=0.1)
    parser.add_argument("--natural-share", type=float, default=0.1)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--bank-rate", type=float, default=0.0,
                        help="share of synthetic programs sampled operator-first on real-shaped arguments")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--coder", default="s2-main", help="whose candidate sets to rank during training")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    phrases = language.load_phrases()
    real_train_texts = [" ".join(r["words"]) for r in json.loads((V2 / "real_dev.json").read_text()) if r["part"] == "train"]
    keyword_texts = [" ".join(language.keyword_phrases(op)) for op in bi.OPERATORS.values()]
    natural = natural_texts() if args.natural_share else []
    vocabulary = data2.Vocabulary(language.build_vocabulary(real_train_texts + keyword_texts
                                                            + [n["description"] for n in natural], phrases),
                                  list(bi.OPERATORS))
    describer = language.Describer(bi.OPERATORS, phrases)
    started = time.time()
    synthetic = []
    bank = argbank.load() if args.bank_rate else None
    for _, _, example in data2.build_synthetic(args.train_size, args.seed + 1, vocabulary, describer,
                                               list(bi.OPERATORS.values()), clean=args.clean, bank=bank,
                                               bank_rate=args.bank_rate):
        tokens, start = review.encode(vocabulary, example["program"], example["literals"], example["args"],
                                      example["expected"], example["words"])
        if len(tokens) <= data2.MAX_LENGTH:
            synthetic.append((tokens, start, {}))
    real = real_rows(vocabulary, "train")
    copies = max(1, round(args.real_share * len(synthetic) / ((1 - args.real_share) * max(1, len(real)))))
    nat = natural_rows(vocabulary, natural) if natural else []
    nat_copies = max(1, round(args.natural_share * len(synthetic) / max(1, len(nat)))) if nat else 0
    train = synthetic + real * copies + nat * nat_copies
    held_real = real_rows(vocabulary, "holdout")
    sets = candidate_sets(args.coder)
    print(f"vocabulary {len(vocabulary)}; synthetic {len(synthetic)}, real {len(real)} x{copies}, natural {len(nat)} "
          f"x{nat_copies}; held-out real {len(held_real)}, ranking sets {len(sets)}; "
          f"generated in {time.time() - started:.0f}s", flush=True)

    config = seedmodel.Config(len(vocabulary), args.d_model, args.layers, args.heads, data2.MAX_LENGTH, "orthogonal")
    net = seedmodel.SeedLM(config, slice(vocabulary.first_op, len(vocabulary)))
    matrices = [p for n, p in net.named_parameters() if p.ndim == 2 and not n.startswith(("embed", "head"))]
    others = [p for n, p in net.named_parameters() if p.ndim != 2 or n.startswith(("embed", "head"))]
    optimizers = [Muon(matrices, lr=args.muon_lr, weight_decay=0.01), torch.optim.AdamW(others, lr=args.lr, weight_decay=0.0)]
    base_rates = [[g["lr"] for g in opt.param_groups] for opt in optimizers]
    budget, log, step, seen, smooth = args.minutes * 60, [], 0, 0, None
    begin, next_eval, done = time.time(), args.minutes * 60 / 4, False
    while not done:
        for batch in batches(train, args.batch, rng):
            elapsed = time.time() - begin
            if elapsed >= budget:
                done = True
                break
            factor = min(1.0, (step + 1) / 100) * max(0.05, 1 - elapsed / budget)
            for opt, rates in zip(optimizers, base_rates):
                for group, rate in zip(opt.param_groups, rates):
                    group["lr"] = rate * factor
            inputs, labels = data2.collate(batch, torch)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss = net(inputs, labels)
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            loss.backward()
            for opt in optimizers:
                opt.step()
            step += 1
            seen += len(batch)
            smooth = loss.item() if smooth is None else 0.98 * smooth + 0.02 * loss.item()
            if step % 100 == 0:
                print(f"step {step} elapsed {elapsed:.0f}s seen {seen} loss {smooth:.3f}", flush=True)
            if elapsed >= next_eval:
                next_eval += budget / 4
                entry = {"step": step, "seconds": round(elapsed), "train_loss": round(smooth, 4),
                         "real_holdout_loss": loss_eval(net, held_real), "ranking": rank_eval(net, vocabulary, sets)}
                log.append(entry)
                print(json.dumps(entry), flush=True)
    final = {"real_holdout_loss": loss_eval(net, held_real), "ranking": rank_eval(net, vocabulary, sets)}
    record = {"name": args.name, "kind": "reviewer", "arguments": vars(args), "parameters": net.parameter_count(),
              "vocabulary": {"size": len(vocabulary), "words": len(vocabulary.words), "operators": len(vocabulary.operators)},
              "real_train_rows": len(real), "real_copies": copies, "natural_train_rows": len(nat),
              "natural_copies": nat_copies, "steps": step, "sequences_seen": seen,
              "train_seconds": round(time.time() - begin), "curve": log, "final": final, "torch": torch.__version__}
    torch.save({"state": net.state_dict(), "config": config.to_dict(), "vocabulary": vocabulary.to_dict(),
                "kind": "reviewer"}, ROOT / "outputs/seedlm" / f"{args.name}.pt")
    (ROOT / "reports/seedlm" / f"{args.name}.json").write_text(json.dumps(record, indent=1) + "\n")
    print("FINAL", json.dumps(final), "steps", step, flush=True)


if __name__ == "__main__":
    main()
