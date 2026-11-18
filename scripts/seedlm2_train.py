"""Train SeedLM-2 from scratch: reads the task description, uses the whole operator library and literals.

Data: synthetic programs over every shipped operator with descriptions composed from teacher-written
phrases, mixed with real development tasks (train split and the extra pool's training part) whose
correct program is known. Muon on hidden matrices, orthogonal initialisation, bfloat16 autocast
(the CPU has AMX), fixed wall-clock budget.
"""
import argparse
import ast
import json
import math
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from mylittlellm.rulecoder import backinfer as bi
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import data2, language, model as seedmodel, propose2
from mylittlellm.seedlm.muon import Muon

V2 = ROOT / "outputs/seedlm/v2"


def batches(rows, batch_size, rng):
    order = list(range(len(rows)))
    rng.shuffle(order)
    chunk = batch_size * 40
    for start in range(0, len(order), chunk):
        block = sorted(order[start:start + chunk], key=lambda i: len(rows[i][0]))
        groups = [block[i:i + batch_size] for i in range(0, len(block), batch_size)]
        rng.shuffle(groups)
        for group in groups:
            yield [rows[i] for i in group]


def real_rows(vocabulary, part, rng=None, named_drop=0.5):
    rows = []
    for item in json.loads((V2 / "real_dev.json").read_text()):
        if item["part"] != part:
            continue
        literals, args, expected = (ast.literal_eval(item[k]) for k in ("literals", "args", "expected"))
        named = item["named"] if rng is None or rng.random() >= named_drop else []
        tokens, start = vocabulary.encode(item["words"], literals, args, expected, named, item["program"])
        if len(tokens) <= data2.MAX_LENGTH:
            rows.append((tokens, start, {"args": args, "literals": literals, "expected": expected,
                                         "program": item["program"], "mode": item["mode"], "task_id": item["task_id"]}))
    return rows


def natural_texts():
    """Teacher-written natural descriptions of sampled programs (outputs/seedlm/v2/natural_*), if present."""
    items = []
    for path in sorted(V2.glob("natural_out_*.json")):
        written = json.loads(path.read_text())
        source = json.loads((V2 / path.name.replace("natural_out_", "natural_in_")).read_text())
        for item in source:
            if item["id"] in written:
                items.append(dict(item, description=written[item["id"]]))
    return items


def natural_rows(vocabulary, items, rng, held=False):
    rows = []
    for item in items:
        # A fixed tenth of the natural descriptions is held out to measure the gap to real tasks.
        if (int(item["id"][1:]) % 10 == 0) != held:
            continue
        literals, args, expected = (ast.literal_eval(item[k]) for k in ("literals", "example_args", "example_result"))
        ops = sorted({t for t in item["program"] if "/" in t})
        named = data2.named_for_training(rng, ops, vocabulary)
        tokens, start = vocabulary.encode(language.words(item["description"]), literals, args, expected, named,
                                          item["program"])
        if len(tokens) <= data2.MAX_LENGTH:
            rows.append((tokens, start, {"args": args, "literals": literals, "expected": expected,
                                         "program": item["program"]}))
    return rows


@torch.no_grad()
def evaluate(net, vocabulary, rows, beam_examples=150, width=8):
    net.eval()
    losses = []
    for start in range(0, len(rows), 64):
        inputs, labels = data2.collate(rows[start:start + 64], torch)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            losses.append(net(inputs, labels).item())
    arities = propose2.arity_table(vocabulary)
    exact = greedy = beam = 0
    subset = rows[:beam_examples]
    for tokens, start, ex in subset:
        found = propose2.beam_search(net, vocabulary, tokens[:start], len(ex["args"]), len(ex["literals"]), width,
                                     arities=arities)
        programs = [p for p, _ in found]
        mode = ex.get("mode", "eq")
        exact += bool(programs) and programs[0] == list(ex["program"])
        greedy += bool(programs) and propose2.reproduces(programs[0], ex["args"], ex["literals"], ex["expected"], mode)
        beam += any(propose2.reproduces(p, ex["args"], ex["literals"], ex["expected"], mode) for p in programs)
    net.train()
    n = max(1, len(subset))
    return {"loss": round(sum(losses) / max(1, len(losses)), 4), "top1_exact": round(exact / n, 3),
            "top1_reproduces": round(greedy / n, 3), f"beam{width}_reproduces": round(beam / n, 3), "examples": len(subset)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=300000)
    parser.add_argument("--real-share", type=float, default=0.1, help="fraction of training rows from real tasks")
    parser.add_argument("--natural-share", type=float, default=0.0,
                        help="fraction of training rows from teacher-written natural descriptions")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-from", help="continue from a checkpoint with the same vocabulary")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    phrases = language.load_phrases()
    real_train_texts = [" ".join(r["words"]) for r in json.loads((V2 / "real_dev.json").read_text()) if r["part"] == "train"]
    keyword_texts = [" ".join(language.keyword_phrases(op)) for op in bi.OPERATORS.values()]
    natural = natural_texts() if args.natural_share else []
    natural_text = [n["description"] for n in natural]
    vocabulary = data2.Vocabulary(language.build_vocabulary(real_train_texts + keyword_texts + natural_text, phrases),
                                  list(bi.OPERATORS))
    describer = language.Describer(bi.OPERATORS, phrases)
    operators = list(bi.OPERATORS.values())
    started = time.time()
    synthetic = data2.build_synthetic(args.train_size, args.seed + 1, vocabulary, describer, operators)
    held_synthetic = data2.build_synthetic(1000, 99991, vocabulary, describer, operators)
    real = real_rows(vocabulary, "train", random.Random(args.seed + 2))
    copies = max(1, round(args.real_share * len(synthetic) / ((1 - args.real_share) * max(1, len(real)))))
    nat = natural_rows(vocabulary, natural, random.Random(args.seed + 3)) if natural else []
    nat_copies = max(1, round(args.natural_share * len(synthetic) / max(1, len(nat)))) if nat else 0
    train = synthetic + real * copies + nat * nat_copies
    held_real = real_rows(vocabulary, "holdout")
    held_natural = natural_rows(vocabulary, natural, random.Random(5), held=True) if natural else []
    generation_seconds = time.time() - started
    print(f"vocabulary {len(vocabulary)} (words {len(vocabulary.words)}, operators {len(vocabulary.operators)}), "
          f"phrases for {len(phrases)} operators; synthetic {len(synthetic)}, real {len(real)} x{copies}, "
          f"natural {len(nat)} x{nat_copies}, held-out real {len(held_real)}, held-out natural {len(held_natural)}; "
          f"generated in {generation_seconds:.0f}s", flush=True)

    config = seedmodel.Config(len(vocabulary), args.d_model, args.layers, args.heads, data2.MAX_LENGTH, "orthogonal")
    net = seedmodel.SeedLM(config, slice(vocabulary.first_op, len(vocabulary)))
    if args.init_from:
        checkpoint = torch.load(ROOT / "outputs/seedlm" / f"{args.init_from}.pt")
        net.load_state_dict(checkpoint["state"])
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
                entry = {"step": step, "seconds": round(elapsed), "seen": seen, "train_loss": round(smooth, 4),
                         "synthetic": evaluate(net, vocabulary, held_synthetic, 60),
                         "real_holdout": evaluate(net, vocabulary, held_real, 60)}
                if held_natural:
                    entry["natural_holdout"] = evaluate(net, vocabulary, held_natural, 60)
                log.append(entry)
                print(json.dumps(entry), flush=True)
    final = {"synthetic": evaluate(net, vocabulary, held_synthetic, 300),
             "real_holdout": evaluate(net, vocabulary, held_real, len(held_real))}
    if held_natural:
        final["natural_holdout"] = evaluate(net, vocabulary, held_natural, len(held_natural))
    record = {"name": args.name, "arguments": vars(args), "parameters": net.parameter_count(),
              "vocabulary": {"size": len(vocabulary), "words": len(vocabulary.words), "operators": len(vocabulary.operators)},
              "phrases": len(phrases), "real_train_rows": len(real), "real_copies": copies,
              "natural_train_rows": len(nat), "natural_copies": nat_copies, "natural_holdout_rows": len(held_natural),
              "real_holdout_rows": len(held_real), "data_generation_seconds": round(generation_seconds, 1),
              "steps": step, "sequences_seen": seen, "train_seconds": round(time.time() - begin), "curve": log,
              "final": final, "torch": torch.__version__}
    (ROOT / "outputs/seedlm").mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "config": config.to_dict(), "vocabulary": vocabulary.to_dict()},
               ROOT / "outputs/seedlm" / f"{args.name}.pt")
    (ROOT / "reports/seedlm" / f"{args.name}.json").write_text(json.dumps(record, indent=1) + "\n")
    print("FINAL", json.dumps(final), "steps", step, "seen", seen, flush=True)


if __name__ == "__main__":
    main()
