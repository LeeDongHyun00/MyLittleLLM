"""Train SeedLM-35's reviewer from scratch: P(description words | pyx program, visible example).

Candidates of one sketch share its fixed tokens and differ only inside the holes, so a coder's preference
for the structure lifts the right and the wrong fillings together (SeedLM-34). The question that separates
them runs the other way: does this program account for every word of the description? This is
coder-reviewer reranking (Zhang et al., 2022). SeedLM-3's reviewer read operator programs and memorized the
real descriptions it saw 63 times; this one reads pyx programs, with the coders' vocabulary (s12-a1's), so
every candidate the coders score can be read, and it learns from the same training sources as the coders
plus the teacher's paraphrases (other wordings of the same program, against memorizing one wording) and the
sketch tasks. Held-out teacher tasks, the development fifth, validation and test are never used.

A row is BOS PROG <program> LIT <literals> ARGS <arguments> OUT <expected> DESC <words> EOS
(mylittlellm/seedlm/review.py); only the words and EOS are trained on.

  python scripts/seedlm35_review_train.py --name r35-a --minutes 40 --save-every 10
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
from mylittlellm.seedlm import argbank, coder3, coder4, data2, language, model as seedmodel, review
from mylittlellm.seedlm.muon import Muon
from seedlm2_train import batches
from seedlm3_train import (assert_examples, dev_tasks, oneliner_items, teacher_held, teacher_task_items,
                           view_program)
from seedlm4_train import variant_held
from seedlm26_finetune import paraphrases

V2 = ROOT / "outputs/seedlm/v2"


def encode_all(vocabulary, examples):
    """review.encode each example; one with a symbol outside the vocabulary or too long is skipped."""
    rows = []
    for example in examples:
        try:
            tokens, start = review.encode(vocabulary, example["program"], example["literals"], example["args"],
                                          example["expected"], example["words"])
        except KeyError:
            continue
        if len(tokens) <= data2.MAX_LENGTH:
            rows.append((tokens, start, {}))
    return rows


def task_examples(task, words, views=3):
    """Each of a checked teacher-style task's inputs as the visible example, with its program for that view."""
    literals = bi.description_literals(task["description"])
    out = []
    for view in range(min(views, len(task["inputs"]))):
        try:
            args = list(ast.literal_eval(task["inputs"][view]))
            expected = ast.literal_eval(task["outputs"][view])
        except (ValueError, SyntaxError):
            continue
        program = task["tokens"] if view == 0 else view_program(task["expression"], len(args), literals, args,
                                                               expected)
        if program is not None:
            out.append({"program": program, "literals": literals, "args": args, "expected": expected, "words": words})
    return out


def real_examples(item, bank_item, words, views=3):
    literals, args, expected = (ast.literal_eval(item[k]) for k in ("literals", "args", "expected"))
    examples = [{"args": args, "expected": expected, "mode": item["mode"]}]
    if bank_item is not None:
        for example in assert_examples(bank_item)[1:views]:
            if coder3.reproduces(item["program"], example["args"], literals, example["expected"], example["mode"]):
                examples.append(example)
    return [{"program": item["program"], "literals": literals, "args": e["args"], "expected": e["expected"],
             "words": words} for e in examples]


def sources(vocabulary, synthetic_count, seed):
    """{source: rows} of every training source, read in the reviewer's direction."""
    describer = language.Describer(bi.OPERATORS, language.load_phrases())
    synthetic = [dict(example) for _, _, example in data2.build_synthetic(
        synthetic_count, seed + 1, vocabulary, describer, list(bi.OPERATORS.values()), clean=True,
        bank=argbank.load(), bank_rate=0.3)]
    bank = dev_tasks()
    real_items = [r for r in json.loads((V2 / "real_dev.json").read_text()) if r["part"] == "train"]
    real = [e for item in real_items for e in real_examples(item, bank.get(item["task_id"]), item["words"])]
    teacher_items = [t for t in teacher_task_items() if not teacher_held(t["id"])]
    teacher = [e for t in teacher_items for e in task_examples(t, language.words(t["description"]))]
    variants = [e for t in teacher_task_items("variant_tasks") if not variant_held(t)
                for e in task_examples(t, language.words(t["description"]))]
    lines = []
    for entry in oneliner_items():
        if entry["part"] == "train":
            example = entry["example"]
            lines.append({"program": entry["tokens"], "literals": entry["literals"], "args": example["args"],
                          "expected": example["expected"], "words": language.words(entry["item"]["description"])})
    real_by_id = {r["task_id"]: r for r in real_items}
    teacher_by_id = {t["id"]: t for t in teacher_items}
    paraphrased = []
    for task_id, (_, texts) in sorted(paraphrases().items()):
        for text in texts:
            words = language.words(text)
            if task_id in real_by_id:
                paraphrased += real_examples(real_by_id[task_id], bank.get(task_id), words)
            elif task_id in teacher_by_id:
                paraphrased += task_examples(dict(teacher_by_id[task_id], description=text), words)
    sketch = [e for t in teacher_task_items("sketch_tasks") for e in task_examples(t, language.words(t["description"]))]
    return {name: encode_all(vocabulary, examples) for name, examples in
            [("synthetic", synthetic), ("real", real), ("teacher", teacher), ("variant", variants),
             ("oneliner", lines), ("paraphrase", paraphrased), ("sketch", sketch)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--vocabulary-from", default="s12-a1")
    parser.add_argument("--minutes", type=float, default=40)
    parser.add_argument("--save-every", type=float, default=10, help="keep a checkpoint every so many minutes")
    parser.add_argument("--d-model", type=int, default=320)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--synthetic", type=int, default=150000)
    parser.add_argument("--shares", default="real=0.1,teacher=0.2,variant=0.1,oneliner=0.03,paraphrase=0.1,sketch=0.07",
                        help="each natural source's share of the mix; synthetic rows fill the rest")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    started = time.time()
    checkpoint = torch.load(ROOT / "outputs/seedlm" / f"{args.vocabulary_from}.pt", weights_only=False)
    vocabulary = coder4.Vocabulary(checkpoint["vocabulary"]["words"], checkpoint["vocabulary"]["operators"])
    rows = sources(vocabulary, args.synthetic, args.seed)
    shares = {k: float(v) for k, v in (part.split("=") for part in args.shares.split(","))}
    other = 1 - sum(shares.values())
    n = len(rows["synthetic"])
    copies = {name: (max(1, round(shares[name] * n / (other * len(r)))) if r else 0) for name, r in rows.items()
              if name != "synthetic"}
    train = list(rows["synthetic"])
    for name, count in copies.items():
        train += rows[name] * count
    print("rows", {k: len(v) for k, v in rows.items()}, "copies", copies, f"total {len(train)}; "
          f"data in {time.time() - started:.0f}s", flush=True)

    config = seedmodel.Config(len(vocabulary), args.d_model, args.layers, args.heads, data2.MAX_LENGTH, "orthogonal")
    net = seedmodel.SeedLM(config, slice(vocabulary.first_op, len(vocabulary)))
    matrices = [p for n_, p in net.named_parameters() if p.ndim == 2 and not n_.startswith(("embed", "head"))]
    others = [p for n_, p in net.named_parameters() if p.ndim != 2 or n_.startswith(("embed", "head"))]
    optimizers = [Muon(matrices, lr=args.muon_lr, weight_decay=0.01),
                  torch.optim.AdamW(others, lr=args.lr, weight_decay=0.0)]
    base_rates = [[g["lr"] for g in opt.param_groups] for opt in optimizers]
    budget, step, seen, smooth = args.minutes * 60, 0, 0, None
    begin, next_save, done = time.time(), args.save_every * 60 if args.save_every else None, False

    def save(name):
        torch.save({"state": net.state_dict(), "config": config.to_dict(), "vocabulary": checkpoint["vocabulary"],
                    "kind": "reviewer-pyx"}, ROOT / "outputs/seedlm" / f"{name}.pt")

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
            if step % 200 == 0:
                print(f"step {step} {elapsed:.0f}s loss {smooth:.3f}", flush=True)
            if next_save and elapsed >= next_save and elapsed < budget - 60:
                save(f"{args.name}-m{round(next_save / 60)}")
                next_save += args.save_every * 60
    save(args.name)
    record = {"name": args.name, "kind": "reviewer-pyx", "arguments": vars(args), "parameters": net.parameter_count(),
              "rows": {k: len(v) for k, v in rows.items()}, "copies": copies, "train_rows": len(train), "steps": step,
              "sequences_seen": seen, "train_seconds": round(time.time() - begin), "final_loss": round(smooth, 4)}
    (ROOT / "reports/seedlm" / f"{args.name}.json").write_text(json.dumps(record, indent=1) + "\n")
    print("FINAL", json.dumps({k: record[k] for k in ("name", "steps", "train_rows", "final_loss")}), flush=True)


if __name__ == "__main__":
    main()
