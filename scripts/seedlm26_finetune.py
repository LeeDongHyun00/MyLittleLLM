"""Fine-tune a post-coder on extra rows of 1,200 training tasks, in their own words or in paraphrases (SeedLM-26).

Both arms start from the same checkpoint and take the same number of steps over the base run's data mix (a
fresh synthetic draw, real rows, teacher tasks, one-liners, variants, at the base run's shares) plus rows of
the same 1,200 training tasks at a tenth of the mix (reports/seedlm/seedlm26_protocol.json):
  original   - each task's own description, twice (to match the count)
  paraphrase - the task's two teacher-written paraphrases (outputs/paraphrase/out_part*.json, only those that
               pass scripts/seedlm26_paraphrase_check.py)
Programs, examples and views are the task's own in both arms; only the words differ.

SeedLM-34 reuses the same loop with two more sources of extra rows (reports/seedlm/seedlm34_protocol.json):
  sketch     - the teacher-written tasks for recurring sketches (data/seedlm/sketch_tasks_*.json, checked and
               decontaminated like every teacher task), every task with its three views
  none       - no extra rows; the base run's shares exactly (the extra share is forced to 0)

  python scripts/seedlm26_finetune.py --base s12-a1 --name s26-p1 --words paraphrase [--steps 400]
  python scripts/seedlm26_finetune.py --base s12-a1 --name s26-o1 --words original
  python scripts/seedlm26_finetune.py --base s12-a1 --name s34-k1 --words sketch
  python scripts/seedlm26_finetune.py --base s12-a1 --name s34-o1 --words none
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
from mylittlellm.seedlm import argbank, coder3, coder4, data2, language, model as seedmodel
from mylittlellm.seedlm.muon import Muon
from seedlm2_train import batches
from seedlm3_train import (assert_examples, dev_tasks, named_in, oneliner_items, oneliner_rows, real_rows,
                           teacher_rows, teacher_task_items)
from seedlm4_train import convert, variant_held
from seedlm26_paraphrase_check import DIR, problems

V2 = ROOT / "outputs/seedlm/v2"


def paraphrases():
    """{task id: (original description, [paraphrase, paraphrase])} for every checked item."""
    out = {}
    for path in sorted(DIR.glob("in_part*.json")):
        tasks = {t["id"]: t for t in json.loads(path.read_text())}
        written = DIR / path.name.replace("in_", "out_")
        if not written.exists():
            continue
        for row in json.loads(written.read_text()):
            task = tasks.get(row.get("id"))
            if task and not problems(task, row.get("paraphrases")):
                out[task["id"]] = (task["description"], row["paraphrases"])
    return out


def real_task_rows(vocabulary, item, bank_item, text, views):
    """real_rows() for one task with its description replaced by `text` (the program and examples kept)."""
    literals, args, expected = (ast.literal_eval(item[k]) for k in ("literals", "args", "expected"))
    examples = [{"args": args, "expected": expected, "mode": item["mode"]}]
    if views > 1 and bank_item is not None:
        for example in assert_examples(bank_item)[1:views]:
            if coder3.reproduces(item["program"], example["args"], literals, example["expected"], example["mode"]):
                examples.append(example)
    rows = []
    for example in examples:
        tokens, start = vocabulary.encode(language.words(text), literals, example["args"], example["expected"],
                                          named_in(text), item["program"])
        if len(tokens) <= coder3.MAX_LENGTH:
            rows.append((tokens, start, {"args": example["args"], "literals": literals, "expected": example["expected"],
                                         "program": item["program"], "mode": example["mode"], "task_id": item["task_id"]}))
    return rows


def extra_rows(vocabulary, words, views):
    if words == "none":
        return [], 0
    if words == "sketch":
        # Every sketch task trains: none of them is an evaluation task, and a held-out tenth would only thin
        # the rare sketches the rows are meant to teach.
        # A task whose program needs a symbol the base vocabulary lacks (a new separator such as '; ') is
        # skipped whole: the fine-tune cannot add embeddings, and half a task's views would tilt its rows.
        rows, used = [], 0
        for task in teacher_task_items("sketch_tasks"):
            try:
                rows += teacher_rows(vocabulary, [task], held=False, views=views, held_out=lambda task: False)
                used += 1
            except KeyError:
                continue
        return convert(vocabulary, rows), used
    found = paraphrases()
    real = {r["task_id"]: r for r in json.loads((V2 / "real_dev.json").read_text()) if r["part"] == "train"}
    bank = dev_tasks() if views > 1 else {}
    teacher = {t["id"]: t for t in teacher_task_items()}
    rows = []
    for task_id, (original, rewritten) in sorted(found.items()):
        texts = [original, original] if words == "original" else list(rewritten)
        for text in texts:
            if task_id in real:
                rows += real_task_rows(vocabulary, real[task_id], bank.get(task_id), text, views)
            elif task_id in teacher:
                rows += teacher_rows(vocabulary, [dict(teacher[task_id], description=text)], held=False, views=views)
    return convert(vocabulary, rows), len(found)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--words", choices=["original", "paraphrase", "sketch", "none"], required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--extra-share", type=float, default=0.1)
    parser.add_argument("--synthetic", type=int, default=60000)
    parser.add_argument("--lr-scale", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.words == "none":
        args.extra_share = 0.0       # the control keeps the base run's shares, not a tenth of nothing
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    started = time.time()
    base = json.loads((ROOT / "reports/seedlm" / f"{args.base}.json").read_text())["arguments"]
    checkpoint = torch.load(ROOT / "outputs/seedlm" / f"{args.base}.pt")
    vocabulary = coder4.Vocabulary(checkpoint["vocabulary"]["words"], checkpoint["vocabulary"]["operators"])
    net = seedmodel.SeedLM(seedmodel.Config(**checkpoint["config"]))
    net.load_state_dict(checkpoint["state"])
    net.blind = False
    describer = language.Describer(bi.OPERATORS, language.load_phrases())
    rng = random.Random(args.seed)
    views = base["views"]
    teacher = teacher_task_items()
    variants = teacher_task_items("variant_tasks") if base.get("variant_share") else []
    synthetic = convert(vocabulary, data2.build_synthetic(args.synthetic, 1000 + base["seed"], vocabulary, describer,
                                                          list(bi.OPERATORS.values()), clean=True, bank=argbank.load(),
                                                          bank_rate=base["bank_rate"]))
    real = convert(vocabulary, real_rows(vocabulary, "train", random.Random(base["seed"] + 2), views=views))
    tasks_train = convert(vocabulary, teacher_rows(vocabulary, teacher, held=False, views=views))
    lines_train = convert(vocabulary, oneliner_rows(vocabulary, oneliner_items(), "train", views=views))
    variants_train = convert(vocabulary, teacher_rows(vocabulary, variants, held=False, views=views,
                                                      held_out=variant_held))
    extra, tasks_with_words = extra_rows(vocabulary, args.words, views)
    n = len(synthetic)
    shares = {"real": base["real_share"], "teacher": base["teacher_share"], "oneliner": base["oneliner_share"],
              "variant": base.get("variant_share", 0.0), "extra": args.extra_share}
    other = 1 - sum(shares.values())

    def copies(share, rows):
        return max(1, round(share * n / (other * max(1, len(rows))))) if rows and share else 0

    train = (synthetic + real * copies(shares["real"], real) + tasks_train * copies(shares["teacher"], tasks_train)
             + lines_train * copies(shares["oneliner"], lines_train)
             + variants_train * copies(shares["variant"], variants_train)
             + extra * copies(shares["extra"], extra))
    print(f"base {args.base}: rows {len(train)} (synthetic {n}, extra {len(extra)} x{copies(shares['extra'], extra)} "
          f"from {tasks_with_words} tasks, words {args.words}); data in {time.time() - started:.0f}s", flush=True)

    tables = ("embed", "head", "notes")
    matrices = [p for name, p in net.named_parameters() if p.ndim == 2 and not name.startswith(tables)]
    others = [p for name, p in net.named_parameters() if p.ndim != 2 or name.startswith(tables)]
    optimizers = [Muon(matrices, lr=base["muon_lr"] * args.lr_scale, weight_decay=0.01),
                  torch.optim.AdamW(others, lr=base["lr"] * args.lr_scale, weight_decay=0.0)]
    base_rates = [[g["lr"] for g in opt.param_groups] for opt in optimizers]
    net.train()
    step, smooth, begin = 0, None, time.time()
    stream = (b for _ in iter(int, 1) for b in batches(train, args.batch, rng))
    while step < args.steps:
        batch = next(stream)
        factor = min(1.0, (step + 1) / 20) * max(0.05, 1 - step / args.steps)
        for opt, rates in zip(optimizers, base_rates):
            for group, rate in zip(opt.param_groups, rates):
                group["lr"] = rate * factor
        inputs, notes, labels = coder4.collate(batch)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = net(inputs, labels, notes)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt in optimizers:
            opt.step()
        step += 1
        smooth = loss.item() if smooth is None else 0.98 * smooth + 0.02 * loss.item()
        if step % 50 == 0:
            print(f"step {step} {time.time() - begin:.0f}s loss {smooth:.3f}", flush=True)
    net.eval()
    record = {"name": args.name, "kind": "post-coder", "base": args.base, "arguments": vars(args), "rows": len(train),
              "extra_rows": len(extra), "tasks_with_words": tasks_with_words, "steps": step,
              "train_seconds": round(time.time() - begin), "final_loss": round(smooth, 4)}
    torch.save({"state": net.state_dict(), "config": checkpoint["config"], "vocabulary": checkpoint["vocabulary"],
                "kind": "post-coder", "blind": False}, ROOT / "outputs/seedlm" / f"{args.name}.pt")
    (ROOT / "reports/seedlm" / f"{args.name}.json").write_text(json.dumps(record, indent=1) + "\n")
    print("FINAL", json.dumps({k: record[k] for k in ("name", "steps", "extra_rows", "tasks_with_words", "final_loss")}))


if __name__ == "__main__":
    main()
