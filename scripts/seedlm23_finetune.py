"""Fine-tune a post-coder with or without being told which programs that fit the example are wrong (SeedLM-23).

Both arms start from the same checkpoint and take the same number of steps over the same batches: the
base run's data mix (a fresh synthetic draw, real rows, teacher tasks, one-liners, variants, at the base
run's shares) plus the right program of every mined pair as an ordinary row (scripts/seedlm23_mine.py).
The contrast arm adds, for a batch of mined pairs each step, an unlikelihood term at the pair's branch
point: where the wrong program's postfix first leaves the right one's, the probability the coder gives
the wrong program's token is pushed down, -log(1 - p). The arms differ only in that term.

  python scripts/seedlm23_finetune.py --base s12-a1 --name s23-u1 --contrast 1.0 [--steps 400]
  python scripts/seedlm23_finetune.py --base s12-a1 --name s23-m1 --contrast 0
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
import torch.nn.functional as F

from mylittlellm.rulecoder import backinfer as bi
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import argbank, coder4, data2, language, model as seedmodel
from mylittlellm.seedlm.muon import Muon
from seedlm2_train import batches
from seedlm3_train import oneliner_items, oneliner_rows, real_rows, teacher_rows, teacher_task_items
from seedlm4_train import convert, variant_held

PAIRS = ROOT / "outputs/seedlm/v23"


def pair_rows(vocabulary, pairs):
    """(positive rows, [(negative row, branch index)]) in SeedLM-4's row format; pairs whose programs the
    vocabulary cannot write, or whose postfix does not branch, are left out."""
    positives, negatives = [], []
    for pair in pairs:
        literals, args, expected = (ast.literal_eval(pair[k]) for k in ("literals", "args", "expected"))
        rows = []
        for program in (pair["positive"], pair["negative"]):
            try:
                tokens, start = vocabulary.encode(pair["words"], literals, args, expected, pair["named"], program)
            except KeyError:
                break
            rows.append((tokens, start, {"args": args, "literals": literals, "expected": expected,
                                         "program": list(program), "mode": "eq"}))
        if len(rows) < 2:
            continue
        converted = convert(vocabulary, rows)
        if len(converted) < 2:
            continue
        good, bad = converted
        start = good[1]
        branch = next((i for i in range(start, min(len(good[0]), len(bad[0]))) if good[0][i] != bad[0][i]), None)
        if branch is None:
            continue
        positives.append(good)
        negatives.append((bad, branch))
    return positives, negatives


def unlikelihood(net, items):
    """Mean of -log(1 - p(wrong token)) at each negative row's branch point."""
    rows = [row for row, _ in items]
    inputs, notes, _ = coder4.collate(rows)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = net(inputs, None, notes)
    index = torch.tensor([branch - 1 for _, branch in items])
    target = torch.tensor([row[0][branch] for row, branch in items])
    log_p = F.log_softmax(logits[torch.arange(len(items)), index].float(), dim=-1)
    p = log_p.gather(1, target[:, None]).squeeze(1).exp()
    return -torch.log1p(-p.clamp(max=1 - 1e-6)).mean(), p.mean().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--contrast", type=float, default=1.0, help="weight of the unlikelihood term (0: control)")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--pair-batch", type=int, default=16)
    parser.add_argument("--positive-share", type=float, default=0.1)
    parser.add_argument("--synthetic", type=int, default=60000)
    parser.add_argument("--lr-scale", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
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
    pairs = json.loads((PAIRS / f"pairs_{args.base}.json").read_text())
    positives, negatives = pair_rows(vocabulary, pairs)
    n = len(synthetic)
    shares = {"real": base["real_share"], "teacher": base["teacher_share"], "oneliner": base["oneliner_share"],
              "variant": base.get("variant_share", 0.0), "positive": args.positive_share}
    other = 1 - sum(shares.values())

    def copies(share, rows):
        return max(1, round(share * n / (other * max(1, len(rows))))) if rows and share else 0

    train = (synthetic + real * copies(shares["real"], real) + tasks_train * copies(shares["teacher"], tasks_train)
             + lines_train * copies(shares["oneliner"], lines_train)
             + variants_train * copies(shares["variant"], variants_train)
             + positives * copies(shares["positive"], positives))
    print(f"base {args.base}: rows {len(train)} (synthetic {n}, positives {len(positives)} "
          f"x{copies(shares['positive'], positives)}), pairs {len(negatives)} of {len(pairs)}; data in "
          f"{time.time() - started:.0f}s", flush=True)

    tables = ("embed", "head", "notes")
    matrices = [p for name, p in net.named_parameters() if p.ndim == 2 and not name.startswith(tables)]
    others = [p for name, p in net.named_parameters() if p.ndim != 2 or name.startswith(tables)]
    optimizers = [Muon(matrices, lr=base["muon_lr"] * args.lr_scale, weight_decay=0.01),
                  torch.optim.AdamW(others, lr=base["lr"] * args.lr_scale, weight_decay=0.0)]
    base_rates = [[g["lr"] for g in opt.param_groups] for opt in optimizers]
    pair_rng = random.Random(args.seed + 1)
    net.train()
    step, smooth, smooth_ul, p_log, begin = 0, None, None, [], time.time()
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
        items = pair_rng.sample(negatives, min(args.pair_batch, len(negatives)))   # drawn in both arms alike
        if args.contrast > 0 and items:
            ul, p_mean = unlikelihood(net, items)
            loss = loss + args.contrast * ul
            smooth_ul = ul.item() if smooth_ul is None else 0.98 * smooth_ul + 0.02 * ul.item()
            if step % 50 == 0:
                p_log.append({"step": step, "p_wrong_token": round(p_mean, 4)})
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt in optimizers:
            opt.step()
        step += 1
        smooth = loss.item() if smooth is None else 0.98 * smooth + 0.02 * loss.item()
        if step % 50 == 0:
            print(f"step {step} {time.time() - begin:.0f}s loss {smooth:.3f}"
                  + (f" unlikelihood {smooth_ul:.3f}" if smooth_ul is not None else ""), flush=True)
    net.eval()
    with torch.no_grad():
        _, p_after = unlikelihood(net, negatives[:256])
    record = {"name": args.name, "kind": "post-coder", "base": args.base, "arguments": vars(args),
              "rows": len(train), "positives": len(positives), "pairs": len(negatives), "steps": step,
              "train_seconds": round(time.time() - begin), "p_wrong_token_after": round(p_after, 4),
              "p_log": p_log, "final_loss": round(smooth, 4)}
    torch.save({"state": net.state_dict(), "config": checkpoint["config"], "vocabulary": checkpoint["vocabulary"],
                "kind": "post-coder", "blind": False}, ROOT / "outputs/seedlm" / f"{args.name}.pt")
    (ROOT / "reports/seedlm" / f"{args.name}.json").write_text(json.dumps(record, indent=1) + "\n")
    print("FINAL", json.dumps({k: record[k] for k in ("name", "steps", "pairs", "p_wrong_token_after", "final_loss")}))


if __name__ == "__main__":
    main()
