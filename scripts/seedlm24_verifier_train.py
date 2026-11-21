"""Train SeedLM-24's verifier from scratch on candidates mined from coders' beams on training tasks.

Every candidate (a program that reproduces its task's visible example) becomes one row: the task as the
coder reads it, the program, its outputs on five probe inputs, and the label (right on every case or not)
at the verdict position (mylittlellm/seedlm/verifier.py). The vocabulary is a coder's (words and program
tokens) plus the verifier's five tokens. A tenth of the training tasks (by id) is kept out to report the
verifier's accuracy and ranking quality; no measurement set is read.

  python scripts/seedlm24_verifier_train.py --name verifier-1 --data train_s12-a1_part0 train_s12-a2_part1 \\
      --vocabulary s12-a1 [--minutes 25]
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import coder4, model as seedmodel, verifier
from mylittlellm.seedlm.muon import Muon

DATA = ROOT / "outputs/seedlm/v24"


def rows_of(vocabulary, tokens, tasks):
    """[(row, label, task id, candidate index)] for every candidate that encodes."""
    out = []
    for task in tasks:
        literals, args, expected, probes = (ast.literal_eval(task[k]) for k in ("literals", "args", "expected", "probes"))
        for index, candidate in enumerate(task["candidates"]):
            try:
                row = verifier.encode(vocabulary, tokens, task["words"], literals, args, expected, task["named"],
                                      candidate["program"], probes, candidate["behaviour"])
            except KeyError:
                continue
            if row is not None:
                out.append((row, candidate["right"], task.get("id", ""), index))
    return out


def kept_out(task_id):
    return int(hashlib.sha256(("verifier-val:" + task_id).encode()).hexdigest(), 16) % 10 == 0


def ranking_quality(probabilities, items):
    """Among validation tasks with both a right and a wrong candidate: how often the top-scored candidate is
    right, against how often the coder's first candidate is."""
    by_task = {}
    for p, (_, label, task_id, index) in zip(probabilities, items):
        by_task.setdefault(task_id, []).append((index, p, label))
    mixed = [c for c in by_task.values() if any(l for _, _, l in c) and not all(l for _, _, l in c)]
    top = sum(max(c, key=lambda x: (x[1], -x[0]))[2] for c in mixed)
    first = sum(min(c)[2] for c in mixed)
    return {"mixed_tasks": len(mixed), "verifier_top_right": top, "coder_first_right": first}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--vocabulary", required=True, help="a coder whose words and program tokens to use")
    parser.add_argument("--minutes", type=float, default=25)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    started = time.time()
    checkpoint = torch.load(ROOT / "outputs/seedlm" / f"{args.vocabulary}.pt")
    vocabulary = coder4.Vocabulary(checkpoint["vocabulary"]["words"], checkpoint["vocabulary"]["operators"])
    tokens = verifier.Tokens(vocabulary)
    tasks = [t for name in args.data for t in json.loads((DATA / f"{name}.json").read_text())]
    train = rows_of(vocabulary, tokens, [t for t in tasks if not kept_out(t["id"])])
    valid = rows_of(vocabulary, tokens, [t for t in tasks if kept_out(t["id"])])
    positives = [r for r in train if r[1]]
    negatives = [r for r in train if not r[1]]
    print(f"tasks {len(tasks)}; train rows {len(train)} (right {len(positives)}), validation rows {len(valid)}; "
          f"data in {time.time() - started:.0f}s", flush=True)

    config = seedmodel.Config(tokens.size, args.d_model, args.layers, args.heads, verifier.CONTEXT, "orthogonal")
    net = seedmodel.SeedLM(config)
    tables = ("embed", "head", "notes")
    matrices = [p for name, p in net.named_parameters() if p.ndim == 2 and not name.startswith(tables)]
    others = [p for name, p in net.named_parameters() if p.ndim != 2 or name.startswith(tables)]
    optimizers = [Muon(matrices, lr=args.muon_lr, weight_decay=0.01), torch.optim.AdamW(others, lr=args.lr, weight_decay=0.0)]
    base_rates = [[g["lr"] for g in opt.param_groups] for opt in optimizers]
    budget, step, smooth, begin, log = args.minutes * 60, 0, None, time.time(), []

    def evaluate():
        net.eval()
        probabilities = verifier.right_probability(net, tokens, [r[0] for r in valid])
        net.train()
        accuracy = sum((p > 0.5) == r[1] for p, r in zip(probabilities, valid)) / max(1, len(valid))
        return dict(accuracy=round(accuracy, 4), **ranking_quality(probabilities, valid))

    net.train()
    while time.time() - begin < budget:
        # Balanced batches: half right, half wrong candidates.
        batch = rng.sample(positives, args.batch // 2) + rng.sample(negatives, args.batch // 2)
        elapsed = time.time() - begin
        factor = min(1.0, (step + 1) / 100) * max(0.05, 1 - elapsed / budget)
        for opt, rates in zip(optimizers, base_rates):
            for group, rate in zip(opt.param_groups, rates):
                group["lr"] = rate * factor
        inputs, target = verifier.collate([r[0] for r in batch], tokens, [r[1] for r in batch])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = net(inputs, target)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt in optimizers:
            opt.step()
        step += 1
        smooth = loss.item() if smooth is None else 0.98 * smooth + 0.02 * loss.item()
        if step % 100 == 0:
            print(f"step {step} {elapsed:.0f}s loss {smooth:.3f}", flush=True)
        if step % 1000 == 0:
            log.append(dict(step=step, **evaluate()))
            print(json.dumps(log[-1]), flush=True)
    final = evaluate()
    record = {"name": args.name, "arguments": vars(args), "parameters": net.parameter_count(), "steps": step,
              "train_rows": len(train), "train_right": len(positives), "validation_rows": len(valid),
              "curve": log, "final": final}
    torch.save({"state": net.state_dict(), "config": config.to_dict(), "vocabulary": checkpoint["vocabulary"],
                "kind": "verifier"}, ROOT / "outputs/seedlm" / f"{args.name}.pt")
    (ROOT / "reports/seedlm" / f"{args.name}.json").write_text(json.dumps(record, indent=1) + "\n")
    print("FINAL", json.dumps(final), "steps", step)


if __name__ == "__main__":
    main()
