"""Add new operator rules to a trained SeedLM and measure what it does to the old ones.

The base model never saw the held-out operators. Two choices are crossed:
  which weights move
    naive / replay           : every weight;
    rows  / rows-replay      : only the embedding and output rows of the new operator tokens,
                               started in the directions the existing operator rows use least;
  which programs are shown
    naive / rows             : only programs that use a new operator;
    replay / rows-replay     : an equal mix of those and programs drawn from the distribution
                               the base model was trained on (the generator makes both for free).
"""
import argparse
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from mylittlellm.rulecoder import backinfer as bi
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import data, model as seedmodel
from seedlm_train import batches, evaluate


def with_new_operator(count, seed, vocabulary, new_names):
    """Sequences whose program uses at least one of the new operators."""
    rng, rows = random.Random(seed), []
    new_ops = [bi.OPERATORS[name] for name in new_names]
    everything = list(bi.OPERATORS.values())
    while len(rows) < count:
        # Sampling half of the pool from the new operators keeps rejection cheap.
        pool = new_ops * max(1, len(everything) // (2 * len(new_ops))) + everything
        example = data.synth.sample(rng, pool)
        if example is None or not set(example["ops"]) & set(new_names):
            continue
        tokens, start = vocabulary.encode(example["args"], example["expected"],
                                          data.named_for_training(example, vocabulary, rng), example["program"])
        if len(tokens) <= data.MAX_LENGTH:
            rows.append((tokens, start, example))
    return rows


def least_used_directions(weight, old_rows, count):
    """Unit vectors in the subspace where the existing operator rows have the least energy."""
    _, _, right = torch.linalg.svd(weight[old_rows], full_matrices=False)
    basis = right.flip(0)                     # smallest singular directions first
    return torch.stack([basis[i % basis.size(0)] for i in range(count)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--method", choices=["naive", "rows", "replay", "rows-replay"], required=True)
    parser.add_argument("--row-init", choices=["least-used", "random"], default="least-used",
                        help="where the new rows start; random is the control for the orthogonal start")
    parser.add_argument("--minutes", type=float, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--save-as")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    checkpoint = torch.load(ROOT / "outputs/seedlm" / f"{args.base}.pt")
    vocabulary = data.Vocabulary(checkpoint["operators"])
    new_names = checkpoint["held_out_operators"]
    if not new_names:
        raise SystemExit("The base model was trained on every operator; nothing to add")
    net = seedmodel.SeedLM(seedmodel.Config(**checkpoint["config"]))
    net.load_state_dict(checkpoint["state"])
    old_ops = [op for name, op in bi.OPERATORS.items() if name not in set(new_names)]
    old_eval = data.build(600, 99991, vocabulary, old_ops)
    new_eval = with_new_operator(400, 77771, vocabulary, new_names)
    train = with_new_operator(20000, 5, vocabulary, new_names)
    if args.method in {"replay", "rows-replay"}:
        train = train + data.build(len(train), 6, vocabulary, old_ops)
    before = {"old": evaluate(net, vocabulary, old_eval, 150), "new": evaluate(net, vocabulary, new_eval, 150)}

    new_tokens = torch.tensor([vocabulary.index[name] for name in new_names])
    if args.method in {"rows", "rows-replay"}:
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        old_tokens = torch.tensor([t for n, t in vocabulary.index.items() if n not in set(new_names)])
        with torch.no_grad():
            for weight in (net.embed.weight, net.head.weight):
                scale = weight[old_tokens].norm(dim=1).mean()
                if args.row_init == "least-used":
                    directions = least_used_directions(weight, old_tokens, len(new_names))
                else:
                    directions = torch.nn.functional.normalize(torch.randn(len(new_names), weight.size(1)), dim=1)
                weight[new_tokens] = directions * scale
        for weight in (net.embed.weight, net.head.weight):
            weight.requires_grad_(True)
        keep = torch.zeros(len(vocabulary), 1)
        keep[new_tokens] = 1.0
        parameters = [net.embed.weight, net.head.weight]
    else:
        keep, parameters = None, list(net.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    rng, begin, step = random.Random(0), time.time(), 0
    while time.time() - begin < args.minutes * 60:
        for batch in batches(train, 64, rng):
            if time.time() - begin >= args.minutes * 60:
                break
            inputs, labels = data.collate(batch, torch)
            loss = net(inputs, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if keep is not None:
                # Every row except the new operators' stays exactly as it was.
                net.embed.weight.grad.mul_(keep)
                net.head.weight.grad.mul_(keep)
            optimizer.step()
            step += 1
    after = {"old": evaluate(net, vocabulary, old_eval, 150), "new": evaluate(net, vocabulary, new_eval, 150)}
    record = {"base": args.base, "method": args.method, "row_init": args.row_init if keep is not None else None,
              "new_operators": new_names, "steps": step, "minutes": args.minutes, "before": before, "after": after}
    tag = args.method + ("" if args.row_init == "least-used" else "-" + args.row_init)
    (ROOT / "reports/seedlm" / f"add_rules_{tag}_{args.base}.json").write_text(json.dumps(record, indent=1) + "\n")
    if args.save_as:
        torch.save({"state": net.state_dict(), "config": checkpoint["config"], "operators": vocabulary.operators,
                    "held_out_operators": []}, ROOT / "outputs/seedlm" / f"{args.save_as}.pt")
    print(json.dumps(record["before"]))
    print(json.dumps(record["after"]))


if __name__ == "__main__":
    main()
