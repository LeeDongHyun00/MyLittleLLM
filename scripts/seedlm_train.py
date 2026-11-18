"""Train SeedLM from scratch on self-generated operator programs for a fixed wall-clock budget (CPU)."""
import argparse
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
from mylittlellm.seedlm import data, model as seedmodel, propose, synth


def batches(rows, batch_size, rng):
    """Shuffle, then group sequences of similar length so little compute goes to padding."""
    order = list(range(len(rows)))
    rng.shuffle(order)
    chunk = batch_size * 40
    for start in range(0, len(order), chunk):
        block = sorted(order[start:start + chunk], key=lambda i: len(rows[i][0]))
        groups = [block[i:i + batch_size] for i in range(0, len(block), batch_size)]
        rng.shuffle(groups)
        for group in groups:
            yield [rows[i] for i in group]


@torch.no_grad()
def evaluate(net, vocabulary, rows, beam_examples=150, width=8):
    net.eval()
    losses = []
    for start in range(0, len(rows), 128):
        inputs, labels = data.collate(rows[start:start + 128], torch)
        losses.append(net(inputs, labels).item())
    arities = propose.arity_table(vocabulary)
    exact = greedy = beam = 0
    subset = rows[:beam_examples]
    for tokens, start, example in subset:
        found = propose.beam_search(net, vocabulary, tokens[:start], len(example["args"]), width, arities=arities)
        programs = [program for program, _ in found]
        exact += bool(programs) and programs[0] == example["program"]

        def reproduces(program):
            try:
                return synth.execute(program, example["args"]) == example["expected"]
            except Exception:
                return False
        greedy += bool(programs) and reproduces(programs[0])
        beam += any(reproduces(program) for program in programs)
    net.train()
    n = len(subset)
    return {"loss": round(sum(losses) / len(losses), 4), "top1_exact": round(exact / n, 3),
            "top1_reproduces": round(greedy / n, 3), f"beam{width}_reproduces": round(beam / n, 3), "examples": n}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--minutes", type=float, default=10)
    parser.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    parser.add_argument("--init", choices=["orthogonal", "default"], default="orthogonal")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--muon-weight-decay", type=float, default=0.01,
                        help="small batches need weight decay on the matrices Muon updates")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=160)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=5)
    parser.add_argument("--train-size", type=int, default=150000)
    parser.add_argument("--eval-size", type=int, default=1000)
    parser.add_argument("--holdout-ops", type=int, default=0,
                        help="operators never shown in training, to be added later as new rules")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    vocabulary = data.Vocabulary()
    names = list(vocabulary.operators)
    held_out = sorted(random.Random(1234).sample(names, args.holdout_ops)) if args.holdout_ops else []
    operators = [op for name, op in bi.OPERATORS.items() if name not in set(held_out)]
    started = time.time()
    train = data.build(args.train_size, args.seed + 1, vocabulary, operators)
    held = data.build(args.eval_size, 99991, vocabulary, operators)     # same held-out set for every run
    generation_seconds = time.time() - started

    config = seedmodel.Config(len(vocabulary), args.d_model, args.layers, args.heads, data.MAX_LENGTH, args.init)
    net = seedmodel.SeedLM(config, slice(data.FIRST_OP, len(vocabulary)))
    matrices = [p for n, p in net.named_parameters() if p.ndim == 2 and not n.startswith(("embed", "head"))]
    others = [p for n, p in net.named_parameters() if p.ndim != 2 or n.startswith(("embed", "head"))]
    if args.optimizer == "muon":
        from mylittlellm.seedlm.muon import Muon
        optimizers = [Muon(matrices, lr=args.muon_lr, weight_decay=args.muon_weight_decay),
                      torch.optim.AdamW(others, lr=args.lr, weight_decay=0.0)]
    else:
        optimizers = [torch.optim.AdamW(matrices + others, lr=args.lr, weight_decay=0.0)]
    base_rates = [[group["lr"] for group in opt.param_groups] for opt in optimizers]

    budget, log, step, seen, smooth = args.minutes * 60, [], 0, 0, None
    begin = time.time()
    next_eval = budget / 4
    done = False
    while not done:
        for batch in batches(train, args.batch, rng):
            elapsed = time.time() - begin
            if elapsed >= budget:
                done = True
                break
            # Warm up for 100 steps, then decay linearly with the wall-clock budget.
            factor = min(1.0, (step + 1) / 100) * max(0.05, 1 - elapsed / budget)
            for opt, rates in zip(optimizers, base_rates):
                for group, rate in zip(opt.param_groups, rates):
                    group["lr"] = rate * factor
            inputs, labels = data.collate(batch, torch)
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
                         **evaluate(net, vocabulary, held, beam_examples=60)}
                log.append(entry)
                print(json.dumps(entry), flush=True)
    final = evaluate(net, vocabulary, held, beam_examples=300)
    record = {"name": args.name, "arguments": vars(args), "parameters": net.parameter_count(),
              "vocabulary": len(vocabulary), "held_out_operators": held_out,
              "data_generation_seconds": round(generation_seconds, 1), "steps": step, "sequences_seen": seen,
              "train_seconds": round(time.time() - begin), "curve": log, "final": final,
              "torch": torch.__version__, "threads": args.threads}
    out = ROOT / "outputs/seedlm"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "config": config.to_dict(), "operators": vocabulary.operators,
                "held_out_operators": held_out}, out / f"{args.name}.pt")
    reports = ROOT / "reports/seedlm"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"{args.name}.json").write_text(json.dumps(record, indent=1) + "\n")
    print("FINAL", json.dumps(final), "steps", step, "seen", seen, flush=True)


if __name__ == "__main__":
    main()
