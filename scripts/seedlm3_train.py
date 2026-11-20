"""Train SeedLM-3's coder from scratch: descriptions to pyx programs (library operators + Python nodes).

Data, in shares of the training rows:
  - synthetic library programs with descriptions from teacher phrases (as SeedLM-2), a part of them
    sampled operator-first on real-shaped arguments (--bank-rate);
  - real development tasks with a known correct library program (outputs/seedlm/v2/real_dev.json);
  - teacher one-liners for development tasks the engine fails (data/seedlm/oneliners_dev_part*.json),
    training part only;
  - teacher-authored tasks with one-line solutions (data/seedlm/teacher_tasks_*.json), minus those
    the decontamination step dropped, minus a tenth held out to measure.
Validation and test tasks, HumanEval+ and the extra pool's held-out fifth are never trained on.
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

from mylittlellm.rulecoder import backinfer as bi
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import argbank, coder3, data2, language, model as seedmodel, pyx
from mylittlellm.seedlm.muon import Muon
from seedlm2_real import held_out
from seedlm2_train import V2, batches
from seedlm3_oneliner_check import cases_for, tasks as dev_tasks
from seedlm3_task_check import check as check_task

DATA = ROOT / "data/seedlm"


def named_in(description):
    return [k for k, op in bi.OPERATORS.items() if op.named_in(description.lower())]


def teacher_task_items(prefix="teacher_tasks"):
    """Checked, decontaminated teacher-authored tasks (data/seedlm/<prefix>_*.json) with their tokens and outputs."""
    dropped_path = DATA / f"{prefix}_dropped.json"
    dropped = {d["id"] for d in json.loads(dropped_path.read_text())} if dropped_path.exists() else set()
    items = []
    for path in sorted(DATA.glob(f"{prefix}_*.json")):
        if path.name.endswith("_dropped.json"):
            continue
        for task in json.loads(path.read_text()):
            if task.get("id") in dropped:
                continue
            ok, _, details = check_task(task)
            if not ok:
                continue
            items.append(dict(task, tokens=details["tokens"], outputs=details["outputs"]))
    return items


def teacher_held(task_id):
    return int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % 10 == 0


def view_program(expression, n_args, literals, args, expected):
    """The expression's tokens when this example is the visible one (o and o! follow the visible verdict),
    or None if they do not reproduce it."""
    try:
        tokens = pyx.serialize(expression, n_args, literals, expected if isinstance(expected, str) else None)
    except pyx.Unsupported:
        return None
    if len(tokens) > coder3.MAX_PROGRAM or not coder3.reproduces(tokens, args, literals, expected):
        return None
    return tokens


def teacher_rows(vocabulary, items, held, views=1, held_out=None):
    """With views > 1, each of a task's inputs is shown as the example in turn (training rows only).
    `held_out(task)` says which part a task belongs to (by default a tenth of the task ids)."""
    held_out = held_out or (lambda task: teacher_held(task["id"]))
    rows = []
    for task in items:
        if held_out(task) != held:
            continue
        literals = bi.description_literals(task["description"])
        for view in range(min(views, len(task["inputs"]))):
            try:
                args = list(ast.literal_eval(task["inputs"][view]))
                expected = ast.literal_eval(task["outputs"][view])
            except (ValueError, SyntaxError):
                continue
            program = task["tokens"] if view == 0 else view_program(task["expression"], len(args), literals, args,
                                                                   expected)
            if program is None:
                continue
            tokens, start = vocabulary.encode(language.words(task["description"]), literals, args, expected,
                                              named_in(task["description"]), program)
            if len(tokens) <= coder3.MAX_LENGTH:
                rows.append((tokens, start, {"args": args, "literals": literals, "expected": expected,
                                             "program": program, "mode": "eq", "task_id": task["id"]}))
    return rows


def assert_examples(item):
    """Every parsed assert of an extra-pool task (the visible one first); the visible one otherwise."""
    lines = item.get("asserts") or [item["visible_assert"]] if item["source"] == "extra" else [item["visible_assert"]]
    examples = [bi.parse_example(line, item["entry_point"]) for line in lines]
    return [e for e in examples if e is not None and e["args"]]


def oneliner_items():
    """Teacher one-liners for failing development tasks, as pyx, with each task's example."""
    bank, items = dev_tasks(), []
    for path in sorted(DATA.glob("oneliners_dev_part*.json")):
        for row in json.loads(path.read_text()):
            if row["class"] != "oneliner" or not row.get("expression"):
                continue
            item = bank[row["task_id"]]
            example = bi.parse_example(item["visible_assert"], item["entry_point"])
            if example is None or not example["args"]:
                continue
            literals = bi.description_literals(item["description"])
            output = example["expected"] if isinstance(example["expected"], str) else None
            try:
                tokens = pyx.serialize(row["expression"], len(example["args"]), literals, output)
            except pyx.Unsupported:
                continue
            if len(tokens) > coder3.MAX_PROGRAM:
                continue
            part = "holdout" if item["source"] == "extra" and held_out(item["task_id"]) else "train"
            items.append({"task_id": item["task_id"], "part": part, "item": item, "example": example,
                          "literals": literals, "tokens": tokens, "expression": row["expression"]})
    return items


def oneliner_rows(vocabulary, items, part, views=1):
    rows = []
    for entry in items:
        if entry["part"] != part:
            continue
        item = entry["item"]
        examples = [entry["example"]] + ([e for e in assert_examples(item)[1:views]] if views > 1 else [])
        for view, example in enumerate(examples):
            program = entry["tokens"] if view == 0 else view_program(entry["expression"], len(example["args"]),
                                                                     entry["literals"], example["args"],
                                                                     example["expected"])
            if program is None:
                continue
            tokens, start = vocabulary.encode(language.words(item["description"]), entry["literals"], example["args"],
                                              example["expected"], named_in(item["description"]), program)
            if len(tokens) <= coder3.MAX_LENGTH:
                rows.append((tokens, start, {"args": example["args"], "literals": entry["literals"],
                                             "expected": example["expected"], "program": program,
                                             "mode": example["mode"], "task_id": entry["task_id"]}))
    return rows


def real_rows(vocabulary, part, rng=None, named_drop=0.5, views=1):
    """Real development rows; with views > 1, an extra-pool task's other asserts become examples too
    when its program reproduces them."""
    bank = dev_tasks() if views > 1 else {}
    rows = []
    for item in json.loads((V2 / "real_dev.json").read_text()):
        if item["part"] != part:
            continue
        literals, args, expected = (ast.literal_eval(item[k]) for k in ("literals", "args", "expected"))
        examples = [{"args": args, "expected": expected, "mode": item["mode"]}]
        if views > 1 and item["task_id"] in bank:
            for example in assert_examples(bank[item["task_id"]])[1:views]:
                if coder3.reproduces(item["program"], example["args"], literals, example["expected"], example["mode"]):
                    examples.append(example)
        for example in examples:
            named = item["named"] if rng is None or rng.random() >= named_drop else []
            tokens, start = vocabulary.encode(item["words"], literals, example["args"], example["expected"], named,
                                              item["program"])
            if len(tokens) <= coder3.MAX_LENGTH:
                rows.append((tokens, start, {"args": example["args"], "literals": literals,
                                             "expected": example["expected"], "program": item["program"],
                                             "mode": example["mode"], "task_id": item["task_id"]}))
    return rows


@torch.no_grad()
def evaluate(net, vocabulary, grammar, rows, limit=150, width=8):
    net.eval()
    losses = []
    for start in range(0, len(rows), 64):
        inputs, labels = data2.collate(rows[start:start + 64], torch)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            losses.append(net(inputs, labels).item())
    exact = first = anywhere = 0
    subset = rows[:limit]
    for tokens, start, ex in subset:
        found = coder3.beam_search(net, vocabulary, grammar, tokens[:start], len(ex["args"]), len(ex["literals"]),
                                   isinstance(ex["expected"], str), width)
        programs = [p for p, _ in found]
        ok = [coder3.reproduces(p, ex["args"], ex["literals"], ex["expected"], ex.get("mode", "eq")) for p in programs]
        exact += bool(programs) and programs[0] == list(ex["program"])
        first += bool(programs) and ok[0]
        anywhere += any(ok)
    net.train()
    n = max(1, len(subset))
    return {"loss": round(sum(losses) / max(1, len(losses)), 4), "top1_exact": round(exact / n, 3),
            "top1_reproduces": round(first / n, 3), f"beam{width}_reproduces": round(anywhere / n, 3),
            "examples": len(subset)}


@torch.no_grad()
def beyond_eval(net, vocabulary, grammar, items, width=32):
    """Held-out failing development tasks with a teacher one-liner: does the coder's first program that
    reproduces the visible example pass the task's own tests? (The one-liners themselves are not used.)"""
    from seedlm2_ceiling import quick_ok
    net.eval()
    solved = tried = 0
    for entry in items:
        if entry["part"] != "holdout":
            continue
        item, example = entry["item"], entry["example"]
        tried += 1
        prefix = vocabulary.condition(language.words(item["description"]), entry["literals"], example["args"],
                                      example["expected"], named_in(item["description"]))
        found = coder3.beam_search(net, vocabulary, grammar, prefix, len(example["args"]), len(entry["literals"]),
                                   isinstance(example["expected"], str), width)
        for program, _ in found:
            if coder3.reproduces(program, example["args"], entry["literals"], example["expected"], example["mode"]):
                output = example["expected"] if isinstance(example["expected"], str) else None
                code = pyx.to_code(item["entry_point"], len(example["args"]), program, entry["literals"], output)
                solved += quick_ok(code, item["entry_point"], cases_for(item) or [])
                break
    net.train()
    return {"tasks": tried, "solved": solved}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--minutes", type=float, default=75)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=300000)
    parser.add_argument("--bank-rate", type=float, default=0.3)
    parser.add_argument("--real-share", type=float, default=0.1)
    parser.add_argument("--teacher-share", type=float, default=0.15)
    parser.add_argument("--oneliner-share", type=float, default=0.03)
    parser.add_argument("--views", type=int, default=1,
                        help="examples per training task: each of a task's inputs or asserts shown in turn (up to 3)")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant-share", type=float, default=0.0,
                        help="share of SeedLM-5's seeded variants (SeedLM-20: the later data for this coder too)")
    parser.add_argument("--compose-share", type=float, default=0.0, help="share of SeedLM-6's composed tasks")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    started = time.time()
    teacher = teacher_task_items()
    oneliners = oneliner_items()
    from seedlm4_train import variant_held      # here: seedlm4_train imports this module
    variants = teacher_task_items("variant_tasks") if args.variant_share else []
    composed = teacher_task_items("compose_tasks") if args.compose_share else []
    phrases = language.load_phrases()
    real_train_texts = [" ".join(r["words"]) for r in json.loads((V2 / "real_dev.json").read_text()) if r["part"] == "train"]
    keyword_texts = [" ".join(language.keyword_phrases(op)) for op in bi.OPERATORS.values()]
    texts = real_train_texts + keyword_texts + [t["description"] for t in teacher if not teacher_held(t["id"])] \
        + [e["item"]["description"] for e in oneliners if e["part"] == "train"] \
        + [t["description"] for t in variants if not variant_held(t)] \
        + [t["description"] for t in composed if not teacher_held(t["id"])]
    extra_tokens = {s for t in teacher + variants + composed for s in t["tokens"]} | \
        {s for e in oneliners for s in e["tokens"]}
    vocabulary = coder3.Vocabulary(language.build_vocabulary(texts, phrases), coder3.program_tokens(extra_tokens))
    grammar = coder3.Grammar(vocabulary)
    describer = language.Describer(bi.OPERATORS, phrases)
    bank = argbank.load() if args.bank_rate else None
    synthetic = data2.build_synthetic(args.train_size, args.seed + 1, vocabulary, describer, list(bi.OPERATORS.values()),
                                      clean=True, bank=bank, bank_rate=args.bank_rate)
    held_synthetic = data2.build_synthetic(1000, 99991, vocabulary, describer, list(bi.OPERATORS.values()))
    real = real_rows(vocabulary, "train", random.Random(args.seed + 2), views=args.views)
    tasks_train = teacher_rows(vocabulary, teacher, held=False, views=args.views)
    lines_train = oneliner_rows(vocabulary, oneliners, "train", views=args.views)
    variants_train = teacher_rows(vocabulary, variants, held=False, views=args.views, held_out=variant_held)
    composed_train = teacher_rows(vocabulary, composed, held=False, views=args.views)
    n = len(synthetic)
    other = (1 - args.real_share - args.teacher_share - args.oneliner_share - args.variant_share
             - args.compose_share)

    def copies(share, rows):
        return max(1, round(share * n / (other * max(1, len(rows))))) if rows else 0

    train = (synthetic + real * copies(args.real_share, real) + tasks_train * copies(args.teacher_share, tasks_train)
             + lines_train * copies(args.oneliner_share, lines_train)
             + variants_train * copies(args.variant_share, variants_train)
             + composed_train * copies(args.compose_share, composed_train))
    held_real = real_rows(vocabulary, "holdout")
    held_teacher = teacher_rows(vocabulary, teacher, held=True)
    print(f"vocabulary {len(vocabulary)} (words {len(vocabulary.words)}, program tokens {len(vocabulary.operators)}); "
          f"synthetic {n}, real {len(real)} x{copies(args.real_share, real)}, teacher tasks {len(tasks_train)} "
          f"x{copies(args.teacher_share, tasks_train)}, dev one-liners {len(lines_train)} "
          f"x{copies(args.oneliner_share, lines_train)}, variants {len(variants_train)} "
          f"x{copies(args.variant_share, variants_train)}, composed {len(composed_train)} "
          f"x{copies(args.compose_share, composed_train)}; held-out real {len(held_real)}, teacher {len(held_teacher)}, "
          f"dev one-liner tasks {sum(e['part'] == 'holdout' for e in oneliners)}; data in {time.time() - started:.0f}s",
          flush=True)

    config = seedmodel.Config(len(vocabulary), args.d_model, args.layers, args.heads, coder3.MAX_LENGTH, "orthogonal")
    net = seedmodel.SeedLM(config, slice(vocabulary.first_op, len(vocabulary)))
    matrices = [p for name, p in net.named_parameters() if p.ndim == 2 and not name.startswith(("embed", "head"))]
    others = [p for name, p in net.named_parameters() if p.ndim != 2 or name.startswith(("embed", "head"))]
    optimizers = [Muon(matrices, lr=args.muon_lr, weight_decay=0.01), torch.optim.AdamW(others, lr=args.lr, weight_decay=0.0)]
    base_rates = [[g["lr"] for g in opt.param_groups] for opt in optimizers]
    budget, log, step, seen, smooth = args.minutes * 60, [], 0, 0, None
    begin, next_eval, done = time.time(), args.minutes * 60 / 4, False
    # The budget counts training only: beam-search evaluation of pyx programs (up to 32 tokens) is
    # slow enough that counting it would shorten training by a quarter.
    evaluating = 0.0
    while not done:
        for batch in batches(train, args.batch, rng):
            elapsed = time.time() - begin - evaluating
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
                pause = time.time()
                entry = {"step": step, "seconds": round(elapsed), "train_loss": round(smooth, 4),
                         "synthetic": evaluate(net, vocabulary, grammar, held_synthetic, 60),
                         "real_holdout": evaluate(net, vocabulary, grammar, held_real, 60),
                         "teacher_holdout": evaluate(net, vocabulary, grammar, held_teacher, 60)}
                evaluating += time.time() - pause
                log.append(entry)
                print(json.dumps(entry), flush=True)
    final = {"synthetic": evaluate(net, vocabulary, grammar, held_synthetic, 300),
             "real_holdout": evaluate(net, vocabulary, grammar, held_real, len(held_real)),
             "teacher_holdout": evaluate(net, vocabulary, grammar, held_teacher, len(held_teacher)),
             "beyond_holdout": beyond_eval(net, vocabulary, grammar, oneliners)}
    record = {"name": args.name, "kind": "pyx-coder", "arguments": vars(args), "parameters": net.parameter_count(),
              "vocabulary": {"size": len(vocabulary), "words": len(vocabulary.words), "program_tokens": len(vocabulary.operators)},
              "rows": {"synthetic": n, "real": len(real), "teacher_tasks": len(tasks_train), "dev_oneliners": len(lines_train),
                       "teacher_tasks_total": len(teacher)},
              "steps": step, "sequences_seen": seen, "train_seconds": round(time.time() - begin - evaluating),
              "curve": log,
              "final": final, "torch": torch.__version__}
    torch.save({"state": net.state_dict(), "config": config.to_dict(), "vocabulary": vocabulary.to_dict(),
                "kind": "pyx-coder"}, ROOT / "outputs/seedlm" / f"{args.name}.pt")
    (ROOT / "reports/seedlm" / f"{args.name}.json").write_text(json.dumps(record, indent=1) + "\n")
    print("FINAL", json.dumps(final), "steps", step, flush=True)


if __name__ == "__main__":
    main()
