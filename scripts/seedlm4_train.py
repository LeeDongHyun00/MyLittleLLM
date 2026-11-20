"""Train SeedLM-4's coder from scratch: the SeedLM-3b data, written as annotated postfix programs.

Every training row of scripts/seedlm3_train.py (synthetic library programs, real development rows,
teacher-authored tasks, development one-liners; same shares, same held-out parts) is rewritten:
the program in postfix order (mylittlellm/seedlm/post.py), each token followed by the kind, the
relation to the expected output and the size of what the partial program computes on the row's
example. Only the program tokens are trained on. Validation and test tasks, HumanEval+ and the extra
pool's held-out fifth are never trained on.

With --variant-share, SeedLM-5's seeded variants join the mix (data/seedlm/variant_tasks_*.json:
teacher-written neighbours of training-pool tasks the engine cannot solve); the variants of a tenth
of the seeds are held out, so the measure on them is on seeds the coder never saw.

SeedLM-6 adds two sources of new compositions (reports/seedlm/seedlm6_protocol.json):
  --graft-share     grafted training programs (scripts/seedlm6_grafts.py) with descriptions composed
                    from words for each piece (mylittlellm/seedlm/pyx_words.py)
  --compose-share   teacher-written tasks that join two training tasks (data/seedlm/compose_tasks_*.json);
                    a tenth of them (by id) is held out

SeedLM-9 (reports/seedlm/seedlm9_protocol.json): with --internal-val, a twentieth of the training tasks
(real rows, teacher tasks, composed tasks; by a hash of the task id, up to 200 rows each) is kept out of training, and with
--checkpoint-minutes the coder is scored on them every so many minutes of training. The best-scoring
checkpoint is saved as <name>.pt and the last one as <name>-last.pt, each with its own report. The
measurement sets are never used to choose.
"""
import hashlib
import argparse
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
from mylittlellm.seedlm import argbank, coder3, coder4, data2, language, model as seedmodel, post
from mylittlellm.seedlm.muon import Muon
from seedlm2_train import V2, batches
from seedlm3_oneliner_check import cases_for
from seedlm3_train import (named_in, oneliner_items, oneliner_rows, real_rows, teacher_held, teacher_rows,
                           teacher_task_items)
from seedlm2_ceiling import quick_ok
from mylittlellm.seedlm import pyx, pyx_words


def convert(vocabulary, rows, blind=False):
    """SeedLM-3 rows (prefix pyx programs) as SeedLM-4 rows (annotated postfix programs); `blind` drops
    every note (the ablation that keeps only the postfix order)."""
    out = []
    for tokens, start, ex in rows:
        try:
            postfix = post.to_postfix(list(ex["program"]))
        except ValueError:
            continue
        if len(postfix) > post.MAX_POST:
            continue
        notes = post.annotated(postfix, ex["args"], ex["literals"], ex["expected"], ex.get("mode", "eq"))
        sequence, side = coder4.encode(vocabulary, tokens[:start], notes)
        if blind:
            side = [coder4.NO_NOTE] * len(side)
        if len(sequence) <= coder4.MAX_LENGTH:
            out.append((sequence, start, dict(ex, postfix=postfix, notes=side)))
    return out


def graft_rows(vocabulary, describer, grafts, rng, views=3):
    """SeedLM-3-style rows (prefix programs) for grafted programs: each of a graft's cases is an example,
    with a description composed from words for the program's pieces (a fresh draw per view)."""
    rows = []
    for graft in grafts:
        for view in range(min(views, len(graft["cases"]))):
            args, expected = graft["cases"][view], graft["outputs"][view]
            words = pyx_words.describe(rng, graft["tokens"], graft["literals"], args, describer=describer)
            tokens, start = vocabulary.encode(words, graft["literals"], args, expected,
                                              named_in(" ".join(words)), graft["tokens"])
            if len(tokens) <= coder3.MAX_LENGTH:
                rows.append((tokens, start, {"args": args, "literals": graft["literals"], "expected": expected,
                                             "program": graft["tokens"], "mode": "eq"}))
    return rows


def internal_val(task_id):
    """A twentieth of the training tasks, kept out of training only to choose a checkpoint (SeedLM-9)."""
    return int(hashlib.sha256(("internal:" + task_id).encode()).hexdigest(), 16) % 20 == 0


def variant_held(task):
    """A tenth of the seeds: all their variants are held out together, so none of them leaks."""
    return int(hashlib.sha256(task.get("seed", task["id"]).encode()).hexdigest(), 16) % 10 == 0


@torch.no_grad()
def evaluate(net, vocabulary, grammar, rows, limit=60, width=8):
    net.eval()
    losses = []
    for start in range(0, len(rows), 64):
        inputs, notes, labels = coder4.collate(rows[start:start + 64])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            losses.append(net(inputs, labels, notes).item())
    exact = first = anywhere = 0
    subset = rows[:limit]
    for tokens, start, ex in subset:
        found = coder4.beam_search(net, vocabulary, grammar, tokens[:start], ex["args"], ex["literals"],
                                   ex["expected"], ex.get("mode", "eq"), width)
        programs = [p for p, _ in found]
        ok = [coder4.reproduces(p, ex["args"], ex["literals"], ex["expected"], ex.get("mode", "eq")) for p in programs]
        exact += bool(programs) and programs[0] == list(ex["postfix"])
        first += bool(programs) and ok[0]
        anywhere += any(ok)
    net.train()
    n = max(1, len(subset))
    return {"loss": round(sum(losses) / max(1, len(losses)), 4), "top1_exact": round(exact / n, 3),
            "top1_reproduces": round(first / n, 3), f"beam{width}_reproduces": round(anywhere / n, 3),
            "examples": len(subset)}


@torch.no_grad()
def beyond_eval(net, vocabulary, grammar, items, width=32):
    """As in scripts/seedlm3_train.py: held-out failing development tasks with a teacher one-liner; does
    the coder's first program that reproduces the visible example pass the task's own tests?"""
    net.eval()
    solved = tried = 0
    for entry in items:
        if entry["part"] != "holdout":
            continue
        item, example = entry["item"], entry["example"]
        tried += 1
        prefix = vocabulary.condition(language.words(item["description"]), entry["literals"], example["args"],
                                      example["expected"], named_in(item["description"]))
        found = coder4.beam_search(net, vocabulary, grammar, prefix, example["args"], entry["literals"],
                                   example["expected"], example["mode"], width)
        for postfix, _ in found:
            if coder4.reproduces(postfix, example["args"], entry["literals"], example["expected"], example["mode"]):
                output = example["expected"] if isinstance(example["expected"], str) else None
                code = pyx.to_code(item["entry_point"], len(example["args"]), post.to_prefix(postfix),
                                   entry["literals"], output)
                solved += quick_ok(code, item["entry_point"], cases_for(item) or [])
                break
    net.train()
    return {"tasks": tried, "solved": solved}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--minutes", type=float, default=120)
    parser.add_argument("--d-model", type=int, default=320)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=300000)
    parser.add_argument("--bank-rate", type=float, default=0.3)
    parser.add_argument("--real-share", type=float, default=0.1)
    parser.add_argument("--teacher-share", type=float, default=0.15)
    parser.add_argument("--oneliner-share", type=float, default=0.03)
    parser.add_argument("--variant-share", type=float, default=0.0,
                        help="share of SeedLM-5's seeded variants (0: the SeedLM-4 mix, vocabulary included)")
    parser.add_argument("--graft-share", type=float, default=0.0, help="share of grafted programs (SeedLM-6)")
    parser.add_argument("--graft-count", type=int, default=20000)
    parser.add_argument("--compose-share", type=float, default=0.0, help="share of composed tasks (SeedLM-6)")
    parser.add_argument("--internal-val", action="store_true",
                        help="keep a twentieth of the training tasks out of training, to choose a checkpoint")
    parser.add_argument("--checkpoint-minutes", type=float, default=0.0,
                        help="score the internal validation rows every so many training minutes (0: never)")
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="tiny data and a few steps, to check the pipeline")
    parser.add_argument("--blind", action="store_true", help="ablation: postfix programs without notes")
    parser.add_argument("--threads", type=int, default=4,
                        help="torch threads; two 4-thread jobs on 4 cores slow small decoding steps many times over")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    if args.smoke:
        args.train_size, args.minutes = 2000, 1.0

    started = time.time()
    teacher = teacher_task_items()
    variants = teacher_task_items("variant_tasks") if args.variant_share else []
    composed = teacher_task_items("compose_tasks") if args.compose_share else []
    if args.graft_share:
        from seedlm6_grafts import load_or_make     # here: seedlm6_grafts imports this module
        grafts = load_or_make(args.graft_count)
    else:
        grafts = []
    oneliners = oneliner_items()
    phrases = language.load_phrases()
    real_train_texts = [" ".join(r["words"]) for r in json.loads((V2 / "real_dev.json").read_text()) if r["part"] == "train"]
    keyword_texts = [" ".join(language.keyword_phrases(op)) for op in bi.OPERATORS.values()]
    texts = real_train_texts + keyword_texts + [t["description"] for t in teacher if not teacher_held(t["id"])] \
        + [e["item"]["description"] for e in oneliners if e["part"] == "train"] \
        + [t["description"] for t in variants if not variant_held(t)] \
        + [t["description"] for t in composed if not teacher_held(t["id"])] \
        + ([" ".join(pyx_words.all_words())] if grafts else [])
    extra_tokens = {s for t in teacher + variants + composed for s in t["tokens"]} | \
        {s for e in oneliners for s in e["tokens"]} | {s for g in grafts for s in g["tokens"]}
    vocabulary = coder4.Vocabulary(language.build_vocabulary(texts, phrases), coder4.program_tokens(extra_tokens))
    grammar = coder4.Grammar(vocabulary)
    describer = language.Describer(bi.OPERATORS, phrases)
    bank = argbank.load() if args.bank_rate else None
    synthetic = convert(vocabulary, data2.build_synthetic(args.train_size, args.seed + 1, vocabulary, describer,
                                                          list(bi.OPERATORS.values()), clean=True, bank=bank,
                                                          bank_rate=args.bank_rate), args.blind)
    held_synthetic = convert(vocabulary, data2.build_synthetic(1000, 99991, vocabulary, describer,
                                                               list(bi.OPERATORS.values())), args.blind)
    real = convert(vocabulary, real_rows(vocabulary, "train", random.Random(args.seed + 2), views=args.views), args.blind)
    tasks_train = convert(vocabulary, teacher_rows(vocabulary, teacher, held=False, views=args.views), args.blind)
    lines_train = convert(vocabulary, oneliner_rows(vocabulary, oneliners, "train", views=args.views), args.blind)
    variants_train = convert(vocabulary, teacher_rows(vocabulary, variants, held=False, views=args.views,
                                                      held_out=variant_held), args.blind)
    composed_train = convert(vocabulary, teacher_rows(vocabulary, composed, held=False, views=args.views), args.blind)
    graft_train = convert(vocabulary, graft_rows(vocabulary, describer, grafts, random.Random(args.seed + 6),
                                                 args.views), args.blind)
    internal = {}
    if args.internal_val:
        def split(rows):
            return ([r for r in rows if not internal_val(r[2].get("task_id", ""))],
                    [r for r in rows if internal_val(r[2].get("task_id", ""))])
        real, internal["real"] = split(real)
        tasks_train, internal["teacher"] = split(tasks_train)
        composed_train, internal["composed"] = split(composed_train)
        internal = {k: v[:200] for k, v in internal.items() if v}     # every internal row (about 440)
    n = len(synthetic)
    other = (1 - args.real_share - args.teacher_share - args.oneliner_share - args.variant_share
             - args.graft_share - args.compose_share)

    def copies(share, rows):
        return max(1, round(share * n / (other * max(1, len(rows))))) if rows else 0

    train = (synthetic + real * copies(args.real_share, real) + tasks_train * copies(args.teacher_share, tasks_train)
             + lines_train * copies(args.oneliner_share, lines_train)
             + variants_train * copies(args.variant_share, variants_train)
             + composed_train * copies(args.compose_share, composed_train)
             + graft_train * copies(args.graft_share, graft_train))
    held_real = convert(vocabulary, real_rows(vocabulary, "holdout"), args.blind)
    held_teacher = convert(vocabulary, teacher_rows(vocabulary, teacher, held=True), args.blind)
    held_variants = convert(vocabulary, teacher_rows(vocabulary, variants, held=True, held_out=variant_held),
                            args.blind)
    held_composed = convert(vocabulary, teacher_rows(vocabulary, composed, held=True), args.blind)
    print(f"vocabulary {len(vocabulary)} (words {len(vocabulary.words)}, program tokens "
          f"{len(vocabulary.operators)}, notes {len(coder4.NOTE_INDEX)}); synthetic {n}, real {len(real)} x{copies(args.real_share, real)}, "
          f"teacher tasks {len(tasks_train)} x{copies(args.teacher_share, tasks_train)}, dev one-liners "
          f"{len(lines_train)} x{copies(args.oneliner_share, lines_train)}, variants {len(variants_train)} "
          f"x{copies(args.variant_share, variants_train)}, composed {len(composed_train)} "
          f"x{copies(args.compose_share, composed_train)}, grafts {len(graft_train)} "
          f"x{copies(args.graft_share, graft_train)}; held-out real {len(held_real)}, variants {len(held_variants)}, "
          f"composed {len(held_composed)}, internal validation {({k: len(v) for k, v in internal.items()})}, "
          f"teacher {len(held_teacher)}, dev one-liner tasks {sum(e['part'] == 'holdout' for e in oneliners)}; "
          f"mean length {sum(len(r[0]) for r in synthetic[:5000]) / max(1, min(5000, n)):.0f}; "
          f"data in {time.time() - started:.0f}s", flush=True)

    config = seedmodel.Config(len(vocabulary), args.d_model, args.layers, args.heads, coder4.MAX_LENGTH, "orthogonal",
                              n_notes=len(coder4.NOTE_INDEX))
    net = seedmodel.SeedLM(config, slice(vocabulary.first_op, len(vocabulary)))
    net.blind = args.blind
    tables = ("embed", "head", "notes")     # embedding tables are not orthogonalised by Muon
    matrices = [p for name, p in net.named_parameters() if p.ndim == 2 and not name.startswith(tables)]
    others = [p for name, p in net.named_parameters() if p.ndim != 2 or name.startswith(tables)]
    optimizers = [Muon(matrices, lr=args.muon_lr, weight_decay=0.01), torch.optim.AdamW(others, lr=args.lr, weight_decay=0.0)]
    base_rates = [[g["lr"] for g in opt.param_groups] for opt in optimizers]
    budget, log, step, seen, smooth = args.minutes * 60, [], 0, 0, None
    checkpoints, best_state, best_score = [], None, None
    next_checkpoint = args.checkpoint_minutes * 60 if args.checkpoint_minutes and internal else None
    begin, next_eval, done = time.time(), args.minutes * 60 / 4, False
    evaluating = 0.0        # beam-search evaluation time is not counted against the training budget

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
            inputs, notes, labels = coder4.collate(batch)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss = net(inputs, labels, notes)
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
            if elapsed >= next_eval and not args.smoke:
                next_eval += budget / 4
                pause = time.time()
                entry = {"step": step, "seconds": round(elapsed), "train_loss": round(smooth, 4),
                         "synthetic": evaluate(net, vocabulary, grammar, held_synthetic, 60),
                         "real_holdout": evaluate(net, vocabulary, grammar, held_real, 60),
                         "teacher_holdout": evaluate(net, vocabulary, grammar, held_teacher, 60)}
                if held_variants:
                    entry["variant_holdout"] = evaluate(net, vocabulary, grammar, held_variants, 60)
                if held_composed:
                    entry["composed_holdout"] = evaluate(net, vocabulary, grammar, held_composed, 60)
                evaluating += time.time() - pause
                log.append(entry)
                print(json.dumps(entry), flush=True)
            if next_checkpoint is not None and elapsed >= next_checkpoint:
                next_checkpoint += args.checkpoint_minutes * 60
                pause = time.time()
                scores = {k: evaluate(net, vocabulary, grammar, rows, len(rows))["top1_reproduces"]
                          for k, rows in internal.items()}
                score = sum(scores.values()) / len(scores)
                checkpoints.append({"minutes": round(elapsed / 60, 1), "step": step, "internal": scores,
                                    "score": round(score, 4)})
                if best_score is None or score > best_score:       # ties keep the earlier checkpoint
                    best_score = score
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                    checkpoints[-1]["best_so_far"] = True
                evaluating += time.time() - pause
                print("CHECKPOINT", json.dumps(checkpoints[-1]), flush=True)
    limit = 20 if args.smoke else None

    def finish(name, selected=None):
        final = {"synthetic": evaluate(net, vocabulary, grammar, held_synthetic, limit or 300),
                 "real_holdout": evaluate(net, vocabulary, grammar, held_real, limit or len(held_real)),
                 "teacher_holdout": evaluate(net, vocabulary, grammar, held_teacher, limit or len(held_teacher)),
                 "beyond_holdout": beyond_eval(net, vocabulary, grammar, oneliners)}
        if held_variants:
            final["variant_holdout"] = evaluate(net, vocabulary, grammar, held_variants, limit or len(held_variants))
        if held_composed:
            final["composed_holdout"] = evaluate(net, vocabulary, grammar, held_composed, limit or len(held_composed))
        record = {"name": name, "kind": "post-coder", "arguments": vars(args), "parameters": net.parameter_count(),
                  "vocabulary": {"size": len(vocabulary), "words": len(vocabulary.words),
                                 "program_tokens": len(vocabulary.operators)},
                  "rows": {"synthetic": n, "real": len(real), "teacher_tasks": len(tasks_train),
                           "dev_oneliners": len(lines_train), "teacher_tasks_total": len(teacher),
                           "variants": len(variants_train), "variant_tasks_total": len(variants),
                           "composed": len(composed_train), "composed_tasks_total": len(composed),
                           "grafts": len(graft_train), "graft_programs": len(grafts),
                           "internal_validation": {k: len(v) for k, v in internal.items()}},
                  "steps": step, "sequences_seen": seen, "train_seconds": round(time.time() - begin - evaluating),
                  "curve": log, "checkpoints": checkpoints, "selected": selected, "final": final,
                  "torch": torch.__version__}
        if not args.smoke:
            torch.save({"state": net.state_dict(), "config": config.to_dict(), "vocabulary": vocabulary.to_dict(),
                        "kind": "post-coder", "blind": args.blind}, ROOT / "outputs/seedlm" / f"{name}.pt")
            (ROOT / "reports/seedlm" / f"{name}.json").write_text(json.dumps(record, indent=1) + "\n")
        print("FINAL", name, json.dumps(final), "steps", step, flush=True)

    if best_state is None:
        finish(args.name)
    else:
        finish(args.name + "-last", selected="last checkpoint (end of training)")
        net.load_state_dict(best_state)
        chosen = max(checkpoints, key=lambda c: (c["score"], -c["minutes"]))
        finish(args.name, selected=f"best internal-validation checkpoint: {chosen['minutes']} minutes, "
                                   f"step {chosen['step']}, score {chosen['score']}")

if __name__ == "__main__":
    main()
