"""Unquantized BF16 LoRA: RAFT + answer/explanation SFT, then OPSD-style JSD.

This CPU development run is deliberately small. It is not QLoRA, a reproduction
of the published training recipe, or a held-out accuracy experiment.
"""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mylittlellm.distillation_data import encode_supervised, messages, sha256_file
from mylittlellm.distillation_loss import jsd_loss
from mylittlellm.retrieval import ROOT, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sft-steps", type=int, default=10)
    parser.add_argument("--opsd-steps", type=int, default=2)
    parser.add_argument("--allow-unreviewed-smoke", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--run-probes", action="store_true", help="Generate development answers before/after, not test accuracy")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; cannot overwrite an adapter or run")
    if args.sft_steps < 0 or args.opsd_steps < 0 or args.sft_steps + args.opsd_steps == 0:
        parser.error("At least one positive training step is required")
    rows = read_jsonl(args.examples)
    if not rows:
        parser.error("Empty examples")
    if not args.allow_unreviewed_smoke and any(r["usage"] != "reviewed_training" for r in rows):
        parser.error("Unreviewed public examples require --allow-unreviewed-smoke")
    config = json.loads((ROOT / "config/distillation.json").read_text())
    import torch
    import transformers
    import peft
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["threads"])
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; no automatic backend substitution")
    path = ROOT / config["local_model"]
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    prepared = [(r, task, encode_supervised(tokenizer, r, task, config["max_sequence_tokens"]))
                for r in rows for task in ["answer", "explanation"]]
    args.output.mkdir(parents=True)
    started = time.monotonic()
    print("Loading unquantized BF16 Qwen3-4B; frozen base, rank-4 LoRA", flush=True)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
                torch_dtype=torch.bfloat16, device_map={"": args.device},
                low_cpu_mem_usage=True, attn_implementation="sdpa")
    model = get_peft_model(model, LoraConfig(r=config["lora_rank"], lora_alpha=config["lora_alpha"],
                          target_modules=config["target_modules"], lora_dropout=0., task_type="CAUSAL_LM"))
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=config["learning_rate"])
    log = []

    def ids(values):
        return torch.tensor([values], dtype=torch.long, device=args.device)

    def record(row):
        row["elapsed_seconds"] = round(time.monotonic()-started, 3)
        log.append(row)
        with (args.output / "steps.jsonl").open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False)+"\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def probe(stage):
        if not args.run_probes:
            return
        row = next((r for r in rows if r['id'] == 'demo-khu-2026-recommendation'), rows[0])
        prompt = tokenizer.apply_chat_template(messages(row), tokenize=True,
                         add_generation_prompt=True, enable_thinking=False)
        model.eval()
        begin = time.monotonic()
        with torch.no_grad():
            generated = model.generate(input_ids=ids(prompt), attention_mask=torch.ones_like(ids(prompt)),
                max_new_tokens=192, do_sample=False, pad_token_id=tokenizer.eos_token_id, use_cache=True)
        new_tokens = generated[0, len(prompt):].tolist()
        result = {'stage': stage, 'id': row['id'], 'question': row['question'],
                  'generated_text': tokenizer.decode(new_tokens, skip_special_tokens=True),
                  'generated_tokens': len(new_tokens),
                  'truncated': bool(new_tokens and new_tokens[-1] != tokenizer.eos_token_id),
                  'elapsed_seconds': round(time.monotonic()-begin, 3),
                  'usage': 'exposed_development_example_not_held_out_accuracy'}
        with (args.output / 'development_probes.jsonl').open('a') as stream:
            stream.write(json.dumps(result, ensure_ascii=False)+'\n')
        print(json.dumps(result, ensure_ascii=False), flush=True)

    probe('base')

    for step in range(args.sft_steps):
        row, task, batch = prepared[step % len(prepared)]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids=ids(batch["input_ids"]), labels=ids(batch["labels"]), use_cache=False)
        loss = output.loss
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite SFT loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
        optimizer.step()
        record({"stage": "raft_step_sft", "step": step+1, "id": row["id"], "task": task,
                "loss": loss.item(), "grad_norm": grad_norm.item(),
                "prompt_tokens": batch["prompt_tokens"], "target_tokens": batch["target_tokens"]})
        del output, loss
    if args.sft_steps:
        model.save_pretrained(args.output / "sft_adapter")
        probe('raft_step_sft')

    # The frozen base policy supplies privileged-context teacher logits. It is
    # fixed throughout training, not the updated student pretending to be gold.
    for step in range(args.opsd_steps):
        row = rows[step % len(rows)]
        prompt = tokenizer.apply_chat_template(messages(row), tokenize=True,
                      add_generation_prompt=True, enable_thinking=False)
        teacher_prompt = tokenizer.apply_chat_template(messages(row, privileged=True), tokenize=True,
                      add_generation_prompt=True, enable_thinking=False)
        if max(len(prompt), len(teacher_prompt)) + config["rollout_tokens"] > config["max_sequence_tokens"]:
            raise ValueError("OPSD context would be truncated; increase max_sequence_tokens")
        model.eval()
        with torch.no_grad():
            generation = model.generate(input_ids=ids(prompt), attention_mask=torch.ones_like(ids(prompt)),
                max_new_tokens=config["rollout_tokens"], do_sample=True, temperature=1., top_p=1., top_k=0,
                pad_token_id=tokenizer.eos_token_id, use_cache=True)
        rollout = generation[0, len(prompt):].tolist()
        if not rollout:
            raise RuntimeError("Student generated an empty rollout")
        with torch.no_grad(), model.disable_adapter():
            teacher = model(input_ids=ids(teacher_prompt+rollout), use_cache=False).logits[
                0, len(teacher_prompt)-1:len(teacher_prompt)+len(rollout)-1].detach()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        student_output = model(input_ids=ids(prompt+rollout), use_cache=False)
        student = student_output.logits[0, len(prompt)-1:len(prompt)+len(rollout)-1]
        loss = jsd_loss(student, teacher)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite OPSD loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
        optimizer.step()
        record({"stage": "opsd", "step": step+1, "id": row["id"], "jsd_loss": loss.item(),
                "grad_norm": grad_norm.item(), "rollout_tokens": len(rollout),
                "rollout_text": tokenizer.decode(rollout),
                "rollout_truncated": rollout[-1] != tokenizer.eos_token_id,
                "teacher_prompt_tokens": len(teacher_prompt), "student_prompt_tokens": len(prompt)})
        del student_output, student, teacher, loss, generation
    probe('opsd_final')
    model.save_pretrained(args.output / "final_adapter")
    tokenizer.save_pretrained(args.output / "final_adapter")
    delta = sum((p.detach().float()-q.float()).square().sum().item() for p,q in zip(trainable, before))**.5
    report = {"config": config, "device": args.device, "torch": torch.__version__,
              "transformers": transformers.__version__, "peft": peft.__version__,
              "examples_sha256": sha256_file(args.examples), "examples": len(rows),
              "sft_steps": args.sft_steps, "opsd_steps": args.opsd_steps,
              "trainable_parameters": sum(p.numel() for p in trainable),
              "adapter_parameter_delta_l2": delta,
              "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "elapsed_seconds": round(time.monotonic()-started, 3),
              "smoke_only": args.allow_unreviewed_smoke,
              "held_out_accuracy": None, "training_completed": True,
              "teacher": "fixed initial BF16 base; reference solution in teacher context only",
              "limitations": ["Tiny public development data; no independent human review or held-out test",
                              "24-token rollout cap may truncate answers; this is an optimizer smoke test",
                              "RAFT and step-by-step are adapted recipes; no reproduction claim",
                              "Unquantized BF16 LoRA; QLoRA deliberately excluded"]}
    (args.output / "run_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
