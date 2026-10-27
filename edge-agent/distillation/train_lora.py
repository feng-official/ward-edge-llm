#!/usr/bin/env python3
"""BF16 LoRA response-distillation trainer for Qwen2.5-1.5B on one 24GB GPU."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


class ConversationDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int):
        self.rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        messages = self.rows[index]["messages"]
        prompt = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        full = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full, add_special_tokens=False, truncation=True, max_length=self.max_length)["input_ids"]
        labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids):]
        if not any(x != -100 for x in labels):
            labels[-1] = full_ids[-1]
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    length = max(len(x["input_ids"]) for x in batch)
    result = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in batch:
        padding = length - len(item["input_ids"])
        result["input_ids"].append(item["input_ids"] + [pad_id] * padding)
        result["attention_mask"].append(item["attention_mask"] + [0] * padding)
        result["labels"].append(item["labels"] + [-100] * padding)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in result.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--resume-adapter", default="")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    args.output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if args.resume_adapter:
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ))
    model.print_trainable_parameters()
    dataset = ConversationDataset(args.train, tokenizer, args.max_length)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
                        collate_fn=lambda b: collate(b, tokenizer.pad_token_id), num_workers=0)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01)
    updates_per_epoch = math.ceil(len(loader) / args.gradient_accumulation)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(5, int(total_updates * 0.05)), total_updates)
    log_path = args.output / "training_log.jsonl"
    model.train()
    optimizer.zero_grad(set_to_none=True)
    update = 0
    started = time.time()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for step, batch in enumerate(loader, 1):
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss / args.gradient_accumulation
            loss.backward()
            running_loss += loss.item() * args.gradient_accumulation
            if step % args.gradient_accumulation == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                row = {
                    "epoch": epoch + 1,
                    "update": update,
                    "total_updates": total_updates,
                    "loss": running_loss / min(args.gradient_accumulation, step),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "elapsed_seconds": round(time.time() - started, 1),
                    "gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3),
                }
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if update % 10 == 0 or update == total_updates:
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                running_loss = 0.0
        checkpoint = args.output / f"checkpoint-epoch-{epoch + 1}"
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    summary = {
        "base_model": args.model,
        "dataset": str(args.train),
        "samples": len(dataset),
        "epochs": args.epochs,
        "optimizer_updates": total_updates,
        "elapsed_seconds": round(time.time() - started, 1),
        "max_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3),
        "method": "offline teacher-response distillation with BF16 LoRA",
        "resumed_from_adapter": args.resume_adapter or None,
    }
    (args.output / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
