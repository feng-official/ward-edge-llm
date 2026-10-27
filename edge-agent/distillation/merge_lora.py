#!/usr/bin/env python3
"""Merge the LoRA adapter into a standalone Hugging Face model."""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    merged = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(args.output)
    (args.output / "distillation_metadata.json").write_text(json.dumps({
        "base_model": args.base,
        "teacher_model": "Qwen2.5-14B-Instruct-AWQ",
        "method": "offline teacher-response distillation with BF16 LoRA",
        "adapter": args.adapter,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
