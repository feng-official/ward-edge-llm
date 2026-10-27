#!/usr/bin/env python3
"""Combine two same-architecture LoRA adapters by rank concatenation.

The resulting adapter represents:
    base + weight_a * delta_a + weight_b * delta_b

This avoids merging a full copy of the base model for every screening weight.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-a", type=Path, required=True)
    parser.add_argument("--adapter-b", type=Path, required=True)
    parser.add_argument("--weight-a", type=float, required=True)
    parser.add_argument("--weight-b", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_a = json.loads((args.adapter_a / "adapter_config.json").read_text())
    cfg_b = json.loads((args.adapter_b / "adapter_config.json").read_text())
    rank_a, rank_b = int(cfg_a["r"]), int(cfg_b["r"])
    scale_a = float(cfg_a["lora_alpha"]) / rank_a
    scale_b = float(cfg_b["lora_alpha"]) / rank_b

    state_a = load_file(str(args.adapter_a / "adapter_model.safetensors"))
    state_b = load_file(str(args.adapter_b / "adapter_model.safetensors"))
    if set(state_a) != set(state_b):
        raise ValueError("Adapters do not have identical tensor keys")

    result: dict[str, torch.Tensor] = {}
    handled: set[str] = set()
    for key in sorted(state_a):
        if key in handled:
            continue
        if not key.endswith(".lora_A.weight"):
            if key.endswith(".lora_B.weight"):
                continue
            if not torch.equal(state_a[key], state_b[key]):
                raise ValueError(f"Unsupported unequal non-LoRA tensor: {key}")
            result[key] = state_a[key]
            handled.add(key)
            continue

        b_key = key.replace(".lora_A.weight", ".lora_B.weight")
        a1, a2 = state_a[key], state_b[key]
        b1, b2 = state_a[b_key], state_b[b_key]

        # The output adapter uses alpha == rank, so its own scaling is 1.
        result[key] = torch.cat((a1, a2), dim=0).contiguous()
        result[b_key] = torch.cat(
            (b1 * (args.weight_a * scale_a), b2 * (args.weight_b * scale_b)),
            dim=1,
        ).contiguous()
        handled.update((key, b_key))

    args.output.mkdir(parents=True, exist_ok=True)
    for source in (args.adapter_a, args.adapter_b):
        for item in source.iterdir():
            if item.is_file() and item.name not in {
                "adapter_model.safetensors",
                "adapter_config.json",
            }:
                shutil.copy2(item, args.output / item.name)

    cfg_out = dict(cfg_a)
    cfg_out["r"] = rank_a + rank_b
    cfg_out["lora_alpha"] = rank_a + rank_b
    (args.output / "adapter_config.json").write_text(
        json.dumps(cfg_out, ensure_ascii=False, indent=2) + "\n"
    )
    save_file(result, str(args.output / "adapter_model.safetensors"))

    manifest = {
        "method": "rank_concatenation",
        "formula": "base + weight_a * delta_a + weight_b * delta_b",
        "adapter_a": str(args.adapter_a),
        "adapter_b": str(args.adapter_b),
        "weight_a": args.weight_a,
        "weight_b": args.weight_b,
        "rank": rank_a + rank_b,
        "lora_alpha": rank_a + rank_b,
    }
    (args.output / "combination_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
