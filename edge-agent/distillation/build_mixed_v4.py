#!/usr/bin/env python3
"""Build a leakage-checked ward/math/code SFT mixture for Ward-v4."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path


SYSTEM = "You are being evaluated. Follow the requested final-answer format exactly."


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def math_row(source: dict, index: int) -> dict:
    match = re.search(r"####\s*([^\n]+)\s*$", source["answer"])
    if not match:
        raise ValueError(f"GSM8K row {index} has no final answer")
    final = match.group(1).strip()
    reasoning = re.sub(r"<<[^<>]*>>", "", source["answer"])
    reasoning = re.sub(r"\n?####\s*[^\n]+\s*$", "", reasoning).strip()
    prompt = (
        "Solve this grade-school mathematics problem. You may reason step by step. "
        "Your final line must be exactly 'FINAL: <number>'.\n\n" + source["question"]
    )
    return {
        "id": f"v4-gsm8k-train-{index:05d}",
        "task": "gsm8k_replay",
        "source": "openai-gsm8k-train",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"{reasoning}\nFINAL: {final}"},
        ],
    }


def literal_assert_examples(source: dict) -> list[tuple[str, str]]:
    code = source.get("code", "")
    try:
        module = ast.parse(code)
    except SyntaxError:
        return []
    names = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
    examples: list[tuple[str, str]] = []
    for test_text in source.get("test_list", []):
        try:
            test_module = ast.parse(test_text)
        except SyntaxError:
            continue
        if not test_module.body or not isinstance(test_module.body[0], ast.Assert):
            continue
        test = test_module.body[0].test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            continue
        if not isinstance(test.left, ast.Call):
            continue
        call_name = test.left.func.id if isinstance(test.left.func, ast.Name) else ""
        if call_name not in names:
            continue
        try:
            expected = ast.literal_eval(test.comparators[0])
        except (ValueError, TypeError):
            continue
        examples.append((ast.unparse(test.left), repr(expected)))
    return examples


def code_rows(sources: list[dict]) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for source in sources:
        code = source.get("code", "").strip()
        for position, (call, answer) in enumerate(literal_assert_examples(source)):
            key = norm(code + "\n" + call)
            if key in seen:
                continue
            seen.add(key)
            prompt = (
                "Read the Python function and predict its exact return value for the given "
                "arguments. Do not run tools. You may reason step by step. Your final line "
                "must be exactly 'FINAL: <valid Python literal>'.\n\n"
                f"```python\n{code}\n```\n\nCall: {call}"
            )
            rows.append(
                {
                    "id": f"v4-mbpp-{source.get('task_id', 'x')}-{position}",
                    "task": "code_output_replay",
                    "source": "google-research-mbpp",
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": f"FINAL: {answer}"},
                    ],
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ward", required=True, type=Path)
    parser.add_argument("--gsm8k", required=True, type=Path)
    parser.add_argument("--mbpp", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ward-count", type=int, default=2000)
    parser.add_argument("--math-count", type=int, default=1500)
    parser.add_argument("--code-count", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    ward = read_jsonl(args.ward)
    gsm = read_jsonl(args.gsm8k)
    mbpp = read_jsonl(args.mbpp)
    benchmark = read_jsonl(args.benchmark)

    test_math = {norm(row["question"]) for row in benchmark if row["task"] == "gsm8k"}
    test_code = {
        norm(row["code"] + "\n" + row["input"])
        for row in benchmark
        if row["task"] == "cruxeval_o"
    }

    eligible_math = [row for row in gsm if norm(row["question"]) not in test_math]
    rng.shuffle(eligible_math)
    math = [math_row(row, index) for index, row in enumerate(eligible_math[: args.math_count])]

    all_code = code_rows(mbpp)
    eligible_code = []
    for row in all_code:
        user = row["messages"][1]["content"]
        if norm(user) not in test_code:
            eligible_code.append(row)
    rng.shuffle(eligible_code)
    code = eligible_code[: args.code_count]

    rng.shuffle(ward)
    ward_selected = ward[: args.ward_count]
    rows = ward_selected + math + code
    rng.shuffle(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    exact_math_overlap = sum(norm(row["messages"][1]["content"].split("\n\n", 1)[-1]) in test_math for row in math)
    manifest = {
        "version": "ward-mixed-v4.0.0",
        "seed": args.seed,
        "rows": len(rows),
        "counts": {
            "ward": len(ward_selected),
            "math_gsm8k_train": len(math),
            "code_mbpp_output": len(code),
        },
        "task_counts": dict(Counter(row.get("task", "unknown") for row in rows)),
        "sources": {
            "ward": {"path": str(args.ward), "sha256": sha256(args.ward)},
            "gsm8k_train": {"path": str(args.gsm8k), "sha256": sha256(args.gsm8k)},
            "mbpp": {"path": str(args.mbpp), "sha256": sha256(args.mbpp)},
            "held_out_benchmark": {"path": str(args.benchmark), "sha256": sha256(args.benchmark)},
        },
        "leakage_checks": {
            "gsm8k_exact_question_overlap": exact_math_overlap,
            "cruxeval_source_used_for_training": False,
            "benchmark_rows_used_for_training": 0,
        },
        "output_sha256": sha256(args.output),
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
