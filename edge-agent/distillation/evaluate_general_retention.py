#!/usr/bin/env python3
"""Reproducible before/after evaluation of general capabilities.

The evaluator talks to an OpenAI-compatible Chat Completions endpoint, so the
same prepared examples and prompts can be used for the original student, the
merged distilled model, and (when served by llama.cpp) the GGUF model.

It intentionally does not execute model-generated code.  CRUXEval-O measures
code execution reasoning by comparing the predicted Python return value with
the published reference value.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import re
import statistics
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from api_config import resolve_api_key


SUITE_VERSION = "general-retention-v1"
SCORER_VERSION = "symmetric-semantic-extractor-v2"
DEFAULT_SEED = 20260818
SOURCE_URLS = {
    "gsm8k": (
        "https://raw.githubusercontent.com/openai/grade-school-math/"
        "master/grade_school_math/data/test.jsonl"
    ),
    "cruxeval_o": (
        "https://raw.githubusercontent.com/facebookresearch/cruxeval/"
        "main/data/cruxeval.jsonl"
    ),
    "bbh": (
        "https://codeload.github.com/suzgunmirac/"
        "BIG-Bench-Hard/zip/refs/heads/main"
    ),
}
TASK_DESCRIPTIONS = {
    "gsm8k": "数学多步推理（GSM8K）",
    "cruxeval_o": "代码理解与执行推理（CRUXEval-O）",
    "bbh": "自然语言与逻辑推理（BIG-Bench Hard）",
}

BBH_TASKS = (
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "logical_deduction_three_objects",
    "navigate",
    "sports_understanding",
    "temporal_sequences",
    "web_of_lies",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def download(url: str, target: Path) -> None:
    if target.is_file() and target.stat().st_size:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "smart-ward-general-retention-eval/1.0"},
    )
    print(f"download: {url}", flush=True)
    with urllib.request.urlopen(request, timeout=300) as response:
        with temporary.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    temporary.replace(target)


def stable_sample(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count <= 0 or count > len(rows):
        count = len(rows)
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    return [rows[index] for index in sorted(indices[:count])]


def prepare_command(args: argparse.Namespace) -> int:
    cache_dir: Path = args.cache_dir
    source_dir = cache_dir / "sources"
    gsm_path = source_dir / "gsm8k-test.jsonl"
    crux_path = source_dir / "cruxeval.jsonl"
    bbh_zip = source_dir / "BIG-Bench-Hard-main.zip"
    download(SOURCE_URLS["gsm8k"], gsm_path)
    download(SOURCE_URLS["cruxeval_o"], crux_path)
    download(SOURCE_URLS["bbh"], bbh_zip)

    examples: list[dict[str, Any]] = []

    gsm_rows = read_jsonl(gsm_path)
    for index, row in enumerate(stable_sample(gsm_rows, args.per_task, args.seed + 11)):
        match = re.search(r"####\s*([^\n]+)\s*$", row["answer"])
        if not match:
            raise ValueError(f"GSM8K answer has no final marker: {index}")
        examples.append(
            {
                "id": f"gsm8k-{index:04d}",
                "source_id": index,
                "task": "gsm8k",
                "question": row["question"],
                "answer": match.group(1).strip(),
            }
        )

    crux_rows = read_jsonl(crux_path)
    for row in stable_sample(crux_rows, args.per_task, args.seed + 23):
        examples.append(
            {
                "id": f"cruxeval-o-{row['id']}",
                "source_id": row["id"],
                "task": "cruxeval_o",
                "code": row["code"],
                "input": row["input"],
                "answer": row["output"],
            }
        )

    bbh_rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(bbh_zip) as archive:
        for task_name in BBH_TASKS:
            candidates = [
                name
                for name in archive.namelist()
                if name.endswith(f"/bbh/{task_name}.json")
            ]
            if len(candidates) != 1:
                raise ValueError(f"BBH task file not uniquely found: {task_name}: {candidates}")
            document = json.loads(archive.read(candidates[0]).decode("utf-8"))
            for index, example in enumerate(document["examples"]):
                bbh_rows.append(
                    {
                        "source_id": f"{task_name}-{index:04d}",
                        "subtask": task_name,
                        "question": example["input"],
                        "answer": example["target"],
                    }
                )
    for row in stable_sample(bbh_rows, args.per_task, args.seed + 37):
        examples.append(
            {
                "id": f"bbh-{row['source_id']}",
                "source_id": row["source_id"],
                "task": "bbh",
                "subtask": row["subtask"],
                "question": row["question"],
                "answer": row["answer"],
            }
        )

    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in examples:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "suite": SUITE_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "per_task": args.per_task,
        "samples": len(examples),
        "task_counts": {
            task: sum(row["task"] == task for row in examples)
            for task in TASK_DESCRIPTIONS
        },
        "dataset_sha256": sha256_file(output),
        "sources": {
            "gsm8k": {"url": SOURCE_URLS["gsm8k"], "sha256": sha256_file(gsm_path)},
            "cruxeval_o": {
                "url": SOURCE_URLS["cruxeval_o"],
                "sha256": sha256_file(crux_path),
            },
            "bbh": {
                "url": SOURCE_URLS["bbh"],
                "sha256": sha256_file(bbh_zip),
                "subtasks": list(BBH_TASKS),
            },
        },
    }
    write_json(output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


def build_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    if row["task"] == "gsm8k":
        content = (
            "Solve this grade-school mathematics problem. You may reason step by step. "
            "Your final line must be exactly 'FINAL: <number>'.\n\n"
            + row["question"]
        )
    elif row["task"] == "cruxeval_o":
        content = (
            "Read the Python function and predict its exact return value for the given "
            "arguments. Do not run tools. You may reason step by step. Your final line "
            "must be exactly 'FINAL: <valid Python literal>'.\n\n"
            f"```python\n{row['code']}\n```\n\nCall: f({row['input']})"
        )
    elif row["task"] == "bbh":
        content = (
            "Solve this language and logic reasoning problem. You may reason step by step. "
            "Your final line must be exactly 'FINAL: <answer>'. For multiple-choice "
            "questions, output only the option label such as (A).\n\n"
            f"{row['question']}"
        )
    else:
        raise ValueError(f"unknown task: {row['task']}")
    return [
        {
            "role": "system",
            "content": "You are being evaluated. Follow the requested final-answer format exactly.",
        },
        {"role": "user", "content": content},
    ]


def call_api(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: int,
    retries: int,
) -> tuple[str, float]:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "seed": DEFAULT_SEED,
            "max_tokens": max_tokens,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    url = base_url.rstrip("/") + "/chat/completions"
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"].strip(), time.perf_counter() - started
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(8, 2**attempt))
    raise RuntimeError(f"API request failed after {retries + 1} attempts: {last_error}")


def strict_final_text(prediction: str) -> str | None:
    matches = re.findall(r"(?im)^\s*FINAL\s*:\s*(.+?)\s*$", prediction)
    return matches[-1].strip() if matches else None


def normalize_decimal(text: str) -> Decimal | None:
    cleaned = text.strip().replace(",", "").replace("$", "").replace("%", "")
    cleaned = cleaned.rstrip(".")
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def normalize_short_answer(text: str) -> str:
    value = text.strip().lower()
    if re.fullmatch(r"\(?[a-z]\)?[.!]?", value):
        return value.strip("().! ")
    return re.sub(r"\s+", " ", value).strip().rstrip(".")


def strip_markdown_wrapper(text: str) -> str:
    value = text.strip().strip("` ")
    value = re.sub(r"^\*+|\*+$", "", value).strip()
    value = re.sub(r"\\text\{[^{}]*\}", "", value).strip()
    value = value.strip("$ ")
    return value.rstrip("。.").strip()


def extract_gsm8k_relaxed(prediction: str) -> str | None:
    patterns = (
        r"(?im)(?:FINAL|final\s+answer|answer)(?:\s+is)?\s*[:=]\s*[^\d+\-]*([+\-]?\d[\d,]*(?:\.\d+)?)\s*%?",
        r"(?is)\\boxed\{\s*([+\-]?\d[\d,]*(?:\.\d+)?)\s*\}",
    )
    for pattern in patterns:
        matches = re.findall(pattern, prediction)
        if matches:
            return matches[-1]
    numbers = re.findall(r"(?<![\w.])[+\-]?\d[\d,]*(?:\.\d+)?(?![\w.])", prediction)
    return numbers[-1] if numbers else None


def literal_value(text: str) -> tuple[Any, str] | None:
    candidate = strip_markdown_wrapper(text)
    attempts = [candidate]
    quoted = re.findall(r"(`[^`\n]+`|'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")", candidate)
    attempts.extend(reversed(quoted))
    for opening, closing in (("[", "]"), ("{", "}"), ("(", ")")):
        if opening in candidate and closing in candidate:
            attempts.append(candidate[candidate.find(opening) : candidate.rfind(closing) + 1])
    for value in attempts:
        value = value.strip().strip("`").rstrip("。.").strip()
        try:
            return ast.literal_eval(value), value
        except (ValueError, SyntaxError):
            continue
    return None


def extract_cruxeval_relaxed(prediction: str) -> tuple[Any, str] | None:
    lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    labelled: list[str] = []
    label_pattern = re.compile(
        r"(?i)(?:FINAL|final\s+answer|answer|return\s+value)(?:\s+is)?\s*:\s*(.+)$"
    )
    for line in lines:
        match = label_pattern.search(line)
        if match:
            labelled.append(match.group(1))
    for candidate in reversed(labelled + lines):
        parsed = literal_value(candidate)
        if parsed is not None:
            return parsed
    return None


def extract_bbh_relaxed(prediction: str, expected: str) -> str | None:
    expected_normalized = normalize_short_answer(expected)
    labelled = re.findall(
        r"(?im)(?:FINAL|final\s+answer|answer)(?:\s+is)?\s*[:=]\s*([^\n]+)",
        prediction,
    )
    candidates = [strip_markdown_wrapper(value) for value in labelled]
    if re.fullmatch(r"[a-z]", expected_normalized):
        for value in reversed(candidates):
            match = re.search(r"(?i)\(?([a-z])\)?", value)
            if match:
                return match.group(1)
        matches = re.findall(r"(?i)(?<![A-Za-z])\(([a-z])\)(?![A-Za-z])", prediction)
        if matches:
            return matches[-1]
        for line in reversed(prediction.splitlines()):
            if re.fullmatch(r"\s*[A-Za-z]\s*", line):
                return line.strip()
    if expected_normalized in {"yes", "no", "true", "false"}:
        for value in reversed(candidates):
            match = re.search(r"(?i)\b(yes|no|true|false)\b", value)
            if match:
                return match.group(1)
        matches = re.findall(r"(?i)\b(yes|no|true|false)\b", prediction)
        if matches:
            return matches[-1]
    if candidates:
        return candidates[-1]
    lines = [strip_markdown_wrapper(line) for line in prediction.splitlines() if line.strip()]
    return lines[-1] if lines else None


def score_candidate(task: str, expected: str, extracted: str | None) -> tuple[bool, bool, str | None]:
    correct = False
    parse_error: str | None = None
    if extracted is None:
        return False, False, "answer not extracted"
    if task == "gsm8k":
        expected_value = normalize_decimal(expected)
        predicted_value = normalize_decimal(extracted)
        correct = expected_value is not None and predicted_value == expected_value
        if predicted_value is None:
            parse_error = "invalid numeric final answer"
    elif task == "bbh":
        predicted_answer = normalize_short_answer(extracted)
        expected_answer = normalize_short_answer(expected)
        correct = predicted_answer == expected_answer
        if not predicted_answer:
            parse_error = "empty short answer"
    elif task == "cruxeval_o":
        try:
            predicted_value = ast.literal_eval(strip_markdown_wrapper(extracted))
            expected_value = ast.literal_eval(expected)
            correct = predicted_value == expected_value and type(predicted_value) is type(expected_value)
        except (ValueError, SyntaxError) as exc:
            parse_error = f"invalid Python literal: {exc}"
    return correct, parse_error is None, parse_error


def score_prediction(row: dict[str, Any], prediction: str) -> dict[str, Any]:
    task = row["task"]
    expected = str(row["answer"])
    strict_extracted = strict_final_text(prediction)
    strict_correct, strict_valid, strict_error = score_candidate(task, expected, strict_extracted)

    if task == "gsm8k":
        relaxed_extracted = extract_gsm8k_relaxed(prediction)
    elif task == "bbh":
        relaxed_extracted = extract_bbh_relaxed(prediction, expected)
    elif task == "cruxeval_o":
        parsed = extract_cruxeval_relaxed(prediction)
        relaxed_extracted = parsed[1] if parsed is not None else None
    else:
        raise ValueError(f"unknown task: {task}")
    relaxed_correct, relaxed_valid, relaxed_error = score_candidate(task, expected, relaxed_extracted)
    return {
        "strict_extracted": strict_extracted,
        "strict_correct": strict_correct,
        "strict_format_valid": strict_valid,
        "strict_parse_error": strict_error,
        "relaxed_extracted": relaxed_extracted,
        "relaxed_correct": relaxed_correct,
        "relaxed_parse_valid": relaxed_valid,
        "relaxed_parse_error": relaxed_error,
        # Compatibility aliases. Accuracy now means semantic accuracy under the
        # symmetric relaxed extractor; formatting is reported separately.
        "extracted": relaxed_extracted,
        "correct": relaxed_correct,
        "format_valid": strict_valid,
        "parse_error": relaxed_error,
    }


def wilson_interval(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [round(max(0.0, centre - margin), 6), round(min(1.0, centre + margin), 6)]


def model_list(base_url: str, api_key: str, timeout: int) -> list[str]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return [str(item["id"]) for item in body.get("data", [])]


def run_command(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.benchmark)
    if args.limit_per_task:
        limited: list[dict[str, Any]] = []
        for task in TASK_DESCRIPTIONS:
            limited.extend([row for row in rows if row["task"] == task][: args.limit_per_task])
        rows = limited
    benchmark_sha = sha256_file(args.benchmark)
    api_key = resolve_api_key(args.api_key_file)

    served_models = model_list(args.base_url, api_key, args.timeout)
    if args.model not in served_models:
        raise RuntimeError(
            f"requested model {args.model!r} is not served; endpoint reports {served_models}"
        )

    raw_path = args.output.with_suffix(".samples.jsonl")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    previous: dict[str, dict[str, Any]] = {}
    if raw_path.exists() and not args.no_resume:
        for item in read_jsonl(raw_path):
            if (
                item.get("model_label") == args.label
                and item.get("model") == args.model
                and item.get("benchmark_sha256") == benchmark_sha
                and item.get("scorer_version") == SCORER_VERSION
            ):
                previous[item["id"]] = item
    pending = [row for row in rows if row["id"] not in previous]
    print(
        f"run label={args.label} model={args.model} total={len(rows)} "
        f"resume={len(previous)} pending={len(pending)}",
        flush=True,
    )

    def evaluate_one(row: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            prediction, latency = call_api(
                args.base_url,
                api_key,
                args.model,
                build_messages(row),
                args.max_tokens,
                args.timeout,
                args.retries,
            )
            scored = score_prediction(row, prediction)
            error = None
        except Exception as exc:  # one request must not discard the full run
            prediction = ""
            latency = time.perf_counter() - started
            scored = {
                "strict_extracted": None,
                "strict_correct": False,
                "strict_format_valid": False,
                "strict_parse_error": "request failed",
                "relaxed_extracted": None,
                "relaxed_correct": False,
                "relaxed_parse_valid": False,
                "relaxed_parse_error": "request failed",
                "extracted": None,
                "correct": False,
                "format_valid": False,
                "parse_error": "request failed",
            }
            error = f"{type(exc).__name__}: {exc}"
        return {
            "id": row["id"],
            "source_id": row.get("source_id"),
            "task": row["task"],
            "subtask": row.get("subtask"),
            "expected": row["answer"],
            "prediction": prediction,
            **scored,
            "error": error,
            "latency_seconds": round(latency, 6),
            "model_label": args.label,
            "model": args.model,
            "benchmark_sha256": benchmark_sha,
            "scorer_version": SCORER_VERSION,
        }

    if pending:
        with raw_path.open("a", encoding="utf-8", newline="\n") as handle:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(evaluate_one, row): row for row in pending}
                for index, future in enumerate(as_completed(futures), 1):
                    item = future.result()
                    previous[item["id"]] = item
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                    handle.flush()
                    if index % 20 == 0 or index == len(futures):
                        print(f"progress {index}/{len(futures)}", flush=True)

    results = [previous[row["id"]] for row in rows]
    task_metrics: dict[str, Any] = {}
    for task in TASK_DESCRIPTIONS:
        items = [item for item in results if item["task"] == task]
        successes = sum(bool(item["correct"]) for item in items)
        strict_successes = sum(bool(item["strict_correct"]) for item in items)
        strict_valid = sum(bool(item["strict_format_valid"]) for item in items)
        relaxed_valid = sum(bool(item["relaxed_parse_valid"]) for item in items)
        task_metrics[task] = {
            "description": TASK_DESCRIPTIONS[task],
            "samples": len(items),
            "correct": successes,
            "metric_used_for_retention": "semantic_accuracy_relaxed_symmetric_extractor",
            "accuracy": round(successes / max(1, len(items)), 6),
            "semantic_accuracy": round(successes / max(1, len(items)), 6),
            "accuracy_95ci_wilson": wilson_interval(successes, len(items)),
            "strict_accuracy": round(strict_successes / max(1, len(items)), 6),
            "format_valid_rate": round(
                strict_valid / max(1, len(items)), 6
            ),
            "relaxed_parse_rate": round(relaxed_valid / max(1, len(items)), 6),
            "conditional_accuracy_among_strict_valid": (
                round(strict_successes / strict_valid, 6) if strict_valid else None
            ),
            "conditional_accuracy_among_relaxed_parsed": (
                round(successes / relaxed_valid, 6) if relaxed_valid else None
            ),
            "request_errors": sum(bool(item["error"]) for item in items),
            "mean_latency_seconds": round(
                statistics.mean(item["latency_seconds"] for item in items), 6
            ) if items else 0.0,
        }
        if task == "bbh":
            subtask_metrics: dict[str, Any] = {}
            for subtask in BBH_TASKS:
                subset = [item for item in items if item.get("subtask") == subtask]
                if not subset:
                    continue
                subtask_successes = sum(bool(item["correct"]) for item in subset)
                subtask_metrics[subtask] = {
                    "samples": len(subset),
                    "correct": subtask_successes,
                    "accuracy": round(subtask_successes / len(subset), 6),
                }
            task_metrics[task]["subtask_metrics"] = subtask_metrics
    macro_accuracy = statistics.mean(value["accuracy"] for value in task_metrics.values())
    report = {
        "suite": SUITE_VERSION,
        "scorer_version": SCORER_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_label": args.label,
        "model": args.model,
        "served_models": served_models,
        "endpoint": args.base_url,
        "benchmark": str(args.benchmark),
        "benchmark_sha256": benchmark_sha,
        "settings": {
            "temperature": 0,
            "seed": DEFAULT_SEED,
            "max_tokens": args.max_tokens,
            "workers": args.workers,
            "limit_per_task": args.limit_per_task,
        },
        "samples": len(results),
        "macro_accuracy": round(macro_accuracy, 6),
        "task_metrics": task_metrics,
        "raw_samples": str(raw_path),
        "notes": [
            "CRUXEval-O is code reasoning; model-generated code is not executed.",
            "This report is independent of the ward-specific template evaluation.",
            "Retention uses semantic_accuracy from the same relaxed extractor for both models.",
            "Conditional accuracy among parseable samples is diagnostic only and is selection-biased.",
        ],
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


def retention_value(before: float, after: float) -> float | None:
    if before <= 0:
        return None
    return after / before


def compare_command(args: argparse.Namespace) -> int:
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    distilled = json.loads(args.distilled.read_text(encoding="utf-8"))
    if baseline.get("benchmark_sha256") != distilled.get("benchmark_sha256"):
        raise ValueError("reports use different prepared benchmark files")
    rows: dict[str, Any] = {}
    for task in TASK_DESCRIPTIONS:
        before = float(baseline["task_metrics"][task]["accuracy"])
        after = float(distilled["task_metrics"][task]["accuracy"])
        retention = retention_value(before, after)
        rows[task] = {
            "description": TASK_DESCRIPTIONS[task],
            "baseline_accuracy": before,
            "distilled_accuracy": after,
            "absolute_delta": round(after - before, 6),
            "retention": round(retention, 6) if retention is not None else None,
            "retention_percent": round(retention * 100, 2) if retention is not None else None,
            "threshold": args.threshold,
            "passes_threshold": retention >= args.threshold if retention is not None else None,
        }
    before_macro = float(baseline["macro_accuracy"])
    after_macro = float(distilled["macro_accuracy"])
    macro_retention = retention_value(before_macro, after_macro)
    comparison = {
        "suite": SUITE_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmark_sha256": baseline["benchmark_sha256"],
        "baseline": {"label": baseline["model_label"], "model": baseline["model"]},
        "distilled": {"label": distilled["model_label"], "model": distilled["model"]},
        "threshold": args.threshold,
        "task_retention": rows,
        "macro": {
            "baseline_accuracy": before_macro,
            "distilled_accuracy": after_macro,
            "absolute_delta": round(after_macro - before_macro, 6),
            "retention": round(macro_retention, 6) if macro_retention is not None else None,
            "retention_percent": round(macro_retention * 100, 2) if macro_retention is not None else None,
            "passes_threshold": macro_retention >= args.threshold if macro_retention is not None else None,
        },
        "acceptance_rule": (
            "Report every category and the macro average. A zero baseline is marked "
            "not assessable instead of being converted to an infinite percentage."
        ),
    }
    write_json(args.output, comparison)

    markdown = [
        "# 蒸馏前后通用能力保留率",
        "",
        f"- 基线模型：`{baseline['model_label']}` / `{baseline['model']}`",
        f"- 蒸馏模型：`{distilled['model_label']}` / `{distilled['model']}`",
        f"- 验收阈值：{args.threshold * 100:.0f}%",
        f"- 固定评测集 SHA-256：`{baseline['benchmark_sha256']}`",
        "",
        "| 类别 | 蒸馏前 | 蒸馏后 | 保留率 | 是否达标 |",
        "|---|---:|---:|---:|:---:|",
    ]
    for task in TASK_DESCRIPTIONS:
        item = rows[task]
        retention_text = (
            f"{item['retention_percent']:.2f}%"
            if item["retention_percent"] is not None
            else "不可计算"
        )
        pass_text = (
            "是" if item["passes_threshold"] is True else "否" if item["passes_threshold"] is False else "—"
        )
        markdown.append(
            f"| {item['description']} | {item['baseline_accuracy']:.2%} | "
            f"{item['distilled_accuracy']:.2%} | {retention_text} | {pass_text} |"
        )
    macro_text = (
        f"{comparison['macro']['retention_percent']:.2f}%"
        if comparison["macro"]["retention_percent"] is not None
        else "不可计算"
    )
    macro_pass = (
        "是" if comparison["macro"]["passes_threshold"] is True else "否" if comparison["macro"]["passes_threshold"] is False else "—"
    )
    markdown.append(
        f"| **宏平均** | **{before_macro:.2%}** | **{after_macro:.2%}** | "
        f"**{macro_text}** | **{macro_pass}** |"
    )
    markdown.extend(
        [
            "",
            "> 保留率 = 蒸馏后准确率 ÷ 原始模型准确率。此表不包含病房专用模板任务分数。",
            "",
        ]
    )
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
    print(f"markdown: {markdown_path}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="download and freeze benchmark examples")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--cache-dir", type=Path, required=True)
    prepare.add_argument("--per-task", type=int, default=200)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.set_defaults(func=prepare_command)

    run = subparsers.add_parser("run", help="evaluate one served model")
    run.add_argument("--benchmark", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--label", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    run.add_argument(
        "--api-key-file",
        type=Path,
        default=None,
    )
    run.add_argument("--workers", type=int, default=6)
    run.add_argument("--max-tokens", type=int, default=512)
    run.add_argument("--timeout", type=int, default=180)
    run.add_argument("--retries", type=int, default=2)
    run.add_argument("--limit-per-task", type=int, default=0)
    run.add_argument("--no-resume", action="store_true")
    run.set_defaults(func=run_command)

    compare = subparsers.add_parser("compare", help="calculate retention rates")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--distilled", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--threshold", type=float, default=0.80)
    compare.set_defaults(func=compare_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
