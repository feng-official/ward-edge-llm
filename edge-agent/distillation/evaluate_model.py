#!/usr/bin/env python3
"""Evaluate Qwen ward behavior on the fixed held-out set through an OpenAI API."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from api_config import resolve_api_key


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def call_api(base_url: str, api_key: str, model: str, messages: list[dict], max_tokens: int) -> tuple[str, float]:
    data = json.dumps({"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=data, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {api_key}"
    })
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as resp:
        obj = json.loads(resp.read().decode("utf-8"))
    return obj["choices"][0]["message"]["content"].strip(), time.perf_counter() - started


def char_bigrams(text: str) -> set[str]:
    text = re.sub(r"\s+", "", text)
    return {text[i:i + 2] for i in range(max(0, len(text) - 1))}


def dice(a: str, b: str) -> float:
    aa, bb = char_bigrams(a), char_bigrams(b)
    return 2 * len(aa & bb) / max(1, len(aa) + len(bb))


def score(row: dict, prediction: str) -> dict:
    expected = row["messages"][-1]["content"].strip()
    task = row["task"]
    item = {"id": row["id"], "task": task, "expected": expected, "prediction": prediction, "similarity": dice(expected, prediction)}
    if task == "cloud_judgment":
        exp_label = expected.split("|", 1)[0].strip()
        pred_label = prediction.split("|", 1)[0].strip().lower()
        item["label_expected"] = exp_label
        item["label_prediction"] = pred_label
        item["label_correct"] = pred_label == exp_label
        item["format_valid"] = bool(re.match(r"^(confirm|reject|escalate)\|(?:0(?:\.\d+)?|1(?:\.0+)?)\|.+", prediction, re.S))
    else:
        match = re.search(r"【(紧急|警告|提醒)】", prediction)
        expected_match = re.search(r"【(紧急|警告|提醒)】", expected)
        item["urgency_expected"] = expected_match.group(1) if expected_match else ""
        item["urgency_prediction"] = match.group(1) if match else ""
        item["urgency_correct"] = bool(match and expected_match and match.group(1) == expected_match.group(1))
        item["format_valid"] = bool(match and len(prediction) <= 120)
    return item


def macro_f1(items: list[dict]) -> float:
    labels = sorted({i.get("label_expected") for i in items if i.get("label_expected")})
    values = []
    for label in labels:
        tp = sum(i.get("label_expected") == label and i.get("label_prediction") == label for i in items)
        fp = sum(i.get("label_expected") != label and i.get("label_prediction") == label for i in items)
        fn = sum(i.get("label_expected") == label and i.get("label_prediction") != label for i in items)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        values.append(2 * precision * recall / max(1e-12, precision + recall))
    return statistics.mean(values) if values else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--api-key-file", type=Path, default=None)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    rows = load_jsonl(args.input)
    if args.limit:
        rows = rows[:args.limit]
    key = resolve_api_key(args.api_key_file)
    results: list[dict] = []

    def run(row: dict) -> dict:
        prediction, latency = call_api(args.base_url, key, args.model, row["messages"][:-1], 180)
        return score(row, prediction) | {"latency_seconds": latency}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, row) for row in rows]
        for index, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if index % 25 == 0 or index == len(futures):
                print(f"evaluation {index}/{len(futures)}", flush=True)
    results.sort(key=lambda x: x["id"])
    clouds = [i for i in results if i["task"] == "cloud_judgment"]
    events = [i for i in results if i["task"] == "event_enhancement"]
    metrics = {
        "label": args.label,
        "model": args.model,
        "samples": len(results),
        "cloud_judgment_accuracy": sum(i["label_correct"] for i in clouds) / max(1, len(clouds)),
        "cloud_judgment_macro_f1": macro_f1(clouds),
        "event_urgency_accuracy": sum(i["urgency_correct"] for i in events) / max(1, len(events)),
        "format_valid_rate": sum(i["format_valid"] for i in results) / max(1, len(results)),
        "mean_reference_similarity": statistics.mean(i["similarity"] for i in results),
        "mean_latency_seconds": statistics.mean(i["latency_seconds"] for i in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"metrics": metrics, "samples": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
