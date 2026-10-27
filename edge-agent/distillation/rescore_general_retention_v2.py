#!/usr/bin/env python3
"""Re-score saved general-benchmark outputs with a symmetric relaxed extractor.

No model inference is repeated.  The exact same baseline and distilled answers
are parsed with the same task-specific rules.  Strict FINAL-format compliance
is retained as a separate instruction-following metric.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from evaluate_general_retention import (
    BBH_TASKS,
    SCORER_VERSION,
    TASK_DESCRIPTIONS,
    score_prediction,
    sha256_file,
    wilson_interval,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def rescore_one(
    benchmark: list[dict[str, Any]],
    benchmark_sha256: str,
    samples_path: Path,
    output_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    benchmark_map = {row["id"]: row for row in benchmark}
    old_samples = read_jsonl(samples_path)
    old_map = {row["id"]: row for row in old_samples}
    missing = sorted(set(benchmark_map) - set(old_map))
    extra = sorted(set(old_map) - set(benchmark_map))
    if missing or extra:
        raise ValueError(f"sample/benchmark mismatch: missing={missing[:5]} extra={extra[:5]}")

    rescored: list[dict[str, Any]] = []
    for row in benchmark:
        old = old_map[row["id"]]
        scored = score_prediction(row, str(old.get("prediction", "")))
        rescored.append(
            {
                **old,
                "v1_correct": bool(old.get("correct")),
                "v1_format_valid": bool(old.get("format_valid")),
                "v1_extracted": old.get("extracted"),
                "v1_parse_error": old.get("parse_error"),
                **scored,
                "scorer_version": SCORER_VERSION,
            }
        )

    raw_output = output_path.with_suffix(".samples.jsonl")
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    with raw_output.open("w", encoding="utf-8", newline="\n") as handle:
        for item in rescored:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    task_metrics: dict[str, Any] = {}
    for task, description in TASK_DESCRIPTIONS.items():
        items = [item for item in rescored if item["task"] == task]
        total = len(items)
        semantic_correct = sum(bool(item["relaxed_correct"]) for item in items)
        strict_correct = sum(bool(item["strict_correct"]) for item in items)
        strict_valid = sum(bool(item["strict_format_valid"]) for item in items)
        relaxed_valid = sum(bool(item["relaxed_parse_valid"]) for item in items)
        v1_correct = sum(bool(item["v1_correct"]) for item in items)
        v1_valid = sum(bool(item["v1_format_valid"]) for item in items)
        metric: dict[str, Any] = {
            "description": description,
            "samples": total,
            "primary_metric": "semantic_accuracy",
            "semantic_correct": semantic_correct,
            "semantic_accuracy": ratio(semantic_correct, total),
            "semantic_accuracy_95ci_wilson": wilson_interval(semantic_correct, total),
            "relaxed_parse_rate": ratio(relaxed_valid, total),
            "strict_correct": strict_correct,
            "strict_accuracy": ratio(strict_correct, total),
            "strict_format_valid_rate": ratio(strict_valid, total),
            "conditional_accuracy_among_strict_valid": ratio(strict_correct, strict_valid),
            "conditional_accuracy_among_relaxed_parsed": ratio(semantic_correct, relaxed_valid),
            "original_v1_accuracy": ratio(v1_correct, total),
            "original_v1_format_valid_rate": ratio(v1_valid, total),
            "format_penalty_recovered": round(
                semantic_correct / max(1, total) - v1_correct / max(1, total), 6
            ),
            "request_errors": sum(bool(item.get("error")) for item in items),
        }
        if task == "bbh":
            metric["subtask_metrics"] = {}
            for subtask in BBH_TASKS:
                subset = [item for item in items if item.get("subtask") == subtask]
                if subset:
                    correct = sum(bool(item["relaxed_correct"]) for item in subset)
                    metric["subtask_metrics"][subtask] = {
                        "samples": len(subset),
                        "correct": correct,
                        "semantic_accuracy": ratio(correct, len(subset)),
                    }
        task_metrics[task] = metric

    semantic_values = [float(value["semantic_accuracy"]) for value in task_metrics.values()]
    strict_values = [float(value["strict_accuracy"]) for value in task_metrics.values()]
    report = {
        "suite": "general-retention-v2",
        "scorer_version": SCORER_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_label": rescored[0].get("model_label") if rescored else "unknown",
        "model": rescored[0].get("model") if rescored else "unknown",
        "benchmark_sha256": benchmark_sha256,
        "source_samples": str(samples_path),
        "samples": len(rescored),
        "primary_metric": "macro_semantic_accuracy",
        "macro_semantic_accuracy": round(statistics.mean(semantic_values), 6),
        "macro_strict_accuracy": round(statistics.mean(strict_values), 6),
        "task_metrics": task_metrics,
        "raw_samples": str(raw_output),
        "methodology": [
            "Both models are rescored from already saved outputs with the identical extractor.",
            "Semantic accuracy uses relaxed task-specific extraction; strict FINAL compliance is separate.",
            "Conditional accuracy among parsed samples is diagnostic and not used for retention.",
            "CRUXEval model-generated code is never executed.",
        ],
    }
    write_json(output_path, report)
    return report, rescored


def retention(before: float, after: float) -> float | None:
    return after / before if before > 0 else None


def build_comparison(
    baseline: dict[str, Any],
    distilled: dict[str, Any],
    baseline_samples: list[dict[str, Any]],
    distilled_samples: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    baseline_map = {row["id"]: row for row in baseline_samples}
    distilled_map = {row["id"]: row for row in distilled_samples}
    tasks: dict[str, Any] = {}
    for task, description in TASK_DESCRIPTIONS.items():
        before = float(baseline["task_metrics"][task]["semantic_accuracy"])
        after = float(distilled["task_metrics"][task]["semantic_accuracy"])
        value = retention(before, after)
        paired = [
            (bool(baseline_map[item_id]["relaxed_correct"]), bool(distilled_map[item_id]["relaxed_correct"]))
            for item_id in baseline_map
            if baseline_map[item_id]["task"] == task
        ]
        tasks[task] = {
            "description": description,
            "baseline_semantic_accuracy": before,
            "distilled_semantic_accuracy": after,
            "absolute_delta": round(after - before, 6),
            "semantic_retention": round(value, 6) if value is not None else None,
            "semantic_retention_percent": round(value * 100, 2) if value is not None else None,
            "passes_threshold": value >= threshold if value is not None else None,
            "baseline_strict_format_valid_rate": baseline["task_metrics"][task]["strict_format_valid_rate"],
            "distilled_strict_format_valid_rate": distilled["task_metrics"][task]["strict_format_valid_rate"],
            "baseline_relaxed_parse_rate": baseline["task_metrics"][task]["relaxed_parse_rate"],
            "distilled_relaxed_parse_rate": distilled["task_metrics"][task]["relaxed_parse_rate"],
            "baseline_only_correct": sum(a and not b for a, b in paired),
            "distilled_only_correct": sum(b and not a for a, b in paired),
        }
    before_macro = float(baseline["macro_semantic_accuracy"])
    after_macro = float(distilled["macro_semantic_accuracy"])
    macro_retention = retention(before_macro, after_macro)
    return {
        "suite": "general-retention-v2",
        "scorer_version": SCORER_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "threshold": threshold,
        "primary_metric": "semantic_accuracy_with_symmetric_relaxed_extraction",
        "baseline": {"label": baseline["model_label"], "model": baseline["model"]},
        "distilled": {"label": distilled["model_label"], "model": distilled["model"]},
        "task_retention": tasks,
        "macro": {
            "baseline_semantic_accuracy": before_macro,
            "distilled_semantic_accuracy": after_macro,
            "absolute_delta": round(after_macro - before_macro, 6),
            "semantic_retention": round(macro_retention, 6) if macro_retention is not None else None,
            "semantic_retention_percent": round(macro_retention * 100, 2) if macro_retention is not None else None,
            "passes_threshold": macro_retention >= threshold if macro_retention is not None else None,
        },
        "warning": (
            "Do not use conditional accuracy among format-valid samples as the main metric; "
            "the selected subsets differ between models."
        ),
    }


def write_markdown(path: Path, comparison: dict[str, Any]) -> None:
    lines = [
        "# 蒸馏前后通用能力保留率（方法修正版 v2）",
        "",
        "主指标使用同一套宽松答案抽取器对两边的已保存输出重新评分；严格格式遵循率单独报告。",
        "",
        "| 类别 | 基线格式率 | 蒸馏格式率 | 基线语义准确率 | 蒸馏语义准确率 | 语义保留率 | 80%阈值 |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for task in TASK_DESCRIPTIONS:
        item = comparison["task_retention"][task]
        retention_text = (
            f"{item['semantic_retention_percent']:.2f}%"
            if item["semantic_retention_percent"] is not None
            else "不可计算"
        )
        passed = "是" if item["passes_threshold"] is True else "否" if item["passes_threshold"] is False else "—"
        lines.append(
            f"| {item['description']} | {item['baseline_strict_format_valid_rate']:.2%} | "
            f"{item['distilled_strict_format_valid_rate']:.2%} | "
            f"{item['baseline_semantic_accuracy']:.2%} | {item['distilled_semantic_accuracy']:.2%} | "
            f"{retention_text} | {passed} |"
        )
    macro = comparison["macro"]
    macro_retention = (
        f"{macro['semantic_retention_percent']:.2f}%"
        if macro["semantic_retention_percent"] is not None
        else "不可计算"
    )
    macro_passed = "是" if macro["passes_threshold"] is True else "否" if macro["passes_threshold"] is False else "—"
    lines.extend(
        [
            f"| **宏平均** | — | — | **{macro['baseline_semantic_accuracy']:.2%}** | "
            f"**{macro['distilled_semantic_accuracy']:.2%}** | **{macro_retention}** | **{macro_passed}** |",
            "",
            "> 语义保留率 = 蒸馏后语义准确率 ÷ 原始模型语义准确率。格式不合规不会自动等同于语义错误。",
            "",
            "> 仅在格式合规样本上计算的条件准确率存在选择偏差，不用于验收。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--baseline-samples", type=Path, required=True)
    parser.add_argument("--distilled-samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.80)
    args = parser.parse_args()

    benchmark = read_jsonl(args.benchmark)
    benchmark_sha256 = sha256_file(args.benchmark)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline, baseline_samples = rescore_one(
        benchmark,
        benchmark_sha256,
        args.baseline_samples,
        args.output_dir / "general-baseline-1.5b-full-v2.json",
    )
    distilled, distilled_samples = rescore_one(
        benchmark,
        benchmark_sha256,
        args.distilled_samples,
        args.output_dir / "general-distilled-1.5b-full-v2.json",
    )
    comparison = build_comparison(
        baseline, distilled, baseline_samples, distilled_samples, args.threshold
    )
    comparison_path = args.output_dir / "general-comparison-full-v2.json"
    write_json(comparison_path, comparison)
    markdown_path = args.output_dir / "general-comparison-full-v2.md"
    write_markdown(markdown_path, comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
