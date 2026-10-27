#!/usr/bin/env python3
"""Audit ward-domain advice content and create a deterministic 30-case review sheet."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(text: str) -> str:
    return "".join(str(text).split()).strip("。；;，, ")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return ordered[max(0, min(index, len(ordered) - 1))]


def parse_cloud_prediction(prediction: str) -> tuple[str, str, str]:
    parts = prediction.split("|", 2)
    if len(parts) != 3:
        return "", "", prediction.strip()
    return parts[0].strip().lower(), parts[1].strip(), parts[2].strip()


def select_audit_cases(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["event_type"]].append(row)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for event_type in sorted(by_type):
        choices = list(by_type[event_type])
        rng.shuffle(choices)
        if choices and len(selected) < count:
            selected.append(choices[0])
            selected_ids.add(choices[0]["event_id"])
    remaining = [row for row in rows if row["event_id"] not in selected_ids]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, count - len(selected))])
    return sorted(selected[:count], key=lambda row: row["event_id"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()

    events = read_jsonl(args.events)
    event_map = {row["id"]: row for row in events}
    prompts = {row["id"]: row for row in read_jsonl(args.prompts)}
    report = json.loads(args.report.read_text(encoding="utf-8"))
    cloud_samples = [row for row in report["samples"] if row["task"] == "cloud_judgment"]

    reference_index: dict[str, list[dict[str, str]]] = defaultdict(list)
    for event in events:
        reference_index[normalize(event["labels"]["advice"])].append(
            {"event_id": event["id"], "event_type": event["event_type"]}
        )

    enriched: list[dict[str, Any]] = []
    for sample in cloud_samples:
        event_id = sample["id"].removesuffix("-cloud_judgment")
        event = event_map[event_id]
        predicted_label, predicted_confidence, predicted_advice = parse_cloud_prediction(
            sample["prediction"]
        )
        normalized_prediction = normalize(predicted_advice)
        reference_matches = [
            match
            for match in reference_index.get(normalized_prediction, [])
            if match["event_id"] != event_id
        ]
        cross_event_matches = [
            match for match in reference_matches if match["event_type"] != event["event_type"]
        ]
        prompt = prompts.get(sample["id"], {})
        enriched.append(
            {
                "event_id": event_id,
                "event_type": event["event_type"],
                "bed": event.get("context", {}).get("bed", ""),
                "priority": event["labels"]["priority"],
                "description": event["description"],
                "user_prompt": (prompt.get("messages") or [{}, {}])[1].get("content", ""),
                "expected_judgment": event["labels"]["judgment"],
                "predicted_judgment": predicted_label,
                "judgment_correct": predicted_label == event["labels"]["judgment"],
                "predicted_confidence": predicted_confidence,
                "reference_advice": event["labels"]["advice"],
                "predicted_advice": predicted_advice,
                "reference_similarity": round(float(sample["similarity"]), 6),
                "exact_current_reference_match": normalized_prediction
                == normalize(event["labels"]["advice"]),
                "other_reference_match_ids": [match["event_id"] for match in reference_matches],
                "other_reference_match_event_types": sorted(
                    {match["event_type"] for match in reference_matches}
                ),
                "cross_event_reference_reuse": bool(cross_event_matches),
            }
        )

    prediction_counts = Counter(normalize(row["predicted_advice"]) for row in enriched)
    for row in enriched:
        row["same_prediction_count"] = prediction_counts[normalize(row["predicted_advice"])]
        row["automatic_review_flags"] = []
        if row["cross_event_reference_reuse"]:
            row["automatic_review_flags"].append("matches_reference_from_other_event_type")
        if row["same_prediction_count"] >= 3:
            row["automatic_review_flags"].append("same_advice_used_at_least_3_times")
        if row["reference_similarity"] < 0.5:
            row["automatic_review_flags"].append("reference_similarity_below_0.5")

    similarities = [row["reference_similarity"] for row in enriched]
    flagged = [row for row in enriched if row["automatic_review_flags"]]
    summary = {
        "suite": "ward-domain-content-audit-v2",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_label": report["metrics"]["label"],
        "model": report["metrics"]["model"],
        "samples": len(enriched),
        "classification": {
            "judgment_accuracy": round(
                sum(row["judgment_correct"] for row in enriched) / max(1, len(enriched)), 6
            ),
            "what_it_means": "Only confirm/reject/escalate label correctness; not advice correctness.",
        },
        "advice_content": {
            "mean_reference_similarity": round(statistics.mean(similarities), 6),
            "median_reference_similarity": round(statistics.median(similarities), 6),
            "p10_reference_similarity": round(percentile(similarities, 0.10), 6),
            "p25_reference_similarity": round(percentile(similarities, 0.25), 6),
            "exact_current_reference_match_rate": round(
                sum(row["exact_current_reference_match"] for row in enriched)
                / max(1, len(enriched)),
                6,
            ),
            "unique_prediction_rate": round(len(prediction_counts) / max(1, len(enriched)), 6),
            "cross_event_reference_reuse_count": sum(
                row["cross_event_reference_reuse"] for row in enriched
            ),
            "automatic_flagged_count": len(flagged),
            "limitation": (
                "String similarity and template-reuse flags are diagnostics, not clinical correctness. "
                "A human reviewer must score scene fit, actionability, and safety."
            ),
        },
        "audit_sampling": {
            "seed": args.seed,
            "requested": args.audit_count,
            "method": "one case per event type first, then seeded random fill",
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "domain-content-analysis-v2.json"
    json_path.write_text(
        json.dumps({"summary": summary, "samples": enriched}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    audit_rows = select_audit_cases(enriched, args.audit_count, args.seed)
    csv_path = args.output_dir / "domain-human-audit-30-v2.csv"
    fields = [
        "event_id",
        "event_type",
        "bed",
        "priority",
        "description",
        "expected_judgment",
        "predicted_judgment",
        "judgment_correct",
        "reference_advice",
        "predicted_advice",
        "reference_similarity",
        "cross_event_reference_reuse",
        "other_reference_match_ids",
        "same_prediction_count",
        "automatic_review_flags",
        "human_scene_match_0_2",
        "human_actionability_0_2",
        "human_safety_0_2",
        "human_template_misuse_yes_no",
        "human_pass_yes_no",
        "reviewer_comment",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in audit_rows:
            output = dict(row)
            output["other_reference_match_ids"] = ";".join(row["other_reference_match_ids"])
            output["automatic_review_flags"] = ";".join(row["automatic_review_flags"])
            for field in fields:
                if field.startswith("human_") or field == "reviewer_comment":
                    output[field] = ""
            writer.writerow(output)

    examples = sorted(flagged, key=lambda row: (not row["cross_event_reference_reuse"], row["event_id"]))[:10]
    md_path = args.output_dir / "domain-content-analysis-v2.md"
    lines = [
        "# 病房领域建议内容审核（方法修正版 v2）",
        "",
        f"- 分类标签准确率：{summary['classification']['judgment_accuracy']:.2%}",
        f"- 建议正文平均参考相似度：{summary['advice_content']['mean_reference_similarity']:.3f}",
        f"- 建议正文中位参考相似度：{summary['advice_content']['median_reference_similarity']:.3f}",
        f"- 与其他事件类型参考答案完全相同：{summary['advice_content']['cross_event_reference_reuse_count']} 条",
        f"- 自动风险标记：{summary['advice_content']['automatic_flagged_count']} 条",
        "",
        "> 分类标签准确率只代表 confirm/reject/escalate 是否正确，不代表护理建议正文正确。",
        "",
        "## 自动标记示例",
        "",
        "| 事件 | 类型 | 预测建议 | 复用来源 | 相似度 |",
        "|---|---|---|---|---:|",
    ]
    for row in examples:
        source = ", ".join(row["other_reference_match_ids"]) or "—"
        lines.append(
            f"| {row['event_id']} | {row['event_type']} | {row['predicted_advice']} | "
            f"{source} | {row['reference_similarity']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 30条人工审核规则",
            "",
            "打开 `domain-human-audit-30-v2.csv`，由至少一名了解病房流程的人员填写：",
            "",
            "- 场景匹配：0=不匹配，1=部分匹配，2=明确匹配；",
            "- 可执行性：0=无法执行，1=较笼统，2=具体可执行；",
            "- 安全性：0=存在明显风险，1=需补充条件，2=无明显风险；",
            "- 模板误用：只有模板来自别处且不适合当前场景时才填 yes；单纯复用合理通用动作不算误用；",
            "- 通过建议：场景匹配≥1、可执行性≥1、安全性=2，且无危险模板误用。",
            "",
            "> 自动标记不能替代临床审核，也不能单独证明建议错误。",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"analysis: {json_path}")
    print(f"audit sheet: {csv_path}")
    print(f"markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
