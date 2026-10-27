#!/usr/bin/env python3
"""Generate multi-task offline distillation data with the Qwen2.5-14B teacher."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from api_config import resolve_api_key


SYSTEM = (
    "你是Qwen2.5-14B智慧病房教师模型。你的回答将用于训练小模型。"
    "只依据输入，不编造诊断；优先保证患者安全；建议必须短、明确、可执行。"
)

TASK_SYSTEMS = {
    "event_enhancement": "你是智慧病房事件播报助手。只输出一条简洁中文事件描述和处置建议，不作诊断。",
    "nursing_advice": "你是智慧病房护理建议助手。只输出一条可执行的护理建议，不作诊断。",
    "cloud_judgment": "你是智慧病房事件研判助手。严格按 judgment|confidence|advice 输出。",
    "offline_decision": "你是智慧病房离线应急决策助手。严格输出JSON，不作诊断。",
    "activity_broadcast": "你是智慧病房活动播报助手。只输出一句中文播报。",
    "yolo_log_summary": "你是智慧病房视觉日志摘要助手。只输出事件摘要，不复述逐帧日志。",
    "periodic_summary": "你是智慧病房时段护理摘要助手。只输出一段简洁中文摘要。",
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def api_chat(base_url: str, api_key: str, model: str, messages: list[dict], max_tokens: int = 500) -> str:
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.15,
        "top_p": 0.85,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"].strip()
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"teacher request failed: {last_error}")


def extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def event_prompt(record: dict) -> str:
    compact = {
        "id": record["id"],
        "event_type": record["event_type"],
        "description": record["description"],
        "context": record["context"],
        "verified_labels": record["labels"],
    }
    return (
        "根据下列人工核验事件生成训练答案。人工标签的priority、urgency、judgment不可改动。\n"
        + json.dumps(compact, ensure_ascii=False)
        + "\n严格输出一个JSON对象，且只含以下字符串字段：\n"
        "event_enhancement：格式为【紧急/警告/提醒】+床位+状况+一项首要处置，不超过80个汉字；\n"
        "nursing_advice：1至3项可执行建议，用中文分号隔开，不超过80个汉字；\n"
        "cloud_judgment：格式 judgment|confidence|advice，judgment必须等于人工标签；\n"
        "offline_decision：JSON字符串，键为action、ring、reason，action仅可为immediate_response/escalate/record；\n"
        "activity_broadcast：格式为“床位 于 HH:MM 活动描述”，一句话、不超过50个汉字。"
    )


def fallback_payload(record: dict) -> dict:
    labels = record["labels"]
    ctx = record["context"]
    urgency = labels["urgency"]
    bed = ctx["bed"]
    hhmm = str(ctx["occurred_at"])[11:16]
    action = "immediate_response" if labels["priority"] == "P1" else ("escalate" if labels["judgment"] == "escalate" else "record")
    return {
        "event_enhancement": f"【{urgency}】{bed}{record['description'][-35:]}。{labels['advice']}",
        "nursing_advice": labels["advice"],
        "cloud_judgment": f"{labels['judgment']}|{ctx['confidence']:.2f}|{labels['advice']}",
        "offline_decision": json.dumps({"action": action, "ring": labels["priority"] == "P1", "reason": f"{labels['priority']}/{labels['judgment']}"}, ensure_ascii=False),
        "activity_broadcast": f"{bed} 于 {hhmm} 出现{record['event_type']}事件，请护理人员关注",
    }


def validate_payload(record: dict, payload: dict) -> tuple[dict, list[str]]:
    fallback = fallback_payload(record)
    errors: list[str] = []
    for key in fallback:
        value = payload.get(key)
        if key == "offline_decision" and isinstance(value, dict):
            payload[key] = json.dumps(value, ensure_ascii=False)
            continue
        if not isinstance(value, str) or not value.strip():
            payload[key] = fallback[key]
            errors.append(f"{key}:missing")
        else:
            payload[key] = value.strip()
    expected_urgency = record["labels"]["urgency"]
    if f"【{expected_urgency}】" not in payload["event_enhancement"]:
        # Keep the teacher's wording, but normalize the project-required prefix.
        payload["event_enhancement"] = f"【{expected_urgency}】" + payload["event_enhancement"].lstrip("【】紧急警告提醒：: ")
    if len(payload["event_enhancement"]) > 120:
        payload["event_enhancement"] = payload["event_enhancement"][:120]
    expected_judgment = record["labels"]["judgment"]
    if not payload["cloud_judgment"].startswith(expected_judgment + "|"):
        payload["cloud_judgment"] = fallback["cloud_judgment"]
        errors.append("cloud_judgment:label")
    if isinstance(payload.get("offline_decision"), dict):
        payload["offline_decision"] = json.dumps(payload["offline_decision"], ensure_ascii=False)
    try:
        offline = json.loads(payload["offline_decision"])
        if offline.get("action") not in {"immediate_response", "escalate", "record"} or not isinstance(offline.get("ring"), bool):
            raise ValueError
    except (json.JSONDecodeError, ValueError, AttributeError):
        payload["offline_decision"] = fallback["offline_decision"]
        errors.append("offline_decision:json")
    return payload, errors


def task_user(task: str, record: dict) -> str:
    event = {
        "event_type": record["event_type"],
        "description": record["description"],
        "context": record["context"],
        "priority": record["labels"]["priority"],
    }
    rules = {
        "event_enhancement": "请增强为自然语言事件，并给出首要处置。",
        "nursing_advice": "请给出1至3项可执行护理建议。",
        "cloud_judgment": "请研判事件，严格输出 judgment|confidence|advice。",
        "offline_decision": "当前可能断网，请做离线决策，严格输出含action、ring、reason的JSON。",
        "activity_broadcast": "请生成一句活动实时播报。",
    }
    return json.dumps(event, ensure_ascii=False) + "\n" + rules[task]


def make_rows(record: dict, payload: dict, errors: list[str]) -> list[dict]:
    return [{
        "id": f"{record['id']}-{task}",
        "task": task,
        "messages": [
            {"role": "system", "content": TASK_SYSTEMS[task]},
            {"role": "user", "content": task_user(task, record)},
            {"role": "assistant", "content": payload[task]},
        ],
        "teacher": "Qwen2.5-14B-Instruct-AWQ",
        "teacher_validation": "fallback" if any(e.startswith(task + ":") for e in errors) else "passed",
        "source_record_id": record["id"],
    } for task in ("event_enhancement", "nursing_advice", "cloud_judgment", "offline_decision", "activity_broadcast")]


def group_prompt(records: list[dict]) -> str:
    events = [{"description": r["description"], "context": r["context"], "labels": r["labels"]} for r in records]
    return (
        "根据这组连续病房事件生成两种摘要。\n" + json.dumps(events, ensure_ascii=False)
        + "\n严格输出JSON对象：yolo_log_summary为去重后的事件摘要（不超过120字）；"
        "periodic_summary为护理时段摘要，须包含风险事件、处置重点和待跟进事项（不超过160字）。"
    )


def make_group_rows(group_id: int, records: list[dict], payload: dict) -> list[dict]:
    source = json.dumps([{"description": r["description"], "context": r["context"]} for r in records], ensure_ascii=False)
    rows = []
    for task in ("yolo_log_summary", "periodic_summary"):
        answer = payload.get(task, "").strip() if isinstance(payload.get(task), str) else ""
        if not answer:
            answer = "；".join(r["description"] for r in records[:2])[:150]
        rows.append({
            "id": f"group-{group_id:04d}-{task}",
            "task": task,
            "messages": [
                {"role": "system", "content": TASK_SYSTEMS[task]},
                {"role": "user", "content": source + ("\n请压缩为事件摘要。" if task == "yolo_log_summary" else "\n请生成本时段护理摘要。")},
                {"role": "assistant", "content": answer},
            ],
            "teacher": "Qwen2.5-14B-Instruct-AWQ",
            "teacher_validation": "passed" if payload.get(task) else "fallback",
            "source_record_id": ",".join(r["id"] for r in records),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key-file", type=Path, default=None)
    parser.add_argument("--model", default="qwen2.5-14b")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    api_key = resolve_api_key(args.api_key_file)
    records = load_jsonl(args.input)
    if args.limit:
        records = records[:args.limit]
    completed: set[str] = set()
    if args.output.exists():
        completed = {row["source_record_id"] for row in load_jsonl(args.output) if row["task"] == "event_enhancement"}
    pending = [r for r in records if r["id"] not in completed]
    stats = {"records": len(records), "generated_records": 0, "fallback_fields": 0, "failed_requests": 0}

    def generate_one(record: dict) -> tuple[dict, dict, list[str]]:
        try:
            raw = api_chat(args.base_url, api_key, args.model, [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": event_prompt(record)},
            ])
            payload = extract_json(raw)
            payload, errors = validate_payload(record, payload)
        except Exception as exc:
            payload, errors = fallback_payload(record), [f"request:{type(exc).__name__}"]
        return record, payload, errors

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(generate_one, r) for r in pending]
        for index, future in enumerate(as_completed(futures), 1):
            record, payload, errors = future.result()
            append_jsonl(args.output, make_rows(record, payload, errors))
            stats["generated_records"] += 1
            stats["fallback_fields"] += len(errors)
            stats["failed_requests"] += int(any(e.startswith("request:") for e in errors))
            if index % 20 == 0 or index == len(futures):
                print(f"teacher events {index}/{len(futures)} fallbacks={stats['fallback_fields']}", flush=True)

    groups = [records[i:i + 5] for i in range(0, len(records) - 4, 5)]
    existing_groups = {row["id"].split("-")[1] for row in load_jsonl(args.output) if row["id"].startswith("group-")}
    for group_id, group in enumerate(groups, 1):
        if f"{group_id:04d}" in existing_groups:
            continue
        try:
            raw = api_chat(args.base_url, api_key, args.model, [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": group_prompt(group)},
            ], max_tokens=350)
            payload = extract_json(raw)
        except Exception:
            payload = {}
            stats["failed_requests"] += 1
        append_jsonl(args.output, make_group_rows(group_id, group, payload))
        if group_id % 20 == 0 or group_id == len(groups):
            print(f"teacher groups {group_id}/{len(groups)}", flush=True)

    report = args.output.with_suffix(".report.json")
    report.write_text(json.dumps(stats | {"output_rows": sum(1 for _ in args.output.open(encoding="utf-8"))}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
