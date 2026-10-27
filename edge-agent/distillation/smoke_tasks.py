#!/usr/bin/env python3
"""Smoke-test the seven distilled ward tasks against the running API."""

import argparse
import json
import os
import re
import urllib.request
from pathlib import Path
from api_config import resolve_api_key


CASES = [
    ("event_enhancement", "你是智慧病房事件播报助手。只输出一条简洁中文事件描述和处置建议，不作诊断。", "{\"event_type\":\"fall_suspected\",\"description\":\"B09患者从床边滑落，摄像头和床垫信号一致\",\"context\":{\"bed\":\"B09\",\"confidence\":0.94},\"priority\":\"P1\"}\n请增强为自然语言事件，并给出首要处置。"),
    ("nursing_advice", "你是智慧病房护理建议助手。只输出一条可执行的护理建议，不作诊断。", "{\"event_type\":\"bed_exit\",\"description\":\"B08夜间离床且无人陪护\",\"context\":{\"bed\":\"B08\",\"confidence\":0.88},\"priority\":\"P2\"}\n请给出1至3项可执行护理建议。"),
    ("cloud_judgment", "你是智慧病房护理助手。根据事件快速输出：状况、紧急程度（紧急/警告/提醒）、最多3条护理建议。只做护理建议，不做诊断。", "你是智慧病房的AI护理助手。边缘端报告了一起安全事件，请研判其真实性。\n\n事件类型: fall_suspected\n优先级: P1\n置信度: 0.91\n床位: B07\n详情: {\"observation_quality\":\"good\",\"posture\":\"falling\",\"fall_score\":0.86,\"on_bed\":false}\n自然语言摘要: B07患者从床边滑落，摄像头与床垫信号一致。\n\n请给出研判结果，格式为: judgment|confidence|advice\njudgment 取值为 confirm(确认)/reject(误报)/escalate(升级)。"),
    ("offline_decision", "你是智慧病房离线应急决策助手。严格输出JSON，不作诊断。", "{\"event_type\":\"fall_suspected\",\"description\":\"B07患者倒地\",\"context\":{\"network_state\":\"disconnected\",\"confidence\":0.93},\"priority\":\"P1\"}\n当前断网，请做离线决策，严格输出含action、ring、reason的JSON。"),
    ("activity_broadcast", "你是智慧病房活动播报助手。只输出一句中文播报。", "{\"description\":\"患者从睡眠中醒来坐起\",\"context\":{\"bed\":\"B01\",\"occurred_at\":\"2026-08-09T14:32:00Z\"}}\n请生成一句活动实时播报。"),
    ("yolo_log_summary", "你是智慧病房视觉日志摘要助手。只输出事件摘要，不复述逐帧日志。", "14:31:58 B01 lying conf=0.91\n14:31:59 B01 lying conf=0.92\n14:32:00 B01 sitting conf=0.89\n14:32:01 B01 sitting conf=0.90\n请压缩为事件摘要。"),
    ("periodic_summary", "你是智慧病房时段护理摘要助手。只输出一段简洁中文摘要。", "14:02 B01翻身；14:11 B02正常离床；14:18 B03疑似跌倒并已通知护士；14:25 B03护士到场。\n请生成本时段护理摘要。"),
]


def valid(task: str, text: str) -> bool:
    if not text.strip():
        return False
    if task == "event_enhancement":
        return bool(re.search(r"【(紧急|警告|提醒)】", text)) and len(text) <= 120
    if task == "cloud_judgment":
        return bool(re.match(r"^(confirm|reject|escalate)\|(?:0(?:\.\d+)?|1(?:\.0+)?)\|.+", text, re.S))
    if task == "offline_decision":
        try:
            obj = json.loads(text)
            return obj.get("action") in {"immediate_response", "escalate", "record"} and isinstance(obj.get("ring"), bool)
        except json.JSONDecodeError:
            return False
    limits = {"nursing_advice": 120, "activity_broadcast": 80, "yolo_log_summary": 160, "periodic_summary": 220}
    return len(text) <= limits[task]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5-1.5b-ward-v4-q6")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--api-key-file", type=Path, default=None)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    key = resolve_api_key(args.api_key_file)
    results = []
    for task, system, user in CASES:
        payload = json.dumps({"model": args.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "temperature": 0, "max_tokens": 180}, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(args.base_url.rstrip("/") + "/chat/completions", data=payload, headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=180) as response:
            answer = json.loads(response.read().decode("utf-8"))["choices"][0]["message"]["content"].strip()
        results.append({"task": task, "passed": valid(task, answer), "answer": answer})
    report = {"passed": sum(r["passed"] for r in results), "total": len(results), "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] == report["total"] else 1)


if __name__ == "__main__":
    main()
