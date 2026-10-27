#!/usr/bin/env python3
"""Build a deterministic, leakage-checked 50-row smart-ward evaluation set."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


SYSTEM = (
    "你是智慧病房护理助手。根据事件快速输出：状况、紧急程度（紧急/警告/提醒）、"
    "最多3条护理建议。只做护理建议，不做诊断。"
)


SCENARIOS = [
    ("B07", "fall_suspected", "疑似跌倒", "P1", 0.94, "紧急", "confirm", "已确认", "立即到场评估并启动跌倒处置流程", "走廊摄像头连续8帧检测到患者倒地，床垫传感器显示离床，画面清晰", "posture=lying_floor; on_bed=false; observation_quality=good; consecutive_frames=8"),
    ("B12", "fall_suspected", "疑似跌倒", "P1", 0.88, "紧急", "reject", "疑似误报", "确认患者安全后关闭误报并继续巡视", "患者在床边由治疗师陪同做下肢训练，床垫与定位信息稳定", "posture=bending; on_bed=true; therapist_present=true; observation_quality=good"),
    ("B03", "fall_suspected", "疑似跌倒", "P1", 0.68, "紧急", "escalate", "证据冲突", "优先人工复核患者位置，必要时按跌倒事件处理", "夜间低照度出现横卧轮廓，但关键点缺失且床垫信号间歇异常", "posture=unknown; on_bed=unknown; observation_quality=degraded; keypoints=missing"),
    ("B18", "fall_suspected", "疑似跌倒", "P1", 0.96, "紧急", "confirm", "已确认", "立即到卫生间查看患者并通知值班护士", "卫生间门口检测到快速下坠，随后地面压力传感器持续触发", "posture=lying_floor; floor_sensor=true; observation_quality=good"),
    ("B05", "fall_suspected", "疑似跌倒", "P1", 0.82, "紧急", "reject", "疑似误报", "确认陪护操作正常后记录并继续观察", "家属协助患者从轮椅转移到床上，护理呼叫记录与画面一致", "caregiver_present=true; transfer_activity=true; on_bed=true; observation_quality=good"),
    ("B21", "fall_suspected", "疑似跌倒", "P1", 0.71, "紧急", "escalate", "证据不足", "立即人工查看遮挡区域并保持跌倒预警", "床帘遮挡大部分画面，仅检测到短暂下坠轨迹，床垫显示离床", "occlusion=high; on_bed=false; observation_quality=poor; fall_score=0.71"),
    ("B09", "fall_suspected", "疑似跌倒", "P1", 0.91, "紧急", "confirm", "已确认", "立即到场保护患者并评估意识与损伤", "患者倒地后按下呼叫器，摄像头与呼叫记录时间一致", "posture=lying_floor; call_button=true; observation_quality=good"),
    ("B14", "fall_suspected", "疑似跌倒", "P1", 0.79, "紧急", "reject", "疑似误报", "移除地面物品并确认患者仍在床上", "落地毛毯被识别为人体轮廓，但床垫、腕带定位均显示患者在床", "object=blanket; on_bed=true; wrist_location=bed; observation_quality=good"),
    ("B02", "bed_exit", "离床", "P2", 0.93, "警告", "confirm", "已确认", "立即前往床旁协助高跌倒风险患者", "认知障碍患者凌晨独自离床，床旁无人且腕带正在向门口移动", "on_bed=false; caregiver_present=false; wrist_moving_to=door; observation_quality=good"),
    ("B16", "bed_exit", "离床", "P2", 0.90, "警告", "reject", "计划内活动", "记录计划内如厕并保持常规观察", "患者按计划由护士陪同前往卫生间，护理任务单已登记", "on_bed=false; nurse_present=true; scheduled_toilet=true; observation_quality=good"),
    ("B11", "bed_exit", "离床", "P2", 0.64, "警告", "escalate", "状态不明", "人工确认患者位置并检查床垫传感器", "床垫报告离床但定位腕带离线，摄像头被设备车遮挡", "on_bed=false; wrist_online=false; occlusion=high; observation_quality=poor"),
    ("B06", "bed_exit", "离床", "P2", 0.89, "警告", "confirm", "已确认", "立即到场劝阻患者自行下床", "高跌倒风险患者持续坐在床沿并尝试站立，床旁无人", "posture=sitting_edge; stand_attempt=true; caregiver_present=false; observation_quality=good"),
    ("B20", "bed_exit", "离床", "P2", 0.77, "警告", "reject", "传感器扰动", "检查床垫连接并继续观察患者", "更换床单时床垫压力短暂归零，画面显示患者由两名护士安全托扶", "on_bed=unknown; nurse_count=2; linen_change=true; observation_quality=good"),
    ("B24", "bed_exit", "离床", "P2", 0.95, "警告", "confirm", "已确认", "立即通知护士在病区出口拦护患者", "患者未经陪同离床并接近病区出口，摄像头与腕带定位一致", "on_bed=false; wrist_location=ward_exit; caregiver_present=false; observation_quality=good"),
    ("B08", "vital_sign_abnormal", "生命体征异常", "P1", 0.97, "紧急", "confirm", "持续心动过速", "立即复测生命体征并通知值班医生", "心率连续5分钟维持在135次/分，电极接触良好且患者清醒不适", "heart_rate=135; duration_min=5; signal_quality=good; electrode_contact=good"),
    ("B01", "vital_sign_abnormal", "生命体征异常", "P1", 0.74, "紧急", "escalate", "血氧读数可疑", "立即人工复测血氧并检查探头位置", "血氧读数降至82%，但指端探头松动且波形质量很差", "spo2=82; probe_loose=true; signal_quality=poor; waveform=unstable"),
    ("B17", "vital_sign_abnormal", "生命体征异常", "P1", 0.95, "紧急", "confirm", "持续低血压", "立即床旁复测并通知医生评估", "三次测量血压均低于85/50mmHg，袖带位置正确", "bp=82/48; repeats=3; cuff_position=correct; signal_quality=good"),
    ("B10", "vital_sign_abnormal", "生命体征异常", "P2", 0.86, "警告", "reject", "测温误报", "重新放置探头后复测体温并记录设备异常", "体温探头已脱离皮肤却报告39.2摄氏度，耳温复测正常", "temperature=39.2; probe_detached=true; recheck_temperature=36.8; signal_quality=invalid"),
    ("B22", "vital_sign_abnormal", "生命体征异常", "P1", 0.69, "紧急", "escalate", "呼吸数据冲突", "立即人工计数呼吸并复核监护设备", "床旁监护显示呼吸频率8次/分，穿戴设备显示18次/分且患者被被褥遮挡", "resp_rate_monitor=8; resp_rate_wearable=18; occlusion=medium; signal_quality=conflict"),
    ("B13", "device_alarm", "输液泵报警", "P2", 0.96, "警告", "confirm", "管路阻塞报警", "立即检查输液管路和穿刺部位并通知护士", "输液泵连续报告下游阻塞，压力曲线同步升高", "alarm=downstream_occlusion; pressure=high; repeats=4; self_test=pass"),
    ("B04", "device_alarm", "监护设备异常", "P3", 0.92, "提醒", "confirm", "电极脱落", "重新粘贴电极并确认心电信号恢复", "胸前电极脱落，监护仪显示导联中断，患者状态稳定", "alarm=lead_off; electrode_contact=false; patient_stable=true"),
    ("B19", "device_alarm", "设备报警", "P3", 0.84, "提醒", "reject", "设备自检误报", "记录自检结果并继续常规监测", "设备重启时短暂报警，自检通过后连续十分钟运行正常", "restart=true; self_test=pass; stable_duration_min=10; alarm_active=false"),
    ("B15", "inactivity", "长时间静止", "P3", 0.88, "提醒", "reject", "正常睡眠", "维持常规巡视并继续监测生命体征", "夜间睡眠两小时体位未变，但呼吸、心率和床垫信号均稳定", "sleep_period=true; duration_min=120; vitals=stable; on_bed=true"),
    ("B23", "inactivity", "长时间静止", "P2", 0.94, "警告", "confirm", "长时间无反应", "立即到床旁确认意识和生命体征", "白天呼叫患者无回应且90分钟无自主活动，画面清晰", "daytime=true; no_response=true; duration_min=90; observation_quality=good"),
    ("B25", "inactivity", "长时间静止", "P2", 0.66, "警告", "escalate", "状态无法确认", "尽快人工巡视并恢复定位与视频观测", "摄像头画面过暗且定位腕带离线，系统无法确认患者是否在床", "observation_quality=poor; wrist_online=false; on_bed=unknown; duration_min=60"),
]


def message_rows() -> list[dict]:
    rows: list[dict] = []
    for i, (bed, event_type, event_cn, priority, confidence, urgency, judgment,
            state, advice, summary, details) in enumerate(SCENARIOS, 1):
        sid = f"ward50-20260824-{i:02d}"
        event_user = (
            f"床位{bed}检测到【{event_cn}】事件。\n"
            f"置信度: {confidence:.0%}，优先级: {priority}\n"
            f"详细数据: {details}\n"
            f"现场摘要: {summary}\n"
            "请给出一句话状况描述和处置建议。"
        )
        event_answer = f"【{urgency}】{bed}{event_cn}{state}。{advice}"
        rows.append({
            "id": sid + "-event_enhancement",
            "task": "event_enhancement",
            "scenario_id": sid,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": event_user},
                {"role": "assistant", "content": event_answer},
            ],
        })

        cloud_user = (
            "你是智慧病房的AI护理助手。边缘端报告了一起安全事件，请研判其真实性。\n\n"
            f"事件类型: {event_type}\n优先级: {priority}\n置信度: {confidence:.2f}\n"
            f"床位: {bed}\n详情: {details}\n自然语言摘要: {summary}\n\n"
            "请给出研判结果，格式为: judgment|confidence|advice\n"
            "judgment 取值为 confirm(确认)/reject(误报)/escalate(升级)。"
        )
        cloud_answer = f"{judgment}|{confidence:.2f}|{advice}"
        rows.append({
            "id": sid + "-cloud_judgment",
            "task": "cloud_judgment",
            "scenario_id": sid,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": cloud_user},
                {"role": "assistant", "content": cloud_answer},
            ],
        })
    return rows


def load_prompts(path: Path) -> set[str]:
    if not path.exists():
        return set()
    prompts = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            for message in row.get("messages", []):
                if message.get("role") == "user":
                    prompts.add(message.get("content", "").strip())
                    break
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = message_rows()
    assert len(rows) == 50
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids))

    dataset_root = args.project_root / "datasets/ward-nlu-500-v1"
    reference_paths = [
        dataset_root / "qwen-sft-train-v1.0.0.jsonl",
        dataset_root / "qwen-sft-test-v1.0.0.jsonl",
        dataset_root / "distillation/student-train-v2.jsonl",
    ]
    existing = set().union(*(load_prompts(path) for path in reference_paths))
    new_prompts = {
        next(m["content"].strip() for m in row["messages"] if m["role"] == "user")
        for row in rows
    }
    overlap = sorted(existing & new_prompts)
    if overlap:
        raise RuntimeError(f"Exact prompt leakage detected: {len(overlap)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "ward-task-50-v1.jsonl"
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    output.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()

    labels = Counter(
        row["messages"][-1]["content"].split("|", 1)[0]
        for row in rows if row["task"] == "cloud_judgment"
    )
    urgencies = Counter(
        row["messages"][-1]["content"].split("】", 1)[0].lstrip("【")
        for row in rows if row["task"] == "event_enhancement"
    )
    manifest = {
        "dataset": "ward-task-50-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "scenarios": len(SCENARIOS),
        "task_distribution": dict(Counter(row["task"] for row in rows)),
        "judgment_distribution": dict(labels),
        "urgency_distribution": dict(urgencies),
        "exact_prompt_overlap_with_existing_sets": len(overlap),
        "reference_sets_checked": [str(path) for path in reference_paths],
        "sha256": digest,
        "notes": [
            "This is a newly authored deterministic holdout set.",
            "It contains 25 scenario pairs: event enhancement plus cloud judgment.",
            "It is not used for training or adapter selection.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
