#!/usr/bin/env python3
"""Build one of seven ward prompts and call any compatible Chat Completions API."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    'event_enhancement': '你是智慧病房事件播报助手。输出一句简洁状况和护理建议，以【紧急】、【警告】或【提醒】开头。只依据输入，不做诊断。',
    'nursing_advice': '你是智慧病房护理建议助手。只依据输入，输出1至3条简洁可执行的护理建议，不做诊断。',
    'cloud_judgment': '你是智慧病房事件研判助手。严格只输出 judgment|confidence|advice。judgment为confirm/reject/escalate，confidence为0至1的数字。',
    'offline_decision': '你是智慧病房离线应急决策助手。严格只输出action、ring、reason三个键的JSON；action仅可为immediate_response/escalate/record，ring为布尔值。只依据输入，不做诊断。',
    'activity_broadcast': '你是智慧病房活动播报助手。只输出一句包含床位、时间和活动事实的中文播报，不编造输入中没有的信息。',
    'yolo_log_summary': '你是智慧病房视觉日志摘要助手。只输出合并重复帧后的事件摘要，保留时间、床位和活动变化，不编造未观察到的事件。',
    'periodic_summary': '你是智慧病房时段护理摘要助手。简要汇总活动、风险事件、已完成处置与待跟进事项，区分已经确认和仍待确认的信息。',
}
EVENT_TASKS = {'event_enhancement', 'nursing_advice', 'cloud_judgment', 'offline_decision'}
EVENT_NAMES = {
    'fall_suspected': '疑似跌倒', 'seizure': '抽搐检测', 'fall_prediction': '坠床预警',
    'nurse_call': '护士呼叫', 'bed_leave': '离床', 'abnormal_posture': '异常姿态',
    'long_still': '长时间静止', 'night_wandering': '夜间游走', 'door_departure': '离开病区',
    'bedsore_risk': '压疮风险', 'environment_anomaly': '环境异常', 'device_fault': '设备故障',
    'node_offline': '节点离线',
}


def validate_event(event: object) -> dict:
    if not isinstance(event, dict):
        raise ValueError('事件输入必须是JSON对象。')
    for key in ('event_type', 'description'):
        if not isinstance(event.get(key), str) or not event[key].strip():
            raise ValueError(f'{key} 必须是非空字符串。')
    if event.get('priority') not in {'P1', 'P2', 'P3'}:
        raise ValueError('priority 必须是P1、P2或P3。')
    ctx = event.get('context')
    if not isinstance(ctx, dict):
        raise ValueError('context 必须是对象。')
    value = ctx.get('confidence')
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('context.confidence 必须是0至1的有限数值。')
    if ctx.get('network_state') not in {'connected', 'degraded', 'disconnected'}:
        raise ValueError('context.network_state 必须是connected、degraded或disconnected。')
    if not isinstance(ctx.get('bed'), str) or not ctx['bed'].strip():
        raise ValueError('context.bed 必须是非空床位编号。')
    if 'details' in ctx and not isinstance(ctx['details'], dict):
        raise ValueError('context.details 必须是对象。')
    return event


def messages_for(task: str, source: str) -> list[dict]:
    if task in EVENT_TASKS:
        event = validate_event(json.loads(source))
        ctx = event['context']
        if task == 'offline_decision' and ctx['network_state'] == 'connected':
            raise ValueError('offline_decision 需要degraded或disconnected网络状态。')
        details = json.dumps(ctx.get('details', {}), ensure_ascii=False)
        if task == 'cloud_judgment':
            user = (f"事件类型: {event['event_type']}\n优先级: {event['priority']}\n"
                    f"置信度: {ctx['confidence']}\n床位: {ctx['bed']}\n详情: {details}\n"
                    f"自然语言摘要: {event['description']}\n请按judgment|confidence|advice格式输出。")
        elif task == 'event_enhancement':
            user = (f"床位{ctx['bed']}检测到【{EVENT_NAMES.get(event['event_type'], event['event_type'])}】事件。\n"
                    f"置信度: {ctx['confidence']:.0%}，优先级: {event['priority']}\n"
                    f"详细数据: {details}\n描述: {event['description']}\n请给出一句话状况描述和处置建议。")
        else:
            user = json.dumps(event, ensure_ascii=False)
    else:
        if not source.strip():
            raise ValueError('输入文本不能为空。')
        user = source.strip()
    return [{'role': 'system', 'content': TASKS[task]}, {'role': 'user', 'content': user}]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=sorted(TASKS), required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path)
    source.add_argument('--text')
    parser.add_argument('--base-url', default=os.getenv('QWEN_BASE_URL', 'http://127.0.0.1:8000/v1'))
    parser.add_argument('--model', default=os.getenv('QWEN_MODEL', 'qwen2.5-1.5b-ward-v4-q6'))
    parser.add_argument('--api-key-file', type=Path)
    parser.add_argument('--max-tokens', type=int, default=180)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--dry-run', action='store_true', help='Print the request without contacting a model.')
    args = parser.parse_args()
    try:
        text = args.input.read_text(encoding='utf-8-sig') if args.input else args.text
        payload = {'model': args.model, 'messages': messages_for(args.task, text),
                   'temperature': 0, 'max_tokens': args.max_tokens, 'stream': False}
        if args.dry_run:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        key = os.getenv('QWEN_API_KEY', '')
        if args.api_key_file and not key:
            key = args.api_key_file.read_text(encoding='utf-8').strip()
        headers = {'Content-Type': 'application/json'}
        if key:
            headers['Authorization'] = f'Bearer {key}'
        req = urllib.request.Request(args.base_url.rstrip('/') + '/chat/completions',
                                     data=json.dumps(payload, ensure_ascii=False).encode('utf-8'), headers=headers)
        with urllib.request.urlopen(req, timeout=args.timeout) as response:
            result = json.loads(response.read().decode('utf-8'))
        print(result['choices'][0]['message']['content'])
        return 0
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        print(f'调用失败：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
