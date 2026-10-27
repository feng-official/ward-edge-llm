# 模型输入与七类任务

推荐由调用程序添加系统提示，不需要使用者每次手动输入角色和格式要求。`scripts/ward_cli.py` 已实现这一过程；`--dry-run` 可以查看实际请求。

## 事件对象

`event_enhancement`、`nursing_advice`、`cloud_judgment`、`offline_decision` 接收事件JSON。

| 字段 | 类型与范围 |
|---|---|
| event_type | 非空字符串，建议采用项目事件枚举 |
| description | 非空现场事实描述 |
| priority | P1 / P2 / P3 |
| context.bed | 非空床位编号 |
| context.confidence | 0至1的有限数值，不是百分比字符串 |
| context.network_state | connected / degraded / disconnected |
| context.details | 可选，传感器证据对象 |
| context.occurred_at | 可选，带时区的ISO 8601时间 |
| context.rule_hits | 可选，触发规则列表 |

完整例子见 `examples/event.json`。P1、P2、P3分别表示立即处置、尽快人工确认、记录观察或常规处理；对应事件增强标签为“紧急”“警告”“提醒”。这属于项目标签规则，不代表模型一定正确遵守。

CLI对事件增强和云端裁决使用原训练数据类似的中文文本模板；其他事件任务将JSON序列化到用户消息。日志与时段摘要则直接使用输入文本。

## 任务及输出

| task | CLI输入 | 预期输出形态 |
|---|---|---|
| event_enhancement | 事件JSON | `【紧急/警告/提醒】床位+状况+首要建议` |
| nursing_advice | 事件JSON | 1至3条简洁护理建议 |
| cloud_judgment | 事件JSON | `confirm/reject/escalate\|0至1置信度\|建议` |
| offline_decision | 网络为degraded/disconnected的事件JSON | `{"action":"escalate","ring":true,"reason":"..."}` |
| activity_broadcast | 活动文本 | 一句床位、时间、活动播报 |
| yolo_log_summary | 帧级日志文本 | 合并重复帧后的事件摘要 |
| periodic_summary | 时段事件文本 | 风险、已完成处置、待跟进事项 |

离线决策的 `action` 只允许 `immediate_response`、`escalate`、`record`；`ring` 必须是JSON布尔值。格式要求只约束模型输出，不会自动执行呼铃或业务动作。

```bash
python scripts/ward_cli.py --task activity_broadcast --input examples/activity.txt
python scripts/ward_cli.py --task yolo_log_summary --input examples/yolo-log.txt
python scripts/ward_cli.py --task periodic_summary --input examples/periodic-events.txt
```

## 直接使用HTTP接口

向 `POST /v1/chat/completions` 发送：

```json
{
  "model": "qwen2.5-1.5b-ward-v4-q6",
  "messages": [
    {"role": "system", "content": "你是智慧病房事件研判助手。严格只输出 judgment|confidence|advice。"},
    {"role": "user", "content": "事件类型: fall_suspected\n优先级: P1\n置信度: 0.93\n床位: B07\n详情: 摄像头和床垫信号一致\n请按judgment|confidence|advice格式输出。"}
  ],
  "temperature": 0,
  "max_tokens": 180,
  "stream": false
}
```

实际文本在 `choices[0].message.content` 中。接口不会根据任意顶层 `task` 字段自动选择任务；任务由CLI或上层路由器转成 `messages`。若服务的模型别名不同，使用 `--model` 指定实际名称。
