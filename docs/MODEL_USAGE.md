# Qwen2.5-1.5B Ward Distilled v4 Balanced

> 独立仓库说明（2026-10-27）：本页保留原服务器部署说明和较完整的任务示例。新设备请优先使用仓库根README中的 `scripts/start_model.py`、`scripts/ward_cli.py`，不依赖原AutoDL包装命令。权重未随仓库提供；新增50条测试及当前限制见根README和 `docs/EVALUATION.md`。本页中的AutoDL路径只作历史位置记录。

## Final deployable model

- GGUF: `/root/autodl-tmp/qwen/models/Qwen2.5-1.5B-Ward-Distilled-v4-GGUF/qwen2.5-1.5b-ward-v4-q6_k.gguf`
- Quantization: `Q6_K`
- File size: `1,272,739,456 bytes` (`1.273 GB`, `1213.78 MiB`)
- SHA-256: `f7ffc9751378a8f67bed3a6a8872f0f20c40b48cc4223b5560e225bd547ca8a0`
- SHA-256 short form: `f7ffc9751378a8f6`

The previous v3 model remains available and was not overwritten.

## What changed

1. Built a leakage-free 5,000-row mixed training set (`v4.1.0`):
   - 2,000 ward-domain rows
   - 1,500 official GSM8K training rows
   - 1,500 MBPP-derived code execution/output reasoning rows
2. Trained a general-capability LoRA from the original Qwen2.5-1.5B-Instruct base.
3. Fused the established ward LoRA and the new general LoRA by exact rank concatenation.
4. Screened multiple fusion weights. The selected balance is:
   - ward delta weight: `0.50`
   - general delta weight: `0.75`
5. Merged the selected adapter and converted it to Q6_K GGUF.

The frozen 600-question benchmark was not used as training data. The builder
reported zero exact overlap with the frozen GSM8K test questions, and CRUXEval
was not used as a training source.

## Full evaluation results

All general-capability figures below use the same frozen 600-question benchmark,
temperature 0, and the same symmetric relaxed answer extractor.

| Model | GSM8K (200) | CRUXEval-O (200) | BBH (200) | Macro |
|---|---:|---:|---:|---:|
| Original Qwen2.5-1.5B | 69.0% | 26.0% | 31.0% | 42.0% |
| Previous ward v3 Q6_K | 47.5% | 16.5% | 41.0% | 35.0% |
| New ward v4 Q6_K | 60.0% | 25.0% | 35.0% | 40.0% |

New v4 versus original 1.5B retention:

- Math: `86.96%`
- Code: `96.15%`
- Natural-language/logic: `112.90%`
- Macro retention: `95.24%`

New v4 versus previous v3 Q6_K:

- Math: `+12.5 percentage points`
- Code: `+8.5 percentage points`
- BBH: `-6.0 percentage points`
- Macro: `+5.0 percentage points`

Ward-domain full evaluation (200 samples):

- Cloud judgment accuracy: `99.0%`
- Cloud judgment macro-F1: `99.14%`
- Event urgency accuracy: `82.0%`
- Format valid rate: `100.0%`
- Mean reference similarity: `0.4673`

This is a trade-off model: math and code improved substantially over v3, while
BBH and ward urgency are slightly lower than v3. The core ward judgment remains
99% accurate on this test set.

## Memory and speed check

Strict single-instance deployment profile:

```bash
--n-gpu-layers 20 --ctx-size 4096 --parallel 1 \
--cache-type-k q8_0 --cache-type-v q8_0
```

- Idle GPU memory: `1438 MiB`
- GPU memory after one request: `1442 MiB`
- One short ward request total time: `0.370 s`
- Prompt processing: `89 ms`
- Generation: about `44.4 tokens/s`

The test establishes runtime GPU memory below 1.5 GiB for this exact profile.
It does not establish a 75% TTFT reduction because a paired original-model TTFT
test has not yet been run.

## 部署与使用

### 1. 部署前准备

推荐使用支持 CUDA 的 `llama.cpp/llama-server` 加载 GGUF。当前验证过的部署条件为：

- 模型文件：`qwen2.5-1.5b-ward-v4-q6_k.gguf`
- 量化方式：`Q6_K`
- 上下文长度：`4096`
- 并发槽位：`1`
- GPU 卸载层数：`20`
- KV Cache：`Q8_0`
- 实测显存：约 `1.44 GiB`

部署前先校验权重，避免文件下载不完整：

```bash
sha256sum qwen2.5-1.5b-ward-v4-q6_k.gguf
```

正确结果应为：

```text
f7ffc9751378a8f67bed3a6a8872f0f20c40b48cc4223b5560e225bd547ca8a0
```

### 2. 在现有 AutoDL 环境启动

服务器已经安装了启动脚本时，可以直接执行：

```bash
qwen-start-v4
qwen-status
```

正常状态示例：

```text
running model=qwen2.5-1.5b-ward-v4-q6 port=8000
```

停止服务：

```bash
qwen-stop
```

### 3. 在其他设备使用 llama-server 部署

将下面的模型路径和 `llama-server` 路径替换为实际位置：

```bash
llama-server \
  --model /path/to/qwen2.5-1.5b-ward-v4-q6_k.gguf \
  --alias qwen2.5-1.5b-ward-v4-q6 \
  --host 127.0.0.1 \
  --port 8000 \
  --n-gpu-layers 20 \
  --ctx-size 4096 \
  --parallel 1 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0
```

默认只监听 `127.0.0.1`。如果需要允许其他设备访问，应另外配置身份验证、防火墙和 TLS，不建议直接把无认证的 `8000` 端口暴露到公网。

检查服务和实际模型名称：

```bash
curl http://127.0.0.1:8000/v1/models
```

接口根地址为：

```text
http://127.0.0.1:8000/v1
```

这是程序调用接口，不是聊天网页。直接在浏览器打开 `/v1` 不能进行对话。

### 4. 命令行快速使用

已有 `qwen-chat` 命令时，可以直接提问：

```bash
qwen-chat 'B07患者疑似跌倒，置信度0.93，网络断开，请给出简洁处置建议。'
```

这种方式适合人工测试。项目正式接入应使用下文的 OpenAI 兼容接口，并由程序添加任务对应的系统提示。

### 5. OpenAI 兼容接口请求格式

请求地址：

```text
POST http://127.0.0.1:8000/v1/chat/completions
Content-Type: application/json
```

外层请求必须使用以下结构：

```json
{
  "model": "qwen2.5-1.5b-ward-v4-q6",
  "messages": [
    {
      "role": "system",
      "content": "你是智慧病房护理助手。只提供护理辅助建议，不做诊断。"
    },
    {
      "role": "user",
      "content": "床位B07发生疑似跌倒事件，置信度0.93，请给出处置建议。"
    }
  ],
  "temperature": 0,
  "max_tokens": 180,
  "stream": false
}
```

调用示例：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-1.5b-ward-v4-q6",
    "messages": [
      {"role": "system", "content": "你是智慧病房护理助手。只提供护理辅助建议，不做诊断。"},
      {"role": "user", "content": "床位B07发生疑似跌倒事件，置信度0.93，请给出处置建议。"}
    ],
    "temperature": 0,
    "max_tokens": 180,
    "stream": false
  }'
```

如果 `/v1/models` 返回的模型名称不同，应把请求中的 `model` 改成接口实际返回的名称。

### 6. 病房事件统一数据格式

项目代码向模型构造提示前，建议先把事件整理成以下 JSON 对象：

```json
{
  "event_type": "fall_suspected",
  "description": "B07患者倒地，摄像头连续8帧检测到地面人体",
  "priority": "P1",
  "context": {
    "ward": "W-01",
    "bed": "B07",
    "node_id": "EDGE-W01-B07",
    "confidence": 0.93,
    "occurred_at": "2026-08-28T14:32:00+08:00",
    "network_state": "disconnected",
    "rule_hits": [
      "camera.posture=lying_floor",
      "bed_sensor.on_bed=false"
    ],
    "details": {
      "posture": "lying_floor",
      "on_bed": false,
      "observation_quality": "good",
      "consecutive_frames": 8
    }
  }
}
```

字段要求：

| 字段 | 类型 | 要求 |
|---|---|---|
| `event_type` | string | 必填，使用稳定的英文事件标识 |
| `description` | string | 必填，简洁描述现场事实，不加入未经确认的诊断 |
| `priority` | string | 必填，只能是 `P1`、`P2`、`P3` |
| `context` | object | 必填，保存床位、置信度、网络状态及传感器证据 |
| `context.ward` | string | 建议填写病区编号 |
| `context.bed` | string | 建议填写床位编号，不要填写患者姓名 |
| `context.node_id` | string | 可选，边缘节点编号 |
| `context.confidence` | number | 必填，范围为 `0.0` 至 `1.0` |
| `context.occurred_at` | string | 建议使用带时区的 ISO 8601 时间 |
| `context.network_state` | string | 必填，只能是 `connected`、`degraded`、`disconnected` |
| `context.rule_hits` | array[string] | 可选，触发的确定性规则 |
| `context.details` | object | 可选，保存不同事件特有的传感器数据 |

训练数据覆盖较多的事件标识包括：

```text
fall_suspected, fall_prediction, seizure, nurse_call,
bed_leave, abnormal_posture, long_still, night_wandering,
door_departure, bedsore_risk, environment_anomaly,
device_fault, node_offline
```

模型也能接收其他事件名称，但没有经过专项评测的事件类型不能宣称具有同等准确率。

优先级语义：

| 优先级 | 含义 | 事件增强等级 |
|---|---|---|
| `P1` | 需要立即处置的高风险事件 | `紧急` |
| `P2` | 需要尽快人工确认 | `警告` |
| `P3` | 记录、观察或常规处理 | `提醒` |

### 7. 七类病房任务输入输出契约

模型本身不会读取 OpenAI 请求体之外自定义的顶层 `task` 字段。项目中的 `task_router` 应先选择任务，再把对应系统提示和事件内容放入 `messages`。

#### 7.1 事件语义增强 `event_enhancement`

系统提示：

```text
你是智慧病房事件播报助手。根据输入事件生成一句简洁中文描述。
输出必须以【紧急】、【警告】或【提醒】开头，不做诊断。
```

用户输入可使用前述事件 JSON，或由程序转换为以下文本：

```text
床位B07检测到【疑似跌倒】事件。
置信度: 93%，优先级: P1
详细数据: posture=lying_floor; on_bed=false; observation_quality=good
请给出一句话状况描述和处置建议。
```

预期输出：

```text
【紧急】B07疑似跌倒信号明确。立即到场评估意识和外伤，并启动跌倒处置流程
```

#### 7.2 护理建议 `nursing_advice`

系统提示：

```text
你是智慧病房护理建议助手。只输出1至3条可执行护理建议，不做诊断。
```

用户输入：一个符合统一格式的事件 JSON，并在末尾注明：

```text
请给出1至3条可执行护理建议。
```

预期输出示例：

```text
立即到床旁确认患者意识和外伤；保持患者原位并通知值班医生；记录事件和生命体征
```

#### 7.3 离线应急决策 `offline_decision`

系统提示：

```text
你是智慧病房离线应急决策助手。当前网络不可用。
严格只输出包含action、ring、reason的JSON，不输出其他文字。
```

输入事件的 `context.network_state` 应为 `degraded` 或 `disconnected`。

输出必须满足：

```json
{
  "action": "escalate",
  "ring": true,
  "reason": "高置信度疑似跌倒，需要立即人工处置"
}
```

字段限制：

- `action`：只能是 `immediate_response`、`escalate`、`record`。
- `ring`：只能是 JSON 布尔值 `true` 或 `false`，不能写成字符串。
- `reason`：简短说明依据，不包含诊断结论。

#### 7.4 云端研判 `cloud_judgment`

系统提示：

```text
你是智慧病房云端事件研判助手。判断事件是确认、误报还是需要升级复核。
严格只输出 judgment|confidence|advice。
```

用户输入：统一事件 JSON，可以附加自然语言摘要和其他观测源证据。

预期输出：

```text
confirm|0.93|立即到场评估意识和外伤，并启动跌倒处置流程
```

标签限制：

- `confirm`：证据充分，确认事件。
- `reject`：正常活动、设备伪影或明确误报。
- `escalate`：证据冲突或不足，需要人工复核。
- 第二段置信度必须是 `0.0` 至 `1.0` 的小数。
- 第三段是可执行建议，内部不要再使用竖线 `|`。

#### 7.5 活动实时播报 `activity_broadcast`

系统提示：

```text
你是智慧病房活动播报助手。只输出一句简洁播报，不做诊断。
```

用户输入：统一事件 JSON，并注明“请生成一句活动实时播报”。

预期输出：

```text
B01于14:32从睡眠中醒来坐起
```

#### 7.6 YOLO日志摘要 `yolo_log_summary`

系统提示：

```text
你是智慧病房视觉日志摘要助手。只输出一段事件摘要，不复述全部原始日志。
```

用户输入必须是 JSON 数组，每个元素是一条结构化事件：

```json
[
  {
    "event_type": "bed_leave",
    "description": "B01离床40秒",
    "priority": "P2",
    "context": {
      "bed": "B01",
      "confidence": 0.86,
      "network_state": "connected",
      "details": {
        "absence_seconds": 40,
        "on_bed": false
      }
    }
  }
]
```

预期输出是一段简洁摘要，不要求固定 JSON 格式。

#### 7.7 时段护理摘要 `periodic_summary`

系统提示：

```text
你是智慧病房时段护理摘要助手。只输出一段简洁护理摘要，突出高风险事件和待处理事项。
```

用户输入同样是事件 JSON 数组，按时间顺序排列。建议每次只汇总一个固定时段，并在进入模型前由程序限制日志数量，避免超过 4096 上下文。

预期输出示例：

```text
本时段B01发生一次高置信度离床事件，已建议床旁确认；B03设备信号短暂中断，需检查摄像头连接，其余床位无高风险告警。
```

### 8. 输出解析要求

| 任务 | 程序解析方式 |
|---|---|
| `event_enhancement` | 提取开头的 `【紧急】`、`【警告】`、`【提醒】` |
| `nursing_advice` | 作为自然语言建议展示，必要时按分号或换行拆分 |
| `offline_decision` | 使用标准 JSON 解析器，并校验三个字段的类型和枚举 |
| `cloud_judgment` | 按前两个 `|` 分割为标签、置信度和建议 |
| `activity_broadcast` | 作为单句文本展示 |
| `yolo_log_summary` | 作为摘要文本展示 |
| `periodic_summary` | 作为摘要文本展示 |

不要根据字符串中是否出现某个关键词来执行不可逆操作。即使输出格式正确，也应在项目代码中再次进行 Schema 校验，并保留人工复核和规则兜底。

### 9. Python调用示例

```python
import requests

event = {
    "event_type": "fall_suspected",
    "description": "B07患者倒地，摄像头连续8帧检测到地面人体",
    "priority": "P1",
    "context": {
        "ward": "W-01",
        "bed": "B07",
        "confidence": 0.93,
        "network_state": "disconnected",
        "details": {
            "posture": "lying_floor",
            "on_bed": False,
            "observation_quality": "good"
        }
    }
}

response = requests.post(
    "http://127.0.0.1:8000/v1/chat/completions",
    json={
        "model": "qwen2.5-1.5b-ward-v4-q6",
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是智慧病房离线应急决策助手。当前网络不可用。"
                    "严格只输出包含action、ring、reason的JSON，不输出其他文字。"
                )
            },
            {
                "role": "user",
                "content": __import__("json").dumps(event, ensure_ascii=False)
            }
        ],
        "temperature": 0,
        "max_tokens": 180,
        "stream": False
    },
    timeout=60
)
response.raise_for_status()
answer = response.json()["choices"][0]["message"]["content"]
print(answer)
```

### 10. 输入安全与使用边界

- 输入前必须校验必填字段、枚举、置信度范围和 JSON 类型。
- 不要把患者姓名、身份证号、电话号码等直接发送给模型，优先使用病区和床位编号。
- 传感器原始帧级日志应先由 `summarize_yolo_log` 压缩，再交给模型。
- `temperature` 推荐设为 `0`，减少同一事件多次调用时的格式波动。
- 模型只能提供护理辅助建议，不能替代诊断、医嘱和医护人员决策。
- 高风险事件不能只依赖模型输出，应保留确定性告警规则、人工复核和超时兜底。
- 新增 50 条测试显示模型对 `reject` 类反证场景识别仍弱，不应让模型无人值守地自动关闭告警。

### 11. 常见问题

**访问 `http://127.0.0.1:8000/v1` 没有聊天页面**  
这是正常现象。它是 API 根地址，应调用 `/v1/chat/completions`，或者另外安装 WebUI。

**接口返回 model not found**  
先调用 `/v1/models`，将请求中的 `model` 改成服务实际返回的名称。

**模型没有严格输出 JSON 或竖线格式**  
确认使用了对应任务的 `system` 提示、`temperature: 0`，并在项目端校验格式；格式不合规时可以重试一次，之后必须交由规则或人工兜底。

**模型把正常护理活动识别成异常**  
这是当前模型已知限制。请在输入的 `context.details` 中明确提供“护士陪同”“计划内训练”“更换床单”“探头脱落”等反证信息，并保留人工复核。

## Evidence files

- General Q6 full report: `datasets/ward-nlu-500-v1/distillation/v4/general-v4-q6-full.json`
- General Q6 raw samples: `datasets/ward-nlu-500-v1/distillation/v4/general-v4-q6-full.samples.jsonl`
- Ward Q6 full report: `datasets/ward-nlu-500-v1/distillation/v4/domain-v4-q6-full.json`
- Selected pre-quantization full report: `datasets/ward-nlu-500-v1/distillation/v4/general-combo-a050-full.json`
- Selected pre-quantization ward report: `datasets/ward-nlu-500-v1/distillation/v4/domain-combo-a050-full.json`
- Mixed-data manifest: `datasets/ward-nlu-500-v1/distillation/v4/manifest-v4.1.0.json`
- Adapter fusion manifest: `/root/autodl-tmp/qwen/models/Qwen2.5-1.5B-Ward-LoRA-v4-combo-a050/combination_manifest.json`
