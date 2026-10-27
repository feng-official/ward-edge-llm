# Ward Edge LLM｜智慧病房轻量模型

基于 Qwen2.5-1.5B-Instruct 的病房任务蒸馏、LoRA 训练、GGUF 部署与评测项目。教师模型为 Qwen2.5-14B-Instruct-AWQ；当前推荐学生产物为 **Ward-v4-Q6_K**。

本仓库是从智慧病房项目中整理的独立模型子项目，包含病房数据集、训练与评测脚本、API 调用示例、历史实验记录及报告图。上层前端、摄像头/YOLO 服务和 MQTT Broker 不在本仓库内。

## 当前交付状态

- 代码、数据、历史结果可以在不启动原 AutoDL 服务器的情况下查看和上传。
- **GitHub仓库不包含 GGUF 权重和 LoRA 权重**。最终GGUF已发布到 [Hugging Face模型仓库](https://huggingface.co/siwuxie27/Qwen2.5-1.5B-Ward-v4-Q6)，下载后即可按下文部署。
- 原冻结 600 题 `benchmark.jsonl` 未保存在本地。本仓库包含准备脚本、历史 manifest、结果与样本级预测；重新准备题库后必须核对哈希，详见 [评测说明](docs/EVALUATION.md)。
- 本次整理只进行了本地完整性和调用接口测试，没有重新运行模型推理或训练。

## 目录

```text
README.md
models/model-manifest.json       最终权重名称、大小、SHA-256
scripts/start_model.py           跨设备启动 llama-server
scripts/ward_cli.py              自动构造任务提示并调用模型
scripts/verify_repository.py     无需模型和网络的本地检查
scripts/generate_ward_nlu_dataset.py
examples/event.json              病房事件输入样例
edge-agent/distillation/         教师生成、LoRA、融合、评测
datasets/ward-nlu-500-v1/         原始500条、训练/测试视图与历史结果
reports/training-v4/             真实训练loss与超参数记录
reports/local-validation.json    本次离线检查结果
docs/                           使用、训练、评测、上传和图表说明
contracts/                      与原智慧病房项目对接的JSON契约
```

## 快速开始

Python 3.10+。调用与离线检查仅使用 Python 标准库，无需安装 PyTorch。

在仓库根目录执行：

```bash
python scripts/verify_repository.py
python scripts/ward_cli.py --task cloud_judgment --input examples/event.json --dry-run
```

第二条命令仅预览发送给模型的请求，不需要权重或服务器。

### 启动最终模型

准备好适合本机的 `llama-server`，从 [Hugging Face](https://huggingface.co/siwuxie27/Qwen2.5-1.5B-Ward-v4-Q6) 下载最终权重，放到：

```text
models/qwen2.5-1.5b-ward-v4-q6_k.gguf
```

也可以在仓库根目录执行以下下载命令（约1.27GB，URL固定到已核对的模型版本）：

```bash
curl -L --fail "https://huggingface.co/siwuxie27/Qwen2.5-1.5B-Ward-v4-Q6/resolve/98e94417c3ced60a4b1197c6fba75ed7ba3fa9a6/qwen2.5-1.5b-ward-v4-q6_k.gguf" -o models/qwen2.5-1.5b-ward-v4-q6_k.gguf
```

Windows PowerShell若把 `curl` 识别成别名，请将命令开头换为 `curl.exe`。正确SHA-256为 `f7ffc9751378a8f67bed3a6a8872f0f20c40b48cc4223b5560e225bd547ca8a0`。

然后执行（将第一处路径换成你的可执行文件）：

```bash
python scripts/start_model.py --llama-server /path/to/llama-server
```

Windows 示例：

```powershell
python scripts/start_model.py --llama-server "C:/llama/llama-server.exe"
```

启动前会校验权重 SHA-256。默认监听 `127.0.0.1:8000`，采用 20 层 GPU 卸载、4096 上下文、单槽、Q8_0 KV Cache。仅使用 CPU 时追加 `--cpu`。这只改变运行设备，不会重新训练模型。

服务在当前终端运行，按 Ctrl+C 关闭。接口地址为 `http://127.0.0.1:8000/v1`。

### 输入事件并提问

另开终端，在仓库根目录执行：

```bash
python scripts/ward_cli.py --task cloud_judgment --input examples/event.json
python scripts/ward_cli.py --task event_enhancement --input examples/event.json
python scripts/ward_cli.py --task nursing_advice --input examples/event.json
python scripts/ward_cli.py --task offline_decision --input examples/event.json
```

脚本会自动补充系统提示、任务要求与输出格式，使用者只需要提供事件。输入示例：

```json
{
  "event_type": "fall_suspected",
  "description": "B07床旁检测到倒地姿态，摄像头与床垫信号一致",
  "priority": "P1",
  "context": {
    "bed": "B07",
    "confidence": 0.93,
    "network_state": "disconnected",
    "details": {"posture": "lying_floor", "on_bed": false}
  }
}
```

云端研判的输出形态为 `judgment|confidence|advice`；离线决策为包含 `action`、`ring`、`reason` 的 JSON。示例形态不代表每次真实生成必然正确。完整七类任务说明见 [输入输出格式](docs/INPUT_FORMAT.md)。

服务启用 API Key 时，先设置环境变量 `QWEN_API_KEY`；脚本也支持 `--api-key-file`。不需要原服务器上的 `qwen-start-v4` 或 `qwen-chat` 命令。

## 已保存的历史结果

### 通用任务：同一600题，统一宽松语义抽取器

| 模型 | GSM8K（200） | CRUXEval-O（200） | BBH（200） | 宏平均 |
|---|---:|---:|---:|---:|
| 原始1.5B | 69.0% | 26.0% | 31.0% | 42.0% |
| 最终v4-Q6_K | 60.0% | 25.0% | 35.0% | 40.0% |
| 14B教师 | 94.0% | 52.5% | 54.0% | 66.83% |

学生相对原始1.5B的宏平均保持率为 **95.24%**，相对14B教师约为 **59.85%**。这两个比较基准不能混用。

### 病房任务：分别报告两套测试

| 指标 | 原200条（100个事件） | 新增50条（25个事件） |
|---|---:|---:|
| 判断准确率 | 99.00% | 68.00% |
| 判断Macro-F1 | 99.14% | 60.47% |
| 紧急度准确率 | 82.00% | 88.00% |
| 格式合规率 | 100.00% | 100.00% |
| 参考答案文字相似度 | 0.4673 | 0.4054 |

判断指标分别使用100条、25条 `cloud_judgment` 样本；紧急度指标分别使用100条、25条 `event_enhancement` 样本。新增测试的误报识别较弱，不能只引用原200条的99%来代表新场景表现。格式合规、文字相似度和建议内容正确性分别是不同指标，尚无人工护理建议内容准确率的充分验证。

历史v4-Q6平均完整请求延迟为0.422秒；历史v2-Q4_K_M的0.156秒不属于当前模型。这些记录不是流式TTFT。约1.44GiB的历史数字是特定配置的GPU显存，不能据此宣称系统总内存不超过1.5GB。

## 文档与上传

- [部署与完整使用说明](docs/MODEL_USAGE.md)
- [统一输入格式与七类任务](docs/INPUT_FORMAT.md)
- [训练流程及本地缺失项](docs/TRAINING.md)
- [评测复现与结果边界](docs/EVALUATION.md)
- [模型权重获取与校验](models/README.md)
- [GitHub上传步骤](docs/GITHUB_UPLOAD.md)
- [第三方来源与许可说明](THIRD_PARTY_NOTICES.md)

权重、运行输出、真实 `.env` 和缓存已在 `.gitignore` 中排除。代码整理包可直接上传，运行模型仍需另外取得权重。
