# 蒸馏与训练流程

本项目采用离线教师响应蒸馏：14B生成病房任务答案，并按项目标签校准；1.5B通过监督微调学习这些答案。不是实时教师logits匹配，也不是把14B参数直接压缩成1.5B。

## 数据和训练资产

1. `datasets/ward-nlu-500-v1`：500个模拟病房事件，400训练、100测试；派生800/200条SFT记录。
2. `distillation/teacher-train-v1.jsonl`：教师生成的多任务数据。历史报告记录400个源事件、2160条输出、392个fallback字段，所以不能声称每个答案都未经规则处理地来自14B。
3. `distillation/student-train-v2.jsonl`：教师与原生提示混合，并平衡部分研判类别。
4. `distillation/v4/mixed-train-v4.1.0.jsonl`：5000条，含2000病房、1500数学、1500代码回放数据。
5. 分别获得病房LoRA和通用回放LoRA，再按0.50/0.75融合，合并到底座并量化为Q6_K。

精确字符串重叠为零只排除了所检查的重复，不能证明语义去重或完全没有模型选择偏差。

## 可执行训练命令

在准备好CUDA/PyTorch、基础模型并安装 `requirements-train.txt` 后，以下命令训练**通用回放适配器**：

```bash
python edge-agent/distillation/train_lora.py --model /path/to/Qwen2.5-1.5B-Instruct --train datasets/ward-nlu-500-v1/distillation/v4/mixed-train-v4.1.0.jsonl --output outputs/lora-v4-general --epochs 1 --batch-size 2 --gradient-accumulation 8 --learning-rate 0.00002 --lora-r 16 --lora-alpha 32
```

训练脚本默认max-length为768；历史摘要没有单独记录是否覆盖该参数，因此上述命令是按现存脚本重建的运行示例，不声称是原始命令逐字还原。历史训练记录见 `reports/training-v4/`：313次更新、683.4秒、峰值GPU显存6.829GiB。

已有两个适配器时可按以下方式融合：

```bash
python edge-agent/distillation/combine_lora_adapters.py --adapter-a /path/to/ward-lora-v2 --adapter-b outputs/lora-v4-general --weight-a 0.50 --weight-b 0.75 --output outputs/lora-v4-combined
python edge-agent/distillation/merge_lora.py --base /path/to/Qwen2.5-1.5B-Instruct --adapter outputs/lora-v4-combined --output outputs/merged-v4
```

融合是rank拼接：单个适配器rank16，最终rank32。GGUF转换和Q6_K量化需要另行准备llama.cpp工具；本仓库没有保存其历史构建版本或可执行文件。

## 生成教师数据

启动真实14B兼容接口后：

```bash
python edge-agent/distillation/generate_teacher_data.py --input datasets/ward-nlu-500-v1/train-v1.0.0.jsonl --output outputs/teacher-train.jsonl --model YOUR_TEACHER_ALIAS --base-url http://127.0.0.1:8000/v1
python edge-agent/distillation/prepare_student_data.py --teacher outputs/teacher-train.jsonl --native datasets/ward-nlu-500-v1/qwen-sft-train-v1.0.0.jsonl --output outputs/student-train.jsonl
```

接口需要认证时设置 `QWEN_API_KEY`，或使用 `--api-key-file`。教师生成脚本请求JSON对象模式，后端需支持该功能。测试集不得作为生成输入。

## 尚未本地保存的资产

- 最终GGUF权重、病房LoRA、通用LoRA、合并后的HF权重。
- 历史训练环境完整依赖锁、CUDA/驱动/llama.cpp构建版本。
- 原600题冻结题库本体及完整原始训练数据下载缓存。

所以本仓库可以运行数据处理和评测程序、继续训练，但**不能在缺少历史适配器时保证逐比特重建最终v4-Q6权重**。发布现有最终模型时应取回记录中的GGUF，并核对哈希。
