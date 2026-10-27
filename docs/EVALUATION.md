# 评测与历史证据

所有命令在仓库根目录执行。无需原AutoDL上的启动包装命令，评测脚本连接一个已运行的兼容接口；未启用认证时无需API Key。

## 不启动模型的本地检查

```bash
python scripts/verify_repository.py --output outputs/local-check.json
```

检查JSON/JSONL、数据数量、ID与场景族隔离、训练/测试提示精确重复、历史结果能否由逐样本记录重算、模型元数据、文件清单哈希、Python语法及常见凭证模式。它不是模型推理评测，也不证明MQTT链路或断网业务保持率。

## 病房原200条与新增50条

```bash
python edge-agent/distillation/evaluate_model.py --input datasets/ward-nlu-500-v1/qwen-sft-test-v1.0.0.jsonl --output outputs/domain-v4-new.json --model qwen2.5-1.5b-ward-v4-q6 --label local-v4 --workers 1
python edge-agent/distillation/evaluate_model.py --input datasets/ward-nlu-500-v1/evaluation/ward-task-50-v1/ward-task-50-v1.jsonl --output outputs/ward50-v4-new.json --model qwen2.5-1.5b-ward-v4-q6 --label local-v4 --workers 1
```

`evaluate_model.py`保留历史评分：判断标签按协议提取、紧急度按标签提取，单独统计严格格式与字符二元组Dice相似度。它适合复跑同格式的学生模型；若比较自由回答的原始模型或教师，应对两边使用相同宽松语义提取方式，不能把解析失败直接当作能力为零。

两个测试集结果应并列报告。原200条为99%判断准确率，新增50条为68%；新增测试 `reject` 召回率为12.5%。不要把格式100%写成建议内容100%正确。

七类任务仅做输出形态烟雾测试：

```bash
python edge-agent/distillation/smoke_tasks.py --output outputs/seven-task-smoke.json
```

## 600题通用能力

原题库不在本地，但保存了原manifest、学生与教师逐题预测，以及统一宽松抽取器评分报告。

学生原始预测日志含739行、600个唯一题号，其中139行为断点续跑的旧记录。历史评测程序用 `previous[id] = item` 聚合，按题号取最后一条。采用相同规则后得到数学120/200、代码50/200、BBH70/200，与60%/25%/35%报告一致。仓库保留原日志，并在 `reports/general-samples/` 提供去重后的600条和审核说明；本次没有重新询问模型。

可重新核查日志（无需联网）：

```bash
python scripts/audit_saved_samples.py
```

准备题库（会联网下载上游公开数据）：

```bash
python edge-agent/distillation/evaluate_general_retention.py prepare --output outputs/benchmark.jsonl --cache-dir cache/general-benchmark --per-task 200 --seed 20260818
```

重新生成后核对SHA-256：

```bash
python -c "import hashlib,pathlib; print(hashlib.sha256(pathlib.Path('outputs/benchmark.jsonl').read_bytes()).hexdigest())"
```

原题库哈希：`890538935892216320d02e461c8c3c6ea65787c77fff16e3eccdb760e2b57e71`。

上游URL指向可变分支，重新下载可能变化。**哈希不一致时，应作为新题库让两个模型重新跑，不可与旧成绩直接计算保持率。**保存好本次生成的题库和新manifest。

为两个模型分别运行同一条命令，只改变模型名和输出文件：

```bash
python edge-agent/distillation/evaluate_general_retention.py run --benchmark outputs/benchmark.jsonl --output outputs/baseline.json --label baseline --model ORIGINAL_MODEL_ALIAS --workers 1
python edge-agent/distillation/evaluate_general_retention.py run --benchmark outputs/benchmark.jsonl --output outputs/student.json --label student --model qwen2.5-1.5b-ward-v4-q6 --workers 1
python edge-agent/distillation/evaluate_general_retention.py compare --baseline outputs/baseline.json --distilled outputs/student.json --output outputs/comparison.json --threshold 0.80
```

旧600题对比也可直接从已保存报告重新汇总（无需模型）：

```bash
python edge-agent/distillation/evaluate_general_retention.py compare --baseline datasets/ward-nlu-500-v1/distillation/general-retention-v1/general-baseline-1.5b-full-v2-normalized.json --distilled datasets/ward-nlu-500-v1/distillation/v4/general-v4-q6-full.json --output outputs/historical-general-comparison.json
```

CRUXEval-O评测预测的Python返回值，不执行模型生成代码。保持率使用两边相同宽松抽取器的全体样本准确率；不能只比较“格式正确样本中的准确率”，这会因样本子集不同产生偏差。

## 图表和限制

`docs/figures/`为此前制作的5张报告图，保留历史口径；说明见 `docs/HISTORICAL_FIGURES.md`。当前README并列披露新增50条表现。图中延迟不是TTFT，显存不是系统总内存；没有本次重新测得的性能或医疗内容准确率。

`reports/training-v4/`保存真实loss记录。5000条manifest文件名含v4.1.0、内部version却为v4.0.0；本次保留原文件和哈希，没有静默改写历史证据。
