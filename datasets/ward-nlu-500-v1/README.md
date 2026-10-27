# Ward NLU 500

13类模拟病房事件，500条主数据，400训练/100测试；同一源事件派生事件增强与云端裁决两类任务，得到800条SFT训练/200条SFT测试。这里的“500条”是主事件数量。

| 文件或目录 | 用途 |
|---|---|
| nlu-dataset-v1.0.0.jsonl / .json | 全部500条主数据 |
| train-v1.0.0.jsonl | 400条训练事件 |
| test-v1.0.0.jsonl | 100条留出事件 |
| qwen-sft-train-v1.0.0.jsonl | 800条训练对话 |
| qwen-sft-test-v1.0.0.jsonl | 200条测试对话 |
| project-inference-test-v1.0.0.jsonl | 原项目100条推理请求夹具 |
| schema-v1.0.0.json / manifest.json | 结构与分布 |
| LABEL_SPEC.md | 标签定义 |
| distillation/teacher-train-v1.jsonl | 教师输出及校准信息 |
| distillation/student-train-v2.jsonl | 学生混合训练数据 |
| distillation/v4/mixed-train-v4.1.0.jsonl | 5000条病房/数学/代码混合训练记录 |
| distillation/v4/*full*.json* | 历史完整评测与逐样本输出 |
| distillation/general-retention-v1/ | 原600题manifest和基线汇总；题库本体未本地保存 |
| evaluation/ward-task-50-v1/ | 新增25个场景、50条任务及真实测试结果 |

全部床位、时间和描述由规则及模板构造，非真实患者记录。数据里的 `source=manual` 是原数据的来源枚举，不代表每条都经过临床专家人工审核。教师答案部分由规则校准或回退生成，详见教师报告。

在仓库根目录运行 `python scripts/verify_repository.py` 可离线核查数量、JSON和划分隔离。`generate_ward_nlu_dataset.py`可重建初版500条，但不要用新生成版本静默覆盖带历史哈希的证据文件；建议输出到 `outputs/rebuilt-ward`。

原完整智慧病房工程的路由器/MQTT兼容测试不在当前独立包的离线检查范围内。模型评测见 `docs/EVALUATION.md`。
