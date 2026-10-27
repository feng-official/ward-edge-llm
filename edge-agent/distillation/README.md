# 训练与评测脚本

此目录保留原项目相对路径，便于追溯历史记录。独立运行不需要原 `edge-agent/src`。

| 脚本 | 用途 |
|---|---|
| generate_teacher_data.py | 14B兼容接口生成多任务答案，并按标签校准 |
| prepare_student_data.py | 混合原生和教师数据、类别平衡 |
| build_mixed_v4.py | 病房/GSM8K训练集/MBPP混合构建 |
| train_lora.py | BF16 LoRA监督训练 |
| combine_lora_adapters.py | 按权重rank拼接两个适配器 |
| merge_lora.py | 合并底座与LoRA |
| evaluate_model.py | 病房两类任务严格格式评测 |
| evaluate_general_retention.py | 准备公开题库、运行统一通用评测及比较 |
| rescore_general_retention_v2.py | 对保留的旧输出进行统一语义重评分 |
| audit_domain_content_v2.py | 导出抽样护理建议内容审核材料 |
| build_ward50_eval.py | 构建新增50条病房任务 |
| smoke_tasks.py | 七类任务输出形态检查 |
| api_config.py | 读取可选环境变量或显式API Key文件 |

本次只调整API Key读取方式及烟雾测试默认模型别名，以支持其他设备；历史评分和训练算法未改动。

使用步骤见仓库根README、`docs/TRAINING.md`和`docs/EVALUATION.md`。旧服务器专用shell包装脚本没有放入本仓库；请用 `scripts/start_model.py`。
