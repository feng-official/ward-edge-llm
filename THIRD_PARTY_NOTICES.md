# 来源和许可

此包从本地保存的智慧病房项目及蒸馏交付材料整理，保留原始数据与实验记录。未找到足以确定整个上游项目授权范围的统一LICENSE，因此本次不替原作者新增或更改项目许可证。发布者应依据团队约定确认自有代码与数据的发布权限；不能仅因上传GitHub就推定全部内容适用某一开源许可证。

第三方模型、数据和依赖分别遵守其原始许可与引用要求：

- Qwen2.5模型：[Qwen2.5官方项目](https://github.com/QwenLM/Qwen2.5)。本包没有附带模型权重。
- GSM8K：[openai/grade-school-math](https://github.com/openai/grade-school-math)。混合训练包含训练集派生样本。
- MBPP：[google-research/google-research/mbpp](https://github.com/google-research/google-research/tree/master/mbpp)。混合训练包含代码输出推理派生样本。
- CRUXEval：[facebookresearch/cruxeval](https://github.com/facebookresearch/cruxeval)。用于评测，不是训练来源。
- BIG-Bench Hard：[suzgunmirac/BIG-Bench-Hard](https://github.com/suzgunmirac/BIG-Bench-Hard)。用于评测。
- LoRA：[LoRA论文](https://arxiv.org/abs/2106.09685)。训练使用PEFT/Transformers/PyTorch。
- GGUF推理：[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)。可执行文件需使用者另行准备。

这里的链接来自既有项目元数据与常用上游标识，本次离线整理未重新核验上游许可文本或固定全部上游版本。`requirements-train.txt`不是原训练环境的锁文件。
