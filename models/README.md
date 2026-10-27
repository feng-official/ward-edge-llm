# 模型权重

GitHub仓库包含使用代码和元数据，**不包含权重文件**。最终权重已发布到 [Hugging Face模型仓库](https://huggingface.co/siwuxie27/Qwen2.5-1.5B-Ward-v4-Q6)。

[下载最终版GGUF（约1.27GB）](https://huggingface.co/siwuxie27/Qwen2.5-1.5B-Ward-v4-Q6/resolve/98e94417c3ced60a4b1197c6fba75ed7ba3fa9a6/qwen2.5-1.5b-ward-v4-q6_k.gguf)

下载地址固定到版本 `98e94417c3ced60a4b1197c6fba75ed7ba3fa9a6`。文件大小及LFS SHA-256已在2026-10-27通过Hugging Face公开API与历史记录核对一致；本次没有完整下载权重或重新运行模型。

最终模型：`qwen2.5-1.5b-ward-v4-q6_k.gguf`，Q6_K量化，历史记录大小1,272,739,456字节。

SHA-256：

```text
f7ffc9751378a8f67bed3a6a8872f0f20c40b48cc4223b5560e225bd547ca8a0
```

在GitHub仓库根目录执行：

```bash
curl -L --fail "https://huggingface.co/siwuxie27/Qwen2.5-1.5B-Ward-v4-Q6/resolve/98e94417c3ced60a4b1197c6fba75ed7ba3fa9a6/qwen2.5-1.5b-ward-v4-q6_k.gguf" -o models/qwen2.5-1.5b-ward-v4-q6_k.gguf
```

Windows PowerShell可使用 `curl.exe` 替代 `curl`。

原AutoDL位置（仅作历史来源记录，下载Hugging Face权重不需要启动AutoDL）：

```text
/root/autodl-tmp/qwen/models/Qwen2.5-1.5B-Ward-Distilled-v4-GGUF/qwen2.5-1.5b-ward-v4-q6_k.gguf
```

取得文件后放入本目录。Linux/macOS用 `sha256sum 文件名`，Windows用 `Get-FileHash 文件名 -Algorithm SHA256` 核验；启动脚本也会核验该文件的哈希。

`model-manifest.json`已填入模型页面、固定版本下载地址、大小与哈希。Hugging Face仓库提供权重下载，本链接不是在线推理API；下载后按GitHub根README启动本地服务。基础Qwen模型不能替代本项目最终蒸馏权重。
