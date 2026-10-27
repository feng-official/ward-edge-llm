# 模型权重

本仓库包含使用代码和元数据，**不包含权重文件**。当前没有已发布的下载链接。

最终模型：`qwen2.5-1.5b-ward-v4-q6_k.gguf`，Q6_K量化，历史记录大小1,272,739,456字节。

历史SHA-256：

```text
f7ffc9751378a8f67bed3a6a8872f0f20c40b48cc4223b5560e225bd547ca8a0
```

原AutoDL位置（服务器重开后才可取回）：

```text
/root/autodl-tmp/qwen/models/Qwen2.5-1.5B-Ward-Distilled-v4-GGUF/qwen2.5-1.5b-ward-v4-q6_k.gguf
```

取得文件后放入本目录。Linux/macOS用 `sha256sum 文件名`，Windows用 `Get-FileHash 文件名 -Algorithm SHA256` 核验；启动脚本也会核验该文件的哈希。上述大小及哈希来自历史文档，本次未在关闭的云服务器上重新计算。

发布者日后把权重上传到自己的模型托管页或GitHub Release附件后，可在 `model-manifest.json` 的 `download_url` 中填写实际链接。请勿把不存在的链接当作可下载地址。基础Qwen模型不能替代本项目最终蒸馏权重。
