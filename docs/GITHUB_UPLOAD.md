# 上传到新的GitHub仓库

本目录已经整理为独立仓库，不需要启动云服务器。本次没有替你创建仓库或推送到任何远程地址。

## 网页上传

1. 解压收到的ZIP，进入 `ward-edge-llm` 文件夹。根目录应直接看到 `README.md`、`scripts`、`datasets` 等。
2. 登录GitHub，新建仓库，例如 `ward-edge-llm`。可先选择Private；不要初始化额外README，以免与本目录重复。
3. 在仓库页面使用“上传已有文件”（uploading an existing file / Add file → Upload files）。
4. 把本文件夹**里面的文件与目录**拖入上传区。不要只上传ZIP，也不要把整个 `ward-edge-llm` 再套一层。
5. 确认 `.gitignore`、`.gitattributes`、`.env.example` 也在文件列表中，再提交（Commit changes）。
6. 如果网页提示一次文件过多，先提交根目录和scripts，再分批上传datasets、docs和其余目录；也可采用下面Git方式。
7. 上传后打开仓库首页，应能直接看到模型介绍及使用步骤。确保 `docs/INPUT_FORMAT.md` 和 `models/README.md` 可以访问。

## 本地Git上传（适合一次上传整个目录）

安装Git后，在 `ward-edge-llm` 文件夹打开终端，执行：

```bash
git init
git add .
git status
git commit -m "Add Ward v4 model code, datasets, evaluation and deployment docs"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/ward-edge-llm.git
git push -u origin main
```

把URL替换成你刚创建的仓库地址。GitHub认证由你自己的浏览器或凭证管理器完成，不要把密码写入命令或文件。

若这些文件要放回原智慧病房仓库，请不要用本仓库根README覆盖整个原项目README；本包是为新建独立模型仓库整理的。

## 后续发布权重

当前包不含权重，使用方需要另行取得最终GGUF。待从AutoDL取回文件后，通过模型托管页或Release附件发布，并更新 `models/model-manifest.json` 的 `download_url`。这一步不影响先上传代码和数据。

`reports/package-manifest.json`为本次打包时的校验清单。以后修改仓库文件时，它会失效；可在项目根目录运行 `python scripts/verify_repository.py --refresh-manifest` 更新清单，再运行普通检查。
