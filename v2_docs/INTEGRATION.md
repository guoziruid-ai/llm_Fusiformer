# V2 与现有 Abstract-Text-Embedding-Module 分支的接入说明

## 本地 VS Code / PyCharm

1. 创建 `Fusiformer-V2-CrossModal-Fusion` 新分支；原分支作为 A0 不动。
2. 将本压缩包内容解压到一个临时目录，与仓库合并（同名原文件除了 `.gitignore` 无修改）。
3. 运行 `python -m pytest -q v2_tests`，提交新文件：

```bash
git status --short
git add .gitignore fusiformer_v2 v2_tests v2_docs \
  tools/precompute_semantic_segments_v2.py train_fusiformer_v2.py README_FUSIFORMER_V2.md
git commit -m "feat(v2): native atom-semantic cross-modal fusion A1-A3"
git push -u origin Fusiformer-V2-CrossModal-Fusion
```

## 服务器

旧 `/root/llm_fusiformer`（Graph-clustering-optimization）不切换、不覆盖。

```bash
cd /root
git clone --branch Fusiformer-V2-CrossModal-Fusion \
  https://github.com/guoziruid-ai/llm_Fusiformer.git llm_fusiformer_v2
cd /root/llm_fusiformer_v2
conda activate fusiformer
```

新目录需要能访问相同的 JSON、MatSciBERT 模型。可以复用原数据路径（修改配置为绝对路径）或用软链接：

```bash
# 首先核对服务器原数据的实际路径，不要覆盖已有 data 目录。
# 示例：ln -s /root/llm_fusiformer/data/dataset_final_merged_with_robocrys.json \
#             /root/llm_fusiformer_v2/data/dataset_final_merged_with_robocrys.json
```

生成缓存、运行 V2 训练见根目录 `README_FUSIFORMER_V2.md`。

## 验证建议

- A0（原 v6） vs. A1 vs. A2 vs. A3: 保持同一数据文件、target、split seed(7)、训练 seed(42)、超参数等；A2 改变旧文本引导的交互位置，因此差值不可归因于单一机制。
- 在 A2 进行 paired/unpaired text shuffle, all-text-mask, segment mask ablation。
- 每个实验保存 `git rev-parse HEAD`、配置、MAE/RMSE、实际数据哈希及 checkpoint。
- 真正 server smoke 测试后若有运行错误，提供 traceback、torch/dgl 版本、inputs 维度，继续精确修复。
