---
experiment: e1
results: research/phase1/results/e1
reviewed: true
---

本文回答 Key／Value 位宽、量化轴与 RoPE 顺序如何影响局部注意力误差，依据 2026-09-25 审核的 `research/phase1/results/e1/`、`research/phase1/guide.md` 和 `research/phase1/configs/quant.md`；这里不推断自由生成质量。

## 范围：局部表示比较了什么？

两款模型各比较 Key 轴、pre／post-RoPE 与 KV8、K8V4、K4V8、KV4 的 16 格主矩阵，另有 Value 分组诊断。选择量为两折平均注意力输出相对误差；容量必须满足指导文档的位宽条件。

## 来源：候选如何追溯？

逐模型 17 格摘要、无效组和留下的表示见 `research/phase1/results/e1/qwen/summary.json`、`research/phase1/results/e1/llama/summary.json`；校准与折划分见 `research/phase1/splits/`。两份摘要登记协议 v7、代码提交 `6ca63fd`。

## 结果：升 Key 位宽的局部误差更低

| 模型 | post-RoPE、per-token K8V4 误差 | 同条件 K4V8 误差 | 留给 E2 的表示 |
|---|---:|---:|---|
| Qwen | 0.005521 | 0.082087 | `k8v4-per_channel-post`、`kv4-per_channel-pre` |
| Llama | 0.005296 | 0.013198 | `k8v4-per_token-post`、`k8v4-per_token-pre` |

Qwen 的 6 bit 档最低误差来自 per-channel／post-RoPE（0.005061），4 bit 档留下 per-channel／pre-RoPE（0.025403）。Llama 的 4 bit 档没有严格优于均匀 KV4 的表示，因此按冻结规则留下 6 bit 档两种。上述数值均是各模型摘要中的局部相对误差，容量与无效组排序也由同一文件记录。

## 边界：局部优势能否迁移？

本实验固定高精度隐藏状态逐层测误差，不让误差沿生成传播。E2 的真实生成可能改变排序；E1 不能单独给出质量保留或服务时间结论。
