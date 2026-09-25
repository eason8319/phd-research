---
experiment: f
results: research/phase1/results/f
reviewed: true
---

本文登记第一阶段最终候选的冻结扩样结果，依据 2026-09-25 审核的 `research/phase1/results/f/`、E2／E3 选择文件和 `research/phase1/configs/quality.md`；“保留”只表示本阶段开发筛选资格，不是正式非劣性证明。

## 范围：哪些候选需要扩样？

对 E2 简单表示及符合 E3 规则的静态分配，按 V 64 题加扩样池至 128、最多 256 题；高精度、KV8 和相关随机对照同步扩样。判断使用每题配对的参考减候选下降，容差为 2 分；GSM8K 用 EM，HotpotQA 用 token-F1。长解码不扩样，32K 检索因高精度不足被排除。

## 来源：结果如何审核？

两模型的最终状态和配对区间见 `research/phase1/results/f/qwen/selection.json`、`research/phase1/results/f/llama/selection.json`；同目录的 `*_n128.json` 与 `*_n256.json` 保存新增题的逐题评分和输入哈希。审核逐题顺序与 `research/phase1/splits/`、同题条件的输入哈希、前 64 条续跑前缀、汇总分数和固定 10,000 次重采样／精确二项区间。Qwen 的 256 题 HotpotQA 高精度合并 F1 为 42.20，仍高于 40 分门槛。

## 结果：只有 Llama 的 HotpotQA 低位宽工作点保留

| 模型、任务、候选 | 总题数 | 对高精度下降点估计，分 | 95% 配对区间，分 | 状态 |
|---|---:|---:|---:|---|
| Qwen GSM8K，KV8 | 256 | −1.17 | [−4.85, 2.60] | 证据不足 |
| Qwen GSM8K，E3 敏感度分配 | 256 | 0.00 | [−4.00, 4.00] | 证据不足 |
| Qwen HotpotQA，KV8 | 256 | 0.62 | [−2.50, 3.83] | 证据不足 |
| Qwen HotpotQA，E3 敏感度分配 | 256 | 1.48 | [−1.67, 4.64] | 证据不足 |
| Llama GSM8K，E2 K8V4 | 256 | 0.78 | [−3.61, 5.12] | 证据不足 |
| Llama HotpotQA，E2 K8V4 | 128 | 0.20 | [−0.01, 0.63] | 保留 |

每行来自相应模型 `selection.json` 的 `versus_bf16`；非 KV8 候选也按同文件的 `versus_kv8` 判定。Llama HotpotQA 对 KV8 的下降区间为 [−1.45, 1.77] 分，两个上界均不超过 2。Qwen 的所有 256 题区间仍跨越 2，按冻结上限保持“证据不足”，`next_expansion` 为空。Qwen 随机层分配在新增 192 题上呈两组近乎失效、一组接近候选的分化，逐题证据保留在对应 `e3-p50_random_*_n256.json`。

## 边界：最终候选意味着什么？

配对区间用于开发筛选，样本上限与 2 分容差均未因结果改变。Qwen 的“证据不足”不能写成保留或淘汰；Llama 的 GSM8K 也不能因 HotpotQA 保留而外推。32K 检索只保留高精度失败边界，无量化保真结论。
