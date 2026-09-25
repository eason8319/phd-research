---
experiment: e2
results: research/phase1/results/e2
reviewed: true
---

本文回答局部低误差表示在真实生成、长历史和近期残留下是否仍可用，依据 2026-09-25 审核的 `research/phase1/results/e2/`、`research/phase1/configs/data.md` 与 `research/phase1/configs/quality.md`；32K 检索高精度基线未达门槛，不对其作量化保真判断。

## 范围：哪些任务参加 H2 判定？

两模型的 GSM8K 短生成和 HotpotQA 32K 多证据问答各用冻结 V 集 64 题。高精度基线分别须达到 EM 50 和 F1 40。32K 检索须达到 EM 90；未达标时只记失败边界。Llama 对 E2 选中的一种低位宽表示比较 $H=0,8,16$；Qwen 低位宽全部淘汰，不补残留搜索。

## 来源：评分、输入与选择在哪里？

逐题评分与选择分别见 `research/phase1/results/e2/qwen/`、`research/phase1/results/e2/llama/` 内的 `v_*.json`、`h0_selection.json` 和 `residual_selection.json`；冻结样本与输入哈希见 `research/phase1/splits/`。长解码诊断另见 `research/phase1/results/e2/llama/long_decode_selection.json`。这些结果遵守题面升版的适用范围：GSM8K 用 v5 后数据，HotpotQA 用 v6 后全文，检索用 v7 全文。

## 结果：Qwen 回退 KV8，Llama 留下无残留 K8V4

| 模型与单元 | 高精度，V 64 题 | KV8，V 64 题 | E2 选择与结果 |
|---|---:|---:|---|
| Qwen GSM8K，EM | 90.63 | 93.75 | 两个 E1 低位宽候选区间均跨 2 分 |
| Qwen HotpotQA，F1 | 50.23 | 46.31 | 两个低位宽候选相对高精度下降分别为 23.49、16.56 分，均淘汰 |
| Llama GSM8K，EM | 90.63 | 90.63 | `k8v4-per_token-post`、$H=0$ 为 90.63 |
| Llama HotpotQA，F1 | 67.70 | 67.62 | 同一表示、$H=0$ 为 67.70 |

Qwen 的 `kv4-per_token-post` 在 GSM8K 前 9 题出现 8 次退化，按预登记严重失效规则提前停止；两种 E1 候选在参加判断的 HotpotQA 单元均淘汰，因此进入 E3 的表示是 KV8，残留位宽为 0。Llama 选择 `k8v4-per_token-post`，短生成和参加判断的检索类均记“无残留可行”；$H=8/16$ 没有形成需要残留才能挽回的类，进入 E3 时仍为 $H=0$。选择依据见两模型的 `h0_selection.json` 与 `residual_selection.json`。

两模型的 32K 检索高精度 EM 分别为 87.50（Qwen）与 28.13（Llama），均低于 90 分门槛。Llama 长解码预定的 8 题中，高精度没有一题生成达到 256 token，合格样本数为 0；该诊断记“未归类”，不改变 E3 配置。离线 scale 范围复核与跨 token 组只作为诊断，来源见同目录的 `range_refit_*.json`、`scale_drift_*.json` 和 `cross_token_gt32.json`。

## 边界：哪些失败不能解释成量化退化？

32K 检索的高精度基线不足，故不能据该单元的候选得分推断量化损害或保真。Qwen 用 V 值重新拟合 scale 的文件显式标为诊断且使用评估值，不能替代 C 校准或改变候选。长解码合格样本为零，也不能给出有无残留的长解码结论。
