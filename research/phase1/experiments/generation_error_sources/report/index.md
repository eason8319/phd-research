---
experiment: generation_error_sources
results: research/phase1/experiments/generation_error_sources/results
reviewed: true
---

# 真实生成与误差来源

本文回答局部低误差表示在真实生成、长历史和近期残留下是否仍可用。更新日期为 2026-09-30，测量数字沿用 2026-09-25 审核的结果。本报告不覆盖最终扩样判定、存储服务成本或正式测试集；32K 检索高精度基线未达门槛，不对该单元作量化保真判断。

来源说明：生成和评分约定见 `research/phase1/configs/data.md`，判断规则见 `research/phase1/configs/quality.md`，实验顺序见 `research/phase1/guide.md`。本实验 `results/{qwen,llama}/` 内的 `v_*.json` 保存逐题评分，`no_residual_selection.json` 和 `residual_selection.json` 保存无残留与残留选择；Llama 长解码结果见 `long_decode_selection.json`。冻结样本及输入校验值见 `research/phase1/splits/`。输入分别使用 GSM8K 的具体答案示例、HotpotQA 去除答案占位符后的全文及要求完整键匹配的检索全文；历史提交号、评分和来源校验值保留实际测量时的记录。

当前指导保留每模型四种 K8V4 候选，即 Key 离线 per-channel／在线 per-token 与 pre／post-RoPE 的四种组合。现有数据属于原“每模型至多两种非基线候选”的历史执行范围：在这四种 K8V4 中，Qwen 只测过离线 per-channel、post-RoPE，Llama 只测过在线 per-token 的 pre／post-RoPE。新四候选规则尚未完成全部质量验证，下文的淘汰、回退与选择仅适用于历史候选集。本次只标明覆盖范围，不补跑质量实验；补齐并审核后再更新本报告及后续实验的选择。

## 实验目的

比较高精度、KV8、KV4 与位宽和表示交互选中的低位宽候选，判断局部误差优势是否迁移到真实生成，并区分近期残留收益与诊断性误差来源。

两模型的 GSM8K 短生成和 HotpotQA 32K 多证据问答各用冻结 V 集 64 题。高精度基线分别须达到 EM 50 和 F1 40；32K 检索须达到 EM 90，未达标时只记失败边界。候选同时相对高精度和 KV8 按配对下降及其区间判定，开发容差为 2 分。Llama 对选中的一种低位宽表示比较 $H=0,8,16$；Qwen 低位宽全部淘汰，不补残留搜索。生成中的首个答案 token 由高精度 prefill 决定，量化 KV 从该 token 的 query 开始读取。

## 实验结果与结论

### Qwen 回退 KV8，Llama 留下无残留 K8V4

下表每个分数由同一冻结 V 单元的全部 64 题汇总；GSM8K 为 EM，HotpotQA 为逐题 token-F1 的平均值，分数均为 0–100。

| 模型与单元 | 高精度，V 64 题 | KV8，V 64 题 | 真实生成与误差来源选择与结果 |
|---|---:|---:|---|
| Qwen GSM8K，EM | 90.63 | 93.75 | 两个位宽与表示交互低位宽候选区间均跨 2 分 |
| Qwen HotpotQA，F1 | 50.23 | 46.31 | 两个低位宽候选相对高精度下降分别为 23.49、16.56 分，均淘汰 |
| Llama GSM8K，EM | 90.63 | 90.63 | `k8v4-per_token-post`、$H=0$ 为 90.63 |
| Llama HotpotQA，F1 | 67.70 | 67.62 | 同一表示、$H=0$ 为 67.70 |

Qwen 的 `kv4-per_token-post` 在 GSM8K 前 9 题出现 8 次退化，按预登记严重失效规则提前停止；两种位宽与表示交互候选在参加判断的 HotpotQA 单元均淘汰，因此进入静态层分配的表示是 KV8，残留位宽为 0。Llama 选择 `k8v4-per_token-post`，短生成和参加判断的检索类均记“无残留可行”；$H=8/16$ 没有形成需要残留才能挽回的类，进入静态层分配时仍为 $H=0$。选择依据见两模型的 `no_residual_selection.json` 与 `residual_selection.json`。

两模型的 32K 检索高精度 EM 分别为 87.50（Qwen）与 28.13（Llama），均低于 90 分门槛。Llama 长解码预定的 8 题中，高精度没有一题生成达到 256 token，合格样本数为 0；该诊断记“未归类”，不改变静态层分配配置。离线 scale 范围复核与跨 token 组只作为诊断，来源见同目录的 `range_refit_*.json`、`scale_drift_*.json` 和 `cross_token_gt32.json`。

## 总结

在历史候选集和参加判断的任务单元中，本实验留下的产出是 Qwen 回退 KV8、Llama 选择无残留的 `k8v4-per_token-post`，并将这两种表示及各自的 $H=0$ 条件带入历史静态层分配。32K 检索基线不足和长解码有效样本为零作为失败边界保留，没有据此形成量化保真或长解码残留结论。

原至多两候选规则下的历史选择保持原义，不能据 Qwen 的历史回退认定其四种 K8V4 均不可行，也不能据 Llama 的历史选择认定其余两种表示已经验证。新四候选的完整质量比较与据此更新的后续选择仍待完成；后续须在相同数据、生成和基线对照下补齐四候选，审核后再更新残留诊断与静态层分配的输入。

32K 检索的高精度基线不足，故不能据该单元的候选得分推断量化损害或保真。Qwen 用 V 值重新拟合 scale 的文件显式标为诊断且使用评估值，不能替代 C 校准或改变候选。长解码合格样本为零，也不能给出有无残留的长解码结论。

“无残留可行”是当前开发规则下的类别，不等于最终扩样后的保留或正式非劣性证明。当前结果目录只有 Llama 的 `trajectories.json`，没有 Qwen 对应的轨迹结果；不能用已有局部诊断替代尚未完成的 Qwen 累积误差归因。T 集未使用，这些 V 集结论仍是开发证据。
