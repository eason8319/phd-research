---
experiment: e3
results: research/phase1/results/e3
reviewed: true
---

本文回答短校准的层／K-V 敏感度能否指导静态 INT4／INT8 分配，依据 2026-09-25 审核的 `research/phase1/results/e3/`、`research/phase1/guide.md` 与 `research/phase1/configs/quality.md`；结论限于本阶段冻结的模型、样本和预算。

## 范围：门槛和迁移怎样分开？

两模型都在 25% 与 50% INT8 单位预算下比较敏感度、重建误差、反向分配和固定随机分配。主预算为 50%。先以 C 的检索子集全层 KL 判断敏感度是否严格优于三个随机分配；再在 V 的 GSM8K 与 HotpotQA 各 64 题查看迁移。高精度 32K 检索未达能力门槛，不参加 V 的质量判断。

## 来源：哪些文件固定分配与对照？

探针、位宽配置、C 门槛、V 逐题评分和最终路线分别在 `research/phase1/results/e3/qwen/`、`research/phase1/results/e3/llama/` 内的 `probe.json`、`allocations.json`、`gate.json`、`v_*.json`、`selection.json`。门槛记录的 C 样本数为数学 64、检索 32，V 每单元为 64 题。随机种子由 `research/phase1/splits/` 固定。

## 结果：Qwen 门槛通过，Llama 保留简单表示

| 模型 | C 上敏感度严格优于三个随机分配 | V 的有效检索单元 | E3 证据与 F 路线 |
|---|---|---|---|
| Qwen | 是 | HotpotQA，64 题 | V 点估计优于三个随机分配；`p50_sensitivity` 进入 F，同时保留 KV8 简单对照 |
| Llama | 否 | HotpotQA，64 题 | 静态层混合证据不足；F 只用 E2 的 `k8v4-per_token-post` |

Qwen 在 V HotpotQA 上，敏感度分配相对高精度的 F1 下降点估计为 −0.66 分，三个随机分配分别为 46.95、49.99、6.74 分；这是 E3 的预定 V 迁移比较，不能替代 F 的扩样区间。Llama 未过 C 门槛，V 改测重建分配与反向分配；不因 V 表现回头改冻结配置。具体区间和各任务状态仍见逐模型 `selection.json`。

## 边界：迁移证据有多强？

层敏感度是局部探针量；C 门槛与 V 的点估计均不是最终保留判定。Qwen 的三组随机对照差异极大，F 扩样后仍需依照配对区间评价候选。32K 检索高精度基线不足，不能从该单元宣称跨任务迁移。
