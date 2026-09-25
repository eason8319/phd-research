---
experiment: e0
results: research/phase1/results/e0
reviewed: true
---

本文回答量化比较开始前的实现正确性门槛，依据 2026-09-25 审核的 `research/phase1/results/e0/` 与 `research/phase1/guide.md`；本报告不把正确性测试当作任务质量通过。

## 范围：哪些实现假象被排除？

检查量化往返、打包与布局、参数异常和组／块边界，并在 Qwen 与 Llama 的短输入和 32K 输入上核对缓存、非方形因果可见性、末位 logits 与 pre-RoPE 对照。

## 来源：通过条件在哪里？

11 类小张量与边界用例及各自 `ok` 值见 `research/phase1/results/e0/quant.json`；两模型各 32 与 32,768 token 的前向核对见 `research/phase1/results/e0/model.json`。两份结果均登记协议 v4，指导文档注明它们在题面后续升版后继续有效。

## 结果：正确性门槛已通过

| 证据 | 样本／用例 | 审核结果 |
|---|---:|---|
| 常量、超界、未满组、闭合组、尾块、非有限值、参数溢出及往返等 | 11 类 | 全部 `ok=true` |
| Qwen 短输入与 32K | 2 个长度 | 缓存、可见掩码、末位 logits、RoPE 均 `ok=true` |
| Llama 短输入与 32K | 2 个长度 | 同上 |

两份结果的总 `passed` 均为 `true`。32K decode 时，两模型可见掩码均覆盖 32,769 个位置，避免单 token query 的错误因果掩码。

## 边界：仍需哪些后续证据？

这些检查确认实现约束与边界处理，不能证明量化后的真实生成质量、长历史迁移或硬件服务收益；这些分别由 E1–E4 给出。
