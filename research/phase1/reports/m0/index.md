---
experiment: m0
results: research/phase1/results/m0
reviewed: true
---

本文核对第一阶段两款模型的运行准入，依据 2026-09-25 审核的 `research/phase1/results/m0/`、`research/phase1/configs/runtime.md`；这里的单次时延不是服务吞吐测量，也不涉及量化质量。

## 范围：两款模型能否运行到 32K？

在同一张 NVIDIA RTX 6000 Ada Generation（49,140 MiB）上，分别检查 Qwen2.5-7B-Instruct 和 Llama-3.1-8B-Instruct 的 BF16 短输入、4K、32K prefill 与单步 decode。只保留末位词表 logits。

## 来源：怎样复核运行条件？

设备、GPU UUID、作业号和准入状态见 `research/phase1/results/m0/device.json`；逐模型长度、时延和显存见 `research/phase1/results/m0/qwen.json` 与 `research/phase1/results/m0/llama.json`。软件版本为 PyTorch 2.5.1+cu121、CUDA 12.1；结果登记为协议 v4，指导文档确认其正确性结果继续有效。

## 结果：32K 冒烟均通过

| 模型 | 32K prefill，单次 | 32K 单步 decode | 32K 峰值显存 | 结果 |
|---|---:|---:|---:|---|
| Qwen | 8.065 s | 0.215 s | 20.30 GiB | 通过 |
| Llama | 9.202 s | 0.260 s | 24.22 GiB | 通过 |

两模型的 32K cache 均为 32,768 token，短输入与 4K 也成功。表中数字来自上述逐模型文件各一条 32K 记录；设备文件的 `passed` 为 `true`。

## 边界：这些数值能说明什么？

这只是指定节点和单次调用的可运行性证据。峰值包含权重、激活和工作区；时延没有重复测量、置信区间或并发负载，不用于 E4 的服务时间结论。
