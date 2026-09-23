# 第一阶段运行配置

本文件只冻结模型和运行环境。实验顺序和通过条件见 [../guide.md](../guide.md)。本文件属于 guide.md 所记的协议版本。

## 模型

第一对候选是 Llama-3.1-8B-Instruct 与 Qwen2.5-7B-Instruct，用于跨模型检验。只从项目内路径加载，不重新下载，不改权重。完整修订号和逐文件校验值在 `research/manifests/models/`。

| 项目 | Llama-3.1-8B-Instruct | Qwen2.5-7B-Instruct |
|---|---:|---:|
| 项目内目录 | `models/Llama-3.1-8B-Instruct` | `models/Qwen2.5-7B-Instruct` |
| 修订号 | `0e9e39f249a16976918f6564b8830bc894c89659` | `16c174980d8a1492910551634b4969e69cdc2444` |
| 层数 | 32 | 28 |
| Query / KV head | 32 / 8 | 28 / 4 |
| Head dimension | 128 | 128 |
| 配置窗口 | 131,072 | 32,768 |
| 权重体积（约） | 14.96 GiB | 14.19 GiB |

32,768 历史 token 的纯 KV 载荷（不含参数、padding、缓冲、工作区）：

| 存储精度 | Llama | Qwen |
|---|---:|---:|
| BF16/FP16 | 4.00 GiB | 1.75 GiB |
| KV8 | 2.00 GiB | 0.875 GiB |
| K8V4 / K4V8 | 1.50 GiB | 0.656 GiB |
| KV4 | 1.00 GiB | 0.438 GiB |

这张表用来判断 32K 纯 KV 能否放进单卡。M0 的峰值显存另测，含权重、激活和工作区。

## 软件与作业

登录节点 Python 3.6.8 不用于实验。环境建在 `research/.runtime/env/`。依赖锁定写入 `research/env/requirements-lock.txt`。`HF_HOME`、`TORCH_HOME`、`PIP_CACHE_DIR`、`TMPDIR` 指向 `research/.runtime/cache` 或 `research/.runtime/tmp`。

| 组件 | 拟定版本（M0 锁定） |
|---|---|
| Python | 3.11 |
| PyTorch | 2.5.1，CUDA 构建先试 cu121，驱动需 ≥ 530 |
| Transformers | 4.46.3 |

- 单卡目标约 48 GiB；实际型号、驱动、显存和 BF16 在 M0 登记。
- 每次一个模型、一个进程。主实验 `batch_size=1`。
- Slurm 初始申请：1 GPU、8 CPU、64 GiB 主机内存。正式任务拆成可恢复小批次。
- 质量实验默认 BF16 权重与激活。若不支持 BF16，两模型全部对照统一改为 FP16，并升协议版本。
- 峰值显存目标 ≤ 设备总显存的 85%。质量参考使用的反量化浮点缓存单独测峰值，不用上面的纯 KV 载荷表代替。
- 轨迹磁盘预算初始 200 GiB。达到上限后停止新增完整轨迹。OOM 时的处理见 guide.md 的停止表。

## 时间估算

M0 之后用实测填写。这个数只用于申请时长：

$$
T_{\mathrm{GPU}} \approx 1.3 \times \sum_{\text{配置},\,\text{样本}} \big(T_{\mathrm{prefill}}(L_{\mathrm{in}}) + T_{\mathrm{decode}} \cdot N_{\mathrm{gen}}\big)
$$

1.3 是调试余量。E1 以轨迹回放估算，E2 以全生成估算。

## 正式运行前必须填完

2026-09-23 在 `node8` 上实测。下表已填完。权重与激活保持 BF16，不升协议版本。环境锁定 PyTorch 2.5.1 的 cu121 构建。模型从项目内 `models/` 加载。200 GiB 轨迹可以保存。

| 事项 | 实测 | 决定 |
|---|---|---|
| GPU 型号、显存、驱动、BF16 | 两张 NVIDIA RTX 6000 Ada，各 49140 MiB，计算能力 8.9，驱动 550.78。8.9 支持 BF16 | 权重格式保持 BF16。M0 冒烟时再确认 PyTorch 的 BF16 dtype |
| 磁盘配额是否支撑 200 GiB 轨迹 | 项目所在文件系统可用约 86 TiB。`quota -v` 没有报告用户限额 | 轨迹保存可执行 |
| 模型清单的完整修订号与逐文件校验 | 两个 `source.json` 中的大小和 SHA256 已在服务器上重新计算，全部一致 | 可以从项目内路径加载 |
| cu121 与驱动是否匹配 | 驱动 550.78，不低于 530；驱动报告的 CUDA 版本为 12.4 | 可以锁定 cu121 |
