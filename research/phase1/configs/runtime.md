# 第一阶段运行配置

本文件只冻结模型和运行环境。实验顺序和通过条件见 [../guide.md](../guide.md)。本文件与 guide.md 共同定义当前实验约定。

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

这张表用来判断 32K 纯 KV 能否放进单卡。实际冒烟峰值见下方准入记录，含权重、激活和工作区。

## 软件与作业

登录节点 Python 3.6.8 不用于实验。环境建在 `research/.runtime/env/`。依赖锁定写入 `research/env/requirements-lock.txt`。`HF_HOME`、`TORCH_HOME`、`PIP_CACHE_DIR`、`TMPDIR` 指向 `research/.runtime/cache` 或 `research/.runtime/tmp`。

| 组件 | 已锁定版本 |
|---|---|
| Python | 3.11 |
| PyTorch | 2.5.1+cu121，驱动需 ≥ 530 |
| Transformers | 4.46.3 |

- 单卡目标约 48 GiB；已测设备、驱动、显存和 BF16 见下方准入记录。更换设备时核对该设备的准入。
- 每次一个模型、一个进程。主实验 `batch_size=1`。
- 计算节点会话遵守 [服务器连接说明](../../../docs/server-access.md)「计算节点执行方式」中的第八节点交互式 `srun` 约定。正式任务拆成可恢复小批次。
- 质量实验默认 BF16 权重与激活。若不支持 BF16，两模型全部对照统一改为 FP16，直接更新本文件并重跑受影响的对照。
- 峰值显存目标 ≤ 设备总显存的 85%。质量参考使用的反量化浮点缓存单独测峰值，不用上面的纯 KV 载荷表代替。
- 轨迹磁盘预算初始 200 GiB。达到上限后停止新增完整轨迹。OOM 时的处理见 guide.md 的停止表。

## 时间估算

用对应运行环境的实测时延估算。这个数只用于申请时长：

$$
T_{\mathrm{GPU}} \approx 1.3 \times \sum_{\text{配置},\,\text{样本}} \big(T_{\mathrm{prefill}}(L_{\mathrm{in}}) + T_{\mathrm{decode}} \cdot N_{\mathrm{gen}}\big)
$$

1.3 是调试余量。位宽与表示交互以轨迹回放估算，真实生成与误差来源以全生成估算。

## 正式运行前必须填完

2026-09-23 在 `node8` 上实测。下表已填完。权重与激活保持 BF16，沿用已冻结的运行配置。环境锁定 PyTorch 2.5.1 的 cu121 构建。模型从项目内 `models/` 加载。200 GiB 轨迹可以保存。

| 事项 | 实测 | 决定 |
|---|---|---|
| GPU 型号、显存、驱动、BF16 | 两张 NVIDIA RTX 6000 Ada，各 49140 MiB，计算能力 8.9，驱动 550.78；PyTorch BF16 dtype 冒烟已通过 | 权重格式保持 BF16 |
| 磁盘配额是否支撑 200 GiB 轨迹 | 项目所在文件系统可用约 86 TiB。`quota -v` 没有报告用户限额 | 轨迹保存可执行 |
| 模型清单的完整修订号与逐文件校验 | 两个 `source.json` 中的大小和 SHA256 已在服务器上重新计算，全部一致 | 可以从项目内路径加载 |
| cu121 与驱动是否匹配 | 驱动 550.78，不低于 530；驱动报告的 CUDA 版本为 12.4 | 可以锁定 cu121 |

## 已通过的准入检查

预备检查于 2026-09-25 审核通过，配置身份为 `phase1-initial-prompts`。环境冒烟的实际执行提交为 `e29000b`，在 `node8` 的一张 NVIDIA RTX 6000 Ada 上运行，作业号为 `20345`；正确性检查的实际执行提交为 `cb9cf28`。已确认 BF16、PyTorch 2.5.1+cu121、CUDA 12.1，以及仅保留末位词表 logits。

两模型均通过 32、4,096、32,768 token 的 prefill 与单步 decode，prefill 缓存长度与输入一致。32K 单次峰值如下；峰值低于该设备总显存的 85%，包含权重、激活和工作区。

| 模型 | 峰值字节 | 峰值 GiB | 占设备显存 |
|---|---:|---:|---:|
| Qwen2.5-7B-Instruct | 21,800,183,296 | 20.3030 | 42.31% |
| Llama-3.1-8B-Instruct | 26,010,360,320 | 24.2240 | 50.48% |

量化往返、常量组、超界、组／块边界、非有限值、scale 下溢与参数溢出等 11 类 CPU 核对均通过。两模型各 32 与 32,768 token 的 4 条模型记录均通过：prefill 捕获全部 28／32 层；第 0 层单步 decode 后的缓存与可见位置分别为 33／32,769；仅保留末位 logits；第 0 层 pre-RoPE 重新旋转与缓存 Key 的最大绝对差为 0。该记录不声称已逐层验证 decode 或 RoPE 等价。

原始结果固定在 Git 提交 `52068a6a14ecbbb182f39ca42d62245a0325cf2a`：环境冒烟来源为该提交的 `research/phase1/results/m0/device.json`、`qwen.json`、`llama.json`；正确性来源为该提交的 `research/phase1/results/e0/quant.json`、`model.json`。这些是历史来源路径，原始协议字段、执行提交和完整数值可按该提交复查；一次性预备代码、结果和报告已清理，不作为当前实验入口。

上述证据只确认指定环境的运行与实现准入，不代表任务质量、长上下文迁移或服务性能通过。单次时延和峰值不外推到其他设备。模型、软件或共享注意力／量化实现发生影响正确性的变化时，在项目运行目录重新核对受影响的准入；长期回归测试保留在 `research/tests/`。

## 离线混合位宽实现核验

2026-09-30 核验执行提交 `c3a2db2ce73ab6316741eaf011d096f0270c6ff3`。在 `node7` 的 RTX 4090 上，74 项长期回归测试全部通过；两模型的 pre／post-RoPE、在线／离线 K8V4 共 8 个短输入对照与整理前实现生成相同 token。4 个离线混合 INT4／INT8 对照使用实际层位宽的全部 C 参数，有效 KV 和生成 token 与手动逐层选择正确参数的参考逐值一致，全层门槛 logits 为有限值。原有量化、生成与 FP32 注意力数值函数的语法树核对通过，忽略文档字符串和异常文字，未改变数值运算。

这次只验证短输入实现与位宽参数选择，不构成 RTX 4090 的 32K 运行准入，也不构成任务质量结论。正式补测使用已准入的 `node8`、RTX 6000 Ada；加载模型前确认选用 GPU 至少有 40 GiB 空闲显存，不缩短上下文或样本。临时核验结果 SHA256 为 `32f8014ea82fb93c758ad6f1b2766d189476ea2f46c0f30b4b3107f85d80392c`；审核后按一次性检查规则清理。持续回归见 `research/tests/test_generation_quantization.py`，中断依赖保护见 `research/tests/test_experiment_completion.py`。

## 静态 Key 分配的共享 prefill 准入

2026-10-09 在 node8 的 NVIDIA RTX 6000 Ada 上，沿用 Python 3.11、
PyTorch 2.5.1+cu121、Transformers 4.46.3 和 BF16 权重，执行提交为
`ef664c0219e2efd774aa037d7ec1de6beadfff15`。两模型各在 64、192 token 输入下核验三个开发格式及
既定 KV8，共 16 个对照；混合 K4/K8、V4 与 H=8、R=128 的近期副本按正式规则构建。
共用高精度 prefill 后独立量化各配置，与各配置单独 prefill 的入口相比，
有效 K/V 缓存及下一步 logits 逐值一致。19 项相关长期回归测试通过。
准入覆盖上述短输入的捕获与构建等价，正式 C 的局部指标和全层 KL 另由正式结果审核。

一次性核验代码 SHA256 为
`f6318816d2a83d164057af7d4a28d4b8ace29dae7f0649bc2c14cd0aee25fe21`，
输出 SHA256 为
`46d9e24ccb3cc3ad956be8d3002c105eba3e9f1310e5e9b0a0aecb4088211026`。
审核后清理一次性代码、输出及独立目录；执行来源和回归记录见
`research/phase1/source_index.json` 的 `static_key_allocation`。
