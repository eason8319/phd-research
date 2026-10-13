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
- 每张 GPU 同时只运行一个模型、一个进程。用户已允许 Qwen 与 Llama 在两张空闲 GPU 上并行；各模型的参考与候选在同一张卡上配对。主实验 `batch_size=1`。
- 计算节点会话遵守 [服务器连接说明](../../../docs/server-access.md)「计算节点执行方式」中的交互式 `srun` 约定，不固定节点。每次工作前估算实际峰值显存并核对实时空闲容量，满足该次估算及不确定性余量即可启动，不设置至少 40 GiB 空闲的统一条件。正式任务拆成可恢复小批次。
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

这次只验证短输入实现与位宽参数选择，不构成 RTX 4090 的 32K 运行准入，也不构成任务质量结论。正式补测的节点与显存条件按本文件「软件与作业」中的当前约定执行，不缩短上下文或样本。临时核验结果 SHA256 为 `32f8014ea82fb93c758ad6f1b2766d189476ea2f46c0f30b4b3107f85d80392c`；审核后按一次性检查规则清理。持续回归见 `research/tests/test_generation_quantization.py`，中断依赖保护见 `research/tests/test_experiment_completion.py`。

## 本次 Qwen 候选补测的 RTX 4090 显存准入

2026-10-06 用户允许在显存可容纳时使用 RTX 4090。于 `node7` 的 RTX 4090 上，
使用同一冻结环境、Qwen BF16 权重及 `k8v4-per_channel-pre`、H=8、R=128 路径，
检查合成 31,744 token 输入、预留 1,024 token 的完整缓存、首步解码及 32,768
可见位置容量下的一步解码。峰值实际分配 22,561,111,552 字节（21.01 GiB），
峰值缓存预留 24,029,167,616 字节（22.38 GiB）；设备总显存 25,386,352,640 字节。
无 OOM，无效组为 0。这是显存检查，不是任务评分、真实长生成或固定回放实验。

仅本次 Qwen 离线 pre、H=8 补测可使用此设备，加载前至少空闲 23 GiB，逐题记录
实际设备与峰值。缓存预留高于通常 85% 的显存目标，保留该事实并按本次授权执行；
不缩短输入、输出预算或样本，不改变生成与质量容差。其他配置不沿用这项显存准入。
执行基准提交 `3284bfe55cea2ce906e553979039e63f4a1cd1c3`，实际工作区代码由
`research/.runtime/candidate-publication.json` 的 SHA256 清单确认。
一次性检查结果 SHA256 为 `1a7d7221fb26c27de628451c34a48593b052c3321dcccfbb4c5ca6fcd812ce7c`，
检查代码 SHA256 为 `e3b2b3b78a51a06701b9332b9f15a6cd71324b55bf45f432cae4308cddeb538f`。
检查审核通过，仅在本节保留准入事实，临时代码与输出按规则清理。

用户最终澄清可进入有 RTX 4090 的节点，node8 只有 RTX 6000 Ada。
正式补测通过相同交互式 `srun` 方式进入 `node[7]`，完成两个任务各 64 题。
此前 node8 尝试在模型加载阶段因其他进程占用显存而 OOM，没有产生正式题目；
失败记录保留在补测恢复状态与日志中。正式结果逐题保留实际显存峰值和设备信息。

## 跨层、跨步传播的 RTX 4090 准入

2026-10-06 本次传播实验在 node7 的 RTX 4090 上运行。Qwen 的完整 8 题已测最高
预留显存为 24,050,139,136 字节（22.40 GiB）。Llama 对最长冻结诊断输入的
31,742 token、1,024 token 生成容量，完成 BF16 及三个候选的 prefill 和解码检查；
峰值分配为 24,016,911,872 字节（22.37 GiB），预留为 24,803,016,704 字节
（23.10 GiB），无无效组。以上容量仍高于通常 85% 目标，按本次用户允许的
4090 范围执行，逐题保存实际峰值，不缩短输入或输出上限。

Llama 使用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；候选前 8 层缓存
按原 FP32 dtype 暂存 CPU，每层计算时原样搬回，pre-RoPE 捕获原值暂存 CPU。
BF16 参考逐层释放已消费的旧缓存。默认常驻 GPU 的共享入口保持可用。
131 步的小型实际模型检查覆盖 128-token 环形窗口跨界、历史重建及混合 CPU/GPU
驻留；真实 Llama 的三个候选短序列也与常驻路径的 token 和有效 KV 逐值相同。
观测回调与原生成入口的真实 Qwen 对照同样逐值通过。

长期检查为 `research/tests/test_cache_offload.py` 与
`research/tests/test_trajectory_results.py`。基准提交仍为
`3284bfe55cea2ce906e553979039e63f4a1cd1c3`；两模型各自的实际源码清单、快照与
检查校验值由 `source_index.json` 的 `propagation_experiment` 追溯。
Llama 长输入准入输出 SHA256 为
`d7f791282417e6d869c4521b25f5093b3250dc6797553179334f7e2e16f01b5c`。
初次默认分配器检查因显存碎片 OOM；启用上述分配方式后通过。node8 的尝试因其他
进程占用而在加载模型时 OOM，未产生该设备的正式测量。

两模型全部 8 题、三个候选的正式传播与自由生成均已完成。Llama 实测最高分配
24,017,165,824 字节（22.37 GiB），最高预留 24,868,028,416 字节（23.16 GiB）；
这是正式长输入运行峰值，区别于上述单步准入峰值。真实 Llama 的一致性检查使用
129-token 输入和 5 步生成。相关 12 项 CPU 与 2 项 CUDA 回归均通过。
成功的一次性检查仅保留本节准入事实与来源索引校验值，未解决的历史预测复现差异
及 node8 加载失败记录保留。报告绘图依赖单独放在服务器运行目录，不修改实验环境。

## 新输入量化的路径解析核验

2026-10-09 在 node8 核验新输入量化的中断修复，执行基准提交为
`56213c3a8be75e571538e8d255eef9e760ae38dd`。四题长解码输入清单中的路径为字符串，
统一从项目根解析后读取；与此前服务器源码相比仅此处改变，数值函数语法树相同。
19 项长期回归测试通过；两模型共 160 份冻结输入的全文校验值、token 数与 BF16
基线一致，其中 128 份检索输入在修复前后逐 token 相同。已完成的 Qwen 五组检索
共 320 次生成及其完整预测的文件校验值保持不变。

本次仅确认输入读取、既有结果复用及实现恢复条件，不构成长解码质量通过结论。
实际执行源码清单、快照及保留结果校验值见服务器项目内
`research/.runtime/new-input-quality/recovery/repair.json`，SHA256 为
`b3a79b3e140910b19fa4c3d328e7e8e01a8d7c8de007fca475517a89d9cc412c`。
一次性输入核验代码 SHA256 为
`02cb64fee28cd4ca41276c9e82c4ba0ddace2d25a8cfb06d0b29948e36d58708`；
审核后清理该入口与成功检查输出，保留正式运行失败记录、恢复来源和长期回归测试。

2026-10-09 恢复前在服务器运行目录建立冻结执行根，保留当前共享源码改动。
冻结清单的 18 份文件及执行副本的 46 份 Python 文件校验通过，10 项新输入回归
测试通过；两模型 160 个 BF16 基线请求及已完成的 Qwen 400 个量化请求通过复用
与完整性检查。输入、量化参数、输出预算与数值实现沿用原冻结来源。
执行根为 `research/.runtime/new-input-quality/recovery/execution_root`，通过项目内
链接使用原模型、运行进度和正式结果；来源与核验索引为
`research/.runtime/new-input-quality/recovery/frozen_execution.json`，SHA256 为
`3f1665e7b71421bd1a5fd2b83616d78f2ec208dd14495a494a621de182f9a5f8`。
成功核验的一次性入口与输出已清理，恢复副本及失败证据保留供实际执行追溯。

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

## 静态 Key 分配的独立映射与内容位置准入

2026-10-09 在 node8、NVIDIA RTX 6000 Ada 的 GPU 1 上，沿用冻结的 Python 3.11、
PyTorch 2.5.1+cu121、Transformers 4.46.3 与 BF16 权重。
准入执行提交为 `46c3a158a3148330a4b0c4ece6d93f936ed51d97`，25 项长期回归通过，
覆盖独立映射、实际分词跨度、正确目标差距、题内等权汇总、持续缓存与预测前测量、
固定 K 预算及残留成本，以及线性 logits 下 KL 的非可加性。
成功输出 SHA256 为 `e0df4a325d7440ec640ba29ba161bd8290c85511c3aaa31f8c3797cb6d0a1a54`；
预备输出已清理，持续回归与来源见 `source_index.json` 的 `static_key_allocation`。

正式执行提交为 `6be99c903005b1ee2004ce7c6861ea967d82f278`，冻结 2K/4K 检索输入、
32-token 自由输出预算及每方案独立缓存。含余量峰值估算为 Qwen 20 GiB、Llama 22.5 GiB，
各模型加载前实际空闲容量满足估算；实测最大分配/预留为 Qwen 15.109/15.348 GiB、
Llama 17.470/18.229 GiB。全矩阵无无效量化、非有限输出或 OOM，内容与单层原始 logits
复算通过。node6 的 GPU 设备访问失败尚未解决，其记录保留在来源索引指定的服务器路径。
本节登记上述输入与实现的运行事实，V 计划保持未冻结。

## 静态 Key 分配的完整路径与配对精度准入

2026-10-10 在 node7、NVIDIA GeForce RTX 4090 的 GPU 0 上，沿用冻结的 Python 3.11、
PyTorch 2.5.1+cu121、Transformers 4.46.3 与 BF16 权重。准入执行提交为
`c03e885a2ff2b6c39a38df16d6f57400f31f9801`，6 项长期回归通过，覆盖精确答案规范化、
首次非正领先量、实测精度包络、共同基线消去、并列平均秩，以及校准关键层保护后的
随机预算嵌套与同预算完整成本。两模型各用第一条冻结 2K 验证输入和同一 BF16 源值，
核对高精度参照、三格式 K8V4 与敏感度 r=2 的原内容位置 logits；当前计算条件的新入口
与原入口逐值一致。实际 FP32 输出头矩阵计算通过 FP64 真值的玩具矩阵核验；
FP32 残差条件的层归一化输入为 FP32，QKV 投影输入为 BF16。

关闭 TF32 和 autocast，FP32 矩阵精度设为 highest。完整路径的前缀、分隔符、内容与
规范终止 token 先测预测，再输入当前目标；各方案持续维护自己的量化缓存。
准入覆盖上述 2K 同源实现与 dtype 边界；正式 2K/4K 完整路径另由原始 logits 审核。
准入代码 SHA256 为
`65ea762f9c2e6f54151a922f49cd218be98845082dc376031f868cd99386e705`，
成功输出 SHA256 为
`a6c046ce40a00e1bd648f8ece20c90c4d995c8e2f31d5804344538785ade1059`。
准入事实及一次性材料清理状态由
`research/.runtime/static-key-numerics/admission.json` 追溯。

原实测保留峰值 Qwen 15.348 GiB、Llama 18.229 GiB，新增实际 FP32 头权重分别为
2.030 GiB、1.957 GiB；再计残差与临时张量余量，冻结含余量估算为 21 GiB、23 GiB。
node8 的目标卡在检查时显存不足，正式测量改用满足估算的 node7 空闲卡。
旧卡历史 NLL 与新卡结果单独记录，三精度诊断始终共用本轮固定 prefill 源值。
原实现与新入口的同源逐值准入不要求不同设备的历史值逐值一致；过严历史值守卫
触发的未完成运行保留在 `research/.runtime/static-key-numerics/failures/`。
正式执行来源为 `83028d8519f9e466bf86bd2b54256f53298aa345`；V 计划保持未冻结。

完整正式测量及 32 题原始输出审核均通过；实测保留峰值为 Qwen 17.480 GiB、
Llama 19.686 GiB，低于含余量估算。指定路径的在线 post logits 在每题、每精度的
两次执行中逐值一致；这项事实只说明执行重复性，精度变化另由配对增量检查。
审核结果见 `experiments/static_layer_allocation/results/numerics_review.json`。

缓存预留容量的补充实现核验使用 Qwen/Llama 的 head 形状、2K/4K 历史、
9/11/13-token 与 32-token 容量。在共同有效前缀下，注意力逐 KV head 的
FP32 K/V 矩阵均连续，GEMM 的形状、步长和对齐一致；36 个检查输出逐值相同。
其最小事实保存在上述 `admission.json`，一次性输出登记后清理。
审核后作图环境另补 matplotlib 3.10.5；模型测量使用的 PyTorch 2.5.1+cu121、
Transformers 4.46.3 保持原版本；作图环境中的 NumPy 为 2.4.6，实际版本由
`research/.runtime/static-key-numerics/figure_environment.json` 追溯。

## 静态分配独立生成收尾准入

2026-10-12 在 node8、NVIDIA RTX 6000 Ada Generation GPU 0，使用既定 Python 3.11.7、
PyTorch 2.5.1+cu121、Transformers 4.46.3 和 BF16 权重；TF32 与 autocast 关闭。
本轮三候选使用实际 V 最长单问输入，保留完整 1024-token 缓存容量，并核验末端
decode。预留峰值为 Qwen 混合 21.244 GiB、Llama KV4 28.936 GiB、Llama 混合
33.354 GiB，低于含余量估算 30/36 GiB；实际权重分片及 tokenizer 校验通过。
24 项相关回归通过，涵盖原静态实验及新请求级配对、候选优先级和停止规则。

准入来源为 `research/.runtime/static-final/admission.json`，SHA256
`f14ef70ee5994653788ff24ff597ada88a25a6dffedee6a6479d1e86617f96b5`；一次性检查
源码 SHA256 为 `50a17bdf74f2939c5fb7094a3a1428f820fdde8690f44174a5cd675f5531058d`，
审核后清理其入口，保留上述最小事实。正式执行根为
`research/.runtime/static-final/execution_root`，实际源码由同运行目录的 publication.json
逐文件校验登记。24 份既有参考的相同样本输入、生成长度与得分均已核对一致，
首轮实体直接复用；仅 BF16、KV8 的必要缺样追加。缺少历史辅助元数据不自动触发
重新生成，复用与重复实体清理记录见静态包 `reference_reuse.json` 和
`reference_cleanup.json`。V 计划和扩样身份已冻结；准入不代替任务质量判定。

恢复 Llama 时，映射权重直接搬至 GPU 出现加载停滞。保留 BF16 权重，在 CPU
逐张量复制并核对 dtype 与逐值相等后再搬至 GPU，恢复加载成功；前向计算与量化
运算保持原实现。该可选加载方式只改变权重传输，执行来源和适用样本范围由
`source_index.json` 追溯。参考复用、缺样追加与统一结果格式的 33 项回归检查通过。

## 公共实现与执行版本核验

2026-10-13 在 node8 的交互式计算节点，使用原 Python 3.11 环境完成 168 项长期
回归检查，165 项通过，3 项因本次隐藏 CUDA 而跳过。共用生成、缓存、量化、BF16
解码和 EOS 实现与原静态收尾快照的计算签名一致；该核验不新增模型生成，也不声称
重新完成 CUDA 准入。原设备准入和任务结果仍沿用上述实际来源。

本次本地基准为 `cbce2edc20183fff42f8444a67b539eb1fbf205d` 加实际文件 SHA256，
服务器代码发布到独立快照；当前索引为 `research/.runtime/phase1-execution/current.json`。
回归记录为 `research/.runtime/phase1-execution/1c5ed8a21f78b127/tests.log`；来源索引
登记逐文件一致性、计算签名和原测量归并依据。共同首轮复用 1,248 个统计单位，
保留必要参考追加 2,496 个、候选测量 1,872 个。原快照、原审核和完整预测归档仍可
追溯；本次只整理执行和结果来源，不改变任务的质量判定。

四个正式实验完成后，使用 `python -m phase1.entry review --experiment all` 在当前
服务器执行快照复查保存的评分、配对统计、扩样停止和成本依赖。统一复查与长期
回归的最新事实保存在 `source_index.json` 的 `phase1_organization`；不把只读复算
记作新的模型生成或 GPU 准入。原始输出审核和实际模型测量继续沿用各自执行来源。

2026-10-13 按当前四份报告的证据范围清理一次性检查及未使用的旧基线。
服务器旧编号结果目录、未报告的检索诊断和构造对照、一过性检查入口及其输出
已清理；历史失败事实、文件原校验和删除依据由 `source_index.json` 的
`report_scope_cleanup` 追溯，未将尚未归因的旧诊断改记为通过。
报告使用的完整预测、logits、校准参数、实际执行源码与长期回归继续保留。
