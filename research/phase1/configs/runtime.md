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
