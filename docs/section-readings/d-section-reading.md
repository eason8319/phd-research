# D 类精读：活跃请求、KV 驻留与完整模型 TPOT

整理日期：2026-09-22。对象是文献清单 **D01–D10**。书目与归类以 [文献清单](../literature-survey-reading-list.md) 为准，本文不另建清单。课题对照口径见 [项目范围](../project-scope.md)。

第 1–5 节回答本类的五个问题。第 6 节是逐篇要点。第 7 节记录不改变上述回答的其他观点。第 8 节分成两段：可以写入课题模型的判断，以及这些论文尚未覆盖、不能写成已经证实的部分。

数字取自各篇开放全文的方法、实验与附录。原文没有的测量写「未报告」。效率倍数只作该论文自己的基线对照，不换算成课题口径，也不直接写成 Attention 引擎的加速。

D 类在课题中是系统边界，不构成必交付的服务框架。数字用于约束工作负载、容量、接口和 TPOT 分解。课题侧核心效率是 $B=1$、$S\in\{32\mathrm{K},64\mathrm{K},128\mathrm{K}\}$ 上连续 256 个 decode 步；完整模型 TPOT 在 Attention 服务时间之外单独计入投影与 FFN。下文的 $B$ 是同时在飞的请求数，$S$ 是当前可见历史长度。

## 阅读边界

阅读范围、版本和不能外推的部分记在这里。第 1–5 节的数字都受这些限制。

- 阅读版本：D01 为 USENIX OSDI 2022 开放 PDF；D02–D09 为 arXiv HTML（FlexGen 2303.06865v2，PagedAttention 2309.06180v1，Splitwise 2311.18677v2，ALISA 2403.17312v1，DistServe 2401.09670v2，InfiniGen 2406.19707v1，CacheGen 2310.07240v2，vAttention 2405.04437v2）；D10 为 arXiv HTML 2412.03131v3，题名已是正式版 DiffKV。
- 图中曲线没有逐点抄录。正文给出的倍数、容量和带宽来自论文文字或表格。
- 第 5 节逐条列出倍数成立的服务条件。换成 $B=1$、32K–128K、GQA 或 INT4/INT8 页时，需要重测。

---

## 1. 有多少活跃请求，长度和 batch 如何变化？

D 类没有一篇把活跃请求数固定在 $B=1$、并把上下文固定在 32K–128K。在线系统里，每一步的 batch 由到达过程、各请求已生成长度和剩余 KV 容量共同决定，会在迭代之间增减。离线吞吐系统则把一批等长 prompt 一次跑完，batch 在一次生成内部基本不变。

| 论文 | 请求如何进 batch | 长度 | batch 如何变 |
|---|---|---|---|
| D01 Orca | 每迭代按到达时间重选，上限 `max_bs`；同时为新请求按 `max_tokens` 预留 KV 槽 | 最大序列 2048。端到端：输入 $\mathrm{U}(32,512)$，生成 $\mathrm{U}(1,128)$，生成到上限为止。微基准输入 32 或 128、生成 32 | `max_bs` 取 1/8/16/32。负载用 Poisson 到达率扫描。低负载时 101B 凑不满 batch，延迟退回引擎本身 |
| D03 vLLM | 迭代级 FCFS；物理块不够时整段请求换出或重算，并暂停接收新请求直到被抢占请求完成 | ShareGPT 的平均输入、输出分别是 Alpaca 的 $8.4\times$、$5.8\times$。OPT 最大序列 2048。聊天实验把 prompt 截到最近 1024，最多再生成 1024 | 同样到达率下，OPT-13B / ShareGPT 上 vLLM 同时处理的请求数是 Orca (Oracle) 的 $2.2\times$、Orca (Max) 的 $4.3\times$。OPT-175B + 短序列 Alpaca 时，空闲 KV 空间很大，系统转为计算受限，分页优势变小 |
| D04 Splitwise | 集群把请求同时分到 prompt 机和 token 机。prompt 机 FCFS，且一批 prompt 的 token 总数限制在 2048。token 机尽量填满，接近显存上限才排队。混合池里 prompt 可抢占 token | Azure 生产痕迹：编程中位 prompt 1500、输出 13；对话中位 prompt 1020、输出 129 | 把痕迹缩到 2 请求/秒后，对话有 60%–70% 的时间一批里只有不超过 20 个活跃 token；编程有超过 20% 的时间 token 阶段只有 1 个活跃 token。prompt 阶段按全部 prompt token 计数，token 阶段每请求只计 1 |
| D06 DistServe | prefill 与 decode 分实例。prefill 在输入短于饱和长度 $L_m$ 时才合并；decode 用最大可容纳 batch。Poisson 到达，历史痕迹拟合后重采样 | ShareGPT、HumanEval、LongBench。13B 上单条 512 token 的 prefill 已使 A100 计算饱和，更大模型饱和所需序列更短。文中 decode 分析例为 batch 128、输入 256 | prefill batch 在真实 prompt（常为数百 token）下保持很小。decode batch 受 KV 容量限制，分离后可以由多台 prefill 向一台 decode 累积 |
| D02 FlexGen | 离线、无限 prompt 池。GPU batch 与 block 内 GPU batch 数分开；有效 batch = 二者之积 | 主实验 prompt 512 或 1024，每条固定生成 32 token，prompt 被 pad 到等长 | OPT-175B、单张 16 GB GPU：无压缩时 GPU batch 32、block $32\times 8$（有效 batch 256）；4-bit 后有效 batch 144。DeepSpeed / Accelerate 的 batch 上限为 2 |
| D05 ALISA | 单机 GPU–CPU，一批等长序列跑完，没有多租户到达过程 | 系统实验输入 128、输出 512；精度实验输入 2048 | batch 从 4 扫到 64。80% KV 稀疏是其精度可接受的上限（多数任务下降小于 5%，Alpaca 约 3%） |
| D07 InfiniGen | 离线批，KV 在 CPU，GPU 只取本步推测出的条目 | 常见配置输入 1920、输出 128，总长 2048；另有总长 512–2048 与 Llama-2-13B 的 4096 | 延迟主图 batch 20；序列扫描 batch 8；模型规模扫描 batch 4 |
| D08 CacheGen | 一次加载一段可复用上下文，再进入未改动的 decode。并发请求共享同一 GPU | 662 段上下文，约 1.4K–16K token。LongChat 中位 9.4K；NarrativeQA 中位 14K | 并发数上升时，单请求分到的 GPU 时间下降，TTFT 上升。短于约 1K token 时改为发送原文并在本地重算 |
| D09 vAttention | 同一 vLLM 调度器。decode 吞吐实验把每请求初始上下文固定，再跑 400 步 | decode：初始上下文 16K，batch 扫到 32（Yi-34B 在 32 时显存不足）。端到端：50 条请求，上下文 32K–128K，prefill:decode token 比 500、100、50 | 动态 OpenChat、7 请求/秒时，2 MB 页的最大 batch 为 Yi-6B 187、Llama-3-8B 203、Yi-34B 56；64 KB 页升到 240、258、68 |
| D10 DiffKV | 每步在显存内尽量多塞请求。动态负载沿用 vLLM 的 Poisson 扫描 | 吞吐实验最大生成长度：QwQ-32B 为 16K，Qwen2.5-32B 为 8K，其余模型 4K；1000 条 MATH 序列，生成常常顶到该上限 | QwQ-32B 上达到的 batch 为 15.9，同设置 vLLM 为 2.7。Llama3-8B、batch 128 的 32 MB 页表是元数据例子，不是该吞吐实验的实测 batch |

对课题的直接读法：$B=1$ 的连续 256 步更接近 Splitwise 编程痕迹里 token 阶段的低占用区间，以及 DistServe 所批评的「decode 专用 batch 很小则 GPU 吃不满」。它不同于 Orca/vLLM 用内存换并发、也不同于 FlexGen 用数百条等长短输出换吞吐。$S$ 在 D 类主实验里多数停在 2K–16K；vAttention 的 64K–192K 和 DiffKV 的 16K 生成是少数长序列点，且都带有较大的 batch 或很长的生成，不能当成 $B=1$、128K 的逐步轨迹。

---

## 2. KV 驻留在哪里，实际可用容量和带宽是多少？

「GPU 显存容量」在这些论文里几乎都不是 KV 的可用容量。权重要先占用一块，Orca/vLLM 式预留和碎片还会再砍掉一块。带宽必须分成四条不同的路：GPU HBM、CPU–GPU、机器间网络、磁盘。论文多数只给出其中一条的实测或峰值，没有给出本课题需要的「扣除权重后的可持续 HBM 带宽」。

**只驻留在 GPU。**

- Orca 把 K/V 放在 GPU。新请求第一次被调度时按 `max_tokens` 预留槽位，槽位总量 `n_slots` 取模型、并行度和显存允许的最大值。实验机为 Azure 8×40 GB A100，机内 NVLink；每台 VM 有 8 条 200 Gbps HDR InfiniBand，机间合计 1.6 Tb/s。参数与激活为 FP16。论文没有报告 HBM 可持续带宽，也没有报告扣掉权重后的 KV 字节数。
- vLLM 的物理 KV 块在 GPU DRAM；换出时才复制到 CPU，且 CPU 换出空间不超过 GPU 上的 KV 物理块总量。表 1 的可用 KV 容量是：OPT-13B 在 40 GB 上参数 26 GB、KV 12 GB、最多 15.7K 个 token 槽；66B 在 160 GB 上参数 132 GB、KV 21 GB、9.7K 槽；175B 在 640 GB（8×80 GB）上参数 346 GB、KV 264 GB、60.1K 槽。OPT-13B 单 token KV 为 $2\times 5120\times 40\times 2=800\,\mathrm{KB}$，一条顶满 2048 的请求为 1.6 GB。既有系统的有效显存比例可以低到 20.4%。默认块大小 16。小块换出会把 PCIe 打成大量小传输，论文没有给出实测 GB/s。
- vAttention 仍把物理页放在 GPU，虚拟地址按最大上下文一次留出。A100 80 GB。Yi-6B、Llama-3-8B、Yi-34B 的每 token KV 分别为 64 KB、128 KB、240 KB。decode 的物理页分配需求饱和在不超过 600 MB/s；64 KB 页时 CUDA 分配带宽为 7.6 GB/s（TP-1），高于分配需求一个数量级。Yi-34B、TP-2、最大上下文 200K、虚拟 batch 500 时，仅 K 或 V 的虚拟缓冲就到每 worker 100 GB、60 层合计 12 TB 虚拟地址；物理页仍按需分配。64 KB 页相对 2 MB 页没有测到 attention kernel 变慢。
- DiffKV 的页也在 GPU。L40 48 GB，权重 FP16。压缩后 KV 约为 FP16 的 $1/5.7$ 到 $1/2.7$（摘要的 $2.7\times$–$5.7\times$）；表 1 里各任务的占用约为 FP16 的 19.3%–36.7%。Llama3-8B、batch 128、32 层、8 个 KV head 时，双向页表共 32 MB，而单请求 KV 约 1 GB。attention kernel 的加速接近字节削减：K8V8 把体积减半，理论上限 $2\times$，实测 $1.7\times$，差额来自量化元数据和反量化。论文没有单独给出 HBM 峰值带宽。

**GPU + CPU，必要时加磁盘。**

- FlexGen 在一张 T4（16 GB）、208 GB DRAM、1.5 TB NVMe 上把权重、激活和 KV 按线性规划拆到三层。SSD 读约 2 GB/s、写约 1 GB/s；附录另有 1.6/1.3 GB/s 与 0.5/0.5 GB/s 两档磁盘。CPU–GPU 带宽只作为代价模型变量，正文没有给出实测常数。OPT-175B 权重大约 325 GB；若 $b=512$、$s=512$、生成 32，KV 为 1.2 TB，是权重的 $3.8\times$。OPT-175B、prompt 512 的时间分解里，prefill 2711 s 中计算 2220 s；decode 11315 s 中计算只有 1498 s，KV cache 读为 7046 s。对应的 GPU 计算占用率是 prefill 82%、decode 13%。KV 在 CPU 时，在 CPU 上算 attention 分数可把搬运量从 $b\times s\times h_1$ 降到 $b\times h_1$，长序列（$s\ge 512$）更合适。
- ALISA 的机器是 7B/13B 用 16/32 GB V100，30B 用 80 GB H100，CPU 为 128 GB DRAM，CPU–GPU 带宽 20 GB/s。权重和激活保持在 GPU。KV 先全在 GPU；超过 GPU 后按 token 拆到 CPU；序列再变长则从 CPU 删掉最老的一部分，用时在 GPU 重算。系统实验把 KV 做到 80% 稀疏，并存成 INT8。单 token KV 大小按 FP16 写成 $4\cdot b\cdot l\cdot h$ 字节。OPT-13B、序列 512、batch 64 的 KV 超过 25 GB，已经大于约 23 GB 的权重。
- InfiniGen 把完整 KV 放在 CPU，GPU 上只保留一个按计数器淘汰的池，并且每层最多把 20% 的 KV 送上 GPU；跨层平均实际参与 attention 的比例低于 10%（partial weight 比例 0.3，OPT 的 alpha 为 4，Llama-2 为 5）。机器是 RTX A6000 48 GB、96 GB DDR4-2666、PCIe 3.0 ×16。论文写出了链路代数，没有写出测得的可持续 GB/s。OPT-30B 放不下时，额外把 30% 权重放到 CPU，这块权重是 KV 体积的 $1.7\times$。

**跨机器或跨网络。**

- Splitwise 的 KV 在 prompt 机 GPU 上逐层生成，再送到 token 机 GPU，之后随生成增长。云集群描述为每对 GPU 25–50 GB/s 的 InfiniBand（论文写作 GBps）。Azure 实测机为 DGX-A100 与 DGX-H100，H100 互联是 A100 的两倍，文中写 200 Gbps 对 400 Gbps。机内用张量并行，机间只传 KV。
- DistServe 的 KV 从 prefill 实例拉到 decode 实例。OPT-66B 上 512 token 的 KV 约 1.13 GB；若 10 请求/秒且要让传输「看不见」，需要约 90 Gbps。实验集群 4 节点 32 张 A100-80GB，跨节点只有 25 Gbps，因此把同一流水级的 prefill/decode 段放在同一台机器上，走 NVLink；论文给出的 A100 NVLink 峰值为 600 GB/s。OPT-175B / ShareGPT 上传输不到总延迟的 0.1%，95% 以上请求的传输短于 30 ms。
- CacheGen 的 KV 事先编码成码流，经网络进入 GPU。服务器是 4×A40、384 GB 主机内存。主 TTFT 曲线取 3 Gbps；敏感性从 0.1–10 Gbps 扫到 10–100 Gbps。文中例子：25 GB 的 KV 在 20 Gbps 上要 10 s，在 100 Gbps 上要 2 s，与同模型把原文 prefill 一遍的时间同量级。压缩后体积相对均匀量化为 $1/4.3$–$1/3.7$。

对本课题：$C_{\mathrm{offchip,avail}}$ 应照 vLLM 表 1 的方式填写，即设备容量减去权重、激活和运行时保留，再减去碎片；不能把标称 HBM/GPU 容量写成 KV 容量。D 类没有一篇给出「长上下文、低比特、规则流式访问」下的可持续片外带宽。能直接引用的链路数字是：ALISA 的 20 GB/s CPU–GPU、FlexGen 的约 2 GB/s 磁盘读、DistServe 的 25 Gbps 跨节点与 600 GB/s NVLink 峰值、Splitwise 的 200/400 Gbps、CacheGen 实验用的 3 Gbps 量级网络。

---

## 3. Prefill 交给 decode 的格式是什么？转换和初始化要多少成本？

这些系统里的「交接」有三种，成本差一个数量级以上。

**同一引擎、同一地址空间，只切换计算形状。** Orca 把第一次迭代称为 initiation：一次吃进全部输入 token，写出 K/V；之后的 increment 每次只吃一个新 token，用请求号和 token 下标向 Attention K/V manager 要地址。控制消息含请求号、当前 token 下标和输入 token 数，走 CPU 侧 gRPC，张量走 NCCL。没有单独的格式转换阶段。vLLM 的 prefill 仍用常规 self-attention（文中指向 FlashAttention）生成 KV，再经融合的 reshape 与 block write 写入块表；块大小默认 16，最后一块可以只填一部分。逻辑块到物理块的映射对 kernel 可见。vAttention 把这块写回简化为一次张量拷贝，因为虚拟地址连续；分页实现则要按块追加。DiffKV 在 prompt 步先按「全部高精」保守分配页，压缩之后立刻回收多占的页；生成步每个 head 最多再要一页，请求结束前不做页回收。它的页内布局是为向量化反量化定的：$K$ 与 $V$ 的排布不同，并带 scale、zero-point、分数和位置。

**换一台机器，KV 内容不变。** Splitwise 声明传输是无损的，精度与单机执行相同；实现基于 vLLM 块，用 MSCCL++ 的单边 put 按块发送，并尽量把同一信号量下的连续块并成一次传输。prompt 计算与逐层发送重叠。H100 上短于 512 token 的 prompt 用串行传输，更长的用逐层传输。没有被计算盖住的残余时间约为 A100 8 ms、H100 5 ms。无 batch 的编程痕迹上，这项开销约占端到端延迟的 0.8%；第二个 token 的延迟增加 16.5%（串行传输则为 64%）。DistServe 交接的是「中间结果，主要是 KV cache」加上第一个 token。跨节点用 NCCL，节点内用异步 `cudaMemcpy`。decode 侧拉取，prefill GPU 暂存来不及消化的 KV。在它的低跨节点带宽集群上，由于改走 NVLink，这项成本落到前面说的 $<0.1\%$ 与 $<30\,\mathrm{ms}$。论文同时写出：若带宽不够，512 token 的 OPT-66B 就要 1.13 GB/请求。

**换一种编码，decode 前要解码。** CacheGen 交给下游的是分块码流，不是 FP16/BF16 张量。每块相对锚点做差分，按层分配量化级别，再按 channel–layer 分组做算术编码；传输中可降级，带宽过低时改发原文并在本地 prefill。码流解码与传输流水重叠，论文认为解码对端到端延迟的影响很小，且远小于从原文重算 KV。离线编码约 200 ms。短于约 1K token 时直接走原文。CacheGen 写明它不改变 decode 过程：KV 进 GPU 之后的逐步生成与原来相同。

**初始化里容易被当成零成本的部分：**

- vLLM 换出与重算的取舍取决于 CPU–GPU 带宽和 GPU 算力。小块换出因为传输次数多而变慢；重算延迟不随块大小变化，并且在其微基准里始终不超过换出延迟的 20%。中等块（16–64）上二者的端到端表现接近。
- vAttention 若在关键路径上同步调用 CUDA 分配，64 KB 页可使单条 16K prefill 变慢到 $1.15\times$，decode 上出现 5–20 ms 的尖峰。把分配与计算重叠、提前映射、推迟回收之后，这些尖峰从关键路径消失。
- DiffKV 的并行压缩整理占 prompt 步延迟的不到 0.2%，占生成步的不到 0.9%。同一逻辑若放在 CPU 上多线程做，生成步的管理时间会超过模型执行时间。管理结构必须留在数据面上，否则压缩省下的时间会被管理吃掉。
- InfiniGen 的交接不是改格式，而是每层只预取下一层推测为重要的 K/V。推测用当前层输入、下一层 query 权重的一部分和下一层 key。partial weight 比例 0.3 对延迟影响很小，因为它不改变传输量；alpha 增大则会多传 KV。部分模型（文中 OPT-6.7B）若不做离线权重 skew，精度会明显掉下去。
- ALISA 的相位切换点、卸载比例和重算比例是离线算好的，在线没有这项搜索开销。INT8 的 scale/zero-point 随 KV 一起存，计算前反量化回 FP16。

课题若规定 prefill 产出的是另一种页布局、精度标签或页统计，这项转换要单列，不能沿用 Splitwise「无损、只多几毫秒」或 DistServe「NVLink 上可忽略」。那两个结果依赖同构 GPU、未改数值格式，以及高带宽互联。

---

## 4. 引擎加速对完整模型 TPOT 能产生多大影响？

D 类没有一篇把「Attention 引擎单独加速 $k$ 倍」映射成 GQA、长上下文、$B=1$ 下的完整模型 TPOT。能读出的是：完整一步里权重与 KV 的搬运经常占主导；attention kernel 的加速会被同一步的其他算子、调度和 prefill 摊薄；系统论文里最大的倍数来自并发和驻留策略。

按论文自己的口径，相关数字是：

- **Orca 的 $36.9\times$ 不是引擎算术加速。** 175B、中位归一化延迟 190 ms 时，FasterTransformer 为 0.185 请求/秒，Orca 为 6.81 请求/秒。归一化方式是每条请求的端到端延迟除以它生成的 token 数。作者把大模型的主要瓶颈写成每一轮从 GPU 全局内存读取参数；selective batching 故意不把 Attention 并成一个大 batch，因为 Attention 没有可复用的参数，微基准里因此只比 FasterTransformer 略慢或相当。175B 上引擎本身最多快 47%，作者归到控制面与数据面分离。低负载、单机 8 GPU 的 101B 上，两端延迟接近，因为 batch 里没有足够请求。
- **vLLM 的请求率提升与 attention kernel 反向。** PagedAttention kernel 比 FasterTransformer 的 attention 慢 20%–26%，作者写明这份开销只落在 attention，不落在 Linear。端到端仍能撑住最高约 $22\times$ 的请求率，因为同时驻留的请求更多。kernel 变慢没有主导完整服务延迟。
- **vAttention 把 kernel、decode 吞吐和端到端拆开了。** vLLM 的 decode kernel 延迟最高是 FlashAttention 的 $2.8\times$（Yi-6B）、$1.5\times$（Llama-3-8B）、$2.5\times$（Yi-34B）。换成 FlashAttention 非分页 kernel 后，decode 吞吐最高是 vLLM 的 $1.99\times$、$1.58\times$、$1.53\times$。FlashAttention 分页与非分页 decode kernel 的延迟几乎一样：decode attention 访存停顿盖住了分页多出来的指令。prefill 是计算受限，分页 kernel 最高慢 28%（FlashAttention）和 24%（FlashInfer）。50 条请求的端到端 makespan 只快到 $1.22\times$（相对 FlashAttention 分页）和 $1.29\times$（相对 FlashInfer 分页），而且 prefill:decode 越高、上下文越长，收益越大。decode 上，更好的 kernel 随 batch 变大更有用，因为 attention 延迟随批内 token 总数增长，在整步延迟里的占比上升；Llama-3-8B 上 batch 从 4 到 32，相对 vLLM 的收益从 $1.05\times$ 增到 $1.58\times$。
- **DiffKV 的 kernel 加速大于整批延迟加速，整批延迟加速又小于吞吐加速。** K8V8 的 kernel 为 $1.7\times$（体积减半的理论上限是 $2\times$）。8 条序列、长度 4096 的整批端到端延迟是 vLLM 的 $1.4\times$–$1.6\times$。吞吐是 $1.9\times$–$5.4\times$，其中 $5.4\times$ 出现在 QwQ-32B、最大生成 16K、MATH 样本；同点 Quest、SnapKV、Atom、KIVI 为 $1.6\times$、$1.8\times$、$2.1\times$、$3.4\times$。Quest 不减少驻留 KV，batch 与 vLLM 相同，加速来自少算一些 token，重要性估计又吃掉一部分。Atom 与 KIVI 的 batch 可以接近或超过 DiffKV，吞吐却不成比例，因为它们缺融合反量化 kernel，框架开销也更大。生成步里模型执行占 92%–93%，内存管理不到 0.9%。论文没有把这 92% 再拆成 attention 与 FFN/投影。
- **带宽型系统里，少搬 KV 才能动完整一步，少做算术不够。** FlexGen 的 decode 计算只占该阶段时间的约 13%，KV 读占 7046/11315 秒。InfiniGen 相对 FlexGen 的加速随序列变长而升到 $5.28\times$（OPT-13B，batch 8，输出 128，输入 384–1920）；INT4 与 H2O 在同图的上限是 $1.92\times$ 与 $3.40\times$。摘要中的 $3.00\times$ 是相对既有 KV 管理方法，正文没有把它钉在单一 $(B,S)$ 上。OPT-30B 还要卸载 30% 权重时，InfiniGen 相对 FlexGen 只剩 $1.34\times$。ALISA 的 token 吞吐包含 prefill 和 decode，80% 稀疏下相对 FlexGen 为 $1.4\times$–$3.0\times$，大 batch 下相对 vLLM 最高 $1.9\times$；小 batch 时 vLLM 更高。
- **阶段干扰可以让 TPOT 的变化远大于引擎本身。** DistServe 定义 TPOT 为除第一个 token 外、每输出 token 的平均时间，请求延迟 $=\mathrm{TTFT}+\mathrm{TPOT}\times$ 生成长度。往 decode batch 里加入一条 prefill，会把该步 TPOT 明显拉长，输入越长越严重。它报告的是达标请求率：ShareGPT 上 $2.0\times$–$3.41\times$，代码补全 $3.2\times$，摘要 $4.48\times$ 请求率，以及摘要上 $10.2\times$ 更紧的 SLO。这些倍数来自分开 prefill/decode 并重配并行度。decode 在 batch 增大后会从带宽受限走向计算受限，此时 intra-op 用来压 TPOT，收益递减；inter-op 几乎线性提高吞吐。论文没有给出 attention 在 TPOT 中的百分比。
- **Splitwise 的 token 阶段对 batch 和功耗都不敏感。** batch 64 时 TBT 只变成约 $2\times$。token 阶段把 GPU 功耗上限从 700 W 降到 350 W，延迟几乎不变；prompt 阶段对功耗上限敏感。BLOOM-176B 上，1500 token 的 prompt 时间等于大约 6 个输出 token 的时间，所以端到端时间仍主要在 token 阶段。TBT 是整步时间，没有拆 attention。

可以带进课题模型、但必须带条件的定量关系：

1. 在 vAttention 的 A100、上下文 16K、batch 至 32 的 decode 上，attention kernel 的延迟差到了约 $2.8\times$，完整 decode 吞吐差到了约 $2\times$，端到端在 prefill 偏重时只到约 $1.2\times$。kernel 倍数不能原样写成 TPOT 倍数。
2. 在 DiffKV 的 L40 上，字节减半带来约 $1.7\times$ 的 attention kernel，整批端到端约 $1.4\times$–$1.6\times$。再大的吞吐倍数依赖 batch 从 2.7 增到 15.9 这一类容量效应。
3. 管理、格式转换和反量化要单独进 TPOT。DiffKV 把管理压到生成步的 0.9% 以下。vAttention 测到 vLLM 的块表准备一度占 decode 迭代的 30%，修复后仍可到 10%。本课题的选页、重排和反量化若超过这一量级，就会在完整 TPOT 里可见。
4. $B=1$、长 $S$ 时，D 类的证据指向「KV 字节和 HBM/片外带宽决定 attention 部分」，权重仍要每步从片外进入。Orca 对大模型的判断是参数读取主导整步。本课题已把投影和 FFN 列在引擎之外，TPOT 必须分项相加，不能用 attention 时间代替。

---

## 5. 哪些结果依赖特定服务条件？

下面每条都只在所列条件下成立。换成长上下文、$B=1$、GQA、低比特页或另一条存储链路时，需要重测，不能外推倍数。

- **Orca $36.9\times$：** GPT 形状（非公开 GPT-3 权重，实验不发射 EOS）、FP16、最大长度 2048、输入与生成长度的均匀分布、Poisson 到达、175B 的 2×8 模型并行、以及与 FasterTransformer 微批次流水线对比。请求同质（输入、生成都是 32 或都是 256）时，FasterTransformer 随 batch 增大明显变好，差距缩小。101B、低负载、单机内时，调度优势消失。
- **vLLM 的 $2\times$–$22\times$ 请求率：** OPT 13B/66B/175B 与表 1 的 KV 余量、ShareGPT/Alpaca 的长度分布、Poisson、1 小时痕迹（175B 为 15 分钟）、自实现的 Orca 三种预留策略。175B + 短 Alpaca 时内存不再是第一约束。共享前缀实验（80 与 341 token）和 beam/parallel sampling 的内存节省（Alpaca 上并行采样 6.1%–9.8%，beam 37.6%–55.2%）依赖请求真的共享前缀或候选。聊天实验故意不在轮次之间保留 KV。
- **FlexGen 的高吞吐倍数：** 同一延迟 5000 s 时超过 DeepSpeed / Accelerate $40\times$；把有效 batch 扩到 256 时最大吞吐为基线的 $69\times$；4-bit、有效 batch 144、延迟 4000 s 时论文给出 $100\times$；prompt 512 的单卡表中，压缩配置相对 DeepSpeed 为 $112\times$。这些点都依赖单张 16 GB T4、208 GB DRAM、约 2 GB/s 读的 SSD、等长 prompt、只生成 32 token，以及允许数千到上万秒延迟。生成吞吐把 prefill 时间算进分母。这是吞吐–延迟换点，不是在线 TPOT。
- **Splitwise 的机器配比与「传输可忽略」：** 编程与对话两种长度分布、混合连续 batching、prompt 批不超过 2048 token、H100 prompt / A100 token 或 token 机 50% 功耗上限、以及 200–400 Gbps 的 InfiniBand。传输残余 5–8 ms 是相对这些 GPU 上的 prompt 计算；0.8% 端到端开销来自无 batch 的编程痕迹。SLO 是相对空载 DGX-A100 的倍率（TTFT 的 P50/P90/P99 为 $2\times/3\times/6\times$，TBT 与 E2E 为 $1.25\times/1.5\times/5\times$），九条同时满足。集群结果主要来自模拟器，性能模型 MAPE 小于 3%，并用超过 5 万次迭代做了端到端核对。
- **DistServe 的 $4.48\times$ 请求率与 $10.2\times$ SLO：** 自定的 TTFT/TPOT（例如摘要 TTFT 15 s、TPOT 0.15 s；聊天 OPT-13B 为 0.2 s 与 0.1 s），90% 请求达标，OPT FP16，Poisson，以及低节点亲和放置。跨节点只有 25 Gbps 时，传输「可忽略」依赖同级 prefill/decode 共用 NVLink。高 InfiniBand 集群上的放置算法是另一套，文中大部分实验没有走那套。模拟器假设工作负载在小时到天的尺度上可预测。
- **ALISA 的 $3\times$ / $1.9\times$：** 80% 稀疏在其精度曲线上仍可接受、输入 128、输出 512、batch 偏大、权重留在 GPU、链路 20 GB/s。稀疏再高或任务对局部/跨步注意力更敏感时，精度先失败。小 batch 在线服务里 vLLM 更好。精度实验用满 2048 上下文，系统实验没有。
- **InfiniGen 的加速：** KV 整表在 CPU、模型尽量在 GPU、PCIe 3.0 ×16、alpha 与 20% 上限按模型调过。加速随序列变长而增大，因为重要 token 数次线性增长：OPT-13B 上长度 512/1024/1536/2048 的重要 token 平均为 37/60/66/73，而 H2O 的 20% 预算在 2048 时要加载 409。权重也大量在 CPU 时，收益降到 $1.34\times$。精度依赖 skew；永久驱逐（H2O）在超过预算后与全量 KV 的注意力相似度下降。
- **CacheGen 的 $2.7\times$–$4.3\times$ TTFT：** 可复用的长上下文、3 Gbps 量级链路、Llama 7B–70B 的长上下文微调、以及质量下降被接受（精度约 2% 以内，F1 小于 0.1%，困惑度小于 0.1）。带宽高于约 20 Gbps 时绝对收益变小。上下文短于约 1K、或上下文不可复用（论文举实时搜索）时，这条路径不成立。它不改善 decode TPOT。
- **vAttention 的 $1.99\times$ decode 与 $1.29\times$ 端到端：** 对比的是 2024 年前后的 vLLM 0.2.7 kernel 与 FlashAttention 2.5.9 / FlashInfer 0.4.0，A100 80 GB，Yi/Llama-3 的 GQA 形状。decode 持平「最好的分页 FlashAttention」，收益相对的是较慢的 vLLM 分页 kernel。端到端收益随 prefill 占比上升。2 MB 页在延迟受限的在线 batch 下够用；吞吐导向时 64 KB 页才提高最大 batch。
- **DiffKV 的精度与 $5.4\times$：** 阈值在 MATH 训练集上按模型选定（Llama3 的 $\alpha_h=1$，Qwen2.5-32B 与 QwQ 的 $\alpha_h=3$，Qwen2.5-7B 关闭低精档）。Qwen2.5-7B 的 GQA 为每组 7 个 query，K4 会使 GSM8K 与 HumanEval+ 精度接近于零；Llama3-8B 的组大小为 4，K4V2 仍保留 FP16 精度的 65% 以上。$5.4\times$ 依赖思考模型的长生成把 KV 打满，从而把 batch 从 2.7 提到 15.9。长思维链上压缩误差会沿生成步累积；长 prompt、短生成里大部分 token 来自原文，同样的压缩更不容易暴露。Quest 的加速不能当成容量节省。

---

## 6. 逐篇要点

每篇只保留与第 1–5 节直接相关的机制、实验范围和失效条件。

### D01 Orca

迭代级选择活跃请求，KV 槽按 `max_tokens` 预留在 GPU。最大序列 2048，参数与激活为 FP16。$36.9\times$ 是相对 FasterTransformer 的请求率，来自 selective batching 与控制面分离；低负载、batch 凑不满时，延迟退回引擎本身。175B 上引擎本身最多快 47%。

### D02 FlexGen

离线把权重、激活和 KV 拆到 GPU、CPU 内存和 NVMe。主实验把等长 prompt 一次跑完，每条生成 32 token。prompt 512 的时间分解里，decode 计算约占该阶段的 13%，KV cache 读占 7046/11315 秒。高倍数依赖 16 GB T4、约 2 GB/s 磁盘读，以及允许数千秒延迟。

### D03 vLLM

物理 KV 块在 GPU，默认块大小 16；可用容量先扣除权重。表 1：OPT-13B 在 40 GB 上 KV 为 12 GB。PagedAttention kernel 比 FasterTransformer 慢 20%–26%，端到端请求率仍可升高，因为同时驻留的请求更多。小块换出会把 PCIe 打成大量小传输；重算延迟在其微基准里不超过换出的 20%。

### D04 Splitwise

KV 在 prompt 机逐层生成后无损送到 token 机。残余传输约 A100 8 ms、H100 5 ms；无 batch 的编程痕迹上约占端到端 0.8%。token 阶段 batch 64 时 TBT 只变成约 $2\times$，功耗上限降到 350 W 时延迟几乎不变。这些比例依赖同构 GPU 和 200–400 Gbps 互联，且数值格式没有改变。

### D05 ALISA

一批等长序列在单机 GPU–CPU 上跑完。权重和激活留在 GPU；KV 超出 GPU 后按 token 放到 CPU，再长则删掉最老的一段并重算。系统实验为 80% 稀疏、INT8，CPU–GPU 带宽 20 GB/s。小 batch 时 vLLM 的吞吐更高。

### D06 DistServe

prefill 与 decode 分实例，交接的是 KV 和第一个 token。OPT-66B 上 512 token 的 KV 约 1.13 GB。实验集群跨节点只有 25 Gbps，因此同一流水级改走 NVLink，传输不到总延迟的 0.1%。报告的倍数是达标请求率。往 decode batch 加入一条 prefill 会拉长 TPOT。

### D07 InfiniGen

完整 KV 在 CPU。GPU 上按计数器保留一个池，每层最多送上 20% 的 KV。加速随序列变长升到 $5.28\times$（OPT-13B，batch 8）；30% 权重也在 CPU 时，相对 FlexGen 只剩 $1.34\times$。重要 token 数随序列次线性增加，固定比例的永久驱逐在超出预算后偏离全量注意力。

### D08 CacheGen

下游收到的是分块码流，不是 FP16/BF16 张量。码流改善可复用长上下文的 TTFT；KV 进入 GPU 之后的 decode 与原来相同。上下文短于约 1K，或带宽高于约 20 Gbps 时，这条路径的收益不成立。

### D09 vAttention

物理页在 GPU，虚拟地址按最大上下文一次留出。decode 上，分页与非分页 FlashAttention kernel 的延迟几乎一样，访存停顿盖住了分页指令。kernel 延迟差到约 $2.8\times$ 时，decode 吞吐约到 $2\times$，端到端 makespan 约 $1.22\times$–$1.29\times$。块表准备修复后仍可占到 decode 迭代的 10%。

### D10 DiffKV

页在 GPU。prompt 步先按高精分配，压缩后回收；K 与 V 的页内排布不同，并带 scale、zero-point、分数和位置。K8V8 的 kernel 为 $1.7\times$，8 条序列的整批延迟为 $1.4\times$–$1.6\times$，吞吐可达 $5.4\times$，后者依赖 batch 从 2.7 增到 15.9。管理放在 CPU 上时，生成步的管理时间会超过模型执行时间。Qwen2.5-7B 上 K4 会使 GSM8K 与 HumanEval+ 精度接近于零。

---

## 7. 其他观点

这些观察与上面五个问题相关，但不是那五个问题的直接答案。

1. **调度与「活跃 token」不是一回事。** Orca 的迭代级 FCFS 允许晚到、生成更短的请求先返回，同时保证早到请求已经执行的迭代数不少于晚到请求。Splitwise 进一步区分三种 batch：请求级、prompt 与 token 分开的连续 batch、以及二者混在同一步的 mixed batch。混步会拉长与 prompt 同批的 token 的 TBT。它还把 prompt 的活跃 token 数记为整段 prompt，把 decode 记为每请求 1。用「batch = 32」描述 decode 步，若不说明这是请求数还是 token 数，会把计算量算错一个数量级。

2. **Attention 没有权重，因此合批的收益与 Linear/FFN 不同。** Orca 明确把 Attention 从合批中拿出去，并在微基准里看到效率损失很小。这与「decode 合批是为了分摊权重读取」一致。本课题 $B=1$ 时没有这项分摊；多请求只在 $B=4$ 的带宽竞争里出现。

3. **分页是地址管理，不是语义稀疏。** vLLM 的块表、引用计数和 copy-on-write 让 prompt、beam 和系统前缀共享物理块，未选中的历史仍然占着块。vAttention 说明分页 kernel 的额外指令在 prefill 里可见、在 decode 里常被访存停顿盖住，并且块表准备本身可占到迭代时间的约 10%。DiffKV 把固定页格式称为内存浪费：若按最高精度 K8V4 留槽，K4V2 的 token 会浪费一半页。它的双向页表让高精页从左增长、低精页从右增长，避免两套页表。

4. **重要 token 集合沿生成步变化，固定比例预算会越来越偏。** InfiniGen 用全量注意力与 H2O 的余弦相似度说明：预算内二者接近，超过预算后 H2O 变差，因为本步不重要的 token 下一步可能变重要。层 0 的注意力更平，达到 0.9 累积权重所需的 token 数分布很宽；较深的层则很尖。重要 token 数随序列次线性增加。因此「每步重选、但历史仍保留」与「超过预算就永久删除」是两条不同的容量账。ALISA 把局部静态 token 留在 GPU、更早的动态 token 放到 CPU，也是同一事实的系统版本。

5. **Key 与 Value 的精度不能对调，GQA 组越大越明显。** DiffKV 在 Llama3 与 Qwen2.5 上比较 K8V4/K4V2 和它们的镜像。镜像配置，尤其是 4-bit key，精度崩溃；Qwen2.5-7B 上 K4V8 接近零。K4V1 也接近零，作者把 2 bit 看作 value 的下界。高 GQA 压缩比会提高对 key 精度的敏感度。长思维链比长 prompt 更会把压缩误差累积进后续 token。

6. **压缩字节只有进入融合 kernel 和更大 batch 之后才变成吞吐。** DiffKV 用 Atom/KIVI 的反例说明：显存省了、batch 上去了，吞吐仍可能停在框架和逐次反量化上。Quest 省的是计算 token 数，不是驻留容量。vLLM 早已说明：20%–26% 的 kernel 变慢可以被「多驻留请求」盖过，反过来，只优化 kernel 也可能被内存碎片抵消。

7. **预取与重算的分界由链路时间决定。** FlexGen 在 $s\ge 512$ 且 KV 在 CPU 时，把 attention 分数改到 CPU 上算，搬运量降为 $1/s$。ALISA 在序列越过 $p_2$ 后删除 CPU 上最老的 KV 并重算，因为再搬一次比重算更贵；这个拐点随 batch 和模型变。vLLM 在小块上选择重算、在大块上选择换出。InfiniGen 则坚持全量保留、只预取，并用离线 SVD skew 让少量部分权重能够代表下一层 query/key。

8. **阶段分离的收益是 SLO 可行域，不是单引擎延迟。** DistServe 与 Splitwise 都表明，把 prefill 和 decode 放在同一步会让 TTFT 与 TPOT 互相让步。DistServe 用 M/D/1 说明：到达率低时 prefill 更适合 intra-op（执行时间短），到达率高时更适合 inter-op（排队时间短）；系数 $K$ 随输入长度、模型、通信带宽和放置变化。token 阶段还可以承受大约 50% 的功耗上限，prompt 阶段不行。这些是集群供电机与异构机器的结论。

9. **KV 码流是网络对象。** CacheGen 观察到 KV 沿 token 有局部性、沿层对误差敏感、沿 channel 分布不同，因此用锚点差分和分层量化做码流，并在传输过程中改压缩级或退回原文。它把 H2O/LLMLingua 之后的浮点 KV 再压 $4.7\times$–$5.5\times$，说明 token 删除与数值编码可以叠用，但 H2O 需要 prompt 的 query，离线阶段本来没有。论文把「streaming」用于网络发送，与滑动窗口 attention 或片上流水不是一回事。

---

## 8. 对本课题的直接含义

下面只写可以沿用的判断，以及这些论文没有在本课题工作点上测过的外推。

### 可以写入课题模型的

- 活跃请求数和每步 token 数会变；在线痕迹里 decode 步经常只有很少的活跃 token。$B=1$ 是这种区间的一个点，不是服务系统的典型饱和点。
- KV 可用容量 = 设备容量 − 权重与运行时保留 − 碎片与预留。vLLM 的 12 GB / 40 GB、21 GB / 160 GB、264 GB / 640 GB 是这种减法的实例。未读历史仍然占容量。
- prefill 到 decode 若只是同构 KV 的块拷贝，高带宽上可以是数毫秒到数十毫秒；若改布局、改精度或改成码流，要另计融合写回、反量化、页统计和元数据。DiffKV 说明管理放错位置后，开销可以超过整个生成步。
- 完整 TPOT 上，attention kernel 的倍数会被同一步的权重搬运和其他算子缩小。目前最干净的对照是 vAttention：kernel 约 $2.8\times$，decode 吞吐约 $2\times$，端到端约 $1.2\times$。D 类没有给出可借用的「attention 占 TPOT 的百分之几」。

### 不能写成已经证实的

- 任一 D 类加速倍数在 $B=1$、$S=32\mathrm{K}$–$128\mathrm{K}$、GQA、INT4/INT8 页上仍然成立。
- NVLink 或 InfiniBand 上的传输比例就是本课题片外通道上的传输比例。
- 分页、预取或码流本身降低了每步必须准备的 KV 字节。分页改变放置，预取改变本步搬运，码流改变进入 GPU 之前的体积；三者与片上引擎的每步访问预算不是同一个量。
