# 长上下文 Decode：精度感知 KV Cache 流式 Attention 架构与映射

文献检索与核对目录。分类及书目信息更新至 **2026-09-21**。对应 BibTeX 见 [literature-survey.bib](literature-survey.bib)，引用键与表内编号一致（`A01`--`K08`）。

## 收集范围与归类规则

以 [项目范围界限](project-scope.md) 为依据，收集 **2022 年以来**与自回归 decode 的 KV Cache、Attention 执行及数字架构映射直接相关的工作，检索截止日期见文首；必要的早期基础文献单独标明。本文是经筛选的研究阅读目录，不声称穷尽全部相关论文。

- **A–D 为四个研究主类**：按主要贡献所改变的对象归类，每篇只保留一个完整书目条目。跨机制工作用“贡献与本项目关系”中的标签和说明表达，不重复列入多个主题表。
- **E 为基础与评测附录**：基础模型、性能分析、质量基准及明确标注的非论文资源，不与方法类别竞争归属。
- **K 单独列预印本与技术报告**：在 K 表注明其主题归属，A–E 不再复制其书目。检索到 arXiv 版本不代表仍未发表；查到正式出处后移入对应主题。
- **正式主会、期刊、研讨会、预印本、技术报告和社区资源分别标明**。有正式研讨会出处的 GEAR 归 A，但不得写作 NeurIPS 主会论文。
- **核心与边界对照分开表述**：GQA 等结构文献用于解释既定模型输入；GPU、TPU、主机卸载和服务系统用于机制或系统边界对照，不改变本项目数字 Attention/KV 引擎的交付范围。
- 不收录训练本身、非 KV 驻留架构、投机解码、视频/多模态专用、PIM/CIM 或纯权重/激活压缩作为本清单的研究主线。更新动作简记在[月度更新记录表](literature-survey-monthly.md)。
- **各表按正式发表时间升序排列**：以出处栏的会议/期刊时间为准；同年按会期或期刊月份，同场次保持既有相对顺序。K 表按 arXiv 标识的年月。新增条目插入对应位置后重编号。

- **名称便于记忆**：文献列采用“简称／方法名 · 一句话记忆点”，不列作者。完整题名保留在首个来源链接的悬停提示中；无通用简称时用描述性短名，正式引用以完整书目信息为准。

| 主类 | 主要决策对象 | 与其他类的边界 |
|---|---|---|
| A. KV 表示与精度 | KV 的张量表示、共享/低秩结构及位宽 | 不以淘汰历史 token 或决定本步参与集合为主要贡献 |
| B. KV 保留与访问策略 | 保留哪些历史 KV，每步读取或计算哪些 KV | 区分“已删除”与“仍驻留但本步不读”；未读取数据仍占容量 |
| C. Attention 执行架构与映射 | kernel、数据通路、布局、tile/bank 绑定及流水执行 | 量化/稀疏可作为联合机制；平台不是独立一级分类 |
| D. 推理系统与资源管理 | 请求组织、缓存分配、设备/存储层驻留和传输 | 主要比较服务与资源组织，不重复收录单个 Attention 执行引擎 |

当前有 **56 篇已有正式出处的文献**（A–D 共 48 篇，E 中 8 篇；含 1 篇研讨会论文）、**8 篇预印本/技术报告**及 **2 项非论文资源**。K 表含 7 篇预印本和 1 篇模型技术报告。

**核验口径：** 对保留及新增条目的题名、作者署名、出处/身份与主题关联进行核对；优先使用论文集、出版者、正式论文及作者机构页面，必要时以 Crossref 核对出版元数据。贡献说明是阅读定位，不代表复现论文实验。K 表“未确认正式出处”表示本次核对未找到可确认的正式版本，不是对未发表状态的绝对证明。

**月度维护：** 按[月度检索约定](literature-survey-monthly.md)查阅来源与原定检索式；新文献并入 A–E，预印本/技术报告仅入 K。每次核对 K 表并检查正式题名变更，按正式发表时间插入并刷新数量与编号，更新动作仅在月度文档追加一行表格，不另建平行论文清单。

---

## A. KV 表示与精度

关注“KV 如何表示”。模型结构只作给定输入或扩展对照；量化方法重点比较 K/V、layer、token、head 等精度粒度及相应存储开销。MQA、MLA 原始技术报告与 MiKV 等预印本集中在 K 节。

| 编号 | 简称与记忆点 | 出处与核对来源 | 贡献与本项目关系 |
|---|---|---|---|
| A01 | **GQA** · 分组共享 KV head | EMNLP 2023；[来源](https://aclanthology.org/2023.emnlp-main.298/ "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints") | 【结构输入】在 MQA 与 MHA 之间设置 KV head 共享组，降低每 token 的 KV 表示规模。本项目使用已有 GQA 模型，按 KV head 组组织存储；不复现其 uptraining。 |
| A02 | **KIVI** · K/V 非对称 2-bit 量化 | ICML 2024；[来源](https://proceedings.mlr.press/v235/liu24bz.html "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache") | 【核心：K/V 不对称量化】Key 按 channel、Value 按 token 量化，并提供执行实现；用于确定精度配置、量化分组及 packing 约束。 |
| A03 | **SKVQ** · 近期高精度，历史低精度 | COLM 2024，主会；[会议目录](https://2024.colmweb.org/AcceptedPapers.html "SKVQ: Sliding-window Key and Value Cache Quantization for Large Language Models")；[论文](https://openreview.net/pdf/8a0a7026e749d0b8c2348205ce0f8617e2db2e90.pdf) | 【核心：窗口混合精度】通道重排与分组裁剪量化，近期窗口保留高精度。窗口用于分配精度，不等同于将窗口外 KV 全部丢弃。 |
| A04 | **KVQuant** · Pre-RoPE 量化与异常值处理 | NeurIPS 2024；[来源](https://proceedings.neurips.cc/paper_files/paper/2024/hash/028fcbcf85435d39a40c4d61b42c99a4-Abstract-Conference.html "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization") | 【核心：低比特表示】结合 Pre-RoPE Key、非均匀码本与 dense-and-sparse 表示。分开核算码本、异常值与反量化成本，容量外推不等于长上下文质量验证。 |
| A05 | **ZipCache** · 按 token 重要性分配位宽 | NeurIPS 2024，主会；[论文集](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7e57131fdeb815764434b65162c88895-Abstract-Conference.html "ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification") | 【核心：token 混合精度】用显著度分配 KV 位宽；主要改变表示精度，不按“永久驱逐”归类。细粒度精度标签及处理开销需要单独比较。 |
| A06 | **MiniCache** · 跨层合并 KV | NeurIPS 2024，主会；[论文集](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fd0705710bf01b88a60a3d479ea341d9-Abstract-Conference.html "MiniCache: KV Cache Compression in Depth Dimension for Large Language Models") | 【表示对照：跨层合并】利用相邻层 KV 的相似性合并表示，对差异较大的状态保留独立信息。用于区分层间表示压缩与层间 token 预算分配。 |
| A07 | **GEAR** · 低比特＋低秩/稀疏残差 | NeurIPS ENLSP-IV Workshop 2024，PMLR 262:305–321；[来源](https://proceedings.mlr.press/v262/kang24a.html "GEAR: An Efficient Error Reduction Framework for KV Cache Compression in LLM Inference") | 【研讨会：误差修正】低比特量化结合低秩与稀疏残差修正；主要改变 KV 表示。需核算残差存储和重建成本，不能作为 NeurIPS 主会论文引用。 |
| A08 | **KVTuner** · 逐层搜索 K/V 位宽 | ICML 2025；[来源](https://proceedings.mlr.press/v267/li25dd.html "KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization for Efficient and Nearly Lossless LLM Inference") | 【核心：层间混合精度】离线搜索 layer-wise K/V 精度对，在线使用固定配置。与本项目静态精度配置直接相关，区别于运行时逐 token 精度决策。 |
| A09 | **GTA / GLA** · 面向解码的 Attention 表示 | COLM 2025；开放版本 arXiv:2505.21487；[来源](https://colmweb.org/2025/AcceptedPapers.html "Hardware-Efficient Attention for Fast Decoding") | 【结构输入／扩展】提出 GTA、GLA，比较 Attention 参数化、KV 存储量与算术强度。用于界定既定模型结构对 decode 映射的影响，不新增模型训练任务。 |
| A10 | **TurboQuant** · 旋转量化与残差校正 | ICLR 2026，主会；[论文集](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5c802ef38ab6e366c2ea06eee554c088-Abstract-Conference.html "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate") | 【表示对照：向量量化】随机旋转、标量量化与残差 QJL 校正，分别处理重建误差和内积失真。用于讨论 KV 表示误差与 Attention 误差，不能直接推定 ASIC 收益。 |

---

## B. KV 保留与访问策略

关注“保留哪些、访问哪些”。窗口、重要性驱逐、head 分派与查询选页统一放在本类；保留集合和本步访问集合分别说明，避免把容量压缩与访存减少混为一谈。

| 编号 | 简称与记忆点 | 出处与核对来源 | 贡献与本项目关系 |
|---|---|---|---|
| B01 | **H2O** · 保留高分 token＋近期窗口 | NeurIPS 2023；[来源](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6ceefa7b15572587b78ecfcebb2827f8-Abstract.html "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models") | 【重要性驱逐】保留 recent 与 heavy hitters，动态淘汰其余 KV；作为内容驱逐对照，不要求为其开发核心 RTL。 |
| B02 | **Scissorhands** · 利用重要性持续性驱逐 KV | NeurIPS 2023；[来源](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a452a7c6c463e4ae8fbdc614c6e983e6-Abstract-Conference.html "Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time") | 【重要性驱逐】基于重要性持续性选择保留 token；用于比较分数维护、选择与缓存预算。可与量化组合，但本条主归保留策略。 |
| B03 | **StreamingLLM** · sink＋滑动窗口 | ICLR 2024，主会；[论文集](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5e5fd18f863cbe6d8ae392a93fd271c9-Abstract-Conference.html "Efficient Streaming Language Models with Attention Sinks") | 【窗口保留】保留起始 sink 与近期窗口，支持持续生成；窗口外被丢弃的信息不可再次检索。流式运行长度不等于可访问历史长度。 |
| B04 | **FastGen** · 按 head 自适应选择保留策略 | ICLR 2024；[来源](https://proceedings.iclr.cc/paper_files/paper/2024/hash/639a9a172c044fbb64175b5fad42e9a5-Abstract-Conference.html "Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs") | 【保留策略自适应】按 head 行为选择局部、特殊 token 或全量保留等策略，包含驱逐。先区分保留集合，再比较其访问与更新成本。 |
| B05 | **Quest** · 按 query 选页 | ICML 2024；[来源](https://proceedings.mlr.press/v235/tang24l.html "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference") | 【核心：查询相关选页】用 page Key min/max 与当前 query 估计重要性，再读取 top-k 页。未选中的历史 KV 仍须计入驻留容量，元数据扫描与选择成本不可省略。 |
| B06 | **SnapKV** · 观察窗口选择 prompt KV | NeurIPS 2024；[来源](https://proceedings.nips.cc/paper_files/paper/2024/hash/28ab418242603e0f7323e54185d19bde-Abstract-Conference.html "SnapKV: LLM Knows What You Are Looking for Before Generation") | 【prompt KV 选择】在 prefill 末通过观察窗口压缩 prompt KV；后续生成 KV 仍会追加。不能把固定 prompt 预算当作完整生成期间的恒定缓存。 |
| B07 | **DuoAttention** · 检索 head＋流式 head | ICLR 2025，主会；[论文集](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5c1ddd2e59df46fd2aa85c833b1b36ed-Abstract-Conference.html "DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads") | 【head 路径分派】retrieval heads 保留全上下文，streaming heads 使用固定长度 cache；整体缓存并非常量。用于 head 组分派对照，GQA 下需统一共享 KV 的策略。 |
| B08 | **LServe** · 流式路径＋动态选页 | MLSys 2025；[来源](https://proceedings.mlsys.org/paper_files/paper/2025/hash/cc8c6b9d89f7a898a29f58869b238e46-Abstract-Conference.html "LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention") | 【核心：静态与动态选择联合】统一 streaming heads 与查询相关 KV 页选择，并有量化及 GPU 执行设计；主归访问策略，作为组合机制对照，不能概括为仅有稀疏算法。 |
| B09 | **PyramidKV** · 按层分配 KV 预算 | COLM 2025，主会；[会议目录](https://colmweb.org/2025/AcceptedPapers.html "PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling")；[论文](https://openreview.net/pdf?id=ayi7qezU87) | 【层间预算与选择】按层分配 KV 预算并选择重要 KV；改变保留数量与集合，不是跨层合并张量表示。 |
| B10 | **RetrievalAttention** · 向量索引检索历史 KV | NeurIPS 2025；[来源](https://proceedings.nips.cc/paper_files/paper/2025/hash/4e36d4049fb0fea195a8267c8dcd0824-Abstract-Conference.html "RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval") | 【检索对照】用 attention-aware ANNS 索引从保留的历史 KV 中检索，并涉及 CPU 驻留。用于比较查询选择与索引成本；片上 ANNS 不属于本项目核心实现。 |
| B11 | **Self-Indexing KVCache** · 压缩 Key 兼作索引 | AAAI 2026，主会，40(33):27675–27683；DOI:10.1609/aaai.v40i33.39988；[论文集](https://ojs.aaai.org/index.php/AAAI/article/view/39988 "Self-Indexing KVCache: Predicting Sparse Attention from Compressed Keys") | 【选择与表示联合】利用压缩 Key 表示直接预测稀疏访问，结合 1-bit 向量量化与 CUDA kernel。主归选择机制；是“精度与选页耦合”主张需要检查的已有工作。 |
| B12 | **LouisKV** · 语义边界触发 KV 检索 | ICLR 2026，主会；[论文集](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6b241c515433caae3051266668d808b7-Abstract-Conference.html "LouisKV: Efficient KV Cache Retrieval for Long Input-Output Sequences") | 【检索时机与粒度】利用 decode 时间局部性在语义边界触发检索，并区分输入、输出 KV 的管理。用于检查长输入和长输出下的访问策略，非恒定成本选页证明。 |

---

## C. Attention 执行架构与映射

关注“如何执行”。GPU kernel、FPGA、数字加速器及数据流映射统一归类；硬件与量化/稀疏联合设计按主要贡献留在本类。早期基础和跨平台工作均注明适用边界。

| 编号 | 简称与记忆点 | 出处与核对来源 | 贡献与本项目关系 |
|---|---|---|---|
| C01 | **SpAtten** · token/head 剪枝硬件 | HPCA 2021；[来源](https://hanlab.mit.edu/projects/spatten "SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning") | 【数字架构／早期机制】级联 token/head pruning、渐进量化及专用 top-k 引擎，包含 GPT-2 评测。保留为选取与量化硬件先例，不冒充现代长上下文 GQA 基线。 |
| C02 | **DFX** · 多 FPGA 文本生成 | MICRO 2022，主会；DOI:10.1109/MICRO56248.2022.00051；[出版页面](https://ieeexplore.ieee.org/document/9923883 "DFX: A Low-latency Multi-FPGA Appliance for Accelerating Transformer-based Text Generation")；[开放版本](https://arxiv.org/abs/2209.10797) | 【FPGA／生成推理】多 FPGA 上按文本生成负载组织数据流及 HBM 访问。保留为 decode 映射起点，其 GPT-2 工作负载与本项目长上下文 GQA 不等价。 |
| C03 | **FlashAttention** · 分块融合，减少访存 | NeurIPS 2022；[来源](https://papers.nips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness") | 【GPU／基础机制】通过 tiling 与 online softmax 降低 HBM 往返，作为精确 Attention 的 I/O 与融合执行依据；完整序列收益不直接套用于单 query decode。 |
| C04 | **FlightLLM** · FPGA 推理映射流程 | FPGA 2024，主会；[正式版作者副本](https://dai.sjtu.edu.cn/my_file/pdf/94c37d8a-7f86-4f95-ae72-05a79da5bb61.pdf "FlightLLM: Efficient Large Language Model Inference with a Complete Mapping Flow on FPGAs")；[开放版本](https://arxiv.org/abs/2401.03868) | 【FPGA／映射流程】可配置稀疏 DSP chain、on-chip decode、混合精度与 length-adaptive compilation；用于映射方法和数据流对照，不能据此假设整个长上下文 KV 常驻片上。 |
| C05 | **FlashAttention-2** · 优化并行与工作划分 | ICLR 2024；[来源](https://proceedings.iclr.cc/paper_files/paper/2024/hash/98ed250b203d1ac6b24bbcf263e3d4a7-Abstract-Conference.html "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning") | 【GPU／基础机制】优化 block/warp 工作划分和线程间通信；用于并行分工对照。短 query 的序列切分应另查附录中的 Flash-Decoding 技术说明。 |
| C06 | **FlashDecoding++** · 异步 softmax 解码 | MLSys 2024；[来源](https://proceedings.mlsys.org/paper_files/paper/2024/hash/5321b1dabcd2be188d796c21b733e8c7-Abstract-Conference.html "FlashDecoding++: Faster Large Language Model Inference with Asynchronization, Flat GEMM Optimization, and Heuristics") | 【GPU／decode】异步 softmax、flat GEMM/GEMV 优化与数据流选择；用于比较流水重叠、双缓冲和形状适配。 |
| C07 | **SOFA** · 跨阶段协同分块 | MICRO 2024；开放版本 arXiv:2407.10416；[来源](https://microarch.org/micro57/program/ "SOFA: A Compute-Memory Optimized Sparsity Accelerator via Cross-Stage Coordinated Tiling") | 【数字架构／稀疏执行】跨阶段协同 tiling，协调动态稀疏 Attention 的计算与存储。用于分析选择、执行及中间数据成本，decode 适用范围须按实际工作负载对齐。 |
| C08 | **LAD** · 跨 decode 步复用 KV | HPCA 2025；[来源](https://hpca-conf.org/2025/main-program/ "LAD: Efficient Accelerator for Generative Inference of LLM with Locality Aware Decoding") | 【数字架构／时间局部性】利用相邻 decode 步的局部性减少重复 KV 访问；与每步重新选页及规则流策略比较，需对齐复用范围和误差条件。 |
| C09 | **QServe** · W4A8KV4 联合执行 | MLSys 2025；[来源](https://proceedings.mlsys.org/paper_files/paper/2025/hash/fbe2b2f74a2ece8070d8fb073717bda6-Abstract-Conference.html "QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving") | 【GPU／量化与执行联合】W4A8KV4、SmoothAttention 与低开销 kernel 协同；本项目重点比较 KV4 的布局和反量化执行，不扩展权重/FFN 算法贡献。 |
| C10 | **FlashInfer** · 可定制 Attention 引擎 | MLSys 2025，主会；[论文集](https://proceedings.mlsys.org/paper_files/paper/2025/hash/dbf02b21d77409a2db30e56866a8ab3a-Abstract-Conference.html "FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving") | 【GPU／Attention 引擎】用可组合 KV 格式、block-sparse 表示、JIT 模板和负载均衡执行应对异构缓存及动态请求。主归算子引擎，不因用于 serving 就移入服务管理。 |
| C11 | **Oaken** · 在线/离线 KV 量化硬件 | ISCA 2025，主会；DOI:10.1145/3695053.3731019；[正式论文](https://jongse-park.github.io/files/paper/2025-isca-oaken.pdf "Oaken: Fast and Efficient LLM Serving with Online-Offline Hybrid KV Cache Quantization") | 【数字架构／KV 量化硬件】在线/离线混合量化、专用量化/反量化单元及内存管理协同；是精度感知 KV 数据通路的重要硬件前作。 |
| C12 | **MLA 硬件分析** · 投影复用与重算取舍 | Electronics Letters 61(1):e70504，2025；DOI:10.1049/ell2.70504；开放版本 arXiv:2506.02523；[来源](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ell2.70504 "Hardware-Centric Analysis of DeepSeek’s Multi-Head Latent Attention") | 【MLA／映射分析】比较不同 MLA 执行顺序及潜在投影的复用、重算与能量带宽取舍。主归映射，MLA 仅作为本项目有条件扩展。 |
| C13 | **BitDecoding** · 低比特 KV 驱动 Tensor Core | HPCA 2026，主会，1–13；DOI:10.1109/HPCA68181.2026.11408481；[出版页面](https://ieeexplore.ieee.org/document/11408481/ "BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache")；[作者实现](https://github.com/OpenBitSys/BitDecoding) | 【GPU／低比特 decode】优化布局、query 变换与反量化流水，利用 Tensor Cores 执行低比特 KV Attention；用于分析压缩收益转为实际吞吐的条件。 |
| C14 | **CD-LLM** · 多 FPGA 批量解码 | ACM TRETS 19(1)，Article 8:1–25，2026；DOI:10.1145/3771288；[来源](https://research.cuhk.edu.hk/en/publications/cd-llm-a-heterogeneous-multi-fpga-system-for-batched-decoding-of-/ "CD-LLM: A Heterogeneous Multi-FPGA System for Batched Decoding of 70B+ LLMs Using a Compute-Dedicated Architecture") | 【多 FPGA／batched decode】混合精度与 memory-aligned packing，协调计算和存储带宽。用于有效位宽与传输对齐对照；与一页的 FMC-LLM 不能混用题名或实验数据。 |
| C15 | **AccLLM** · 稀疏＋低比特 FPGA | IEEE TVLSI 34(4):1217–1227，2026；DOI:10.1109/TVLSI.2026.3658524；开放版本 arXiv:2505.03745；[来源](https://api.crossref.org/works/10.1109/TVLSI.2026.3658524 "AccLLM: Accelerating Long-Context LLM Inference Via Algorithm-Hardware Co-Design") | 【FPGA／联合设计】剪枝、Λ 形 Attention 与 W2A8KV4 的联合实现；是算法与硬件协同的直接前作，不能把组合这些机制本身当作新颖性。 |
| C16 | **SnapStream** · 长序列数据流解码 | ISC High Performance 2026 Research Paper Proceedings，1–14；DOI:10.23919/ISC.2026.11520481；开放版本 arXiv:2511.03092；[来源](https://api.crossref.org/works/10.23919/ISC.2026.11520481 "SnapStream: Efficient Long Sequence Decoding on Dataflow Accelerators") | 【数据流加速器／联合实现】结合 SnapKV、StreamingLLM、静态图及 continuous batching，研究长序列 decode 的部署。与算法、架构联合主张直接相关。 |

---

## D. 推理系统与资源管理

关注“请求和 KV 如何在系统资源间组织”。本类均为系统或接口对照；主机卸载、集群调度和完整 serving 框架不因此成为项目必交付。

| 编号 | 简称与记忆点 | 出处与核对来源 | 贡献与本项目关系 |
|---|---|---|---|
| D01 | **Orca** · 逐迭代批处理调度 | OSDI 2022；[来源](https://www.usenix.org/conference/osdi22/presentation/yu "Orca: A Distributed Serving System for Transformer-Based Generative Models") | 【服务边界：批处理】iteration-level scheduling 与 selective batching；说明活跃请求变化如何影响 Attention 利用率，不新增服务调度器交付。 |
| D02 | **FlexGen** · GPU/CPU/磁盘协同卸载 | ICML 2023；[来源](https://proceedings.mlr.press/v202/sheng23a.html "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU") | 【服务边界：卸载】联合 GPU、CPU/磁盘的张量放置与压缩，以吞吐为主要目标。用于层次存储对照，不要求实现主机侧卸载方案。 |
| D03 | **PagedAttention（vLLM）** · KV 分页管理 | SOSP 2023；[来源](https://sigops.org/s/conferences/sosp/2023/toc.html "Efficient Memory Management for Large Language Model Serving with PagedAttention") | 【GPU／分页管理】KV 动态分配与共享降低碎片；比较逻辑页、物理分配和 Attention kernel 接口，不把分页本身当作语义稀疏。 |
| D04 | **Splitwise** · 按推理阶段分配异构机器 | ISCA 2024；[来源](https://www.iscaconf.org/isca2024/program/ "Splitwise: Efficient Generative LLM Inference Using Phase Splitting") | 【服务边界：异构分配】按推理阶段组织异构机器和集群资源；主归系统组织，不因发表在 ISCA 就重复列入芯片架构类。 |
| D05 | **ALISA** · 稀疏感知 KV 调度 | ISCA 2024，主会；[会议目录](https://www.iscaconf.org/isca2024/program/ "ALISA: Accelerating Large Language Model Inference via Sparsity-Aware KV Caching")；[正式论文](https://www.unarylab.com/_files/ugd/032fba_8d2a891a29194980a087f56bbd3b60c0.pdf) | 【服务边界：稀疏缓存管理】Sparse Window Attention 与 token 级动态调度结合，协调缓存、卸载及重算。主归系统管理，保留稀疏机制说明。 |
| D06 | **DistServe** · 分离 prefill 与 decode | OSDI 2024；[来源](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving") | 【服务边界：阶段分离】拆分 prefill/decode 并规划资源与并行策略；说明 decode 引擎的系统接口，不能将 goodput 与单引擎延迟直接等同。 |
| D07 | **InfiniGen** · 预取下一层重要 KV | OSDI 2024；[来源](https://www.usenix.org/conference/osdi24/presentation/lee "InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management") | 【服务边界：预取】推测下一层重要 KV，减少从 host 内存获取的数据；主归跨层次驻留与预取，不将 CPU 卸载策略纳入核心 RTL。 |
| D08 | **CacheGen** · 压缩 KV 码流传输 | SIGCOMM 2024；[来源](https://cs.stanford.edu/~keithw/sigcomm2024/sigcomm24-final1571-acmpaginated.pdf "CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving") | 【服务边界：KV 传输】将 KV 压缩为码流并自适应网络带宽，降低上下文加载延迟。streaming 指网络传输，不是窗口 Attention 或芯片流水。 |
| D09 | **vAttention** · 虚拟连续，物理按需分配 | ASPLOS 2025，主会；DOI:10.1145/3669940.3707256；[作者机构页面](https://www.microsoft.com/en-us/research/publication/vattention-dynamic-memory-management-for-serving-llms-without-pagedattention/ "vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention")；[正式论文](https://apanwariisc.github.io/publications/asplos-2025-vattention/vattention-asplos25.pdf) | 【GPU／虚拟内存】保留 KV 虚拟地址连续性，通过物理内存按需分配管理容量；用于区分页表接口与 kernel 布局选择。 |
| D10 | **DiffKV** · 差异化 KV 存储与并行整理 | SOSP 2025，主会，431–445；DOI:10.1145/3731569.3764810；[正式论文](https://ipads.se.sjtu.edu.cn/zh/publications/sosp25-zhang.pdf "DiffKV: Differentiated Memory Management for Large Language Models with Parallel KV Compaction")；[作者机构书目](https://research.cuhk.edu.hk/en/publications/diffkv-differentiated-memory-management-for-large-language-models/) | 【GPU／异构精度与内存管理】区分 K/V、token 重要性和 head 稀疏性，用并行 KV compaction 管理不规则内存。早期 arXiv:2412.03131 题名为 LeanKV，同一工作只登记正式版。 |

---

## E. 基础与评测附录

本节不构成第五条优化主线。Online Softmax 的预印本书目在 K 节；此处只保留正式基础文献、评测论文，以及单独标注的社区工具/技术说明。

### 基础与性能分析

| 编号 | 简称与记忆点 | 出处与核对来源 | 用途与边界 |
|---|---|---|---|
| E01 | **Roofline** · 计算与带宽性能上界 | CACM 52(4):65–76，2009；DOI:10.1145/1498765.1498785；[来源](https://aiichironakano.github.io/cs596/Williams-Roofline-CACM09.pdf "Roofline: An Insightful Visual Performance Model for Multicore Architectures") | 【基础】运算强度与计算/带宽上限；用于分析 Attention 的访存约束及性能上界。 |
| E02 | **Transformer** · 自注意力基础 | NeurIPS 2017；[来源](https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html "Attention Is All You Need") | 【基础】Attention 记号与运算定义；作为背景，不列为近年 decode 优化进展。 |
| E03 | **Transformer 推理扩展** · 并行与性能建模 | MLSys 2023 / arXiv:2211.05102；[来源](https://proceedings.mlsys.org/paper_files/paper/2023/hash/c4be71ab8d24cdfb45e3d06dbfca2780-Abstract-mlsys2023.html "Efficiently Scaling Transformer Inference") | 【性能分析】生成式推理的延迟、MFU 与分区取舍；用于工作负载及硬件瓶颈建模，不能把其上下文配置直接当成本项目配置。 |
| E04 | **RoFormer / RoPE** · 旋转位置编码 | Neurocomputing 568:127063，2024（早期 arXiv:2104.09864）；[来源](https://www.sciencedirect.com/science/article/pii/S0925231223011864 "RoFormer: Enhanced Transformer with Rotary Position Embedding") | 【基础】RoPE 位置编码及其与 Key 量化顺序的关系；不将 RoFormer 模型训练纳入研究任务。 |

### 长上下文评测

| 编号 | 简称与记忆点 | 出处与核对来源 | 用途与边界 |
|---|---|---|---|
| E05 | **LongBench** · 双语长上下文多任务 | ACL 2024；[来源](https://aclanthology.org/2024.acl-long.172/ "LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding") | 【质量基准】双语多任务长上下文理解；登记实际长度、任务子集与评分，不能仅凭基准名称声称覆盖全部长度档。 |
| E06 | **InfiniteBench（∞Bench）** · 超过 100K 的任务评测 | ACL 2024（长文）；DOI:10.18653/v1/2024.acl-long.814；[来源](https://aclanthology.org/2024.acl-long.814/ "∞Bench: Extending Long Context Evaluation Beyond 100K Tokens") | 【质量基准】补充超过 100K token 的长上下文任务；与核心模型实际有效上下文窗口共同确认可测范围。 |
| E07 | **RULER** · 测试有效上下文长度 | COLM 2024；[来源](https://openreview.net/pdf?id=kIoBbc76Sy "RULER: What’s the Real Context Size of Your Long-Context Language Models?") | 【质量基准】可控长度、任务复杂度、检索与聚合等测试；用于冻结长上下文质量约束。 |
| E08 | **SCBench** · 围绕 KV 生命周期评测 | ICLR 2025，主会；[论文集](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a540b17fb2295c736d5afd6c507acf66-Abstract-Conference.html "SCBench: A KV Cache-Centric Analysis of Long-Context Methods") | 【质量与使用场景】从 KV 生成、压缩、检索、加载及共享上下文角度评测长上下文方法。补充多轮与缓存复用下的失效情形，不新增本项目服务框架交付。 |

### 非论文资源

| 编号 | 简称与记忆点 | 出处与核对来源 | 用途与边界 |
|---|---|---|---|
| E09 | **Flash-Decoding** · 短 query 沿 KV 序列并行 | 作者技术说明，PyTorch Blog，2023；[原文](https://pytorch.org/blog/flash-decoding/ "Flash-Decoding for long-context inference")；核对日期 2026-09-21 | 【技术说明，非会议论文】短 query 沿 KV 序列并行并合并局部结果；与 FlashAttention-2、FlashDecoding++ 分开引用。 |
| E10 | **Needle-in-a-Haystack** · 长上下文检索探针 | 始于 2023；[原始项目及实现](https://github.com/gkamradt/needle-in-a-haystack "Needle In A Haystack -- Pressure Testing LLMs")；访问日期 2026-09-21 | 【社区工具，非论文】测试不同长度与位置的信息检索；记录实现版本和评分方式，不能替代多任务长上下文质量验证。 |

---

## K. 预印本与技术报告

本节独立于正式发表文献。主题归属只用于检索，不在 A–E 复制条目。已找到正式出处的新收录工作（如 SKVQ、TurboQuant、Self-Indexing KVCache、DiffKV）直接进入主题表，不作为预印本重复登记。

| 编号 | 简称与记忆点 | 标识与版本 | 主题归属 | 状态与核对日期 | 贡献与本项目关系 |
|---|---|---|---|---|---|
| K01 | **Online Softmax** · 在线归一化 | arXiv:1805.02867（2018）；[来源](https://arxiv.org/abs/1805.02867 "Online Normalizer Calculation for Softmax") | E：基础；关联 C | 预印本；未确认正式出处（2026-09-21） | 在线 softmax 归一化的基础算法，为分块、融合执行提供依据；不是近年进展。 |
| K02 | **MQA** · 所有 query head 共享一组 KV | arXiv:1911.02150（2019）；[来源](https://arxiv.org/abs/1911.02150 "Fast Transformer Decoding: One Write-Head is All You Need") | A：KV 表示与精度 | 预印本；未确认正式出处（2026-09-21） | 多 query head 共享 KV，是 MQA 的结构起点；只在本节保留书目。 |
| K03 | **MiKV** · 重要 KV 高精度，其余低精度 | arXiv:2402.18096（2024）；[来源](https://arxiv.org/abs/2402.18096 "No Token Left Behind: Reliable KV Cache Compression via Importance-Aware Mixed Precision Quantization") | A：KV 表示与精度 | 预印本；未确认正式出处（2026-09-21） | 重要 KV 用较高精度，其余信息以低精度保留；用于区分重要性混合精度与永久删除 token。 |
| K04 | **DeepSeek-V2 / MLA** · 低秩压缩 KV | arXiv:2405.04434（2024）；[来源](https://arxiv.org/abs/2405.04434 "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model") | A：KV 表示与精度 | 模型技术报告；未确认正式出处（2026-09-21） | MLA 低秩联合压缩 KV 与 weight absorption；本项目只使用 Attention 结构与映射信息，不扩展 MoE 专家调度。 |
| K05 | **Ragged Paged Attention** · TPU 不等长分页 Attention | arXiv:2604.15464（2026）；[来源](https://arxiv.org/abs/2604.15464 "Ragged Paged Attention: A High-Performance and Flexible LLM Inference Kernel for TPU") | C：执行架构与映射 | 预印本；未确认正式出处（2026-09-21） | TPU 上面向不等长分页 KV 的 tiling、KV 更新与 Attention 融合流水，以及分工作负载编译；作为页映射和流水实现对照。 |
| K06 | **RateQuant** · 率失真指导位宽分配 | arXiv:2605.06675，v2（2026）；[来源](https://arxiv.org/abs/2605.06675 "RateQuant: Optimal Mixed-Precision KV Cache Quantization via Rate-Distortion Theory") | A：KV 表示与精度 | 预印本；未确认正式出处（2026-09-21） | 校准各量化器的失真模型，用率失真分配 head 级位宽；用于精度预算对照。题名中的 optimal 受其模型假设限制，不能视为硬件联合搜索全局最优。 |
| K07 | **AATC** · 变换编码与精度分配 | arXiv:2608.14191（2026）；[来源](https://arxiv.org/abs/2608.14191 "KV Cache Compression Through the Lens of Transform Coding") | A：KV 表示与精度 | 预印本；未确认正式出处（2026-09-21） | 在量化噪声模型下分析 Attention 失真，并据校准数据分配位宽；用于比较表示误差与任务质量约束，不将白噪声假设当作普遍事实。 |
| K08 | **HPC-Ops Top-K** · 采样辅助的精确 top-k | arXiv:2609.08450（2026）；[来源](https://arxiv.org/abs/2609.08450 "Sample-Guided Exact Top-K Selection for Long-Context Sparse Attention") | C：执行架构与映射 | 预印本；未确认正式出处（2026-09-21） | 优化已有分数行的精确 top-k 实现，采样后仍需全行核验及必要回退。精确的是选择算子，不是完整 Attention；评分生成与扫描成本不能忽略。 |
