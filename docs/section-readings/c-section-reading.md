# C 类精读：片外存储、SRAM、反量化、计算与归约

整理日期：2026-09-22。对象是文献清单 **C01–C16**。书目与归类以 [文献清单](../literature-survey-reading-list.md) 为准，本文不另建清单。课题对照口径见 [项目范围](../project-scope.md)。

第 1–5 节回答本类的五个问题。第 6 节是逐篇要点。第 7 节记录不改变上述回答的其他观点。第 8 节分成两段：可以写入课题模型的判断，以及这些论文尚未覆盖、不能写成已经证实的部分。

数字取自各篇开放全文的方法、实验与附录。原文没有的测量写「未报告」。效率倍数只作该论文自己的基线对照，不换算成课题口径，也不直接写成 Attention 引擎的加速。

跨 GPU、FPGA 与 ASIC 的整机加速只说明该论文的评测口径。本课题的对照工作点是长上下文 GQA decode、KV 为 INT4/INT8、规则流式加上有界页检索、batch 以 1 为主。C 类里多数加速发生在别的工作点上：全量稠密注意力、大 batch 线性层、权重压缩，或 GPU Tensor Core。

## 阅读边界

阅读范围、版本和不能外推的部分记在这里。第 1–5 节的数字都受这些限制。

- 精读使用各篇开放全文：C02、C03、C04、C05、C06、C07、C09、C12、C13、C15、C16 为 arXiv PDF；C01 为 arXiv:2012.09852；C11 为作者公开的 ISCA 2025 PDF；C14 为与 ACM TRETS 正式版对应的作者 PDF。开放版本与会议或期刊排版可能有字句差别，数字以所读 PDF 正文为准。
- **C08 LAD 没有可核的公开全文。** 下文不描述其片上缓冲、反量化或流水。摘要只说明：作者认为若干 decode 步内部分位置的 attention score 数值稳定，因此可以降低这些位置的 KV 读取频率。摘要中的加速比、能量比和 ROUGE 未对照正文，不纳入通路结论。
- 凡给出倍数、带宽、位宽或 batch，都带论文自己的模型、长度和平台。

---

## 1. 数据实际经过哪些级？

把 Attention 的一次 decode 步写成固定的五级，再看每篇在哪一级动手：

$$
\text{片外 KV/权重}
\rightarrow \text{片上缓冲}
\rightarrow \text{反量化或解包}
\rightarrow QK^\top,\ \mathrm{softmax},\ AV
\rightarrow \text{沿序列或跨块归约，并写回新 KV}
$$

C 类没有一篇同时具备「低比特 KV + 查询相关选页 + 片上 bank 调度 + 在线 softmax」。它们各自改其中一截。

### 1.1 精确注意力：不反量化，用分块把 $N\times N$ 矩阵留在片上

**C03 FlashAttention、C05 FlashAttention-2** 的片外对象是完整的 $Q,K,V,O$。片上只驻留当前行块 $Q_i$、列块 $K_j,V_j$，以及在线统计 $m,\ell$ 和未最终缩放的输出。$S$ 和 $P$ 不写回片外。Softmax 在块内更新最大值和归一化项，循环结束后才做一次缩放。归约就是这条在线统计，数学上等价于整行 softmax。没有 KV 量化，也没有单独的 decode GEMV 数据通路；训练和长序列 prefill 是主要场景。C05 把并行从「块内切 $K$」改成「沿序列切 $Q$ 的行块」，让长序列、小 batch 也能占满 GPU，并推迟对输出的缩放，减少非矩阵乘。

**C06 FlashDecoding++** 开始显式区分 prefill GEMM 和 decode 的 flat GEMM/GEMV，KV cache 在 decode 时追加。Softmax 的局部块不再各自取 max 再同步更新，而是用一个统一上界 $\phi$ 让各段异步累加分子和分母，最后再除。分数超出 $\phi$ 所假定的范围时，整行退回同步 softmax 并重算。没有反量化。

**C10 FlashInfer** 把 KV 组织成逻辑页 / 物理页，再看成块稀疏矩阵。页从片外拷进 shared memory 后做稠密矩阵乘。在线 softmax 的输出是局部状态 $[O, \mathrm{LSE}]$，跨 chunk 用可结合算子归约。评测是 FP16，无反量化。共享前缀只改索引，不搬 KV 本体。

**C02 DFX** 把权重和 KV tile 放在 HBM，输入输出 token 放在 DDR，全程 FP16，无反量化。片上有操作数双缓冲和 KV buffer。生成阶段是矩阵–向量：每步追加一行 KV。Softmax 由向量指令拼出来（先归约最大值，再减最大值后做 exp）。多 FPGA 时每层自注意力和 FFN 各做跨卡同步，结果在环网上拼回完整向量。Value 在写入时按 tile 转置，避免在片上转置整块矩阵。

### 1.2 低比特 KV：反量化插在「读出之后、乘加之前」

这是和本课题最接近的一组。差别在反量化坐在 DMA、寄存器还是 CUDA core，以及乘加吃的是定点、FP16 还是 Tensor Core。

**C11 Oaken** 的片外 KV 分成两份：middle 组（默认约占 90%）以 4-bit 稠密存放，inner/outer 组（默认约 6% 与 4%）量化到 5-bit，再拆进稠密矩阵里被置零的 4 bit 和一条 8-bit 对齐的 COO（6 bit 下标 + 1 bit 组号 + 1 bit 符号）。阈值离线标定，每组的 scale 用该组在线 min/max 计算。新 token 的 KV 在核内量化后写回；历史 KV 由 **DMA 内的反量化引擎流式读出**，不必先把整段 KV 放进片上 SRAM。反量化后的流进入矩阵–向量单元做 $QK^\top$ 和 $AV$。写的顺序是按 head 分页、沿序列追加，这样下一步可以把该 head 的全部历史一次性突发读出。Softmax 仍在注意力核里，论文没有展开到寄存器级。

**C13 BitDecoding** 的片外对象是打包的低比特 KV（INT4/INT2，以及 Blackwell 上的 MXFP4/NVFP4），外加一小段 FP16 residual，scale 以 half 精度存放，粒度可以是通道或张量。读入 shared memory 后，在 **CUDA core 寄存器**里用位运算反量化到 FP16，再交给 Tensor Core 做 $QK^\top$ 和 $PV$。打包发生在生成新 KV 的时候：`ldmatrix` 已经把数据放进 Tensor Core 的交织 fragment，每个线程就地量化，写回的布局因此直接能被下一次 `ldmatrix` 用。Residual 长度由 Tensor Core 的 warp tile 和打包比决定，

$$
N_r = P_n \times W_n \times R, \quad R = \omega / \beta
$$

其中 $\beta$ 是 KV 位宽，$\omega$ 是打包字长。Decode 的 query 长度通常小于 16，所以 warp 只沿 $N$（序列）方向加多，用来并行反量化。Softmax 的行最大值因此跨 warp，要经 shared memory 里的 $s_{\mathrm{TMP}}$ 归约；注意力分数还要经 $s_{\mathrm{Acc}}$ 再装回 Tensor Core 能接受的布局。Blackwell 上若指令直接吃低精度，这条 `lop3` 反量化可以去掉，但布局和 scale 约束还在。

**C09 QServe** 把权重做成 W4、激活 A8、KV 做成 KV4。线性层的反量化在寄存器里把 INT4 解成 INT8，再走 INT8 Tensor Core；保护区间 $[-119,119]$ 让「先乘后减」仍落在 INT8 里。KV4 的反量化把 INT4 解回 FP16，注意力本身在 CUDA core 上以 FP16 做融合 GEMV，而不是用 INT4 Tensor Core。分页 KV 的每个 head 带自己的动态 scale / zero-point。SmoothAttention 把量化难度从 Key 挪到不量化的 Query 上。论文故意把融合注意力的运算强度降到内存界限以内，使 KV4 少读的字节变成实测加速。

**C15 AccLLM** 的片外分工是：HBM 放权重和 KV，DDR 放较小的输入输出激活。权重组大小 64，带 scale 和 zero-point；激活 per-token INT8；K/V 为 per-token、同一 head 共享 scale 的 INT4。乘加不先把所有数扩成同一种浮点，而是直接在 DSP 上做 A8×W2、Q8×K4、S8×V4。注意力融合成五段：$QK^\top$、$\exp(S')$、累加 $\sum\exp$、$\exp(S')\,V$、再累加，全部行走完才做一次除法。Decode 时层间激活向量留在片上，本层输出直接当下一层输入。$\Lambda$ 形注意力只保留 sink 和近期窗口，因此 KV 的驻留长度被钉死；评测里这个窗口对应固定约 1 GB，即 $2+2044$ 个 token，再叠加 KV4 后约 0.25 GB。

**C14 CD-LLM** 把注意力和线性层拆到不同 FPGA。Master 用 HBM 做注意力和浮点 Softmax / SiLU；Slave 用 DDR 做大 batch 线性层，权重是按重要性分成的 INT3/INT4。反量化被吸收进存储格式：每个 AXI 128-bit 节拍里已经带上该组的 zero-point 和 INT8 scale，组间再共享一个列级 FP16 scale，MAC 在定点完成。KV 侧论文给出的是 KV4 与 W3.45 的联合量化，没有查询相关的页选择，也没有把反量化引擎拆成 Oaken 那种独立 DMA 级。注意力与线性层按微批交替流水。

### 1.3 先减少访问集合或计算量，再进入上面的通路

**C01 SpAtten** 在算注意力的同时累加 token 和 head 的重要性，用 top-k 丢掉一部分 token/head，并可以只取高位、置信度不够再取低位重算。片上有 Key/Value SRAM 和双缓冲；摘要和生成的差别是：摘要阶段存活的 K/V 可以留在片上跨 query 复用，生成阶段单个 query 对 K/V 没有这种复用，按修剪后的地址从片外取。这是输入相关的近似，不是现代长上下文 KV cache 的静态低比特格式。

**C07 SOFA** 面向长序列上许多 query 同时存在的动态稀疏（论文动机包括 prefill 和把 decode 变成 prefill 的投机解码）。预测器用对数域移位加估计分数，正式计算仍用较高位宽。跨阶段 tile 让预测、排序和 FlashAttention 式更新共用一块数据，避免把整行 pre-attention 写回 DRAM。Top-k 之后才按需生成或读取真正参与的 $K,V$。Softmax 按分数降序更新，以减少指数和比较。片上有 token、权重和临时缓冲，以及被选中的 K/V。这不是逐 token decode 的 GEMV，也不是 INT4 KV 反量化。

**C16 SnapStream** 在 prefill 做一次 SnapKV，decode 用 StreamingLLM 式环缓。片外是加速器的 HBM/DDR；片上是数据流处理器的大 SRAM，融合核尽量不把完整 $P$ 写出去。Prefill 的 MHA 把 K/V 切成 8192 的块。压缩核会 **重算** 观察窗口与待驱逐段的 $QK^\top$ 才能做 top-k。Decode 只把新 token scatter 进环上下标，sink 和 top-k 段保持不动，之后对这条定长缓存做稠密访问。论文正文没有 INT4/INT8 KV 反量化；平台叙述是 BF16 算力。DeepSeek 路径压缩的是 MLA 的低秩潜向量沿序列的长度，不是全维 K/V。

**C12** 不设计新的存储格式。它比较 MLA 的两种乘法顺序：把升维矩阵吸收进权重再乘潜向量，运算强度高，但吸收后的权重必须留在片上；或者少做这次吸收、多读潜空间缓存。缓存对象是 $C_{\mathrm{KV}}$ 而不是全维 K/V。分析省略了 RoPE，也没有量化、分页或 softmax 的微观实现。

**C04 FlightLLM** 的大块权重和 KV 在 HBM，小而频繁的查找表在 DDR。权重混合精度经专用单元扩成 INT8 再进稀疏 DSP 链；激活在 decode 时尽量常驻片上，一层消费完再给下一层，推理步结束才写回。Softmax 在 SFU 上分两阶段（先整向量归约，再逐元）。块稀疏注意力可以按 mask 跳过加载。KV 本身不是一条独立的 4-bit 反量化流水。

---

## 2. 各级吞吐是否匹配，等待出现在哪里？

Decode 的注意力几乎总是先碰到带宽，而不是峰值算力。等待出现在四类地方：反量化比乘加慢、归约必须等完全部局部结果、片上装不下所以要重算或再取数、以及跨请求或跨芯片的同步。

| 等待从哪来 | 论文里怎么表现 | 论文怎样处理 |
|---|---|---|
| 片外字节多于算力能消化的 | DFX 的生成阶段、AccLLM 的 decode 向量–矩阵、CD-LLM 的注意力、Oaken 的大 batch 注意力、BitDecoding / QServe 的长 KV | 减字节（量化、窗口、剪枝）或把算力单元让给带宽（QServe 故意留在内存界） |
| 反量化落在慢单元上 | QServe：A100 上一次 CUDA core 操作约相当于 50 次 INT4 Tensor Core 操作；反量化开销约 20%–90%。Oaken 把同一算法放到 GPU 上时，分组导致 warp 发散，量化和反量化变长 | Oaken 把引擎放进 DMA 并跨请求重叠；BitDecoding 用更多 warp 和软件流水把反量化盖在下一块 MMA 下面；QServe 把主 GEMM 留在 INT8 Tensor Core，并降低 KV 注意力的运算强度 |
| Softmax / 跨块归约 | C06：同步的 partial softmax 在 Llama2-7B、A100、输入长度 1024 时约占 attention 的 18.8%。BitDecoding：warp 沿 $N$ 切开后，寄存器内 softmax 不再自洽，必须经 shared memory | 统一上界后异步累加，失败则整行重算；BitDecoding 用很小的 $W_n$ 块做跨 warp max |
| 片上容量不够 | FlashAttention 的块大小受 SRAM 限制，块太小则 HBM 往返增加。AccLLM 的 prefill 中间结果装不下就要访问片外；decode 的向量装得下，才能做层融合。SpAtten 在置信度不够时丢掉当前概率、再取低位重算 | 加大块、融合、或接受重算 |
| 跨层、跨卡、跨请求 | DFX 每层多次环网同步，LayerNorm 不做跨卡并行。CD-LLM 的空闲主要来自 slave 网口上的传输。Oaken 每个核独占自己请求的 KV 带宽 | 用另一请求的计算盖住本请求的搬运；CD-LLM 用微批把 master 注意力和 slave 线性层叠起来 |

已经核对过的「盖住了没有」：

- **Oaken**，Llama2-7B，batch 64：量化占端到端延迟 1.29%，反量化占 3.23%。论文写明这两段与其他请求的 DMA 读和注意力重叠，并且是流式的，不需要整段 KV 到位才开始。注意力时间相对未做该 KV 量化的 LPU 平均短 55.0%。这是「反量化没有成为新的长级」的直接证据，条件是 ASIC DMA，不是 GPU。
- **QServe** 的设计目标写在引言里：推迟 CUDA core 屋顶的拐点，同时降低 KV4 注意力的运算强度，使算子留在内存界，低比特才能换成吞吐。若反量化把算子推回算力界，KV4 的字节优势不会变成延迟优势。
- **BitDecoding** 的软件流水是：当前 slice 做 `mma` 时，下一切片做加载和反量化。Query 很短，所以不能靠 $M$ 维摊开反量化，只能增加沿序列的 warp 数，让 SM 调度器掩盖停顿。
- **AccLLM** 的屋顶图给出 decode 相对峰值大约 90% 的性能跌落，原因是向量–矩阵几乎没有复用。后续消融说明：KV4 把注意力硬件利用率提高约 2 倍，但端到端只有 1.05 倍，因为注意力占的计算份额小；真正拉开端到端的是 W2（1.91 倍）和 DSP 打包（1.28 倍）。等待被掩盖的是线性层带宽，不是长 KV 的页检索。
- **CD-LLM** 用计算强度说明 batch 不能同时喂饱两种层。线性层的 $X=1$，注意力的 $X=B$（KV 随 batch 长大）：

$$
\mathrm{CI} = \frac{B \cdot D_i \cdot D_o}{B \cdot (D_i + D_o) + X \cdot D_i \cdot D_o}
$$

同构系统上注意力利用率很低（文中 8 卡 A100 的 MHA 为 3.72%，线性层为 28.11%–66.16%）。拆到 HBM master 与 DDR slave 之后，两边可以按微批重叠。小 batch（$\le 8$）时 slave 的 DDR 带宽先成为等待；大 batch（$\ge 32$）才吃到 59.90 TOPS 的峰值。评测点是 Llama-3.1-70B、输入/输出 $[1024,256]$、batch 256 时 2721.79 token/s。
- **C06** 的 flat GEMM 在 $N$ 大、偏带宽时用 shared memory 双缓冲重叠加载和 Tensor Core。统一 $\phi$ 去掉的是段间同步，不是片外等待。OPT-6.7B 因为分数范围过大，论文关闭了统一 max。
- **SnapStream** 的 prefill 是算力界（融合把 HBM 往返压下去），decode 是访存界。表 V 的吞吐提升来自压缩后最大 batch 变大，而不是单请求注意力核更快。128K 压到 32K 时，最大 batch 从 16 到 64，吞吐从 434 到 1832 token/s（4.2 倍）。结论里的 4.3 倍对应表中另外两行，并带「prefill 延迟至多增加 5%」。

论文没有把流水气泡逐周期量化的，正文只给了利用率或屋顶，不能改写成「某一级空等了多少拍」。C08 属于这一类里的空白：没有全文，就不能说它的等待在片上还是片外。

---

## 3. 哪些布局和调度是被位宽、页大小、端口数和带宽逼出来的？

这些选择不是风格偏好。换一个约束，原布局就会多一次搬运、一次填充或一次重排。

### 位宽

- **对齐单位把元数据焊进数据拍。** CD-LLM 的 4-bit 模式里，一个 128-bit AXI 拍装两组，每组 13 个 INT4、1 个 INT4 zero-point、1 个 INT8 scale；3-bit 模式是 39 个 INT3 加 zero-point 和 scale。平均位宽因此是 3.94 bit，而不是名义上的 3.45 bit 权重。分开存放 scale 时，FlightLLM 的有效带宽利用率被论文记为 51.77%。Oaken 把 outlier 的 COO 固定成 8 bit；改成两组或四组就会多出填充位，有效位宽并不随「再分一组」下降（表 3）。
- **乘加端口的位宽决定打包方式。** AccLLM 的 DSP48E2 是 $(A+D)\times B$。8×4 可以在同一条路径上塞两路；2-bit 权重又和 2:4 剪枝绑在一起，相邻行的输入不再相同，输入复用消失，于是把两个输入打进 $B$、两个权重打进 $A$ 和 $D$，一次 DSP 做四路积再挑出对角线项。位宽和稀疏模式一起改写了端口分配。
- **Tensor Core 的 fragment 决定 KV 的物理布局。** BitDecoding 不在全局内存里另做一次重排，而是在 `ldmatrix` 之后就地打包。Residual 长度 $N_r$ 随 $\beta$ 改变：INT2 和 INT4 的尾块大小不同。QServe 的权重按 32×32 tile 重排，是因为 INT4 权重和 INT8 激活的 `ldmatrix` 字节语义对不齐；shared memory 还要 swizzle 掉 bank conflict。
- **反量化的中间位宽决定能不能走快单元。** QServe 先量化到带保护区间的 INT8，再量化到 INT4，这样主循环留在 INT8 Tensor Core，而不是 INT4 Tensor Core 加 CUDA core 上的部分和反量化。

### 页和块

- **页在 C 类里多数是容量和地址单位，不是查询相关的读取单位。** Oaken 的 MMU 用稠密、稀疏两张表记录虚实地址和传输长度，按 head 把当前 token 写入不同页，并紧挨上一 token，以便 **整段历史** 突发读出。FlashInfer 的物理页是 $(H,D)$ 一块，$B_c$ 由 KV 管理器决定，计算时再映成块稀疏 tile。SnapStream 的 8192 是 MHA 的计算 tile，压缩之后的 sink、环缓、top-k 在 decode 期间地址固定。AccLLM 的「页」实际上是被 $\Lambda$ 窗口截断后的定长 KV，没有逐拍选页。
- **计算块受片上容量和 head 维约束。** FlashAttention 的 $B_c$、$B_r$ 由 SRAM 容量和 $d$ 决定；$d$ 变大则块变小、HBM 往返变多。DFX 把 tile 定为 $(d,l)=(64,16)$，因为 head 维约为 64，再加大就会让 $QK^\top$ 和 $SV$ 填不满。FlightLLM 的稀疏块是 64×64，N:M 块是 16×16，长度自适应指令按这些块对齐。

### 端口数

- DFX 的 DMA 接满 HBM 通道，加载粒度跟着通道宽度走。
- CD-LLM 把 INT3 和 INT4 分到不同 DDR 通道，通道在离线就绑定到计算单元；slave 只有两个网口，所以批量小时网络等待盖不住。
- BitDecoding 和 QServe 的 shared memory bank 冲突用异或或 swizzle 消掉，这是端口冲突，不是算法稀疏。
- AccLLM 的 2:4 剪枝直接毁掉相邻 PE 的输入广播，端口复用方案必须改。

### 带宽

- 带宽低于算力时，调度改成「少发、对齐发、让计算等搬运」。QServe 降低注意力运算强度；Oaken 用突发读全部历史；CD-LLM 把 scale 并进同一拍；FlightLLM 和 AccLLM 把 decode 激活留在片上，避免每层把向量写回 HBM/DDR。
- 带宽很高时，同样的低比特布局会从「省带宽」变成「反量化和 Tensor Core 利用率」。BitDecoding 在不同 GPU 上的加速差，论文归因于反量化能否被 Tensor Core 吃掉，而不只是字节比。
- 短序列、小 batch 时线性层仍可能是算力界。Oaken 在 Llama2-13B、batch 16、总长低于 8K 时，QServe 和 vLLM 可以超过 Oaken，因为可批的计算占比更大；总长上去之后 HBM 容量先满（文中超过 16K 难以完成该 batch），LPDDR 版可以到 32K。

---

## 4. 收益来自算法少干活，还是硬件把同样的活干得更有效？

两件事要分开记。算法少干活包括：少读 token、少读 bit、少做乘加、用低秩潜向量代替全维 KV。硬件执行效率包括：融合后少写中间矩阵、布局对齐后突发利用率上升、反量化与乘加重叠、并行划分把单元喂饱。多数论文两样都有，但 **端到端数字里占大头的那一项** 并不相同。

| 编号 | 少干活 | 执行更有效 | 端到端里更明显的一项 |
|---|---|---|---|
| C01 | 丢掉 token/head；低置信度时少取位 | 全流水、并行 top-k、按通道散布随机读 | 两者都大；生成阶段更接近带宽，修剪直接少搬 |
| C02 | 无近似、无量化 | tile、满 HBM、生成阶段的矩阵–向量、转置与计算重叠 | 只有执行效率；相对 GPU 的差距主要是生成阶段利用率 |
| C03、C05 | 稠密情况下 FLOP 不减；因果掩码可跳过约一半块 | 不写 $N\times N$、片上在线 softmax、更好的线程划分 | 执行效率。反向甚至用更多 FLOP 换更少的 HBM |
| C06 | 统一 $\phi$ 在分布内与精确 softmax 等价，不减少分数个数 | 去掉段间同步、flat GEMM 少填充、按形状选 CUDA core 或 Tensor Core | 执行效率 |
| C07 | top-k 跳过大量 $QK$；对数域预测没有乘法 | 跨阶段 tile、按需 KV、降序 softmax、复用调度 | 算法少算是前提；tile 和按需读取决定少算能不能少访问 DRAM |
| C09 | W4 与 KV4 减少字节 | 保护区间下的寄存器级反量化、权重重排、把注意力留在内存界 | 两者缺一不可。只量化而反量化落在 CUDA core 上，论文认为吞吐可以更差 |
| C10 | 不减少算术；块稀疏和共享前缀减少冗余搬移 | JIT tile、负载均衡、融合 RoPE | 执行效率与地址布局 |
| C11 | 4/5-bit 与融合编码减少 KV 字节和容量 | DMA 流式量化/反量化、按 head 追加以便突发 | 少字节是主因（注意力时间平均短 55%）。反量化本身在 batch 64 时只有几个百分点，说明硬件把这项开销压住了，而不是用它制造加速 |
| C12 | 潜空间缓存比全维 K/V 小 | 吸收升维矩阵会增加计算、减少带宽 | 少字节为主；执行顺序决定少字节会不会被重算抵消 |
| C13 | 低比特 KV 减少读字节 | fragment 内打包、warp 重划分、反量化与 MMA 流水 | 相对 FP16 的加速同时含两部分。论文的系统贡献是让少掉的字节不被反量化停顿吃掉 |
| C14 | 权重约 3.45 bit、KV4 | AXI 对齐打包、DSP/BRAM/LUT 同时做 MAC、注意力与线性层异质流水 | 消融若把三者乘在一起会重复计算。正文能分开的是：小 batch 时带宽和网络限制峰值，大 batch 时峰值算力才显现。表 6 的 2721.79 token/s 是整系统 |
| C15 | 2:4 剪枝、$\Lambda$ 窗口、W2A8KV4 | DSP 打包、注意力五段融合、decode 层融合 | 端到端加速主要来自 W2 和 DSP 打包。KV4 对端到端只有 1.05 倍，对容量是 75%（相对 FP16），再加窗口后 KV 约 0.25 GB |
| C16 | prefill 一次 SnapKV，decode 只保留定长环 | 静态图融合、K/V 按 8192 分块、decode 改数据并行以免每卡复制整批 KV | 表 V 的 4.2 倍是最大 batch 从 16 增到 64 的系统吞吐，不是 batch=1 的核加速 |

对本课题最有用的拆法是 C09、C11、C13、C15 这一组：

- 字节减少是注意力延迟下降的必要条件。
- 反量化、对齐填充和归约若比省下的搬运更贵，必要条件就不充分。
- KV 压缩对 **整模型 token/s** 的可见度，取决于注意力在该工作点占多少。AccLLM 在 7B、窗口已经很短时，KV4 几乎不改变端到端延迟；Oaken 在长生成、大 batch 上，注意力才是大头。

---

## 5. 模型形状、访问集合或精度一变，原设计的优势是否还在？

### 模型形状

- **序列变长** 加强「少读 KV」的论文（Oaken 在 8K 以上超过短序列时占优的 GPU；BitDecoding 的单 batch 128K；SnapStream 的 128K→32K），削弱「假设 KV 能进片上 SRAM」的设计（SpAtten 按约 1024 token 配 Key/Value SRAM；FlightLLM 的长度桶规划到约 2048；AccLLM 评测到约 7K，窗口把 KV 钉在 $2+2044$）。FlashAttention 的 IO 好处随序列变长更明显，但 $d$ 变大时块变小，好处变弱。
- **Batch 变大** 改变瓶颈位置，而不只是放大同一曲线。CD-LLM 在 batch $\le 8$ 时不如对照，batch $\ge 32$ 才显出峰值；其评测点 batch 256、上下文只有 1024/256。Oaken 的 1.79 倍（相对 vLLM）和 1.58 倍（相对 QServe）是 **batch 256、输入输出各 1K** 的平均吞吐。SnapStream 的 4.2 倍要求压缩后能把 batch 从 16 加到 64。本课题核心是 batch 1、32K–128K，更接近这些论文里的带宽端，而不是它们报最大加速的大 batch 端。
- **GQA / MQA** 减少 KV 头，直接减少字节，量化的相对收益变小。Oaken 写明 Mistral-7B、Mixtral-8x7B、Llama2-70B 使用 GQA；在 Mixtral 上，短生成时量化相对全精度的增益很小，生成变长或 batch 变大后才明显。BitDecoding 用 query 重排把一组 query head 填进 Tensor Core tile，GQA 的组大小 $g$ 因此变成布局参数。FlashInfer 把 decode 的运算强度写成大约 $O(g \cdot l_{qo})$。C03、C05、C06 没有把 GQA 当作主变量；C15 只测了 Llama-2-7B 的多头注意力（其 7K KV 容量公式按 32 头、head 维 128、FP16 写成 3.5 GB）。
- **MLA** 改变的是缓存的秩，不是位宽。C12 说明吸收或重算哪一个划算，取决于片上能不能放下吸收后的权重，以及平台的运算强度拐点。C16 在 DeepSeek 上沿序列压缩潜向量。两者都没有测 INT4 潜向量加页检索。
- **Head 维** 进入 tile 公式。DFX 和 FlashAttention 都表明 $d$ 离开设计点后，阵列或 SRAM 块会填不满。

### 访问集合

- **全量顺序历史** 是 Oaken 突发读、QServe 分页但仍全读、BitDecoding 的 packed KV 扫描能成立的条件。页在这些设计里用来对齐和分配，不用来跳过。
- **定长窗口或一次 top-k**（AccLLM 的 $\Lambda$，SnapStream 的 prefill 压缩）把后续 decode 变回稠密、定长、地址稳定的访问。优势在压缩当时已经兑现。Decode 每步不再支付选页和 gather 的成本，也不再拥有「本步换一批页」的能力。
- **每步都变的稀疏下标** 会打掉为连续突发做的布局。CD-LLM 和 Oaken 都把元数据与数据排进固定节拍；随机页会回到它们批评的细粒度访问。FlashInfer 写明非仿射稀疏不能用 TMA。SOFA 的按需 KV 是为动态 top-k 做的，但是在大量 query 并行的稀疏 prefill 上，不是 batch-1 的页检索 decode。
- **C08** 若正文成立，它减少的是「跨步重复读同一位置」，访问集合是时间上的复用，不是本步的 query-aware 页。没有全文，不能判断复用失效时是否退回全读，以及退回是否计入延迟。

### 精度

- 布局参数跟着位宽走。BitDecoding 的 $N_r$ 含 $\beta$；AccLLM 的 DSP 打包把 8×8、8×4、8×2 分成不同接法；Oaken 的 4-bit 稠密槽用来塞 5-bit outlier 的低 4 位，改位宽就要改融合编码；CD-LLM 的 3-bit 与 4-bit 组大小不同，才能填满 128 bit。
- 精度升高，字节优势下降，反量化若仍按低比特流水来做，会变成纯开销。QServe 的论点是：KV4 必须让注意力留在内存界，否则少掉的 bit 测不出来。AccLLM 去掉 KV4 后端到端只有 1.05 倍的损失，说明在该 7B 窗口工作点，INT4 的价值主要是容量。
- 精度再降到 INT2 或 FP4 时，BitDecoding 仍用同一套 warp 和流水，但反量化或后续把 $P$ 再量化会更重；Blackwell 原生低精度可以跳过一部分解包。这些格式不是本课题的 INT4/INT8 静态配置。C03–C06、C10、C12、C16 没有测低比特 KV，它们的执行效率优势不依赖于位宽，也 **不会** 自动在 INT4 上保留，因为 INT4 会插入它们没有的反量化级。
- 量化分组方式一变，等待位置就变。QServe 的 per-group 零点必须留在主循环里；Oaken 的组比例改变有效 bit 和 COO 对齐。论文里的一组 Pareto 点（Oaken 的 4%/90%/6%）不能换成任意 per-token 或 per-channel 仍保持同一突发效率。

---

## 6. 逐篇要点

每篇只保留与第 1–5 节直接相关的通路、等待、约束、收益归属和失效条件。C08 单独标明未取得全文。

### C01 SpAtten

片外按 MSB/LSB 分区存放 Q/K/V，修剪后按地址读取。片上 Key/Value SRAM 含双缓冲；摘要阶段复用存活 K/V，生成阶段基本不复用。计算是定点乘加，softmax 在浮点，概率再量化回定点。渐进量化在置信度低时多取低位并重算。收益同时来自少算少搬和专用流水、并行 top-k。GQA、32K 以上上下文、静态 INT4 KV 页均未作为主实验。生成阶段已经是内存界，但工作负载是 GPT-2 量级，不能当作长上下文 GQA 基线。

### C02 DFX

HBM 放权重和 KV tile，DDR 放 token 与嵌入，FP16，无反量化。生成是矩阵–向量，操作数双缓冲，Value 的转置藏在先算 V 的调度里。等待在每层跨 FPGA 同步；LayerNorm 不跨卡切。Tile $(64,16)$ 服从 head 维和 HBM 通道宽度。没有任何算法少干活。非 batch、头维 64、中等序列的 GPT 式模型是其设计点；GQA、低比特、页式稀疏都未测。

### C03 FlashAttention

HBM 存 $Q,K,V,O$ 和统计量，SRAM 存当前块。在线 softmax 使结果精确且不物化 $N\times N$。块越大越少访问 HBM，直到变成算力界。收益是 IO 和融合，稠密 FLOP 不减。$d$ 增大或 SRAM 变小则加速变弱。未测 GQA 与 INT KV。单 query decode 不能直接套用「整段序列都在做大块 GEMM」的占用假设。

### C04 FlightLLM

HBM 放大块权重和 KV，DDR 放查找表。权重混合精度扩成 INT8 进入可配置稀疏 DSP；decode 激活向量尽量留在片上。块稀疏和 N:M 少算，always-on-chip 与融合提高带宽利用率。长度指令按 64 与 16 的块对齐。评测 batch 1、长度到约 2048 的规划；GQA、70B、纯 INT4 KV 通路未作为主结果。不能假设整个长上下文 KV 常驻片上。

### C05 FlashAttention-2

仍是精确分块，无反量化。改动是序列维并行、warp 间少交换、因果块可跳、输出缩放推迟到循环末。块大小受寄存器、shared memory 和 $d$ 限制，常见 64 或 128。收益是执行效率。低比特 KV 和页检索未测。可借鉴的是在线统计与「沿长序列切分」，不是存储格式。

### C06 FlashDecoding++

Decode 的 KV 追加在 cache 中。统一 $\phi$ 让 partial softmax 异步，溢出则整行重算；OPT-6.7B 因范围过大不使用该技巧。Flat GEMM 只填充到 Tensor Core 的 $M=8$，大 $N$ 时 shared memory 双缓冲。启发式在 CUDA core 与 Tensor Core 间切换。收益是执行效率。无 INT KV，无页选择。Llama2-7B 上同步 softmax 约占 attention 的 18.8%（A100，输入 1024），说明归约等待在 decode 里真实存在。

### C07 SOFA

动态稀疏的预测、排序、正式注意力按 tile 串起来，避免整行 pre-attention 进出 DRAM。预测走 4-bit 级对数域移位加，正式路径位宽更高；top-k 之后才取 K/V。降序更新减少 softmax 的乘法和比较。收益首先是少算，tile 和按需读取让少算变成少访问。主场景是大量 token 并行，不是 batch-1 GEMV。GQA 与 INT4 KV 页未测。128K 只出现在动机里的 profiling，不表示加速器在该长度上跑通。

### C08 LAD

未取得全文，不做通路判断。摘要声称利用相邻 decode 步 attention score 的数值局部性，减少对应位置的 KV 访问，并保持较高的序列相似度。复用范围、失效时是否全量回读、以及与每步重选页的关系，都需要正文才能核对。

### C09 QServe

W4 权重在寄存器里解到 INT8 后进 Tensor Core；KV4 解到 FP16 后在 CUDA core 上做融合注意力。分页存放 per-head 的 scale 和 zero-point。保护区间 $[-119,119]$ 使反量化可以先乘后减并做寄存器级并行。论文明确要求把 KV4 注意力留在内存界。引言中的反量化开销是 20%–90%；A100 上一次 CUDA core 操作约等于 50 次 INT4 Tensor Core 操作。表 4 的最大吞吐相对 TensorRT-LLM 最优配置，Llama-3-8B 在 A100 上为 1.20 倍、L40S 上为 1.39 倍，Qwen1.5-72B 为 2.38 倍与 3.47 倍。测了 GQA 模型，但是全量注意力加分页，没有有界检索。精度离开 W4A8KV4 后，保护区间和屋顶论点都要重做。

### C10 FlashInfer

逻辑页/物理页进入块稀疏格式，tile 拷进 shared memory 再计算。在线 softmax 状态跨 chunk 归约。FP16 评测。共享前缀是索引组合。非仿射稀疏不能用 TMA。收益来自布局、调度和融合，不来自少 bit。INT4 不能直接沿用其拷贝和 MMA 假设。

### C11 Oaken

见第 1.2 节。额外核对过的条件：外/中/内组 4%/90%/6%，中组 4-bit，内外组 5-bit；融合编码把 outlier 从 23 bit 降到 8 bit；峰值 FP16 270 TFLOPS，1 GHz；HBM 版 80 GB、2.0 TB/s，LPDDR 版 256 GB、1.1 TB/s。Batch 256、1K:1K 时 LPDDR 版相对 vLLM 平均 1.79 倍、相对 QServe 平均 1.58 倍。相对 FP16 的平均精度损失论文记为 0.87 个百分点，并比 KVQuant 低 0.54 个百分点。稀疏是 outlier 的 COO，历史 KV 仍然整段突发读出。访问集合改成每步一小撮不连续页时，这条写序和突发假设不成立。

### C12 MLA 硬件分析

比较潜空间缓存上的两种乘法顺序，用屋顶和能量模型而不是一块芯片。吸收升维权重能提高运算强度，但权重要留在片上；另一种顺序更省计算、更吃带宽。未建模量化、分页和 RoPE。只说明 MLA 的形状会移动瓶颈，不提供 INT4 数据通路。

### C13 BitDecoding

见第 1.2 节。相对 FP16 FlashDecoding-v2 的平均 7.5 倍、Blackwell NVFP4 上至多 8.6 倍、Llama-3.1-8B 128K 单 batch 延迟降到三分之一，都是论文摘要中的总结果，同时包含少字节和提高 Tensor Core 利用率。GQA 通过把 query 重排成 $[g, h_{\mathrm{kv}}]$ 来填 tile。页式管理在评测里出现，但是容量管理，不是内容选页。位宽一变，$N_r$ 和反量化指令都要变。

### C14 CD-LLM

见第 1.2 与第 2 节。不要和 FMC-LLM 混用。KV4 与 W3.45 是算法少字节；128-bit 对齐和异质流水是执行效率。优势区域是大 batch、千级上下文的 70B 批解码。Batch 1、32K–128K 的单请求 decode 正是文中 slave DDR 和 master HBM 会先饱和的区域，论文没有把该点作为优势场景测过。

### C15 AccLLM

见第 1.2 节。Llama-2-7B，U280。相对 FlightLLM 在 U280 上的吞吐是 2.98 倍、能效是 4.07 倍，论文把原因归于 W2 缓解带宽和 DSP 打包。消融：剪枝 1.39 倍，W2 1.91 倍，KV4 1.05 倍，DSP 打包 1.28 倍。窗口在结果图的正文说明里是 $2+2044$ 个 token、约 1 GB；加 KV4 后约 0.25 GB。7K 例子里 FP16 KV 为 3.5 GB、窗口后约 1 GB（少 71.4%）。没有 GQA，没有 32K–128K，没有逐拍选页。KV4 的端到端速度优势在这个短窗口上几乎不成立，容量优势成立。

### C16 SnapStream

Prefill batch 1、TP16，插入环缓 gather 和 SnapKV（含一次额外 $QK^\top$）；decode 对定长缓存做融合注意力，新 token 只 scatter 进环。K/V 计算块为 8192。没有低比特反量化。表 V 是 DeepSeek-R1-0528、SN40L-16：128K 压到 32K 时最大 batch 16→64、吞吐 434→1832 token/s。这个倍数依赖 batch 放大。压缩算法本身仍减少单请求的 KV 字节，但论文没有把 batch=1 的核加速当作结果。静态图使「每步改访问集合」代价很高，所以选择被提前到 prefill。

---

## 7. 其他观点

这些观察与上面五个问题相关，但不是那五个问题的直接答案。每条都依赖论文自己的假设。

1. **反量化的位置是架构变量，不是量化算法的附属。** Oaken 把它放进 DMA 并流式化，QServe 和 BitDecoding 把它放在寄存器 / CUDA core 并要求与乘加重叠。同一套 4-bit KV，放错层级就会把内存界算子推回算力界。假设：乘加单元与反量化单元的吞吐比已知，且可以流水。

2. **有效位宽必须把 scale、zero-point、下标和填充算进去。** CD-LLM 的名义 3.45-bit 权重在 AXI 拍里平均是 3.94 bit；Oaken 的 outlier 对齐会在分组方案改变时被迫加填充。假设：片外端口有固定节拍，元数据不能另开细粒度随机读。

3. **写顺序是为读突发服务的。** Oaken 按 head 分页并沿时间追加，是因为下一步要读 **全部** 历史。这个写序对「每步只读 $K_{\max}$ 个不连续页」没有好处，还可能让未选中的页仍然占据连续容量。假设：访问集合等于驻留集合。

4. **Batch 改变的是哪一层先饱和，不是同一瓶颈的缩放。** CD-LLM 的计算强度公式里，线性层随 $B$ 变算力界，注意力的强度不随 $B$ 上升，因为 KV 流量也乘 $B$。用大 batch 吞吐论证长上下文单请求引擎，会选错该加算力还是该加带宽的 FPGA/ASIC。

5. **KV 量化对端到端 token/s 的可见度取决于注意力占比。** AccLLM 在窗口已经把 KV 钉死后，KV4 的端到端只有 1.05 倍，容量却降到约 0.25 GB。若优化目标是能量和字节而不是峰值 token/s，这个「几乎不加速」仍然是正结果。假设：线性层仍是带宽大头。

6. **定长压缩把动态访问变成静态稠密访问。** SnapStream 在 prefill 付一次重算和 gather，decode 之后地址稳定，才能放进静态图。吞吐表体现的是 batch 容量，不是选页器的每步成本。假设：生产图不能每步改注意力的稀疏模式。

7. **在线 softmax 的异步化依赖分数范围，不依赖稀疏。** C06 的统一 $\phi$ 在 Llama2 上成立，在 OPT 上被关掉，失败路径是整行重算。低比特 KV 会改变分数动态范围，不能默认这个异步技巧仍然几乎不触发回退。

8. **GQA 会缩小量化的相对收益。** Oaken 在 Mixtral 的短生成上观察到全精度与量化的差距变小，因为 KV 头已经共享。本课题若按 query head 重复计算压缩比，会把 GQA 已经省掉的字节再算一遍。

9. **MLA 的「少字节」和「少计算」不能同时取最大。** C12 的两种顺序一个偏计算、一个偏带宽，而且吸收后的大权重必须留在片上，否则重算的收益消失。分析省略 RoPE。本课题若把 MLA 当作有条件扩展，需要把潜向量位宽、升维权重驻留和 RoPE 顺序一起算，不能只引用潜维 512。

10. **预测式稀疏要把错误估计的代价做进流水。** SOFA 用对数域近似决定 top-k，再用正式位宽计算，并用最大值修正兜底。预测若不能按 tile 与正式计算重叠，预测器本身会变成新的长级。该设计的并行度来自同时存在的大量 query，不是单 query 扫完全部历史页。

11. **渐进取位是用重算换带宽的早期形式。** SpAtten 先读高位，置信度不够再读低位并重算概率。它说明「位宽可以是运行时的第二次访问」，但对象是短序列 QKV，不是可再被选中的历史 KV 页。

12. **DSP 或 Tensor Core 的端口几何会把稀疏模式和位宽焊在一起。** AccLLM 因为 2:4 破坏输入复用，才把两个激活打进同一个 18-bit 端口。若本课题的访问稀疏是页级而不是 2:4，这套打包不能沿用；能沿用的是「端口位宽必须在布局阶段就冻结」。

---

## 8. 对本课题的直接含义

下面只写可以沿用的判断，以及这些论文没有在本课题工作点上测过的外推。

C 类已经覆盖了本课题数据通路上的若干零件，但没有覆盖它们在同一工作点上的耦合。

### 可以写入课题模型的

- 低比特 KV 的读出、反量化与乘加如何重叠，Oaken、QServe、BitDecoding、AccLLM 已各给一种放置方式。
- 页作为地址和突发单位，Oaken 与 FlashInfer 已给出；页作为 **本步访问子集**，这两篇都不是。SnapStream 和 AccLLM 的子集在 decode 期间是静态的。
- 位宽、zero-point 与端口节拍的对齐浪费，CD-LLM 和 Oaken 已经把它算进有效位宽。
- 注意力占比很小时，KV 精度几乎不改变 token/s，这是 AccLLM 的消融，不是实现失误。

### 不能写成已经证实的

- 不宜把「量化 + 窗口 + 硬件融合」写成新的组合贡献。尚待用本课题自己的工作点检验的是：INT4/INT8 静态配置、GQA 共享、batch 1、32K–128K，以及 **每步重新选择的有界页** 是否还能维持 Oaken/CD-LLM 那种连续突发；反量化之后，选页元数据扫描和跨页归约会不会成为新的等待级。
- C08 在拿到全文之前，不能当作「跨步复用已经解决重复读取」的证据。
