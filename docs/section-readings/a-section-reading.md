# A 类精读：误差敏感位置、精度时序与读出代价

整理日期：2026-09-22。对象是文献清单 **A01–A10**。书目与归类以 [文献清单](../literature-survey-reading-list.md) 为准，本文不另建清单。课题对照口径见 [项目范围](../project-scope.md)。

第 1–5 节回答本类的五个问题。第 6 节是逐篇要点。第 7 节记录不改变上述回答的其他观点。第 8 节分成两段：可以写入课题模型的判断，以及这些论文尚未覆盖、不能写成已经证实的部分。

数字取自各篇开放全文的方法、实验与附录。原文没有的测量写「未报告」。效率倍数只作该论文自己的基线对照，不换算成课题口径，也不直接写成 Attention 引擎的加速。

不把摘要里的加速或「近乎无损」直接当成硬件结论。GEAR（A07）是 NeurIPS ENLSP-IV 研讨会论文，引用时保持研讨会身份。第 8 节是对照本课题数据通路作出的格式判断，不是论文原句。

## 阅读边界

阅读范围、版本和不能外推的部分记在这里。第 1–5 节的数字都受这些限制。

- 本次未读 K03 MiKV、K04 MLA 技术报告、K06 RateQuant、K07 AATC。ZipCache 与 KVTuner 把 MiKV 当作逐 token 混合精度对照；GTA/GLA 把 MLA 当作潜变量结构对照。需要这些机制的原文数字时再读 K 节。
- 部分表格来自 arXiv HTML 或 PDF 抽取，复杂公式可能丢符号。通道缩放、GEAR 表头和 MiniCache 内存式已按正文与附录 E 核对；若与出版社 PDF 的排版不一致，以出版社版本为准。
- SKVQ 原文写 Mistral-7B-Instruct-v0.2 使用 multi-query attention。本文保持原文用语，不把它改写成 GQA。
- KVQuant 的 10M 与 SKVQ 的 1M 是容量估计。任务分数停在各自论文实际跑过的长度上。

---

## 1. 哪些 K/V、层或 head 对误差敏感？在哪些模型、任务和长度上验证？

各篇测的不是同一种「敏感」。下表只记录该文实际比较过的轴。

| 敏感对象 | 论文中的结论 | 验证范围 |
|---|---|---|
| **Key 比 Value 更不能降位宽** | 同样总比特下，降低 Key 精度使注意力分数误差和最终困惑度上升更明显。KVTuner 在 per-token 非对称量化下，把 Key 从 8 bit 降到 4 bit、再降到 2 bit，Llama-3.1-8B-Instruct 上平均注意力分数误差分别约为 $13.9\times$ 与 $4.6\times$（GSM8K 前 20 条、无跨层误差累积的模拟）。词困惑度上 K8V4 接近 KV8，K4V8 明显变差。 | KVTuner：Llama-3.1-8B-Instruct、Qwen2.5 3B–32B、Mistral-7B-Instruct-v0.3；WikiText 词困惑度，GSM8K 4/8/16-shot 与多轮 CoT，Qwen2.5-7B-Instruct 的 20 个 LongBench 任务。敏感分析用 GSM8K 短提示，LongBench 才是长上下文生成。 |
| **Key 应按通道分组，Value 应按 token 分组** | Key 有固定大通道；按通道量化把误差关在通道内。Llama-2-13B 上 Key 的相对注意力分数误差：per-token $47.00$，per-channel $9.60$（全层全 head 平均）。Value 的重构误差两种分组接近（$4.57$ 与 $3.73$），但注意力输出相对误差 per-token $3.55$、per-channel $49.89$。原因是注意力稀疏，输出是少数 token 的加权和，per-token 量化把误差留在各自 token 内。 | KIVI：分布图为 Llama-2-13B、Falcon-7B。假量化对照为 Llama-2-13B 的 CoQA、TruthfulQA，组大小 32。主实验见下。KVQuant 附录 G 在 LLaMA-7B、Wikitext-2、3 bit 上复现同一方向：Key per-channel + Value per-token 的困惑度 $7.05$，二者都 per-channel 为 $223$，都 per-token 为 $10.87$。 |
| **检索型、非稀疏 head 比流式 head 更敏感** | 高误差 head 的注意力不稀疏。KVTuner 引理表述为：只有稀疏、集中的注意力 head 对低精度 KV 稳定。图例是 Llama-3.1-8B-Instruct 的 layer-2 streaming head 与 layer-13 retrieval head（GSM8K）。层敏感模式在不同提示间保持，作者将其视为模型属性。 | 同上。图 2 另给 Qwen2.5-7B-Instruct、GSM8K、第 79 个 query token 的 layer-0 query head-2 与 layer-21 query head-4，4 bit / 2 bit Key 会造成关键 Key 漏检或误检。 |
| **浅层与个别层的 Pareto 位宽对不同** | 多数层的 Pareto 对是 KV8、K8V4、KV4、K4V2、KV2。Llama-3.1-8B 的第 0 层在 per-token 模式下改为保留 K4V8。Qwen2.5 的前几层和个别中后层候选集不同。 | 见表 4 的层号。搜索用 GSM8K 前 200 条 4-shot；最终精度在 GSM8K 与 LongBench 上报告。 |
| **近期窗口，以及困难生成任务** | 全 token 2 bit（Key 通道、Value token）在 GSM8K 上掉点大；保留长度 $R$ 的 FP16 残留后，掉点缩小到约 2 个百分点以内（Llama/Mistral）。KIVI 认为 Key 的局部高精度窗长期望长度约为 $R/2$，Value 约为 $R$。SKVQ 消融：RTN 2 bit、组 32，LongBench 平均分 35.55；加窗口 128 后升到 45.73，通道重排再升到 47.99，5 个 sink token 再升 0.15。 | KIVI：Llama-2 7B/13B、Falcon-7B、Mistral-7B 的 CoQA、TruthfulQA、GSM8K；LongBench 最大长度 Mistral 8192、其余 4096；NIAH 图以词数作轴。SKVQ：Llama-2 与 Mistral 系列 LongBench；NIAH 用 Llama2-7B-80k，上下文 1k–32k、20 个长度、每长度 15 个插入位置。 |
| **序列开头的 sink token** | KVQuant：第一 token 对量化误差不成比例地敏感，只保留它为 FP16 在 2 bit、且不做稀疏异常值时收益最大。SKVQ：固定保留前 5 个 token 为高精度，收益小于滑动窗口。 | KVQuant 附录 K：LLaMA、Llama-2、Llama-3 的 Wikitext-2 困惑度。SKVQ 表 3：Mistral-7B-Instruct-v0.2，LongBench 平均分。 |
| **RoPE 之后的 Key** | Pre-RoPE Key 的异常通道跨 token 稳定；RoPE 把通道对旋在一起后，异常通道不再固定。LLaMA-7B、3 bit、Key per-channel：post-RoPE 困惑度 $7.05$，pre-RoPE $6.23$。 | 分布示例为 LLaMA-7B、Wikitext-2 一条 2K 样本。困惑度按模型原生窗口：LLaMA 2K、Llama-2 4K、Llama-3 与 Mistral 8K。长上下文质量在 LLaMA-2-7B-32K 与 Llama-2-70B-32K 上做到 passkey / LongBench / RULER 的 32K，不是 10M。 |
| **浅层 KV 不适合跨层合并；中后层相邻 KV 更相似** | MiniCache 从 $L/2$ 开始合并。浅层角距离大。少量角距离大的 token 对合并敏感，用阈值 $\gamma$ 留下。 | 相似度：LLaMA-3-70B 的 COQA、GSM8K、TruthfulQA。合并曲线：Phi-3-Mini、Mixtral-8x7B、LLaMA-3 8B/70B。LongBench：Llama-2-7B/13B-Chat、Mistral-7B 与 Instruct。 |
| **结构共享本身有质量代价，但不是量化误差** | GQA：一组 query head 共享一对 KV head。T5-XXL 上 GQA-8 的平均开发集分数 47.1，MHA-XXL 为 47.2，MQA-XXL 为 46.6。GTA 把 K/V 绑成同一状态，并只对一半通道做 RoPE；作者报告与同组数 GQA 的困惑度接近，但是在自训的 433M–1.47B 模型上，不是现成 GQA 大模型的量化实验。 | GQA：T5 Large/XXL；摘要输入 512 或 2048，输出 32–512。GTA/GLA：FineWeb-Edu 上训练的中小模型，解码基准在 H100 上测算术强度与延迟。 |

KIVI 主实验的模型与任务：Llama-2-7B/13B、Llama-2-7B/13B-Chat、Falcon-7B、Mistral-7B；CoQA、TruthfulQA、GSM8K；LongBench 八个任务。Falcon-7B 是 MQA、只有一个 KV head，2 bit 掉点大于 Llama/Mistral，4 bit 才接近 16 bit。NIAH 模型为 Llama-3-8B-Instruct 与 Mistral-7B-Instruct-v0.2。默认 $G=32$、$R=128$。

KVQuant 的质量验证与容量外推要分开。Wikitext-2 / C4 困惑度覆盖 LLaMA 7B–65B、Llama-2 7B/13B/70B、Llama-3 8B/70B、Mistral-7B。Passkey、LongBench、RULER 的最长设定是 32K（LLaMA-2-7B-32K；70B 只有 passkey）。1M 与 10M 是按压缩后的 KV 字节估计能否放进 A100 或 8 卡，不是这些长度上的任务分数。

TurboQuant 不按层或 head 分配误差预算。它把失真拆成向量 MSE 与内积失真：MSE 最优标量量化对内积有偏，残差上再做 1 bit QJL 才给出无偏内积估计。实验：Llama-3.1-8B-Instruct 的 NIAH（4k–104k，压缩到全精度 KV 的 0.25，TurboQuant 召回 0.997，与全精度相同）；LongBench 上 Llama-3.1-8B-Instruct 与 Ministral-7B-Instruct，2.5 bit 与 3.5 bit。

---

---

## 2. 精度配置是静态确定，还是运行时改变？

要分开三件事：**位宽或残留策略**、**scale / 零点 / 码本**、**哪些元素落在高精度区**。

| 编号 | 位宽或结构 | 高精度区是否在 decode 中移动 | scale / 码本 |
|---|---|---|---|
| A01 GQA | 训练时固定 KV head 组数 | 无量化位宽 | 无 |
| A02 KIVI | 整网 2 bit 或 4 bit，组大小与残留长度是超参 | 最近 $R$ 个 token 保持 FP16，组满后量化并滑出。策略固定，成员随生成变化 | 组内 min/max 在该组量化时计算，之后该组参数不变 |
| A03 SKVQ | 低比特档固定（主实验 K2V2；另有 K2 与 V1.5） | 最近 $w$ 个 token 为全精度；滑出窗口才量化。前 5 个 sink 位置固定保留高精度 | 通道置换与组裁剪系数 $\alpha$ 离线校准；组 scale / 零点按运行时 min/max。可用 FP8 存这两个参数 |
| A04 KVQuant | 每层一个 nuq 位宽；1% 异常值与首 token FP16 是固定策略 | 首 token 位置固定。Value 的异常值按下个 token 在线判定 | Key 的 per-channel scale / 零点与每层码本离线固定，新 token 不回写旧 scale。Value 的 scale 与异常阈值在线 |
| A05 ZipCache | 显著 token 4 bit、其余 2 bit；显著比例是超参（实验 40%–70%） | 每生成 100 个 token 重做一次压缩，显著 token 集合会变 | 通道 scale 与 token scale 都随数据计算 |
| A06 MiniCache | 不分配位宽。从 $L/2$ 起两层合并，插值 $t=0.6$，$\gamma=0.05$ | 角距离超过阈值的 token 在运行时留下，索引随 KV 变化 | 每 token 存两层幅度和夹角 |
| A07 GEAR | 主干 4 bit 或 2 bit；$s=2\%$，$r=4$（prefill）或 $2$（decode 缓冲），缓冲 $n_b=20$ | 新 token 先入 FP 缓冲，每 $n_b$ 步压缩一次；低秩因子只对新 token 重算 | 分组 scale 跟随所选主干（KCVT 或 KIVI） |
| A08 KVTuner | **层×K/V 位宽对离线搜索，在线直接加载**，decode 不再改历史位宽 | 无逐 token 位宽决策 | 底层仍是 KIVI 或 per-token 非对称量化，scale 仍按该量化器计算 |
| A09 GTA/GLA | 结构在训练时固定 | 无 | 无 KV 量化参数。RoPE 只作用于 Key 的一半或独立的半幅投影 |
| A10 TurboQuant | 目标位宽固定。2.5 bit 的例子是 128 维中 32 个通道 3 bit、96 个通道 2 bit | 通道划分是固定规则，不按样本重选 | 码本按 Beta 分布离线 Lloyd-Max，与数据无关。每向量另存 $L_2$ 范数。旋转是数据无关的随机旋转 |

和本课题「layer × 路径 × K/V 静态位宽，decode 不改历史页精度」最接近的是 **KVTuner 的位宽对**。KIVI / SKVQ / GEAR 的位宽档是静态的，但最近窗口的高精度成员在滑动。ZipCache 会在运行时改历史 token 的位宽并重打包。

---

---

## 3. 分组、scale、异常值和高精度残留占多少空间？

下列比例都带条件。序列变长时，固定长度的 FP 窗口占比下降；按通道的 scale 占比也下降；按 token 的 scale、逐 token 位宽标签和稀疏索引不下降。

**KIVI。** 默认组大小 $G=32$，残留长度 $R=128$，且 $R$ 须整除意义下与 $G$ 对齐（$R$ 可被组划分）。Key 与 Value 各自最多 $R$ 个 token 为全精度。GSM8K、Llama-2-13B：$G\in\{32,64\}$ 分数接近（20.77 与 21.00），$G=128$ 降到 17.29；$R\in\{32,96,128\}$ 接近，$R=64$ 为 19.86。正文写长上下文下这段 FP 相对可忽略，**没有**把 scale 与零点算进平均位宽。非对称量化每个组有一个 scale 和一个零点。SKVQ 用同一存储假设给出：2 bit、组 32、FP16 的 scale 与零点时，平均位宽为

$$
2+\frac{16\times 2}{32}=3
$$

改用 FP8(E4M3) 则为 $2+8\times 2/32=2.5$。这是 SKVQ 的公式，可用来估计 KIVI 的元数据，不能写成 KIVI 论文已经报告的平均位宽。

**SKVQ。** 主实验平均组大小 128（重排后各组不等长）、窗口 128、K/V 均为 2 bit。表 4 在 KV 2 bit、窗口 128 下给出平均位宽：组 128 为 2.125 bit，组 64 为 2.25 bit，组 32 为 2.5 bit（GovReport 与 MultiFieldQA-zh 的平均分分别为 35.365、35.805、36.51）。通道置换矩阵融合进 $W_k,W_v$，推理时不另存完整置换矩阵；图 1 的存储统计包含量化参数和重排索引。5 个 sink token 为 FP16，相对长序列很小。

**KVQuant。** 主方法不做细分组。附录 M：整数量化假定低精度零点加 16 bit scale；NUQ 的零点与偏移各 16 bit。稀疏矩阵用 32 bit 的 per-token 索引（为了长序列），元素值和元素级索引各 16 bit。因此 1% 异常值的空间不是「1% 个 FP16」。LLaMA-7B、序列 128K 的 KV 容量：FP16 64.0 GB；nuq4 16.0 GB，nuq4-1% 17.3 GB；nuq3 12.0 GB，nuq3-1% 13.3 GB；nuq2 8.0 GB，nuq2-1% 9.3 GB。128K 上的平均位宽：nuq4 为 4.00–4.02，加 1% 异常值为 4.32–4.35；nuq3-1% 为 3.32–3.35；nuq2-1% 为 2.32–2.35。32K passkey 表里 nuq4-1% / nuq3-1% / nuq2-1% 写为 4.33 / 3.33 / 2.33 bit。首 token FP16 另计，短序列上占比高于长序列。

**ZipCache。** 记张量形状为 $b\times h\times l\times d$，组大小 $n$。全精度参数个数：逐 token 为 $2bl$；细分组为 $2bhld/n$（文中表 1 把 K 与 V 都分组时写成 $4bhld/n$）。通道可分离方案的参数个数为 $hd+2bl$。他们采用的配置是 Key 逐通道、Value 通道可分离后再逐 token，参数个数 $3hd+2bl$。表 1 的压缩比按 $b=8$、$hd=l=4096$、$n=32$、4 bit 计算：细分组 $3.2\times$，该配置 $4.00\times$（LLaMA3-8B，GSM8K 准确率 54.74%，细分组 54.51%）。显著 token 实验把一部分 token 放到 4 bit、其余 2 bit，压缩比随显著比例变化；GSM8K 平均输入长度取 $l=840$ 时，Mistral-7B 上 60% 显著、4/2 bit 的压缩比为 $4.98\times$。位宽标签本身是逐 token 的，长度增加时标签存储不消失。

**GEAR。** 表内 「KV size」 是相对 FP16 的剩余比例，已包含主干、低秩和稀疏。CoT 任务上，2 bit、KIVI 组 64、$s=2\%$、$r=4$ 的 GEAR 剩余 27.6%；4 bit KCVT 主干、$s=2\%$、$r=4$ 剩余 31.0%。低秩按 head：$A_h\in\mathbb{R}^{n\times r}$，$B_h\in\mathbb{R}^{d_H\times r}$。例子：$n=1024$、$d_H=128$、$r=4$。稀疏是每个通道（Key）或每个 token（Value）两侧各抽出 $s/2$ 比例的极值，用稀疏矩阵存 FP 值。流式缓冲 $n_b=20$ 的新 token 为全精度；KIVI 式残留必须是组大小的倍数，GEAR 用粗粒度 KCVT 时缓冲长度可以不等于组大小。2 bit 主干仍退回 KIVI，$G=64$、残留 64。

**MiniCache。** 全量 FP16 KV 记为 $4brh(s+n)$（$r$ 层数，$h$ 隐藏维，$s$ 输入长度，$n$ 输出长度）。前一半层不合并，占 $2brh(s+n)$；后一半每两层存一份共享状态，占 $brh(s+n)$。恢复项：合并层的幅度标量 $2br(s+n)$，以及按 $\gamma=0.05$ 留下的 token。附录把留下部分算成 $0.1brh(s+n)$（两个相邻层都留）。合计

$$
br(s+n)(3.1h+2)
$$

相对 $4brh(s+n)$，隐藏维 4096 时剩余约 $3.1/4=77.5\%$，再加 $2/(4h)$ 的标量项。这是合并本身，不含量化。与 4 bit 量化合用时，正文报告 ShareGPT 上最高 $5.02\times$、显存降 41%、吞吐约 $5\times$（LLaMA-2-7B，A100 80GB，平均输入 161、输出 338）。LongBench 表里单独的 KIVI-2 行压缩比为 $3.95\times$，MiniCache 行是 $5.02\times$；表题写 MiniCache 建在 4 bit KIVI 之上。两行不要当成同一量化配置。

**TurboQuant。** 位宽 $b$ 表示每个坐标平均 $b$ bit，另存每向量 FP 范数以便反量化后缩放。2.5 bit 的通道分裂：$32\times 3+96\times 2$ 再除以 128。3.5 bit 是另一组通道比例，正文没有写出第二个比例的具体通道数。码本按位宽预计算，不随 token 数增长。QJL 残差占用目标位宽中的 1 bit，已经含在 $b$ 里，不是额外的 FP 残差区。

**GQA / GTA / GLA。** 节省的是 KV head 个数或 K 与 V 是否分存，不是 scale。GQA 把 KV 头数从 $H$ 降到组数 $G$。GTA 相对同组数 GQA 把 $m_{kv}$ 从 2 降到 1，KV 字节大约减半；另外还有一个广播到所有组的半幅 RoPE Key，形状为 $B\times L\times 1\times d_h/2$。GLA-2 的潜变量总宽与 MLA 的 $4d_h$ 相同，但分成两个 $2d_h$ 头，TP $\ge 2$ 时每设备缓存可减半。

---

## 4. 读出后要做哪些解码、反量化或重建？

| 编号 | 读出后的运算 |
|---|---|
| A02 KIVI | 分组反量化与矩阵乘在 tile 内融合（文中的 Q_MatMul）：$\hat x=(q-z)\cdot s$。Key 的分组部分与 FP 残留分别与 query 相乘再拼接。Value 同样分量化段与 FP 队列。Prefill 当下层仍走全精度激活，写入缓存的是量化结果。 |
| A03 SKVQ | 算法 1 先对已存缓存 `dequant`，与新 token 拼接，再按离线置换重排后做 $QK^\top$ 与 $AV$。滑出窗口的 token 做裁剪量化：用离线 $\alpha$ 缩放该组 min/max，再量化；mask 命中的 sink 保持原值。置换已融入 $W_k,W_v,W_o$，注意力内仍有一次与分组对齐的重排。 |
| A04 KVQuant | 4 bit 元素作查找表下标，取出 FP16 重建值。Key 在反量化之后、与 query 相乘之前按元素形式补 RoPE。异常值走 CSR 或 CSC 的稀疏乘，再与稠密低比特结果合并。Value 写入前在线做 top-k 以定异常阈值。 |
| A05 ZipCache | 通道归一化 $X_i\leftarrow X_i/c_i$，量化，读出后把 $c_i$ 乘回，再做 token 维的 scale / 零点反量化。显著度用探针 query 的注意力（默认最近 5% 加随机 5%）近似，其余 token 可走 FlashAttention。每 100 个新 token 重压一次，读路径要能识别该 token 当前是 4 bit 还是 2 bit。 |
| A06 MiniCache | 合并时对相邻层做归一化、夹角和 SLERP（$\sin$、$\arccos$）。解码前按该层幅度把共享方向向量缩放回原范数，再按索引把未合并 token 写回。两层不能直接共用一份已还原向量。 |
| A07 GEAR | 重建为量化主干 $D$、低秩 $L=\mathrm{Concat}(A_h B_h^\top)$ 与稀疏 $S$ 之和。低秩前向先算 $q_h^\top B_h$，再乘 $A_h^\top$。稀疏是按索引的 FP 修正。反量化与矩阵乘可融合。缓冲未满时新 token 直接以 FP 参与注意力。 |
| A08 KVTuner | 在线没有额外的位宽搜索。读出运算等于底层量化器：KIVI 的融合反量化，或 per-token 非对称的 $(q-z)\cdot s$。不同层的 K/V 位宽不同，反量化数据通路要按层切换位宽。 |
| A09 GTA | 从内存加载一份 tied KV。Value 用全维；Key 取前一半，与广播的 RoPE 半幅拼接。没有量化解码。GLA 在解码时对潜变量直接做注意力，上投影吸收进 query / 输出矩阵；TP 上再 AllReduce。 |
| A10 TurboQuant | 反量化 $Q^{-1}$：码本把整数坐标映回实数，乘回存储的 $L_2$ 范数，并施加旋转的逆（或在 query 侧吸收同一旋转）。内积版还要把 1 bit QJL 残差估计加回。作者强调可向量化；残差校正是第二段读和加，不是一次 scale。 |
| A01 GQA | 无反量化。一个 KV head 广播到组内多个 query head。 |

---

## 5. 这些格式是否适合连续传输、bank 划分和并行计算？

这里的「适合」指：历史 KV 按 token 顺序追加、突发传输时有效载荷连续、同一 bank 内位宽与地址规则固定、反量化能并进 MAC 流水。判断如下。

**适合规则连续流的部分**

- KVTuner 的层位宽在服务期间不变，历史页不必重打包。同一层内 K 与 V 可以是不同位宽，但各自沿序列均匀。
- KIVI 的 Value、以及 SKVQ 滑出窗口之后的历史段，都是沿 token 追加的均匀低比特。Key 的 per-channel 组在组闭合后也沿序列追加。
- KVQuant 的 Key 用离线 per-channel scale，新 token 只追加低比特元素，不必重写旧 scale。稠密 NUQ 数组本身规则。
- TurboQuant 逐向量、数据无关，码本全局共享，适合在线追加。
- GQA 的 KV head 组是固定的连续张量，组内 query 广播，不引入逐 token 间接寻址。

**会打断连续突发或固定 bank 的部分**

- KIVI / SKVQ / GEAR 的 FP 窗口与低比特历史是两种宽度。窗口边界每个 decode 步移动，物理上要么保留一块 FP 区并把滑出 token 转码，要么接受页内混合位宽。
- Key 的 per-channel 组要攒满 $G$ 个 token 才能量化。未满的残留不能并进已打包的历史突发。
- SKVQ 重排后各组长度不相等，按通道打包时组边界不对齐。
- KVQuant 的 1% 异常值是 CSR/CSC。追加 token 会改稀疏行或列，不能和稠密 4 bit 数组做成同一条对齐突发。RoPE 在反量化后对通道对做交叉，通道维打包要保证一对通道同拍可读。
- ZipCache 每 100 token 改一批历史 token 的 4/2 bit 归属，页内位宽随时间变化，和「历史页精度静态」冲突。逐 token 标签也使 bank 步长不恒定。
- GEAR 的低秩因子长度随 $n$ 增长，稀疏索引不规则；重建是三段数据相加，不是读出即 MAC。
- MiniCache 的保留 token 靠索引散射写回，共享状态与保留状态地址不连续。SLERP 用三角函数，不适合并进整数 MAC。
- GLA 的潜变量可按 head 分到不同设备且避免整份复制，这是并行划分上的优点；它改变的是模型结构，不是在已有 GQA 权重上的存储格式。

**并行计算**

- KIVI、GEAR 已把反量化融合进分块矩阵乘，这是数字阵列可沿用的形式：scale 广播，整数乘加。
- KVQuant 认为 KV 加载受带宽约束，非均匀查表的额外计算不一定增加延迟。该论点依赖 GPU 上查表相对访存很便宜；ASIC 上查表 ROM、稀疏指针和 RoPE 都要单独计入节拍。
- GTA/GLA 的系统优化说明：分页 KV 使连续块加载变难，64 bit 地址计算本身很贵，需要多线程分摊。规则页比逐元素 gather 更适合他们的异步拷贝。
- 层间不同位宽（KVTuner）可以按层切换 MAC 位宽，不必在一个 tile 内混合 2/4/8 bit。Head 间不同位宽会在 GQA 共享组内把同一 KV 的多个消费者拆开，A 类论文没有把位宽配到 query head。

---

## 6. 逐篇要点

每篇只保留与第 1–5 节直接相关的机制、实验范围和失效条件。

### A01 GQA

结构输入，不是量化。Query head 分成 $G$ 组，每组共享一个 Key head 和一个 Value head。GQA-1 即 MQA，GQA-$H$ 即 MHA。从 MHA 检查点转换时，组内原来的 K/V 投影做均值池化，再用约 5% 原预训练步数 uptraining。KV 缓存与每步加载量按 head 数从 $H$ 降到 $G$。大模型按 head 切分时，MQA 的单份 KV 会被复制到每个分片；GQA 用「KV head 数不少于分片数」避免这份复制。

实验是 T5.1.1 Large / XXL，解码器自注意力和交叉注意力，编码器自注意力不变。任务为 CNN/DailyMail、arXiv、PubMed、MediaSum、Multi-News、WMT14 En-De、TriviaQA。输入长度 512 或 2048，输出 256、512 或 32。8 块 TPUv4，每块 batch 不超过 32。GQA-8-XXL 推理时间 0.28 s/样本，MQA-XXL 0.24，MHA-XXL 1.51。没有 K/V 位宽、scale 或反量化。

### A02 KIVI

量化轴是全文的核心证据，见第 1 节的误差表。算法把 Key 每 $G$ 个 token 分成一组，沿通道量化；组未满的后缀留在 FP16。Value 用长度 $R$ 的 FP 队列，挤出的 token 按 token 量化后追加。$R\le 128$，实验取 $R=128$、$G=32$。融合反量化矩阵乘用 CUDA，组量化核用 Triton，可与 weight-only 量化同时使用。

效率负载是 ShareGPT，平均输入 161、输出 338，Llama-2-7B，单卡 A100 80GB。相对 FP16，峰值内存相近时 batch 最大约 $4\times$，吞吐 $2.35\times$–$3.47\times$。这是短上下文服务负载，不是 32K–128K 的长上下文测量。

### A03 SKVQ

两段机制。第一段：按通道统计做 KMeans，把相似通道分到一组，并用等价置换融合进投影，使 $QK^\top$ 与 $AVW_o$ 不必在运行时显式乘置换矩阵。组内仍有离群点，于是每组一个裁剪系数 $\alpha\in(0,1]$，离线最小化注意力输出 MSE，全 token 共享该 $\alpha$。第二段：最近 $w$ 个 token 保持全精度；可选过滤器决定滑出窗口后是否继续高精度。实验启用的过滤器是前 5 个 attention sink。作者试过把 heavy hitter 留在高精度，但收益不大，而且 FlashAttention 不易取出全部分数，所以没有放进主实验。

校准：WikiText-2 训练集 256 条、长度 4096。LongBench 主表：平均组 128、K2V2、窗口 128。更低比特为 Key 2 bit、Value 1.5 bit、组 64。NIAH：Llama2-7B-80k，1k–32k。容量分析写 7B 模型在单张 A100 80GB 上可放到 1M 上下文；batch 128、序列 200k 的解码加速是理论值 $7\times$，放在附录，不是实测长序列质量。

### A04 KVQuant

四件事叠在一起：Key 沿通道、Value 沿 token；Key 在 RoPE 之前量化；每层用 Fisher 信息加权的 k-means 得到非均匀码本（nuqX）；按量化向量单独取异常阈值，1% 数值存 FP16 稀疏矩阵。Per-vector 阈值与 per-matrix 阈值内存相同，但 LLaMA-7B、3 bit、Wikitext-2 上困惑度从 5.85 降到 5.75（都是 1% 异常值）。

离线与在线的分工写得很明确。Per-channel scale 若在线更新，每来一个 token 就要改该通道的 scale 并重写历史 Key，因此 Key 用校准集。16 条长度 2K 的 WikiText-2。去掉 1% 异常值后，Key 的离线 scale 与在线 scale 困惑度同为 5.75；不做异常值分离时离线为 5.94、在线为 5.91。Value 的离群 token 使离线 scale 困难，所以 scale 和阈值在线算。LLaMA-7B 上 1% top-k 在 CPU 为 0.026 ms，QKV 投影 0.172 ms，两者重叠后 0.173 ms。

核只实现了 4 bit：查表、稀疏乘、Key 侧在线 RoPE。A6000、batch 1、LLaMA-2-7B-32K，序列 2K/4K/16K 的 Key 乘加延迟从 FP16 的 33.3/59.1/219.4 µs 降到 25.6/39.9/126.3 µs；Value 从 26.0/50.2/203.7 µs 降到 22.1/37.9/124.5 µs。RULER 32K 上，KVQuant-3bit-1% 平均 53.65，KIVI-2（组 32、残留 128，平均约 3.05 bit）为 39.78，FP16 为 56.40。作者把差距归因于 KIVI 把 FP16 留给局部窗，而 KVQuant 对全部 token 一视同仁。这是 32K 检索任务上的比较，不是对局部窗在 GSM8K 上作用的否定。

### A05 ZipCache

通道与 token 拆开，是为了少存量化参数，同时挡住通道离群点。Key 的 token 间差异小，所以 Key 只做通道缩放；Value 既有通道离群又有 token 差异，所以先通道缩放再逐 token 量化。文中通道因子为该通道绝对值最大元的平方根。

显著度不用累积注意力。累积分数偏向序列开头，因为下三角前面的 token 被加的次数更多，且短行的 softmax 更大。归一化分数是该列之和除以该列非零个数。探针默认最近 5% 加随机 5%。LLaMA3-8B、GSM8K、40% token 为 4 bit、60% 为 2 bit、探针占 10% 时：全 token 探针 52.54%，最近 token 51.10%，最近加随机 52.08%，随机 47.46%，特殊 token 46.78%。

生成设置：显著 token 4 bit，其余 2 bit，Key 通道量化，Value 通道可分离逐 token。每 100 个新 token 重压。模型为 Mistral、LLaMA2、LLaMA3。任务为 GSM8K CoT、HumanEval、Line Retrieval。摘要中的效率数字：LLaMA3-8B、输入 4096，prefill 延迟降 37.3%，decode 延迟降 56.9%，GPU 内存降 19.8%。GSM8K 上 Mistral-7B 从 41.62% 到 41.24%，压缩 $4.98\times$。这些长度是短上下文或检索探针，不是 32K–128K 的矩阵。

### A06 MiniCache

压缩的是深度，不是位宽。中后层相邻 KV 角距离小，从 $L/2$ 起把相邻层合成一份方向向量，并保留两层各自的 $\ell_2$ 范数和夹角。直接平均会损失离群通道上的幅度，所以还原时按范数重新缩放。$\gamma=0.05$ 以下的角距离阈值内的 token 两层都原样保留。$t=0.6$ 在 LLaMA-3-8B、从第 16 层起合并的消融里优于 $t=0.5$。

还原发生在每层解码之前，共享存储不等于两层可以读同一份向量进 MAC。与 KIVI 正交：量化负责元素位宽，合并负责少存一层。显存与吞吐数字见第 3 节。

### A07 GEAR

量化残差再拆成低秩部分和稀疏部分。低秩捕捉跨 token 共享的残差主方向，稀疏拾取各组里的极值。只做低秩的版本称为 GEAR-L。残差谱在 LLaMA2-7B、一条 GSM8K 样本的第一层 Value 上快速下降，因此 $r=4$ 被当作足够。

主干：4 bit 用无细分组的 KCVT（Key 整通道一组，Value 整 token 一组）；2 bit 时 KCVT 不够，改回 KIVI，$G=64$、残留 64。$s=2\%$ 指每个向量两侧极值的比例。流式缓冲 $n_b=20$，decode 时低秩只对新 token、且 $r=2$。

模型：LLaMA2-7B/13B、Mistral-7B、LLaMA3-8B。困难生成是 GSM8K、AQuA、BBH 的 8-shot CoT，平均 prefill 约 900、1304、1021，生成 256 token。较易设置是 LongBench（平均输入 3642）和 GSM8K 5-shot。效率：LLaMA2-7B，输入 1000、生成 500，V100 16GB，权重量到 8 bit 后 FP16 KV 的最大 batch 为 3，GEAR 为 18，吞吐最高 $5.07\times$。时间分解认为低秩和稀疏不是主项，主项仍是模型前向。该分解是这条短序列、大 batch 设置，不是长上下文单请求。

### A08 KVTuner

动机是在线逐 token 选位宽难以接入 FlashAttention 和静态图。因此只搜索粗粒度对象：一整层的 Key 与 Value 各取 $\{2,4,8\}$ bit。候选 $9^{L}$。Llama-3.1-8B 为 32 层，约 $3.4\times 10^{30}$。层内按「等价位宽–注意力输出误差」剪掉非 Pareto 对，多数层剩下 5 个；再按误差把层聚类，例如剪到 $5^{6}=15625$。在线只加载结果。

误差沿层和沿生成步同时累积。文中例子：Llama-2-13B-chat、GSM8K 15-shot，KIVI-2 把一个减号翻成加号，$20-4-4$ 变成 $20+4+4$，最终答案错误。校准故意在 prefill 就用反量化后的 KV，让层间误差显现，并用数学推理这种容易翻号的任务。

报告的工作点：Llama-3.1-8B-Instruct 上 KIVI 模式的 KVTuner-C3.25，GSM8K 平均 0.7925，BF16 为 0.8038，KIVI-4 为 0.8011。Qwen2.5-7B-Instruct 更敏感，均匀 KV4 在 per-token 模式下接近 0 分，KVTuner 用约 4 bit 等价位宽保持接近 KV8。吞吐相对 KIVI-KV8 最高 +21.25%，正文说覆盖多种上下文长度；本文没有逐项核对其附录里的长度列表。

### A09 GTA / GLA

问题定义是解码时每加载 1 字节做多少 FLOP。BF16 下单个 query token 对 KV 的算术强度约为每字节 1 次 MAC 量级，远低于 H100 的约 295 FLOP/byte。组大小 $g_q=h_q/h_{kv}$ 把强度提高到约 $2g_q/m_{kv}$（$L\gg h_q$）。$m_{kv}=2$ 表示 K 与 V 分开存，$m_{kv}=1$ 表示同一状态兼作 K 和 V。

GTA：一份 tied KV。Value 用全部 $d_h$；Key 的前 $d_h/2$ 不旋转，另一半来自单独的单 head RoPE 投影并广播。相对同组数 GQA，缓存约减半、算术强度约加倍。依据是 Key 在 RoPE 前更低秩，以及只旋转一部分通道仍够用。这些是结构论文中的引用观察，本篇的质量证据来自自己训练的模型。

GLA：潜变量分成 $h_c$ 个头，每头宽 $2d_h$。GLA-2 与 MLA 的总缓存同为 $4d_h$，但 TP $\ge 2$ 时不必每卡复制整份潜变量。解码时上投影吸收进 $W^Q$ 与 $W^O$，注意力直接打在潜变量上。$L_q=1$ 时双潜头的算术强度低于单潜头 MLA；$L_q\ge 2$（例如投机解码的查询长度）时 GLA 仍可能更快，因为 MLA 先顶到计算屋顶。

质量数字是小模型：XL 1.47B 上 GTA 困惑度 10.12、GQA 10.20；GLA 下游平均 60.0%、困惑度 10.21，MLA 为 59.1% 与 10.25。没有现成 7B–14B GQA 模型上的量化实验。分页 KV 与 warp 分工、软件流水见第 5 节与第 7 节第 11 条，说明连续块加载比逐元素地址计算更适合他们的异步拷贝。

### A10 TurboQuant

随机旋转后，单位球上的坐标服从缩放 Beta 分布，高维下接近 $\mathcal{N}(0,1/d)$，坐标近乎独立，于是每坐标用预计算的 Lloyd-Max 标量量化。MSE 最优量化对内积有偏，且偏差随内积变大。内积版用 $b-1$ bit 的 MSE 量化，再对残差做 1 bit QJL，估计无偏。MSE 与信息论下界相差不超过约 $2.7$ 倍；1 bit 时约 $1.45$ 倍。单位范数不成立时，存 FP 范数，反量化后乘回。

KV 实验把通道分成离群与非离群，两套 TurboQuant。2.5 bit 的 32/96 分裂见第 3 节。NIAH 与 LongBench 见第 1 节。压缩比 0.25 的 NIAH 对照包括 SnapKV 0.858、PyramidKV 0.895、KIVI 0.981、PolarQuant 0.995、全精度 0.997、TurboQuant 0.997。LongBench 上 Llama-3.1-8B-Instruct：全精度平均 50.06，TurboQuant 3.5 bit 为 50.06，2.5 bit 为 49.44，KIVI 3 bit 为 48.50。作者写 TurboQuant 在生成过程中也量化新 token，而 KIVI 与 PolarQuant 留下未量化的生成 token。这是实验协议差异，比较平均分时要一起看。

---

## 7. 其他观点

这些观察与上面五个问题相关，但不是那五个问题的直接答案。

1. **Value 的分组选择不是因为 Value 没有离群点。** KIVI 的 Value 重构误差在两种分组下接近，注意力输出误差差一个数量级。KVQuant 附录 G 的解释与之一致：per-channel 的 Value 误差会堆到输出向量的固定通道上，并传到后面的层。

2. **局部 FP 窗口和「所有 token 同等精度」解决的是不同误差。** KIVI 用窗口保住 GSM8K 这种逐步推演；KVQuant 在 RULER 32K 上认为窗口保护不了远距离检索。两项结果都成立，不能用其中一个取消另一个。

3. **Key 的量化轴会改变「Key 更重要」这一排序。** KVTuner 表 4：per-token 与 per-channel 下，第 0 层的 Pareto 集合不同，有的层保留 K4V8 或 K2V4。位宽搜索必须绑定量化轴。

4. **层敏感可以离线测，而且作者声称对提示不敏感。** 证据是 Llama-3.1-8B-Instruct、Qwen2.5-7B-Instruct、Mistral-7B-Instruct-v0.3 上，层误差曲线的形状随 Key 位宽变化，不随他们所试的提示变化。这支持静态层配置，还不等于在 RULER 的 128K 上已经验证过同一组层位宽。

5. **非稀疏 head 是低比特的难点。** KVTuner 明确说 KIVI 的静态前缀加最近块盖不住 Qwen2.5 的检索 head。这和本课题把 head 分成流式路径与检索路径是同一现象，但是 KVTuner 的应对是提高整层位宽，不是给检索 head 另一套访问预算。

6. **RoPE 与量化的顺序是存储格式的一部分。** Pre-RoPE 让通道 scale 稳定，代价是每次使用 Key 都要在反量化后做旋转。Post-RoPE 可以直接点积，但通道统计被旋转打乱。GTA 采用第三种：只旋转单独的半幅，tied 的一半永不旋转，避免「旋转后再逆旋转供 Value 使用」。

7. **异常值是相对的。** SKVQ 认为最大通道相对中等通道是离群，中等通道相对小通道也是离群，所以用分组而不是单独摘出 1% 通道。KVQuant 则在每个量化向量内部摘数值极值。两种「异常值」不是同一存储对象。

8. **MSE 小不等于注意力内积无偏。** TurboQuant 把这件事写成量化器设计：残差的 1 bit 是为了消偏，不是为了再降一点 MSE。数字实现若只做重建误差最小的均匀量化，内积偏差仍然存在。

9. **跨层可共享的是中后层方向，不是浅层，也不是全部 token。** MiniCache 的还原依赖幅度和索引。共享存储之后，两层的 MAC 输入仍要分别装配。

10. **把 K 与 V 绑成一次加载，算术强度大约翻倍。** 这是 GTA 相对 GQA 的全部额外压缩来源之一，发生在表示维度，不依赖低比特。它要求模型按此结构训练或转换，不能直接套到已有的独立 K、V 投影上。

11. **分页之后，地址计算会成为加载开销。** GLA/GTA 的实现用多线程合作算页地址，再用异步拷贝重叠矩阵乘。规则、等长、等宽的页比位宽随 token 变化的页更符合这条流水。

---

## 8. 对本课题的直接含义

下面只写可以沿用的判断，以及这些论文没有在本课题工作点上测过的外推。

这些格式判断供静态 INT4/INT8 与页式访问使用。

### 可以写入课题模型的

- 静态配置应落在 **层 × K/V**，必要时再加路径。KVTuner 说明层间位宽差是稳定的；它没有证明同一层里每个 head 都需要独立位宽。检索 head 更敏感这一观察，可以用「整层抬高 Key 位宽」或「检索路径单独一条位宽」来吸收，二者的存储对齐成本不同。
- Key 用通道组、Value 用 token 组时，历史 Key 的打包粒度是 $G$ 个 token。$G=32$ 或 $64$ 在 KIVI 的 GSM8K 上可用，$G=128$ 变差。页内 token 数需要是这个组的整数倍，否则组 scale 跨页，或者页尾留下 FP 残留。
- 最近窗口的 FP16 是质量项，不是免费对齐填充。它的长度在 KIVI 里最多 $R$ 个 token、在 SKVQ 里是 $w$ 加少数 sink。若核心格式只允许 INT4/INT8，这块区域要么显式做成第二种静态精度，要么接受这些论文在困难生成上的掉点。
- 1% 稀疏异常值、逐 token 4/2 bit 标签、跨层保留索引，都不适合作为核心页的连续突发。它们的索引宽度在 KVQuant 里按 32 bit 设计，长序列下不可忽略。
- 读出后的最低运算是 per-group $(q-z)\cdot s$，并可与 MAC 融合。Pre-RoPE、查表、低秩外积、SLERP 和稀疏合并都是额外流水级，不能算进「反量化很便宜」。
- GQA 只减少 KV head 份数。压缩比必须按 KV head 而不是 query head 计算。GTA/GLA 的进一步减半来自 K=V 或潜变量，属于结构扩展，不是本课题核心权重上的精度映射。

### 不能写成已经证实的

- KVQuant 的 10M 与 SKVQ 的 1M 是容量估计。任务分数停在各自论文实际跑过的长度上。
- 层误差曲线不随所试提示变化，支持静态层配置；这还不等于同一组层位宽已经在 RULER 的 128K 上验证过。
- KVTuner 把检索 head 的敏感吸收为整层更高位宽，没有给检索 head 单独的访问预算，也没有把位宽配到 query head。
- GTA/GLA 的质量数字来自自训的 433M–1.47B 模型。没有现成 7B–14B GQA 模型上的量化实验，不能把 K=V 或潜变量减半写成已有独立 K、V 投影上的精度映射。
