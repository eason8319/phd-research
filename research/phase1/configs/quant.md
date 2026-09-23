# 第一阶段表示与量化配置

本文件只冻结表示的定义、默认数值，以及容量和传输怎么计。哪种表示是主候选、每一步跑哪些格子，见 [../guide.md](../guide.md)。本文件属于 guide.md 所记的协议版本。

## 表示

“静态”分别登记位宽、scale 和存储成员。跨 token 组长记为 $G_t$，单 token 内通道组长记为 $G_c$。

| 表示 | 参数何时确定 | 追加时做什么 |
|---|---|---|
| Key 离线 per-channel | 每层、每 KV head、每通道的 $x_{\min}$、$x_{\max}$ 来自 C 里划定的校准样本，不来自被评估的样本 | 参数在 decode 中固定；新 token 立即量化。记录超界与饱和 |
| 在线 per-token | 每个 token 沿通道分组，参数由该 token 计算 | 旧 token 参数不变。K 和 V 都可以用 |
| Key 在线跨 token | 同一通道上 $G_t$ 个 token 闭合后计算参数 | 闭合前以高精度副本参加 Attention，计入残留。闭合时由该高精度值写入低位宽，不经过另一档低位宽。副本按 $G_t$ 整组滑出，窗口内 token 数在 $R$ 与 $R+G_t-1$ 之间 |
| 近期高精度残留 | 写入低位宽主存时另存一份副本；滑出窗口只丢弃副本 | 主存由高精度值直接量化。三种残留条件下的低位宽主存相同，差异只在是否读取副本 |

每个配置还登记：对称或非对称、量化公式、整数码范围、参数存储类型、舍入、裁剪、饱和、零动态范围、FP8 scale 的下溢与溢出、RoPE 顺序、尾块策略。浮点偏移不称作整数零点。

未闭合的跨 token 组以高精度副本参加 Attention，计入残留的质量、容量和读取。逐 token 表示的残留窗口是可见上下文末尾的 $R$ 个 token，包含 prefill 的末尾；Key 和 Value 一起保留。$H=0$ 不保留这份副本。实验里的 $R$ 与 $H$ 见 guide.md。当前步必须能读到全部可见 KV。在线参数不使用尚未生成的 token。近期残留与序列开头的 sink 不是同一份数据。

## 默认数值

guide.md 的实验表没有另写的项，用这里的值。

- 均匀非对称量化。码为 $0 \ldots 2^b-1$。码用最近偶数舍入，再饱和裁剪。参数的舍入另定，见下。
- 参数为 FP16 scale 加 FP16 浮点偏移，每组按 4 字节计。实现保留 FP16 次正规数。量化与反量化只用存储后的参数，反量化结果为 FP32。
- 记一组的最小值、最大值为 $x_{\min}$、$x_{\max}$。$z_{\mathrm{stored}}$ 是 $x_{\min}$ 向 $-\infty$ 舍入到 FP16 的结果。$s_{\mathrm{real}}=(x_{\max}-z_{\mathrm{stored}})/(2^b-1)$，$s_{\mathrm{stored}}$ 是 $s_{\mathrm{real}}$ 向 $+\infty$ 舍入到 FP16 的结果。$s_{\mathrm{real}}>0$ 时，$s_{\mathrm{stored}}$ 至少为 FP16 最小正次正规数，不并入常量组。

$$
q=\mathrm{clip}\Big(\mathrm{round}\big((x-z_{\mathrm{stored}})/s_{\mathrm{stored}}\big),\,0,\,2^b-1\Big),\qquad
\hat x=q\,s_{\mathrm{stored}}+z_{\mathrm{stored}}.
$$

- $s_{\mathrm{real}}=0$ 时是常量组：$q=0$，$\hat x=z_{\mathrm{stored}}$，并计数。组内有非有限值，或 $z_{\mathrm{stored}}$、$s_{\mathrm{stored}}$ 不是有限 FP16 时，该组无效并计数，不改写成普通数。无效组进入质量分数的方式见 quality.md。
- 离线 per-channel 的 $x_{\min}$、$x_{\max}$ 只来自 C 的校准样本。哪一折或是否用全部 C，按 guide.md 的 E1、E2 分开。主方案不搜索裁剪百分位。
- 在线 per-token 默认 $G_c=32$。离线 per-channel 的 pre-RoPE 与 post-RoPE 分别校准。
- 主物理块 64 token。跨 token 分组只用能容纳完整组的块。尾块保存有效长度。E4 另扫的块长写在 guide.md。
- K/V 分存。INT4 每字节两个码，较小索引在低四位。
- 成本模型默认 128 字节对齐，并标明这是接口假设。

## 三本账

E0 用这三本账核对实现是否可数。E4 用它们判断名义位宽的排序是否等于传输或服务时间的排序。假量化只证明数值，不能填入传输或加速。

1. 驻留容量：KV 载荷、scale、浮点偏移、标签、padding、缓冲和对照中的重复副本。区分峰值与稳态。
2. 片外传输：测量区间内的读写，按新生成 token 数归一化。同一笔写入不按“元素”和“参数”各算一次。
3. 片上访问：缓冲、解包、反量化中间量和参数缓存。不直接加进片外 bytes/token。

名义位宽按元素数加权。容量有效位宽与传输 bytes/token 分列。本阶段没有页选择器，结果标为 dense KV 表示及读出边界。prefill 转换的一次性开销单列。每项结果标明来源：软件实测、布局计数，或带假设的性能模型。

存储和传输按 KV head 计数。误差可以按 query head 细分。GQA 的共享消费者数不重复计入压缩收益。

## 成本度量

INT4/INT8 混合的名义位宽：

$$
\bar b = 4(1-p)+8p = 4+4p.
$$

只计低比特载荷、且每份 KV 对组内 query 被完整复用时，单段理想算术强度（一次 MAC = 2 FLOP）为：

$$
I_{\mathrm{payload}} = \frac{16 g_q}{b}\ \mathrm{FLOP/byte},\qquad g_q = h_q / h_{kv}.
$$

实际强度按实际传输重算。设 K/V 读取字节为 $D_K$、$D_V$。主比较保持 $W_K+W_V=W$：

$$
T_{\mathrm{shared,lb}}=\frac{D_K+D_V}{W},\qquad
T_{\mathrm{split,lb}}=\max\left(\frac{D_K}{W_K},\frac{D_V}{W_V}\right).
$$

这是内存下界。完整服务时间还要计入 QK → softmax → AV 的依赖。没有测量依据时，不把融合反量化写成零额外开销，也不由 GPU kernel 时间推断 ASIC 延迟或功耗。
