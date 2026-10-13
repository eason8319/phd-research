---
experiment: representation_interactions
results: research/phase1/experiments/representation_interactions/results
reviewed: true
---

# 位宽与表示交互

本文回答 K8V4 相对 K4V8 的局部输出误差优势是否稳定，以及分数扰动如何经过注意力概率和 Value 影响输出。更新日期为 2026-10-13。测量对象是固定 query 的逐层注意力输出。

**来源说明**

- 指导与配置：`research/phase1/guide.md`、`research/phase1/configs/data.md`、`research/phase1/configs/quant.md` 和 `research/phase1/configs/representation_diagnostics.json` 规定输入、量化格式与配对指标。
- 执行代码：`research/phase1/experiments/representation_interactions/code/`；`evaluate.py` 负责捕获与评估，`metrics.py` 定义指标，`results.py` 审核并导出统计。
- 正式结果与审核：结果根目录见文首 YAML。每模型的 `summary.json` 提供“冻结主矩阵”“Value 分组对照”和“配对传播诊断”，并保存来源与审核记录；`samples.json` 保存 96 题 × 4 条件的 384 条题目级配对评分。
- 详细记录：完整逐层数据留在服务器，由 `summary.json` 中的 `paired_diagnostic.provenance` 及其 `review.detail_checks` 索引；阶段来源见 `research/phase1/source_index.json`。

## 实验目的

比较相同名义位宽预算的 K8V4 与 K4V8，检查 Key 离线 per-channel／在线 per-token 和 pre／post-RoPE 四种条件。主判据是注意力输出相对平方误差；K/V 增量、概率变化及输出传播预测用于解释共同 KV8 参照下的误差差距。后续实验保留这四种 K8V4，Value 均为在线 per-token，两模型使用同一候选集。

模型为 Qwen2.5-7B-Instruct（下称 Qwen）与 Llama-3.1-8B-Instruct（下称 Llama）。每模型使用冻结 C 集 96 题：64 道数学题、32 道检索题，两折各 48 题。Qwen 有 28 层、28 个 query head；Llama 有 32 层、32 个 query head。离线参数在互补折拟合。每层独立干预，误差不向后传播；query 为高精度轨迹中的首个生成 token，历史含该 token。Value 均采用在线 per-token 配置，Value 分组对照另测离线 per-channel。冻结主矩阵使用 FP32 注意力路径；配对传播诊断读取同一份 FP32 Q 和反量化 K/V，以 FP64 计算分数、概率与诊断输出，高精度参考输出保持 FP32。

## 实验结果与结论

### 位宽分配与表示方式的输出误差

下表来自冻结主矩阵，每组均为 96 题，误差无量纲。每层先以参考输出能量归一化平方误差，每题平均有效层后，再对两折等权汇总。

| 模型 | Key 方式 | RoPE | KV8 | K8V4 | K4V8 | KV4 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Qwen | 离线 per-channel | pre | 0.000134 | 0.005065 | 0.020443 | 0.025403 |
| Qwen | 离线 per-channel | post | 0.000128 | 0.005061 | 0.026363 | 0.031363 |
| Qwen | 在线 per-token | pre | 0.000552 | 0.005494 | 0.079865 | 0.085141 |
| Qwen | 在线 per-token | post | 0.000579 | 0.005521 | 0.082087 | 0.087474 |
| Llama | 离线 per-channel | pre | 0.000088 | 0.005318 | 0.016503 | 0.021876 |
| Llama | 离线 per-channel | post | 0.000104 | 0.005331 | 0.025758 | 0.031243 |
| Llama | 在线 per-token | pre | 0.000066 | 0.005307 | 0.013928 | 0.019287 |
| Llama | 在线 per-token | post | 0.000060 | 0.005296 | 0.013198 | 0.018529 |

八组中 K8V4 的输出误差均小于 K4V8，KV8 的误差则均小于这两种混合位宽方案。K8V4 的误差在不同表示条件下变化较小，K4V8 的变化更明显：Qwen 在线 per-token 的误差高于离线 per-channel，Llama 的比较方向相反；pre/post-RoPE 也改变差距大小，但没有改变 K8V4 与 K4V8 的排序。这表明表示条件影响降位宽的失真程度，影响幅度随模型变化。

在当前相同名义位宽预算的比较中，结果支持在 6 bit 档优先保留 K8。后续实验使用 Key 离线 per-channel／在线 per-token 与 pre／post-RoPE 交叉形成的四种 K8V4，Value 均为在线 per-token。KV8 和 KV4 继续作为均匀基线，K4V8 及其他 KV4 格保留为本实验的数值对照。

### Value 分组方式的对照

下表为冻结的 Value 分组对照，每模型 96 题，误差无量纲，汇总口径与主矩阵相同。两列均采用 KV4、Key 离线 per-channel、pre-RoPE，仅改变 Value 的分组方式与参数来源。

| 模型 | Value 在线 per-token | Value 离线 per-channel |
| --- | ---: | ---: |
| Qwen | 0.025403 | 0.116655 |
| Llama | 0.021876 | 0.464878 |

两个模型改用 Value 离线 per-channel 后，输出误差均增大。当前实验的 Value 采用在线 per-token。

### 输出误差差异的传播分析

主矩阵已给出 K8V4 的输出误差小于 K4V8 的结果。以下以共同 KV8 参照分离两路降位宽增量，再结合分数、概率与输出的实测指标，分析这一差距的传播路径及其逐层分布。

令 $o_{88}$、$o_{48}$、$o_{84}$ 分别为同一次捕获下的 K8V8、K4V8、K8V4 输出，$o_{\mathrm{ref}}$ 为高精度参考。记 $\delta s=s_4-s_8$、$p_8=\operatorname{softmax}(s_8)$、$p_4=\operatorname{softmax}(s_4)$。对单个 query head，固定 V8 或固定 K8 时的输出增量分别为

$$
\delta o_K=o_{48}-o_{88}=V_8^\top(p_4-p_8),\qquad
\delta o_V=o_{84}-o_{88}=(V_4-V_8)^\top p_8.
$$

K 降位宽通过注意力概率的重新分配改变输出，V 降位宽则在同一组概率下改变被加权的内容。整层输出由各 head 拼接，以共同参考能量定义

$$
D_K=\frac{\|\delta o_K\|^2}{\|o_{\mathrm{ref}}\|^2},\qquad
D_V=\frac{\|\delta o_V\|^2}{\|o_{\mathrm{ref}}\|^2}.
$$

记 $E_{48}=\|o_{48}-o_{\mathrm{ref}}\|^2/\|o_{\mathrm{ref}}\|^2$，$E_{84}$ 同理。将总误差写成 KV8 基线误差与增量之和，展开平方范数可得

$$
E_{48}-E_{84}
=D_K-D_V+
\frac{2(o_{88}-o_{\mathrm{ref}})^\top(\delta o_K-\delta o_V)}
{\|o_{\mathrm{ref}}\|^2}.
$$

交叉项把总误差差距分成两路增量与 KV8 基线的交叉贡献。

进一步看 K 路的传播，softmax 的精确关系为

$$
p_{4,i}=\frac{p_{8,i}\exp(\delta s_i)}
{\sum_jp_{8,j}\exp(\delta s_j)}.
$$

所有分数增加相同常数时，概率不变，因此真正影响概率的是位置间的相对分数变化。令 $\mu=\sum_i p_{8,i}\delta s_i$，加权方差 $W=\sum_i p_{8,i}(\delta s_i-\mu)^2$。由上式可得

$$
D_{\mathrm{KL}}(p_8\|p_4)
=\log\sum_i p_{8,i}\exp(\delta s_i-\mu)
\approx \frac{W}{2}.
$$

近似式来自中心化扰动足够小时的二阶展开，加权一阶项为零。softmax 在 $s_8$ 处的雅可比为 $J_8=\operatorname{diag}(p_8)-p_8p_8^\top$，因此小扰动下 $\delta p_i\approx p_{8,i}(\delta s_i-\mu)$。代入输出关系，得到

$$
\widehat{\delta o_K}
=\sum_i p_{8,i}(v_{8,i}-o_{88})\delta s_i,
\qquad o_{88}=\sum_i p_{8,i}v_{8,i}.
$$

该式把分数扰动、概率权重和 Value 相对当前输出的内容差异联系起来。下面用实测残差和余弦检验这一局部展开。

用 $E_{\mathrm{pred}}=\|\widehat{\delta o_K}-\delta o_K\|^2/\|o_{\mathrm{ref}}\|^2$ 和 $\rho=\overline E_{\mathrm{pred}}/\overline D_K$ 衡量汇总预测残差，同时记录预测与实测增量的余弦 $c$。以下输出指标均先合并整层全部 head 的能量再归一化。

下表展示配对传播诊断的输出增量，每组 96 题。每题平均层后，两折等权；比值为两个汇总均值之比。差值均值的 95% 配对 bootstrap 区间在两折内按题重采样，固定种子 8319、2,000 次，设置来源为配对传播诊断配置与正式汇总。

| 模型 | Key 方式 | RoPE | $D_K$ | $D_V$ | $\overline D_K/\overline D_V$ | 正差题数 | $D_K-D_V$ 均值的 95% 区间 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen | 离线 per-channel | pre | 0.019759 | 0.004925 | 4.01 | 96/96 | [0.01445, 0.01521] |
| Qwen | 离线 per-channel | post | 0.025892 | 0.004922 | 5.26 | 96/96 | [0.02037, 0.02156] |
| Qwen | 在线 per-token | pre | 0.079081 | 0.004934 | 16.03 | 96/96 | [0.07125, 0.07684] |
| Qwen | 在线 per-token | post | 0.081775 | 0.004931 | 16.58 | 96/96 | [0.07407, 0.07928] |
| Llama | 离线 per-channel | pre | 0.016397 | 0.005232 | 3.13 | 96/96 | [0.01083, 0.01152] |
| Llama | 离线 per-channel | post | 0.025542 | 0.005230 | 4.88 | 96/96 | [0.01960, 0.02102] |
| Llama | 在线 per-token | pre | 0.013861 | 0.005242 | 2.64 | 96/96 | [0.00823, 0.00904] |
| Llama | 在线 per-token | post | 0.013117 | 0.005240 | 2.50 | 96/96 | [0.00750, 0.00828] |

八组均为 96/96 题 $D_K>D_V$，差值区间均高于零，汇总 K 增量为 V 增量的 2.50–16.58 倍。基线交叉项对汇总总误差差距的绝对贡献均小于 0.65%，逐层分解最大绝对残差低于 $8\times10^{-16}$。K4V8 与 K8V4 的差距主要来自 K 降位宽产生的输出增量。$D_V$ 在四种 Key 条件下变化较小，$D_K$ 的变化更明显，表示条件之间的差距位于 K 路。

下表将概率变化与输出预测放在同一条件下核对，均来自配对传播诊断，每组 96 题。$W$ 与 KL 先按题平均层和 query head，再对两折等权；KL 单位为 nat，$\mathrm{KL}/(W/2)$ 为汇总均值之比。输出指标先按题平均层，再对两折等权，$\rho$ 按归一化 K 增量误差加权。

| 模型 | Key 方式 | RoPE | $W$ | KL（nat） | $\mathrm{KL}/(W/2)$ | $\rho$ | 平均余弦 $c$ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen | 离线 per-channel | pre | 0.049422 | 0.024555 | 0.994 | 0.0496 | 0.9849 |
| Qwen | 离线 per-channel | post | 0.058642 | 0.029258 | 0.998 | 0.0513 | 0.9827 |
| Qwen | 在线 per-token | pre | 0.393991 | 0.272550 | 1.384 | 1.2253 | 0.9212 |
| Qwen | 在线 per-token | post | 0.367382 | 0.260609 | 1.419 | 1.0836 | 0.9216 |
| Llama | 离线 per-channel | pre | 0.023111 | 0.011562 | 1.001 | 0.0268 | 0.9895 |
| Llama | 离线 per-channel | post | 0.028254 | 0.014205 | 1.006 | 0.0281 | 0.9889 |
| Llama | 在线 per-token | pre | 0.015546 | 0.007778 | 1.001 | 0.0235 | 0.9901 |
| Llama | 在线 per-token | post | 0.015389 | 0.007698 | 1.000 | 0.0236 | 0.9900 |

Llama 四组和 Qwen 离线两组的 $\mathrm{KL}/(W/2)$ 为 0.994–1.006，$\rho$ 约为 0.023–0.051，平均余弦约为 0.983–0.990。概率的二阶近似与输出的一阶预测同时接近实测结果，支持在这些条件下，用“相对分数扰动改变权重，再由 Value 内容差异形成输出增量”的局部传播关系解释 K 路失真。

Qwen 在线两组的 $W$ 与 KL 均高于其离线条件，$\mathrm{KL}/(W/2)$ 为 1.384 和 1.419，二阶近似低估了概率变化；$\rho$ 为 1.2253 和 1.0836，一阶输出预测残差大于将增量预测为零的残差。这两组的输出差距用精确 softmax 传播描述。下表列出 Qwen 第 0 与第 27 层，即首尾层的贡献。每组为 96 题 × 28 层，占比按逐层参考输出归一化后的指标之和计算。

| Qwen 在线 RoPE | 首尾层占 $D_K$ | 首尾层占 $E_{\mathrm{pred}}$ | 首层 / 末层平均余弦 | $r^2>1$ 的题目—层数 |
| --- | ---: | ---: | ---: | ---: |
| pre | 69.23% | 94.99% | 0.362 / 0.165 | 194/2688 |
| post | 66.46% | 93.99% | 0.362 / 0.149 | 176/2688 |

其中 $r^2=\|\widehat{\delta o_K}-\delta o_K\|^2/\|\delta o_K\|^2$ 为逐层平方相对残差。首尾层贡献了约 66%–69% 的归一化 K 增量误差和约 94%–95% 的预测残差，其平均余弦低于总体均值。Qwen 在线条件的主要 K 路失真及一阶预测偏差集中在首尾层。

上述测量将 K8V4 的局部误差优势与 K 路的概率重分配及 Value 重加权联系起来：这一路的实测输出增量大于 V 路；六组条件的增量符合局部一阶传播近似，Qwen 在线两组的高失真和传播偏差集中在首尾层。

## 总结

本实验在两模型的四种表示条件下均观察到 K8V4 的局部输出误差小于 K4V8。配对测量将误差差距定位到 K 路的概率重分配与 Value 重加权，Qwen 在线条件的主要失真及一阶预测偏差集中在首尾层。后续两模型共同保留离线 per-channel／在线 per-token × pre／post-RoPE 的四种 K8V4。

下一项“真实生成与误差来源”实验在两个模型上用同一四种候选，与高精度、KV8 和 KV4 对照，每模型共 7 个配置，在短生成与 32K 任务中检验这一误差优势是否迁移，再按任务质量选择进入后续诊断的表示。
