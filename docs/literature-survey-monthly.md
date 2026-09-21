# 文献清单：月度检索约定与更新记录

文献分类目录见 [文献检索目录](literature-survey-reading-list.md)，完整书目见 [BibTeX 文件](literature-survey.bib)。本文规定检索、预印本核查、两份文件同步及简要更新记录。新条目并入目录 A–E 与 K，不在本文另建平行清单。不要为执行本文而在仓库里添加检索脚本。

对助手说 **做本月文献更新**。每次更新先读「检索约定」、目录与 BibTeX，再同步修改并核对，最后在本文「更新记录」追加一行；只改目录不算完成。

检索窗口：最近一次实际完成文献检索的记录日期（若缺，则用目录文首「更新至」）至本次核对日，包含边界日期并去重。范围审查或规则调整不推进该窗口。正式出版来源额外回看最近 90 天，补查延迟收录；上次未完成或新增来源的回溯范围记在表格备注中。

---

## 检索约定

归类以目录文首的边界定义为准：**A KV 表示与精度、B KV 保留与访问策略、C Attention 执行架构与映射、D 推理系统与资源管理**；**E** 是基础与评测附录，**K** 是预印本与技术报告的独立列表。按主要贡献确定唯一归属，检索命中的关键词或发表平台不直接决定类别。

### 核对已有预印本 / 技术报告

每月检查 **K 节全部条目**，不受新增文献检索窗口限制：

1. 有 arXiv 标识时打开 `https://arxiv.org/abs/<id>`，核查 Journal ref、DOI、Comments；忽略 `10.48550/` 的 arXiv 自身 DOI。无 arXiv 标识的报告从原始发布页核查。
2. 用题名或标识交叉查 Semantic Scholar、DBLP、Crossref，并以会议论文集或出版者记录确认出处；留意改题名与版本变更。
3. 确认发表后移入 A–D 的唯一主类，基础/评测论文移入 E，删除 K 中原条目；arXiv 可作为开放版本保留。有可核正式研讨会论文集的工作明确标为「研讨会论文」，不得混称主会；技术报告、模型卡不能仅因分配 DOI 就视为正式论文。
4. 未确认正式出处时保留 K，并更新实际完成核对的日期；访问失败导致未完成核对时保留原核对日期，在表格备注中简记「核查受限」。未检索到或缺少 arXiv 出版元数据不等于未发表。

### 检索来源与覆盖

以下为每月按主题检查的固定入口；会议只需核查窗口内已公开的程序、录用目录及论文集，期刊检查新发表与 Early Access 条目。无新目录须说明无更新，不能把未来会议内容当成已发表论文。

| 来源组 | 固定检查入口 |
|---|---|
| 综合发现与身份核对 | arXiv、Semantic Scholar、DBLP、Crossref；搜索引擎补充发现，正式身份回到出版者或会议来源核验 |
| 架构、设计自动化与 FPGA | ISCA、MICRO、HPCA、ASPLOS、DAC、DATE、ICCAD、FPGA、FPL、FCCM、FPT |
| 算子执行、映射与并行计算 | MLSys、PPoPP、CGO、SC、ISC、ICS |
| KV 管理与服务系统 | OSDI、SOSP、USENIX ATC、EuroSys、SIGCOMM |
| 表示、访问策略与评测 | NeurIPS、ICML、ICLR、COLM、AAAI、ACL、EMNLP、NAACL；同时检查相关 Findings 论文集 |
| 架构与硬件期刊 | IEEE TVLSI、TCAD、TC、TPDS；ACM TRETS、TACO；Electronics Letters |

固定入口不是 venue 白名单。综合数据库和引用追踪发现其他来源时，按同一纳入规则核查；直接涉及数字 Attention/KV 数据通路的 ISSCC、VLSI Symposium、ISLPED、JSSC、TCAS、IEEE CAL 等工作可作补充。E 中历史基础文献按需回溯，不要求每月遍查其原期刊；作者代码库、模型报告和技术说明可补充实现信息，须区分论文、技术报告及非论文资源。

- **新稿与修订分开查：** arXiv 分别按首次提交日和最后更新日浏览窗口内条目；旧稿的新版本可更新题名、方法或出处，按同一 arXiv 号去重。K 全表身份检查仍单独执行。
- **按公开记录发现、按证据定身份：** 检查当月新公开的程序、录用目录、Early Access 和出版记录；仅确认录用、尚无正式出版记录的工作暂留 K，标「已录用，待出版」，不得填造卷页。旧 arXiv 稿对应的新正式版本仍需核查，不能被首次提交日过滤掉。
- **处理漏收与延迟：** 新增来源首次回溯 2022 年至本次核对日；延迟收录补查不能只依赖出版年份。未完成的来源/区间留在表格备注中，下一次优先补查。
- **翻页与引用追踪：** 查到覆盖完整时间窗口为止，不以固定前 25/50 条代表全部结果；对核心文献检查参考文献和新增引用，发现换用术语或发表于其他 venue 的直接相关工作。结果截断、验证页或数据库不可用时记明缺口。
- **抽查覆盖：** 用清单各主类的代表作及本轮发现的相关候选，核查至少有一个检索式、正式来源或引用入口可以发现它；实际未检索返回的条目不能仅凭关键词重合记为已召回。此检查只能暴露漏检风险，不能证明文献穷尽。

### arXiv 检索式

A–D、E 使用下表；K 按主题使用同一组查询。已包含 `KV cache`、`KV-cache`、`KVCache`、`key-value cache` 等写法，不以 `cs.AR` 作为所有主题的硬筛选。硬件基础算子允许用 Attention/Transformer 检索，再核查其 decode 适用性。不要预先用训练、GPU、PIM 等否定关键词过滤，以免删掉正文仅提及这些对照的相关论文；范围判断在阅读后完成。

各字段和布尔运算按 [arXiv 官方检索说明](https://info.arxiv.org/help/api/user-manual.html)书写；`all:` 是元数据字段查询，不是全文检索。按最后更新日补查时使用对应排序并核对更新日期，不叠加排除旧稿的首次提交日期条件。网页可使用相应字段与日期控件，不在仓库添加检索脚本。

| 对应主题 | 检索式 |
|---|---|
| A01 表示与精度：量化 | `(all:"KV cache" OR all:"KV-cache" OR all:KVCache OR all:"key-value cache" OR all:"key value cache") AND (all:quantization OR all:quantisation OR all:"low-bit" OR all:"mixed precision" OR all:"mixed-precision")` |
| A02 表示与精度：结构压缩 | `(all:"KV cache" OR all:"KV-cache" OR all:KVCache OR all:"key-value cache" OR all:"key value cache") AND (all:compression OR all:"low-rank" OR all:"low rank" OR all:"cross-layer" OR all:merging OR all:factorization)` |
| A03 表示与精度：共享与潜在表示 | `(all:GQA OR all:MQA OR all:"grouped-query" OR all:"grouped query" OR all:"multi-query" OR all:"latent attention") AND (all:inference OR all:decode OR all:decoding OR all:cache)` |
| A04 表示与精度：误差与校准 | `(all:"KV cache" OR all:"KV-cache" OR all:KVCache OR all:"key-value cache" OR all:"key value cache") AND (all:outlier OR all:rotation OR all:sensitivity OR all:calibration OR all:RoPE OR all:"error correction")` |
| B01 保留与访问：稀疏与驱逐 | `(all:"KV cache" OR all:"KV-cache" OR all:KVCache OR all:"key-value cache" OR all:"key value cache") AND (all:sparse OR all:sparsity OR all:eviction OR all:pruning OR all:retention)` |
| B02 保留与访问：窗口与流式 | `(all:attention OR (all:"KV cache" OR all:"KV-cache" OR all:KVCache OR all:"key-value cache" OR all:"key value cache")) AND (all:streaming OR all:"sliding window" OR all:"sliding-window" OR all:"attention sink" OR all:"attention sinks")` |
| B03 保留与访问：检索与选择 | `(all:"KV cache" OR all:"KV-cache" OR all:KVCache OR all:"key-value cache" OR all:"key value cache") AND (all:retrieval OR all:"page selection" OR all:"token selection" OR all:"query-aware" OR all:"query aware" OR all:"min-max")` |
| B04 保留与访问：预算与复用 | `((all:"KV cache" OR all:"KV-cache" OR all:KVCache OR all:"key-value cache" OR all:"key value cache") OR all:"sparse attention") AND (all:budget OR all:head OR all:"temporal locality" OR all:reuse OR all:importance)` |
| C01 执行架构与映射：硬件 | `(all:attention OR (all:"KV cache" OR all:"KV-cache" OR all:KVCache OR all:"key-value cache" OR all:"key value cache")) AND (all:hardware OR all:accelerator OR all:FPGA OR all:ASIC OR all:NPU OR all:TPU)` |
| C02 执行架构与映射：kernel 与并行 | `all:attention AND (all:kernel OR all:FlashAttention OR all:"flash decoding" OR all:"flash-decoding" OR all:FlashDecoding OR all:FlashInfer OR all:tiling OR all:"split-k" OR all:"split-kv")` |
| C03 执行架构与映射：布局与传输 | `(all:attention OR (all:"KV cache" OR all:"KV-cache" OR all:KVCache OR all:"key-value cache" OR all:"key value cache")) AND (all:layout OR all:packing OR all:bank OR all:burst OR all:alignment OR all:DMA OR all:HBM OR all:channel)` |
| C04 执行架构与映射：选择与归约单元 | `(all:attention OR all:transformer) AND (all:softmax OR all:"top-k" OR all:topk OR all:"top k" OR all:selection) AND (all:hardware OR all:accelerator OR all:kernel OR all:"low precision" OR all:"numerical stability")` |
| C05 执行架构与映射：映射与建模 | `(all:attention OR (all:"KV cache" OR all:"KV-cache" OR all:KVCache OR all:"key-value cache" OR all:"key value cache")) AND (all:dataflow OR all:mapping OR all:scheduling OR all:"design space" OR all:roofline OR all:"performance model" OR all:"energy model")` |
| D01 系统与资源管理：分页与分配 | `all:"paged attention" OR all:PagedAttention OR ((all:"KV cache" OR all:"KV-cache" OR all:KVCache OR all:"key-value cache" OR all:"key value cache") AND (all:"memory allocation" OR all:"memory management" OR all:compaction OR all:fragmentation OR all:"virtual memory"))` |
| D02 系统与资源管理：驻留与传输 | `(all:"KV cache" OR all:"KV-cache" OR all:KVCache OR all:"key-value cache" OR all:"key value cache") AND (all:offloading OR all:offload OR all:prefetch OR all:prefetching OR all:transfer OR all:transmission OR all:"tiered memory")` |
| D03 系统与资源管理：共享与复用 | `((all:"KV cache" OR all:"KV-cache" OR all:KVCache OR all:"key-value cache" OR all:"key value cache") OR all:"prefix cache" OR all:"prefix caching" OR all:RadixAttention) AND (all:reuse OR all:sharing OR all:caching OR all:"cache hit")` |
| D04 系统与资源管理：请求与阶段组织 | `(all:attention OR (all:"KV cache" OR all:"KV-cache" OR all:KVCache OR all:"key-value cache" OR all:"key value cache")) AND (all:batching OR all:disaggregation OR all:disaggregated OR all:"phase splitting" OR all:"prefill-decode" OR all:serving)` |
| E01 基础与评测：长输入质量 | `(all:LLM OR all:LLMs OR all:"language model" OR all:"language models" OR all:transformer) AND (all:"long-context" OR all:"long context" OR all:"long-sequence" OR all:"long sequence") AND (all:benchmark OR all:benchmarking OR all:evaluation OR all:"effective context")` |
| E02 基础与评测：长输出与多轮 | `(all:LLM OR all:LLMs OR all:"language model" OR all:"language models" OR all:transformer) AND (all:"long-form" OR all:"long form" OR all:"long-output" OR all:"long output" OR all:"long generation" OR all:"multi-turn") AND (all:benchmark OR all:benchmarking OR all:evaluation OR all:quality OR all:consistency)` |
| E03 基础与评测：KV 方法可靠性 | `(all:"KV cache" OR all:"KV-cache" OR all:KVCache OR all:"key-value cache" OR all:"key value cache") AND (all:benchmark OR all:evaluation OR all:robustness OR all:"error propagation" OR all:"failure mode")` |
| 补查：KV 同义词宽检索 | `(all:"KV cache" OR all:"KV-cache" OR all:KVCache OR all:"key-value cache" OR all:"key value cache")` |
| 补查：decode Attention | `(all:LLM OR all:LLMs OR all:"language model" OR all:"language models" OR all:transformer) AND all:attention AND (all:decode OR all:decoding OR all:autoregressive) AND (all:inference OR all:efficient OR all:acceleration)` |

去重顺序：arXiv 号，其次 DOI，再次规范化完整题名（使用来源链接提示或 BibTeX 中的题名，不能仅按显示简称去重）。目录中已有条目更新出处、身份或说明，不另起一行；改题名仍按同一工作去重。每篇只在一个主类保留完整书目；跨机制用标签或「见 X 节」表达。K 中预印本仅标主题归属，不在 A–E 重复列出。

### 纳入与排除

以 [项目范围界限](project-scope.md)为最终依据，检索覆盖核心、直接前作与必要边界对照，但不增加项目交付范围。

- **核心与直接比较：** 自回归 decode 的 KV 表示/量化、保留/访问、GQA 共享关系；规则窗口与页检索；量化分组、页元数据、top-k、online softmax、反量化、packing、bank/通道/突发对齐、映射与流水；质量约束下的延迟、带宽及能量建模。
- **基础与评测：** 长输入理解、长输出与多轮生成、KV 压缩/稀疏的误差与可靠性；区分实际质量实验、数值分析和容量外推。不能只用长输入检索评测代表连续生成稳定性。
- **标为扩展或边界：** MLA、动态精度、GPU/TPU kernel、主机卸载、跨请求缓存与 prefill/decode 系统组织；需要训练的结构方法可作给定模型或机制对照，不能当作无需训练的核心方案。Prefill 格式、元数据和转换开销属于接口，纯 prefill 加速不纳入主线。
- **排除主线：** 纯训练、纯编码器；线性 Attention/SSM 等非 KV 驻留架构；投机解码；多模态/视频专用；PIM/CIM/模拟计算；纯权重/激活压缩、MoE 专家调度，以及缺乏直接 Attention/KV 关系的通用编译器、服务系统和存储论文。

每条收录须有可核 URL。禁止编造题名、作者、venue、DOI、arXiv 号；Findings、主会、期刊、研讨会与报告身份分别注明，背景文献不写入「近年进展」。

### 同步目录、BibTeX 与更新记录

1. 将新条目按主要贡献并入 A–D；基础、评测及明确标注的非论文资源入 E；预印本/技术报告仅入 K，并注明主题归属。 文献列统一用「简称／方法名 · 简短记忆点」，不列作者；完整题名保留在首个来源链接的提示中，无通用简称时用描述性短名。
2. 刷新目录文首「更新至 **YYYY-MM-DD**」与各类数量，按正式发表时间插入并检查编号、来源链接、身份标注及全表去重。转正式发表只计为迁移，不计新增；论文与非论文资源分别统计。
3. **同步 `literature-survey.bib`：** 新增条目同时新增书目；剔除、改题名和转正式发表同步修改对应记录。BibTeX 保留完整题名、完整作者、年份及可核 URL；正式论文补齐来源已提供的 venue、DOI、卷期、页码或文章号，保留已核实的 arXiv 开放版本。按来源选择 `@inproceedings`、`@article` 等类型，预印本、技术报告和非论文资源明确标识，不冒充正式论文；来源未提供的字段不推测填入。
4. **同步引用键：** 引用键与表内编号一一对应；排序、迁移或剔除导致编号改变时，先按 DOI/arXiv 号/完整题名建立旧键到新键的映射，再同时更新 BibTeX 和仓库内已有正文引用。不得只对键名机械重排，或让旧引用指向别的工作。仍被引用的剔除项先核对并解决引用归属，未解决前不得删键或宣称同步完成。
5. **完成前核对：** 检查 BibTeX 语法及必需字段，确保无重复键、无同一工作的重复版本；全表编号集合、条目数、完整题名、年份、身份与标识逐项对应，现有引用无悬空或错配。同步刷新 BibTeX 文件头的日期与编号范围。访问失败、来源冲突或暂不能核实的字段保留待核标记和来源说明，不写成已验证。
6. 在本文「更新记录」**只追加一行表格**，简记窗口、新增、身份变化与必要备注，并注明「BibTeX 已同步」或具体未解决项；不再追加分节说明、逐篇新增/剔除清单或检索过程。无变化也须检查对应关系，写「无新收录 / 预印本状态未变；BibTeX 一致」。

---

## 更新记录

仅用下表记录每次更新；必要的分类调整、剔除数量和未完成核查简记在「备注」。范围审查只记为范围审查，不代表完成当月检索或 K 节重核；未完成的回溯缺口在完成前继续保留。

| 月份 | 检索窗口 | 新增 | 预印本转正式发表 | 仍按预印本 / 技术报告列出 | 核对日期 | 备注 |
|---|---|---|---|---|---|---|
| 2026-09 | 目录初建及 K 节核对 | 初建目录 | 无 | 4 篇 | 2026-09-21 | — |
| 2026-09 | 2026-01-01—2026-09-21；回溯 2024–2026 正式论文集 | 14 篇：正式 10、预印本 4 | 无 | 8 篇：预印本 7、技术报告 1 | 2026-09-21 | 重分类去重；退出 13 篇；整理资源 1 项；Semantic Scholar 补查无结果 |
| 2026-09 | 无新检索 | 无 | 无 | 8 篇：预印本 7、技术报告 1 | 2026-09-21 | A–E、K 各表按正式发表时间升序重排并重编号 |
| 2026-09（范围审查） | 范围与来源抽查，非完整月度检索 | 未补录 | 未重核 | 未重核 | 2026-09-21 | 扩充来源、同义词与底层执行/长输出检索；新增来源待补查 2022 年至本日 |
| 2026-09（展示调整） | 无新检索 | 无 | 未重核 | 8 篇，未重核 | 2026-09-21 | 统一为简称＋记忆点；保留完整题名与来源；不推进检索窗口 |
| 2026-09（书目核查） | 现有 66 条书目核验，非新增检索 | 无 | 未做全面转刊复核 | 8 篇，保留原身份 | 2026-09-21 | 修补 9 条书目；BibTeX 与目录一致；月度流程强制同步；新增来源回溯仍待补 |
