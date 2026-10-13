# 第一阶段正式结果格式

## 测量身份与复用

四个实验共享 `phase1.results` 的正式实体目录与审核回执，以及 `phase1.reuse` 的
测量身份判定。测量身份由模型与权重来源、冻结全文与
分词、任务和统计单位、实际量化配置与层表、残留、输出预算、数值实现及执行软件
共同确定。源码文件校验用于追溯；仅迁移函数、整理导入或修改说明，不使已有测量
失效。计算实现的兼容核验保留执行语句、常量和参数，设备与软件差异另行检查。

运行前分别记录 `reuse`（使用已完成实体）、`append`（只补缺少的冻结前缀）、
`measure`（尚无测量）、`review`（来源不足或执行条件需要核查）、`remeasure`
（已确认测量身份不同）。`review` 不自动转为重跑；先查原发布快照、输入、设备和
原始输出。确认不同的输入或配置保留独立来源，不按同题号合并。

完整只读审核通过后，`source_index.json/reuse_review` 绑定当前正式实体索引、
共享容量输入、实际计算实现和审核依赖。四个包的已完成结果及真实生成的单个条件
均由同一回执决定复用；实体、输入、模型清单、配置或实现变化时先核验，不能只按
文件存在或行数足够判断复用。执行入口和说明编辑不触发模型重跑。回执为当前整理与
分析的核验依据；每份测量的真实执行提交、设备、源码段和原始输出来源仍保留原值。

共同 BF16、KV8、K8V4 首轮的唯一维护位置仍是 `generation_error_sources/results/`。
静态收尾的参考追加行归入 `static_layer_allocation/results/reference_expansion/`，
以 `reference_source` 对应原实体和校验。已完成的 K8V4 追加测量保留并可复用；
后续调度不为与候选同步而强制追加 K8V4。候选累计实体放在 `final_quality/`。
参考的拼接统一由 `phase1.results.reference_view` 完成，核对首轮校验、模型、任务、
条件、逐题输入身份和追加行数；不复制首轮到消费者目录。主矩阵 C 容量输入的最小
事实统一放在 `source_index.json/shared_capacity_inputs`，保留原执行来源与校验。
先前已经清理的额外测量沿用历史索引，不因目录整理重新生成。

`reference_reuse.json` 逐份保留实际核验依据；`reference_cleanup.json` 记录归并前
实体的原校验、服务器原字节归档、同输入得分与长度核对及正式归属。历史未知停止
原因仍为 null，不从重测推填。完整预测实体包含多轮测量时，`raw_selection.ids`
明确当前结果引用的行；原始完整实体与校验继续保留。

同一累计测量跨兼容源码恢复时，以 `source_segments` 的半开行范围记录各段实际
发布清单。审核逐段核对原快照和计算实现，不把旧行标成新源码测量。新增源码使用
`research/.runtime/phase1-execution/current.json` 指向的快照，正式结果来源仍追溯
实际测量的旧或新快照。结构整理不生成实验报告。

## 实验结果格式

静态层分配新测量使用 schema 为 key-downgrade-allocation 的 probe.json、
allocations.json 和 gate.json，仍放各模型原有结果目录。旧模式的 K/V 探针与
p25/p50 分配保留其实际执行来源，不能转换字段后当作新测量。

probe.json 按三格式分别保存逐题、逐层的 sensitivity、reconstruction、k8v4_error、
k4v4_error，均为无量纲相对平方误差。分母和汇总顺序见指导；null 不填零，
保留零能量组、无效量化组与各指标有效题数。reconstruction_location 明确旋转后
主存、残留替换前的口径。rankings_descending 保存高到低的排名，分配代码对原指标
重新按低到高排序，固定并列层序。rank_agreement 登记相关性和高敏感集合重合。

allocations.json 记录每格式、预算的六策略、实际 K4 层名单、固定随机排列、
INT8 数量与比例、完整成本和重复配置别名。参数、padding、残留容量、prefill 转换、
decode 读取两种界及主存／残留写入以字节分列；pre-RoPE 操作量单列，来源为布局计数。
别名只在同格式、同残留下复用。低位宽主存名单不含近期副本。

gate.json 的 KL 单位为 nat，全部对照统一使用 FP64 稳定计算。每题保留同输入身份、
共同首 token、KL 和无效量化组；无效题不删除，相关均值记 null。
mean_next_token_kl 分数学与检索；budget_comparisons 按格式、预算登记三个比较判断；
paired_differences 保存相对同格式 K8V4 及策略间的逐题差值、方向、最大三题贡献。
budget_trends 按 K4 层数登记均值及与上一预算的变化；existing_quality 只承接
同格式、同 H 的已有质量状态及来源，未测单元保留未测。
题目为独立单位；层、head 与生成位置不增加样本数。全部预算保留，
quality_threshold 为 null，不生成 V 选择名单。

三文件均登记配置、实际工作区代码、冻结输入／模板／模型清单与离线参数来源，
同时保存基准提交和实际校验。恢复运行要求输入、代码、校准与配置一致。
新结果完成回传及审核后更新 source_index.json；修订代码不改写旧结果的来源记录。

每个实验只有一个 `results/`。逐题质量 JSON 使用 `result_format_version: 1`，由
`phase1/result_format.py` 写入；旧测量整理时补充相同字段，既有分数、协议标识和
`commit` 保留实际测量来源，不把历史提交改成补测提交。

`n` 是本文件的逐题行数。扩样文件另外记录 `extra_n` 与 `total_n_with_v`，只存新增题；
同条件只保留最大的已完成扩样实体，64→128→256 的判定从其冻结前缀重建。
`selection.json` 的 `expansion_history` 保留各轮计划。归并前必须核对配置摘要和逐行值。

质量汇总 `em`、`f1` 和下降区间的单位均为 0–100 分；`correct` 汇总为答对题数。
逐题 `correct`、`em` 为布尔值，逐题 `f1` 为 0–100 分，token 与无效组计数为整数。
每行包含题号、输入 token 数、全文 SHA256、实际生成 token 数、停止原因和无效组数。
高精度结果无量化无效组，计为 0；历史未记录的停止原因使用 `null`，不能由长度猜测。

四问联答 `v_hotpot_four_bf16.json` 的每行是一个请求，`n` 为请求数，
`n_questions` 为其包含的问题数，`statistical_unit` 为 `request`。
每行 `answer_f1`、`answer_em`、`extracted_answers` 各保存四题记录；`f1` 是四题均值。
每行 `em` 表示四题全部正确，汇总 `em` 是全部正确的请求比例；汇总 `answer_em`
是请求内题级 EM 先平均再平均请求。长度准入、请求均值和缺失答案规则见
`long_decode.json`。历史单问长解码保留原行含义，不与四问请求拼接或直接配对。

`new_input_quality.json` 比较新检索与四问联答上的三个 K8V4 工作点、同输入 KV8，
以及离线 pre 的 H=0。检索行仍是题，四问行仍是请求。汇总同时给出相对 BF16、KV8 的下降、
类通过和状态表；离线 pre 另给 H=8 减 H=0 的配对变化。本文件不授予扩样后的保留。
旧 `v_retrieval_bf16.json` 与历史单问长解码不参与这些配对。

`input_identity_source` 区分 `recorded`、`reconstructed_from_frozen_inputs` 和 `unrecorded`。
重建输入须核验冻结正文、模板和 tokenizer；已有哈希发生冲突时停止整理。
`calibration_commit` 不适用时为 `null`；H=0 的 `residual_bits`、`residual_tokens` 都为 0，
H=8/16 的副本长度为冻结的 128 token。字段统一不改变旧 `bf16-h0` 或 `e2`／`e3-*`
的实际历史身份；文件按研究内容命名，读取入口明确映射这些身份。

主矩阵与配对传播维持其已有模式，布局保留字节、秒和十进制 TB/s，不能将质量评分、
局部误差、容量或假设带宽相混。`source_index.json` 登记规范实体、原始校验、实际提交、
归并前缀及审核事实；运行快照在服务器 `research/.runtime/` 保留，不建立第二套正式结果。

`generation_error_sources/results/<model>/trajectories.json` 使用 `kv-propagation-v1`，
保留冻结病例身份、输入校验、源码发布清单校验、同次 BF16 分数和三个共同候选。
`steps` 记录完整 BF16 token 回放的逐步 KL、argmax、预测间隔和末层相对平方误差。
`layer_mean_rse` 只保留层末隐藏状态的相对平方误差，先在同题生成步内等权平均。
`history_reset` 是首次分歧位置的同格式 BF16 来源历史重建对照；没有分歧时为 null。
生成位置从 0 计数，prefill 给出位置 0 的共同首 token，decode 步 0 预测位置 1。

`paired_free_generation` 保存同次自由生成的分数、目标减 BF16 的差值、长度与停止原因。
GSM8K 使用 0/100 的 EM，HotpotQA 使用 0–100 的 token-F1。`free_quality` 引用用于病例
选择的历史评分，`historical_bf16_score_length_match` 明确其与当次参考分数及长度是否一致。
历史主矩阵不因诊断重测而覆盖。完整预测和层步矩阵仅存服务器，由 `raw_source` 的项目相对
路径与 SHA256 追溯。`trajectory_review.json` 集中保存八题题面与标准答案、完整预测重新评分、来源审核和题间等权汇总。

## 静态分配独立质量收尾记录

三套冻结候选的累计实体归入静态包 `results/final_quality/<模型>/`，任务与条件构成
唯一文件名；每次追加沿用原前缀。逐题或请求记录输入身份、全文校验、实际分词长度、
生成长度、停止原因、无效量化、评分、缺失答案、输出校验和实际峰值。完整预测、
生成 token 及输入留在服务器；累计实体记录其来源、设备、数值软件和实际源码清单。

已有参考的唯一首轮实体保留在真实生成包。静态包 `results/reference_expansion/<模型>/`
只保存 BF16、KV8 尚未测过的追加行，以 `reference_source` 追溯首轮、以
`total_n_with_reference` 记录累计范围；K8V4 只在已有共同样本上比较。
`results/reference_reuse.json` 逐份登记复用或补测决定及理由。四问先平均请求内四题，
只按 BF16 长度准入子集配对。`results/expanded_quality/<模型>/selection.json` 保存
每任务实际样本数、两参考下降及区间、质量状态、下一档扩样、停止原因和档位历史；
候选收敛停止不改记淘汰。单问与四问保持不同的统计单位和结果实体。

收尾审核从完整预测重算评分、输入配对与区间，并核对预先冻结层表、扩样前缀、
运行准入和所有提前停止理由。来源索引保留实际执行根与源码校验；成本包依赖该
选择实体和实际层表，保留质量失败端点并区分布局计数、服务模型与实测生成。

## 成本接口与执行模型记录

成本包的两份 `layout.json` 和 `review.json` 保留已审核的 73 行容量、有效读写与
带宽下界，不因新增接口分析重生成。`service.json` 引用其校验值，在 49 个 decode
工作点上保存两种假设接口的 98 行传输及 49 行操作、系数和片上缓冲记录。
每行单列 K/V 载荷、参数、追加写入、读改写、尾部浪费与缓冲供读；B=4 同时保存
整批每步及每生成 token 字节。prefill 转换与 decode 分开，128K 不新增 decode。

服务模型记录逐层有限流水及所需吞吐。缺少实测吞吐时，延迟字段保持空值；稳态
资源门槛与可加开销预算不作为实测时间。质量选择及其停止状态直接复用，审核元数据
更新时核对实体校验和选择内容，并记录历史、当前审核依赖，不改写原布局的来源。
`service_review.json` 保存独立周期公式、来源与质量一致性、边界和依赖回归结果。
本轮纯 CPU 计数按用户明确授权在本地执行，记录实际运行环境与逐文件来源。
