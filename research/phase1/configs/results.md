# 第一阶段正式结果格式

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
`layer_mean_rse` 先在同题生成步内等权平均；`layer_snapshots` 保留首步、末步及首次分歧层数据。
`history_reset` 是首次分歧位置的同格式 BF16 来源历史重建对照；没有分歧时为 null。
生成位置从 0 计数，prefill 给出位置 0 的共同首 token，decode 步 0 预测位置 1。

`paired_free_generation` 保存同次自由生成的分数、目标减 BF16 的差值、长度与停止原因。
GSM8K 使用 0/100 的 EM，HotpotQA 使用 0–100 的 token-F1。`free_quality` 引用用于病例
选择的历史评分，`historical_bf16_score_length_match` 明确其与当次参考分数及长度是否一致。
历史主矩阵不因诊断重测而覆盖。完整预测和层步矩阵仅存服务器，由 `raw_source` 的项目相对
路径与 SHA256 追溯。人工答案核验仍与自动评分分开保存。
