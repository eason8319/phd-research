# 第一阶段执行提示

本文件只规定代理如何把第一阶段跑完。科学决定、实验顺序、通过条件和停止条件以 [guide.md](guide.md) 与 [configs/](configs/) 为准，协议版本以 guide.md 所记为准。本文件与那些文档冲突时，以那些文档为准。

下次开跑时，把下面这句话作为任务交给执行代理：

> 按 `research/phase1/run-prompt.md` 执行第一阶段。先占住同一张约 48 GiB 的 GPU，再按当前文档从 M0 做到结果回传。中途不释放这张卡。

## 读什么

开始任何写入、同步或占卡之前，先读完：

| 文件 | 用途 |
|---|---|
| `research/phase1/guide.md` | 顺序、做法、记录、通过和停止 |
| `research/phase1/configs/runtime.md` | 模型、软件、资源、准入清单 |
| `research/phase1/configs/data.md` | 生成、划分、评分、基线下限 |
| `research/phase1/configs/quant.md` | 表示、数值、容量与传输 |
| `research/phase1/configs/quality.md` | 下降、区间、保留 / 淘汰 / 证据不足 |
| `research/phase1/pending-review.md` | 各版审核决定写在哪里 |
| `.cursor/rules/experiment-management.mdc` | 代码位置、结果回传、一次性测试 |
| `.cursor/rules/experiment-code.mdc` | 代码与注释 |
| `docs/server-access.md` | 唯一的 SSH 连接方式 |
| `AGENTS.md` | 工作区隔离与同步边界 |

`docs/experiments/phase1-kv-precision.md` 是更早的计划稿。执行时以 `research/phase1/` 的当前文档为准。

文档里没有写明的决定，执行中不补做选择。跑到必须用那项决定才能继续的步骤时，把该步骤记为阻塞，其余已经写明的步骤继续。

看过结果之后不改协议：不缩小样本，不放宽质量容差或基线下限，不把失败配置扩展到 INT2、训练或更多模型，不升协议版本。要改配置才能继续时，停止并报告，由人改文档。

## 做到哪里停

按 guide.md 的执行顺序做完：M0 → 数据清单与 `splits/` → E0 → E1 → E2 → E3 → F → E4。每一步内先 Qwen，再 Llama。每一步的通过条件满足后，立刻开始下一步。数据集按 data.md 下载到服务器 `research/.runtime/data/`；模型权重不重新下载。

正式结果整理后覆盖回传到本地 `research/phase1/results/<实验名>/`。回传完成即本轮结束。运行过程不写报告，不生成图片。`reports/` 等审核通过后再写。

下列情况停止本轮，并留下已完成步骤、失败命令、日志路径和阻塞原因：

- guide.md 停止表要求停止，且无法在不改协议的前提下消除原因。E0 未通过时先修实现；修好后从 E0 继续。修不好则停止后续质量结论。
- 准入清单、样本划分或未决事项使下一步无法开始，而又不能在现有文档内补齐。
- 分配到的 GPU 经 `nvidia-smi` 核对不是约 48 GiB。
- 原卡丢失后无法占回同一张卡。
- 项目目录进不去，或 SSH 按 `docs/server-access.md` 失败且重试仍失败。

普通实现错误、可恢复的作业失败、以及 guide.md 允许的减临时张量、分块或少存轨迹，都在原卡上修好后继续，不因此结束本轮。

## 服务器与代码

本地 `F:\phd-research` 是代码、配置和指导文档的维护处。计算在 SSH 主机 `myserver` 的 `/cluster/home/zengy/phd-research`。

每条 SSH 都使用 `docs/server-access.md` 里的命令，远程命令以 `cd /cluster/home/zengy/phd-research || exit 1` 开头。不新增连接脚本，不读取私钥。服务器当前不是 Git 仓库。同步时用同一套 SSH 参数做 `scp`，把本轮需要的代码、配置、指导拷到服务器项目内；`splits/` 回传本地之后，也以本地为准同步。排除 `models/` 与 `research/.runtime/`。服务器上已有的 `models/` 保持不动，权重只从这两项目录加载：

- `models/Llama-3.1-8B-Instruct`
- `models/Qwen2.5-7B-Instruct`

登录节点的 Python 3.6.8 不跑实验。环境、缓存、占卡状态和一次性测试都放在服务器 `research/.runtime/`，不回传，不进入 Git。

代码放在 `research/src/kvstudy/` 与 `research/phase1/src/phase1/`，并遵守 experiment-code.mdc。E0 的长期回归写入 `research/tests/`。

用户已授权本轮在本地提交 Git：每次把代码同步到服务器之前提交一次，结果登记这个提交号；`splits/`、`research/manifests/`、环境锁和正式结果回传本地之后也各提交一次。只提交本仓库，不推送 GitHub，不改写已有提交。

## 占住同一张 48 GiB 卡

2026-09-23 在本集群核对过：分区 `gpu` 的时限是 unlimited，`OverSubscribe=NO`。`node8` 上有两张 NVIDIA RTX 6000 Ada，`nvidia-smi` 的 `memory.total` 为 49140 MiB。`node6`、`node7` 上能查询到的是 RTX 4090，24564 MiB。分区 `Jupyter` 与 `gpu` 共用这些节点，实验作业提交到 `gpu`。

开跑前先占卡，再写后续计算。整个 M0 到结果回传期间维持这一份分配，写代码和步骤之间的空隙也占着。全部正式计算和需要 GPU 的一次性测试都在这份分配里跑。

提交一份 Slurm 作业，资源按 runtime.md 的初始申请：

```text
sbatch -p gpu -w node8 --gres=gpu:1 -c 8 --mem=64G
```

不设置会中途杀掉作业的时限。作业脚本放在服务器 `research/.runtime/hold/`，不放入 Git。脚本进入项目目录后立刻做这几件事：

1. 用 `nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader` 核对 Slurm 分配到的那张卡。节点未按 cgroup 隔离设备、`nvidia-smi` 列出多张卡时，只核对 `CUDA_VISIBLE_DEVICES` 指向的那一张。`memory.total` 须为 49140 MiB 这一档。对不上则马上退出，不在该卡上加载模型。
2. 把作业号、节点、GPU index、UUID、名称、显存写入 `research/.runtime/hold/state.txt`。这张卡就是本轮唯一的计算卡。
3. 进入等待循环：看到 `research/.runtime/hold/step.cmd` 时，先把它改名为 `step.running` 再执行，避免同一条命令被执行两次；把标准输出和标准错误写入 `research/.runtime/hold/step.log`，把退出码写入 `research/.runtime/hold/step.status`，然后回到等待。没有命令时每 10 秒看一次。循环本身不释放 GPU。

`state.txt` 里已有未结束的作业时，接上那份作业，不再提交第二份 GPU 作业。主机内存不够而进程被杀时，只提高 `--mem`，GPU 数量保持 1，并在 M0 记录实际申请。仍遵守 guide.md 的显存停止规则：减临时张量、分块或轨迹，不缩短上下文，不减少样本。

原作业异常退出时，先按 `state.txt` 的 UUID 重新占同一张卡，占到之后再跑下一步。占到的是另一张卡时立刻退出并重试。同一张卡无法占回时停止本轮。24 GiB 卡不作为替代。

结果已经覆盖回传到本地，或本轮已按上一节确认无法继续时，结束等待循环，确认 `squeue` 里这份作业已消失，并在 `state.txt` 写下释放时间。

## 等待计算时

GPU 步骤一旦提交到上述循环，监控就是当前任务，直到该步骤写出 `step.status`。

- 用 `docs/server-access.md` 的 SSH 查看作业是否仍在、`step.status` 是否出现、`step.log` 的末尾。
- 预计超过 10 分钟的步骤，每 60 秒查一次。预计更短的步骤每 20 秒查一次。
- 用实测进度决定下一次查询，不用 runtime.md 的时间估算公式当作睡眠时长。
- `step.status` 出现后，先读退出码和日志里的失败原因，再写下一份 `step.cmd`。中间不插入报告、作图或与本轮无关的修改。
- 监控用的 SSH 失败时，按 server-access.md 重试。查询失败不作为释放 GPU 的理由。
- 本次对话若必须结束，保留服务器上的占卡作业和 `state.txt`。下次从该文件恢复，不另占一张卡。

## 一次性测试

guide.md 要求写入 `results/` 的运行是正式结果，保留并回传。

正式协议以外的试探只写在服务器 `research/.runtime/scratch/`。一次试探结束，确认正式结果没有依赖这些文件之后，立刻删除该次试探的目录。不回传，不放进 `phaseN/`，不进入 Git。本轮释放 GPU 前，`scratch/` 应为空。环境目录 `research/.runtime/env/` 与占卡目录 `research/.runtime/hold/` 保留到本轮结束。

## 结果

每个实验按 guide.md 写入服务器上的 `research/phase1/results/<实验名>/`，再覆盖回传到本地同路径。同一实验重跑时覆盖整个目录。

完整 prompt、完整预测文本、KV 和激活留在服务器。单文件超过 10 MiB 的实体不回传；结果索引写项目相对路径和校验值。回传不改写 `reports/`。
